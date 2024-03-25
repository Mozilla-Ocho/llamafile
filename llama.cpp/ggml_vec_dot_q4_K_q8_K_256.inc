static void ggml_vec_dot_q4_K_q8_K(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __riscv_v_intrinsic

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        size_t vl = 8;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
        vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
        vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
        vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
        vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        vl = 32;

        int32_t sum_1 = 0;
        int32_t sum_2 = 0;

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        for (int j = 0; j < QK_K/64; ++j) {
            // load Q4
            vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

            // load Q8 and multiply it with lower Q4 nibble
            vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
            vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
            vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

            sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

            // load Q8 and multiply it with upper Q4 nibble
            vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
            vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
            vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
            vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

            sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

            q4 += 32;    q8 += 64;

        }

        sumf += d*(sum_1 + sum_2);

    }

    *s = sumf;

#else


    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}
