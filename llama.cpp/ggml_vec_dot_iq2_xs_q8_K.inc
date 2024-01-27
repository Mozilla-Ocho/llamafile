static void ggml_vec_dot_iq2_xs_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_iq2_xs * restrict x = vx;
    const block_q8_K   * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    int32x4x4_t scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        const uint8x8_t scales8 = vld1_u8(x[i].scales);
        const uint8x8_t scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        const uint8x8_t scales_h = vshr_n_u8(scales8, 4);
        uint8x16_t scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        const uint16x8_t scales1 = vmovl_u8(vget_low_u8(scales));
        const uint16x8_t scales2 = vmovl_u8(vget_high_u8(scales));
        scales32.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        scales32.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        scales32.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        scales32.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[0] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[1] & 511))));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[2] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[3] & 511))));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[4] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[5] & 511))));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[6] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[7] & 511))));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[0] >> 9))), vld1_s8((const void *)(signs64 + (q2[1] >> 9))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[2] >> 9))), vld1_s8((const void *)(signs64 + (q2[3] >> 9))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[4] >> 9))), vld1_s8((const void *)(signs64 + (q2[5] >> 9))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[6] >> 9))), vld1_s8((const void *)(signs64 + (q2[7] >> 9))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[3], q8b.val[3]);
            const int32x4_t p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));
            sumi = vmlaq_s32(sumi, p, scales32.val[ib64]);
            q2 += 8;
        }
        sumf += d*vaddvq_s32(sumi);
    }
    *s = 0.125f * sumf;

#elif defined(__AVX2__)

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m127 = _mm_set1_epi16(127);

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m128i aux_gindex, aux_sindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;
    const uint16_t * sindex = (const uint16_t *)&aux_sindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32__F16C(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m128i q2_data = _mm_loadu_si128((const __m128i*)q2);  q2 +=  8;
            aux_gindex = _mm_and_si128(q2_data, m511);
            aux_sindex = _mm_and_si128(_mm_srli_epi16(q2_data, 9), m127);
            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]], iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]], iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[sindex[3]], signs64[sindex[2]], signs64[sindex[1]], signs64[sindex[0]]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[sindex[7]], signs64[sindex[6]], signs64[sindex[5]], signs64[sindex[4]]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const uint8_t  * restrict sc = x[i].scales;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}
