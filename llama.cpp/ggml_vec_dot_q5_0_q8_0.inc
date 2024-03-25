static void ggml_vec_dot_q5_0_q8_0(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(qk == QK5_0);

    const block_q5_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q5_0 * restrict x1 = &x[i + 1];
        const block_q8_0 * restrict y0 = &y[i];
        const block_q8_0 * restrict y1 = &y[i + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        // extract the 5th bit via lookup table ((!b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_1[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint32_t qh;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    for (int i = 0; i < nb; ++i) {
        const block_q5_0 * restrict x0 = &x[i];
        const block_q8_0 * restrict y0 = &y[i];

        const v128_t m4b  = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_1[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_1[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_1[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_1[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const v128_t v0lf = wasm_i8x16_sub(v0l, qhl);
        const v128_t v0hf = wasm_i8x16_sub(v0h, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_add(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d))));
    }

    *s = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
         wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32_avx2(x[i].qs);
        __m256i bxhi = bytes_from_bits_32_avx2(x[i].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        bx = _mm256_or_si256(bx, bxhi);

        __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx2(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8((char)0xF0);

    // Main loop
    for (int i = 0; i < nb; i++) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        __m256i bx = bytes_from_nibbles_32_avx(x[i].qs);
        const __m256i bxhi = bytes_from_bits_32_avx(x[i].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx);
        __m128i bxh = _mm256_extractf128_si256(bx, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx = MM256_SET_M128I(bxh, bxl);

        const __m256i by = _mm256_loadu_si256((const __m256i *)y[i].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx(bx, by);

        /* Multiply q with scale and accumulate */
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    uint32_t qh;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    // These tempory registers are for masking and shift operations
    vuint32m2_t vt_1 = __riscv_vid_v_u32m2(vl);
    vuint32m2_t vt_2 = __riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2(1, vl), vt_1, vl);

    vuint32m2_t vt_3 = __riscv_vsll_vx_u32m2(vt_2, 16, vl);
    vuint32m2_t vt_4 = __riscv_vadd_vx_u32m2(vt_1, 12, vl);

    for (int i = 0; i < nb; i++) {
        memcpy(&qh, x[i].qh, sizeof(uint32_t));

        // ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
        vuint32m2_t xha_0 = __riscv_vand_vx_u32m2(vt_2, qh, vl);
        vuint32m2_t xhr_0 = __riscv_vsrl_vv_u32m2(xha_0, vt_1, vl);
        vuint32m2_t xhl_0 = __riscv_vsll_vx_u32m2(xhr_0, 4, vl);

        // ((qh & (1u << (j + 16))) >> (j + 12));
        vuint32m2_t xha_1 = __riscv_vand_vx_u32m2(vt_3, qh, vl);
        vuint32m2_t xhl_1 = __riscv_vsrl_vv_u32m2(xha_1, vt_4, vl);

        // narrowing
        vuint16m1_t xhc_0 = __riscv_vncvt_x_x_w_u16m1(xhl_0, vl);
        vuint8mf2_t xh_0 = __riscv_vncvt_x_x_w_u8mf2(xhc_0, vl);

        vuint16m1_t xhc_1 = __riscv_vncvt_x_x_w_u16m1(xhl_1, vl);
        vuint8mf2_t xh_1 = __riscv_vncvt_x_x_w_u8mf2(xhc_1, vl);

        // load
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        vuint8mf2_t x_at = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_lt = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vuint8mf2_t x_a = __riscv_vor_vv_u8mf2(x_at, xh_0, vl);
        vuint8mf2_t x_l = __riscv_vor_vv_u8mf2(x_lt, xh_1, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 16, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 16, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d)) * sumi;
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            sumi += (x0 * y[i].qs[j]) + (x1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*GGML_FP16_TO_FP32(y[i].d)) * sumi;
    }

    *s = sumf;
#endif
}
