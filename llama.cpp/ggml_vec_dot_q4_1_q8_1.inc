static void ggml_vec_dot_q4_1_q8_1(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);

    const block_q4_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

    // TODO: add WASM SIMD
#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    assert(nb % 2 == 0); // TODO: handle odd nb

    for (int i = 0; i < nb; i += 2) {
        const block_q4_1 * restrict x0 = &x[i + 0];
        const block_q4_1 * restrict x1 = &x[i + 1];
        const block_q8_1 * restrict y0 = &y[i + 0];
        const block_q8_1 * restrict y1 = &y[i + 1];

        summs += GGML_FP16_TO_FP32(x0->m) * y0->s + GGML_FP16_TO_FP32(x1->m) * y1->s;

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*y0->d);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*y1->d);
    }

    *s = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    // Main loop
    for (int i = 0; i < nb; ++i) {
#ifdef __AVX2__
        const float d0 = GGML_FP16_TO_FP32__F16C(x[i].d);
#else
        const float d0 = GGML_FP16_TO_FP32(x[i].d);
#endif
        const float d1 = y[i].d;

#ifdef __AVX2__
        summs += GGML_FP16_TO_FP32__F16C(x[i].m) * y[i].s;
#else
        summs += GGML_FP16_TO_FP32(x[i].m) * y[i].s;
#endif

        const __m256 d0v = _mm256_set1_ps( d0 );
        const __m256 d1v = _mm256_set1_ps( d1 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
#if defined(__AVX2__)
        const __m256i bx = bytes_from_nibbles_32_avx2(x[i].qs);
#else
        const __m256i bx = bytes_from_nibbles_32_avx(x[i].qs);
#endif
        const __m256i by = _mm256_loadu_si256( (const __m256i *)y[i].qs );

#if defined(__AVX2__)
        const __m256 xy = mul_sum_us8_pairs_float_avx2(bx, by);
#else
        const __m256 xy = mul_sum_us8_pairs_float_avx(bx, by);
#endif

        // Accumulate d0*d1*x*y
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
#else
        acc = _mm256_add_ps( _mm256_mul_ps( d0d1, xy ), acc );
#endif
    }

    *s = hsum_float_8(acc) + summs;
#elif defined(__riscv_v_intrinsic)
    float sumf = 0.0;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (int i = 0; i < nb; i++) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[i].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[i].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[i].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t v0 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t v1 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#else
    // scalar
    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        int sumi = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[i].qs[j] & 0x0F);
            const int v1 = (x[i].qs[j] >>   4);

            sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk/2]);
        }

        sumf += (GGML_FP16_TO_FP32(x[i].d)*y[i].d)*sumi + GGML_FP16_TO_FP32(x[i].m)*y[i].s;
    }

    *s = sumf;
#endif
}
