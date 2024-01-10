static void ggml_vec_dot_iq2_xxs_q8_K(const int n, float * restrict s, const void * restrict vx, const void * restrict vy) {
    assert(n % QK_K == 0);

    const block_iq2_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 0])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 1])));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 2])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 3])));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 8])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 9])));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[10])), vld1_s8((const void *)(iq2xxs_grid + aux8[11])));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >>  7) & 127))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >> 21) & 127))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]), q2u.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]), q2u.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[1] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[3] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.25f * sumf;

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32__F16C(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}
