#include <cosmo.h>
#include <sys/auxv.h>
#include <libc/sysv/consts/hwcap.h>
#include "ggml-quants.h"

extern "C" void quantize_row_q4_0_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_amd_avx512(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_amd_avx2(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_amd_avx(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_amd_k8(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_ref_arm80(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q4_1_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_amd_avx512(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_amd_avx2(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_amd_avx(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_amd_k8(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_ref_arm80(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_0_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_amd_avx512(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_amd_avx2(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_amd_avx(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_amd_k8(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_ref_arm80(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_1_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_amd_avx512(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_amd_avx2(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_amd_avx(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_amd_k8(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_ref_arm80(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_0_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_amd_avx512(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_amd_avx2(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_amd_avx(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_amd_k8(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_ref_arm80(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_1_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_amd_avx512(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_amd_avx2(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_amd_avx(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_amd_k8(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_ref_arm80(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q2_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_ref_arm80(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q3_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_ref_arm80(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q4_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_ref_arm80(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_ref_arm80(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q6_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_ref_arm80(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_K_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_amd_avx512(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_amd_avx2(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_amd_avx(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_amd_ssse3(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_amd_k8(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_ref_arm80(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq3_xxs_ref_amd_avx512vl(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_amd_avx512(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_amd_avx2(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_amd_avx(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_amd_ssse3(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_amd_k8(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_ref_arm80(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq4_nl_ref_amd_avx512vl (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_amd_avx512 (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_amd_avx2 (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_amd_avx (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_amd_ssse3 (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_amd_k8 (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_ref_arm80 (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq4_xs_ref_amd_avx512vl (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_amd_avx512 (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_amd_avx2 (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_amd_avx (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_amd_ssse3 (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_amd_k8 (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_ref_arm80 (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq3_s_ref_amd_avx512vl  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_amd_avx512  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_amd_avx2  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_amd_avx  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_amd_ssse3  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_amd_k8  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_ref_arm80  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq2_s_ref_amd_avx512vl  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_amd_avx512  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_amd_avx2  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_amd_avx  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_amd_ssse3  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_amd_k8  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_ref_arm80  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q4_0_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_0_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q4_1_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_1_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_0_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_0_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_1_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_1_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_0_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_0_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_1_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_1_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q2_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q2_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q3_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q3_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q4_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q4_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q5_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q5_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q6_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q6_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_q8_K_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_q8_K_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq3_xxs_amd_avx512vl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_amd_avx512(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_amd_avx2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_amd_avx(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_amd_ssse3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_amd_k8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_xxs_arm80(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq4_nl_amd_avx512vl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_amd_avx512 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_amd_avx2 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_amd_avx (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_amd_ssse3 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_amd_k8 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_nl_arm80 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq4_xs_amd_avx512vl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_amd_avx512 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_amd_avx2 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_amd_avx (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_amd_ssse3 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_amd_k8 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq4_xs_arm80 (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq3_s_amd_avx512vl  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_amd_avx512  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_amd_avx2  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_amd_avx  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_amd_ssse3  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_amd_k8  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq3_s_arm80  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void quantize_row_iq2_s_amd_avx512vl  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_amd_avx512  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_amd_avx2  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_amd_avx  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_amd_ssse3  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_amd_k8  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
extern "C" void quantize_row_iq2_s_arm80  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q4_0_amd_avx512vl(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_amd_avx512(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_amd_avx2(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_amd_avx(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_amd_ssse3(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_amd_k8(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_0_arm80(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q4_1_amd_avx512vl(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_amd_avx512(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_amd_avx2(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_amd_avx(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_amd_ssse3(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_amd_k8(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_1_arm80(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q5_0_amd_avx512vl(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_amd_avx512(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_amd_avx2(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_amd_avx(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_amd_ssse3(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_amd_k8(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_0_arm80(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q5_1_amd_avx512vl(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_amd_avx512(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_amd_avx2(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_amd_avx(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_amd_ssse3(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_amd_k8(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_1_arm80(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q8_0_amd_avx512vl(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_amd_avx512(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_amd_avx2(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_amd_avx(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_amd_ssse3(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_amd_k8(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_0_arm80(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q2_K_amd_avx512vl(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_amd_avx512(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_amd_avx2(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_amd_avx(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_amd_ssse3(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_amd_k8(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q2_K_arm80(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q3_K_amd_avx512vl(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_amd_avx512(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_amd_avx2(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_amd_avx(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_amd_ssse3(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_amd_k8(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q3_K_arm80(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q4_K_amd_avx512vl(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_amd_avx512(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_amd_avx2(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_amd_avx(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_amd_ssse3(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_amd_k8(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q4_K_arm80(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q5_K_amd_avx512vl(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_amd_avx512(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_amd_avx2(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_amd_avx(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_amd_ssse3(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_amd_k8(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q5_K_arm80(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q6_K_amd_avx512vl(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_amd_avx512(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_amd_avx2(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_amd_avx(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_amd_ssse3(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_amd_k8(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q6_K_arm80(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_q8_K_amd_avx512vl(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_amd_avx512(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_amd_avx2(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_amd_avx(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_amd_ssse3(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_amd_k8(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_q8_K_arm80(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq2_xxs_amd_avx512vl(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_amd_avx512(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_amd_avx2(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_amd_avx(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_amd_ssse3(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_amd_k8(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xxs_arm80(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq2_xs_amd_avx512vl (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_amd_avx512 (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_amd_avx2 (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_amd_avx (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_amd_ssse3 (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_amd_k8 (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_xs_arm80 (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq2_s_amd_avx512vl  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_amd_avx512  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_amd_avx2  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_amd_avx  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_amd_ssse3  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_amd_k8  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq2_s_arm80  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq3_xxs_amd_avx512vl(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_amd_avx512(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_amd_avx2(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_amd_avx(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_amd_ssse3(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_amd_k8(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_xxs_arm80(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq1_s_amd_avx512vl  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_amd_avx512  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_amd_avx2  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_amd_avx  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_amd_ssse3  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_amd_k8  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_s_arm80  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq1_m_amd_avx512vl  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_amd_avx512  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_amd_avx2  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_amd_avx  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_amd_ssse3  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_amd_k8  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq1_m_arm80  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq4_nl_amd_avx512vl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_amd_avx512 (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_amd_avx2 (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_amd_avx (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_amd_ssse3 (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_amd_k8 (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_nl_arm80 (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq4_xs_amd_avx512vl (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_amd_avx512 (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_amd_avx2 (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_amd_avx (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_amd_ssse3 (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_amd_k8 (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq4_xs_arm80 (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void dequantize_row_iq3_s_amd_avx512vl  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_amd_avx512  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_amd_avx2  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_amd_avx  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_amd_ssse3  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_amd_k8  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
extern "C" void dequantize_row_iq3_s_arm80  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

extern "C" void ggml_vec_dot_q4_0_q8_0_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_0_q8_0_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q4_1_q8_1_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_1_q8_1_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q5_0_q8_0_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_0_q8_0_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q5_1_q8_1_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_1_q8_1_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q8_0_q8_0_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q8_0_q8_0_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q2_K_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q2_K_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q3_K_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q3_K_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q4_K_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q4_K_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q5_K_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q5_K_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_q6_K_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_q6_K_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xxs_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_avx512vl (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_avx512 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_avx2 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_avx (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_ssse3 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_amd_k8 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_xs_q8_K_arm80 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_avx512vl  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_avx512  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_avx2  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_avx  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_ssse3  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_amd_k8  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq2_s_q8_K_arm80  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_avx512vl(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_ssse3(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_amd_k8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_xxs_q8_K_arm80(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_avx512vl  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_avx512  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_avx2  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_avx  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_ssse3  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_amd_k8  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_s_q8_K_arm80  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_avx512vl  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_avx512  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_avx2  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_avx  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_ssse3  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_amd_k8  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq1_m_q8_K_arm80  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_avx512vl (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_avx512 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_avx2 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_avx (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_ssse3 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_amd_k8 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_nl_q8_0_arm80 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_avx512vl (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_avx512 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_avx2 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_avx (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_ssse3 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_amd_k8 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq4_xs_q8_K_arm80 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_avx512vl  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_avx512  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_avx2  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_avx  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_ssse3  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_amd_k8  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
extern "C" void ggml_vec_dot_iq3_s_q8_K_arm80  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

extern "C" size_t quantize_iq2_xxs_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xxs_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq2_xs_amd_avx512vl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_amd_avx512 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_amd_avx2 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_amd_avx (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_amd_ssse3 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_amd_k8 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_xs_arm80 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq2_s_amd_avx512vl  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_amd_avx512  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_amd_avx2  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_amd_avx  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_amd_ssse3  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_amd_k8  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq2_s_arm80  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq3_xxs_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_xxs_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq1_s_amd_avx512vl  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_amd_avx512  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_amd_avx2  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_amd_avx  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_amd_ssse3  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_amd_k8  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_s_arm80  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq1_m_amd_avx512vl  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_amd_avx512  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_amd_avx2  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_amd_avx  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_amd_ssse3  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_amd_k8  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq1_m_arm80  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq4_nl_amd_avx512vl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_amd_avx512 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_amd_avx2 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_amd_avx (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_amd_ssse3 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_amd_k8 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_nl_arm80 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq4_xs_amd_avx512vl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_amd_avx512 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_amd_avx2 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_amd_avx (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_amd_ssse3 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_amd_k8 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq4_xs_arm80 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_iq3_s_amd_avx512vl  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_amd_avx512  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_amd_avx2  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_amd_avx  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_amd_ssse3  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_amd_k8  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_iq3_s_arm80  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q2_K_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q2_K_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q3_K_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q3_K_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q4_K_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_K_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q5_K_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_K_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q6_K_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q6_K_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q4_0_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_0_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q4_1_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q4_1_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q5_0_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_0_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q5_1_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q5_1_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" size_t quantize_q8_0_amd_avx512vl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_amd_avx512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_amd_avx2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_amd_avx(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_amd_ssse3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_amd_k8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
extern "C" size_t quantize_q8_0_arm80(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

extern "C" void iq2xs_init_impl_amd_avx512vl(enum ggml_type type);
extern "C" void iq2xs_init_impl_amd_avx512(enum ggml_type type);
extern "C" void iq2xs_init_impl_amd_avx2(enum ggml_type type);
extern "C" void iq2xs_init_impl_amd_avx(enum ggml_type type);
extern "C" void iq2xs_init_impl_amd_ssse3(enum ggml_type type);
extern "C" void iq2xs_init_impl_amd_k8(enum ggml_type type);
extern "C" void iq2xs_init_impl_arm80(enum ggml_type type);

extern "C" void iq2xs_free_impl_amd_avx512vl(enum ggml_type type);
extern "C" void iq2xs_free_impl_amd_avx512(enum ggml_type type);
extern "C" void iq2xs_free_impl_amd_avx2(enum ggml_type type);
extern "C" void iq2xs_free_impl_amd_avx(enum ggml_type type);
extern "C" void iq2xs_free_impl_amd_ssse3(enum ggml_type type);
extern "C" void iq2xs_free_impl_amd_k8(enum ggml_type type);
extern "C" void iq2xs_free_impl_arm80(enum ggml_type type);

extern "C" void iq3xs_init_impl_amd_avx512vl(int grid_size);
extern "C" void iq3xs_init_impl_amd_avx512(int grid_size);
extern "C" void iq3xs_init_impl_amd_avx2(int grid_size);
extern "C" void iq3xs_init_impl_amd_avx(int grid_size);
extern "C" void iq3xs_init_impl_amd_ssse3(int grid_size);
extern "C" void iq3xs_init_impl_amd_k8(int grid_size);
extern "C" void iq3xs_init_impl_arm80(int grid_size);

extern "C" void iq3xs_free_impl_amd_avx512vl(int grid_size);
extern "C" void iq3xs_free_impl_amd_avx512(int grid_size);
extern "C" void iq3xs_free_impl_amd_avx2(int grid_size);
extern "C" void iq3xs_free_impl_amd_avx(int grid_size);
extern "C" void iq3xs_free_impl_amd_ssse3(int grid_size);
extern "C" void iq3xs_free_impl_amd_k8(int grid_size);
extern "C" void iq3xs_free_impl_arm80(int grid_size);

extern "C" bool ggml_validate_row_data_amd_avx512vl(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_amd_avx512(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_amd_avx2(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_amd_avx(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_amd_ssse3(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_amd_k8(enum ggml_type type, const void * data, size_t nbytes);
extern "C" bool ggml_validate_row_data_arm80(enum ggml_type type, const void * data, size_t nbytes);

static const struct QuantFuncs {
    typeof(quantize_row_q4_0_ref) *ptr_quantize_row_q4_0_ref;
    typeof(quantize_row_q4_1_ref) *ptr_quantize_row_q4_1_ref;
    typeof(quantize_row_q5_0_ref) *ptr_quantize_row_q5_0_ref;
    typeof(quantize_row_q5_1_ref) *ptr_quantize_row_q5_1_ref;
    typeof(quantize_row_q8_0_ref) *ptr_quantize_row_q8_0_ref;
    typeof(quantize_row_q8_1_ref) *ptr_quantize_row_q8_1_ref;
    typeof(quantize_row_q2_K_ref) *ptr_quantize_row_q2_K_ref;
    typeof(quantize_row_q3_K_ref) *ptr_quantize_row_q3_K_ref;
    typeof(quantize_row_q4_K_ref) *ptr_quantize_row_q4_K_ref;
    typeof(quantize_row_q5_K_ref) *ptr_quantize_row_q5_K_ref;
    typeof(quantize_row_q6_K_ref) *ptr_quantize_row_q6_K_ref;
    typeof(quantize_row_q8_K_ref) *ptr_quantize_row_q8_K_ref;
    typeof(quantize_row_iq3_xxs_ref) *ptr_quantize_row_iq3_xxs_ref;
    typeof(quantize_row_iq4_nl_ref) *ptr_quantize_row_iq4_nl_ref;
    typeof(quantize_row_iq4_xs_ref) *ptr_quantize_row_iq4_xs_ref;
    typeof(quantize_row_iq3_s_ref) *ptr_quantize_row_iq3_s_ref;
    typeof(quantize_row_iq2_s_ref) *ptr_quantize_row_iq2_s_ref;
    typeof(quantize_row_q4_0) *ptr_quantize_row_q4_0;
    typeof(quantize_row_q4_1) *ptr_quantize_row_q4_1;
    typeof(quantize_row_q5_0) *ptr_quantize_row_q5_0;
    typeof(quantize_row_q5_1) *ptr_quantize_row_q5_1;
    typeof(quantize_row_q8_0) *ptr_quantize_row_q8_0;
    typeof(quantize_row_q8_1) *ptr_quantize_row_q8_1;
    typeof(quantize_row_q2_K) *ptr_quantize_row_q2_K;
    typeof(quantize_row_q3_K) *ptr_quantize_row_q3_K;
    typeof(quantize_row_q4_K) *ptr_quantize_row_q4_K;
    typeof(quantize_row_q5_K) *ptr_quantize_row_q5_K;
    typeof(quantize_row_q6_K) *ptr_quantize_row_q6_K;
    typeof(quantize_row_q8_K) *ptr_quantize_row_q8_K;
    typeof(quantize_row_iq3_xxs) *ptr_quantize_row_iq3_xxs;
    typeof(quantize_row_iq4_nl) *ptr_quantize_row_iq4_nl;
    typeof(quantize_row_iq4_xs) *ptr_quantize_row_iq4_xs;
    typeof(quantize_row_iq3_s) *ptr_quantize_row_iq3_s;
    typeof(quantize_row_iq2_s) *ptr_quantize_row_iq2_s;
    typeof(dequantize_row_q4_0) *ptr_dequantize_row_q4_0;
    typeof(dequantize_row_q4_1) *ptr_dequantize_row_q4_1;
    typeof(dequantize_row_q5_0) *ptr_dequantize_row_q5_0;
    typeof(dequantize_row_q5_1) *ptr_dequantize_row_q5_1;
    typeof(dequantize_row_q8_0) *ptr_dequantize_row_q8_0;
    typeof(dequantize_row_q2_K) *ptr_dequantize_row_q2_K;
    typeof(dequantize_row_q3_K) *ptr_dequantize_row_q3_K;
    typeof(dequantize_row_q4_K) *ptr_dequantize_row_q4_K;
    typeof(dequantize_row_q5_K) *ptr_dequantize_row_q5_K;
    typeof(dequantize_row_q6_K) *ptr_dequantize_row_q6_K;
    typeof(dequantize_row_q8_K) *ptr_dequantize_row_q8_K;
    typeof(dequantize_row_iq2_xxs) *ptr_dequantize_row_iq2_xxs;
    typeof(dequantize_row_iq2_xs) *ptr_dequantize_row_iq2_xs;
    typeof(dequantize_row_iq2_s) *ptr_dequantize_row_iq2_s;
    typeof(dequantize_row_iq3_xxs) *ptr_dequantize_row_iq3_xxs;
    typeof(dequantize_row_iq1_s) *ptr_dequantize_row_iq1_s;
    typeof(dequantize_row_iq1_m) *ptr_dequantize_row_iq1_m;
    typeof(dequantize_row_iq4_nl) *ptr_dequantize_row_iq4_nl;
    typeof(dequantize_row_iq4_xs) *ptr_dequantize_row_iq4_xs;
    typeof(dequantize_row_iq3_s) *ptr_dequantize_row_iq3_s;
    typeof(ggml_vec_dot_q4_0_q8_0) *ptr_ggml_vec_dot_q4_0_q8_0;
    typeof(ggml_vec_dot_q4_1_q8_1) *ptr_ggml_vec_dot_q4_1_q8_1;
    typeof(ggml_vec_dot_q5_0_q8_0) *ptr_ggml_vec_dot_q5_0_q8_0;
    typeof(ggml_vec_dot_q5_1_q8_1) *ptr_ggml_vec_dot_q5_1_q8_1;
    typeof(ggml_vec_dot_q8_0_q8_0) *ptr_ggml_vec_dot_q8_0_q8_0;
    typeof(ggml_vec_dot_q2_K_q8_K) *ptr_ggml_vec_dot_q2_K_q8_K;
    typeof(ggml_vec_dot_q3_K_q8_K) *ptr_ggml_vec_dot_q3_K_q8_K;
    typeof(ggml_vec_dot_q4_K_q8_K) *ptr_ggml_vec_dot_q4_K_q8_K;
    typeof(ggml_vec_dot_q5_K_q8_K) *ptr_ggml_vec_dot_q5_K_q8_K;
    typeof(ggml_vec_dot_q6_K_q8_K) *ptr_ggml_vec_dot_q6_K_q8_K;
    typeof(ggml_vec_dot_iq2_xxs_q8_K) *ptr_ggml_vec_dot_iq2_xxs_q8_K;
    typeof(ggml_vec_dot_iq2_xs_q8_K) *ptr_ggml_vec_dot_iq2_xs_q8_K;
    typeof(ggml_vec_dot_iq2_s_q8_K) *ptr_ggml_vec_dot_iq2_s_q8_K;
    typeof(ggml_vec_dot_iq3_xxs_q8_K) *ptr_ggml_vec_dot_iq3_xxs_q8_K;
    typeof(ggml_vec_dot_iq1_s_q8_K) *ptr_ggml_vec_dot_iq1_s_q8_K;
    typeof(ggml_vec_dot_iq1_m_q8_K) *ptr_ggml_vec_dot_iq1_m_q8_K;
    typeof(ggml_vec_dot_iq4_nl_q8_0) *ptr_ggml_vec_dot_iq4_nl_q8_0;
    typeof(ggml_vec_dot_iq4_xs_q8_K) *ptr_ggml_vec_dot_iq4_xs_q8_K;
    typeof(ggml_vec_dot_iq3_s_q8_K) *ptr_ggml_vec_dot_iq3_s_q8_K;
    typeof(quantize_iq2_xxs) *ptr_quantize_iq2_xxs;
    typeof(quantize_iq2_xs) *ptr_quantize_iq2_xs;
    typeof(quantize_iq2_s) *ptr_quantize_iq2_s;
    typeof(quantize_iq3_xxs) *ptr_quantize_iq3_xxs;
    typeof(quantize_iq1_s) *ptr_quantize_iq1_s;
    typeof(quantize_iq1_m) *ptr_quantize_iq1_m;
    typeof(quantize_iq4_nl) *ptr_quantize_iq4_nl;
    typeof(quantize_iq4_xs) *ptr_quantize_iq4_xs;
    typeof(quantize_iq3_s) *ptr_quantize_iq3_s;
    typeof(quantize_q2_K) *ptr_quantize_q2_K;
    typeof(quantize_q3_K) *ptr_quantize_q3_K;
    typeof(quantize_q4_K) *ptr_quantize_q4_K;
    typeof(quantize_q5_K) *ptr_quantize_q5_K;
    typeof(quantize_q6_K) *ptr_quantize_q6_K;
    typeof(quantize_q4_0) *ptr_quantize_q4_0;
    typeof(quantize_q4_1) *ptr_quantize_q4_1;
    typeof(quantize_q5_0) *ptr_quantize_q5_0;
    typeof(quantize_q5_1) *ptr_quantize_q5_1;
    typeof(quantize_q8_0) *ptr_quantize_q8_0;
    typeof(iq2xs_init_impl) *ptr_iq2xs_init_impl;
    typeof(iq2xs_free_impl) *ptr_iq2xs_free_impl;
    typeof(iq3xs_init_impl) *ptr_iq3xs_init_impl;
    typeof(iq3xs_free_impl) *ptr_iq3xs_free_impl;
    typeof(ggml_validate_row_data) *ptr_ggml_validate_row_data;

    QuantFuncs() {
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX) && X86_HAVE(AVX2) && X86_HAVE(AVX512F) && X86_HAVE(AVX512BW) && X86_HAVE(AVX512DQ) && X86_HAVE(AVX512VL)) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_avx512vl;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_avx512vl;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_avx512vl;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_avx512vl;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_avx512vl;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_avx512vl;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_avx512vl;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_avx512vl;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_avx512vl;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_avx512vl;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_avx512vl;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_avx512vl;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_avx512vl;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_avx512vl;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_avx512vl;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_avx512vl;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_avx512vl;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_avx512vl;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_avx512vl;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_avx512vl;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_avx512vl;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_avx512vl;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_avx512vl;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_avx512vl;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_avx512vl;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_avx512vl;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_avx512vl;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_avx512vl;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_avx512vl;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_avx512vl;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_avx512vl;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_avx512vl;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_avx512vl;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_avx512vl;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_avx512vl;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_avx512vl;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_avx512vl;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_avx512vl;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_avx512vl;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_avx512vl;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_avx512vl;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_avx512vl;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_avx512vl;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_avx512vl;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_avx512vl;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_avx512vl;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_avx512vl;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_avx512vl;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_avx512vl;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_avx512vl;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_avx512vl;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_avx512vl;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_avx512vl;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_avx512vl;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_avx512vl;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_avx512vl;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_avx512vl;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_avx512vl;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_avx512vl;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_avx512vl;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_avx512vl;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_avx512vl;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_avx512vl;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_avx512vl;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_avx512vl;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_avx512vl;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_avx512vl;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_avx512vl;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_avx512vl;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_avx512vl;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_avx512vl;
            ptr_quantize_q2_K = quantize_q2_K_amd_avx512vl;
            ptr_quantize_q3_K = quantize_q3_K_amd_avx512vl;
            ptr_quantize_q4_K = quantize_q4_K_amd_avx512vl;
            ptr_quantize_q5_K = quantize_q5_K_amd_avx512vl;
            ptr_quantize_q6_K = quantize_q6_K_amd_avx512vl;
            ptr_quantize_q4_0 = quantize_q4_0_amd_avx512vl;
            ptr_quantize_q4_1 = quantize_q4_1_amd_avx512vl;
            ptr_quantize_q5_0 = quantize_q5_0_amd_avx512vl;
            ptr_quantize_q5_1 = quantize_q5_1_amd_avx512vl;
            ptr_quantize_q8_0 = quantize_q8_0_amd_avx512vl;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_avx512vl;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_avx512vl;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_avx512vl;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_avx512vl;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_avx512vl;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX) && X86_HAVE(AVX2) && X86_HAVE(AVX512F)) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_avx512;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_avx512;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_avx512;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_avx512;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_avx512;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_avx512;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_avx512;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_avx512;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_avx512;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_avx512;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_avx512;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_avx512;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_avx512;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_avx512;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_avx512;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_avx512;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_avx512;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_avx512;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_avx512;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_avx512;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_avx512;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_avx512;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_avx512;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_avx512;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_avx512;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_avx512;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_avx512;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_avx512;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_avx512;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_avx512;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_avx512;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_avx512;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_avx512;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_avx512;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_avx512;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_avx512;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_avx512;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_avx512;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_avx512;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_avx512;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_avx512;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_avx512;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_avx512;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_avx512;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_avx512;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_avx512;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_avx512;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_avx512;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_avx512;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_avx512;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_avx512;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_avx512;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_avx512;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_avx512;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_avx512;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_avx512;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_avx512;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_avx512;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_avx512;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_avx512;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_avx512;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_avx512;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_avx512;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_avx512;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_avx512;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_avx512;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_avx512;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_avx512;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_avx512;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_avx512;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_avx512;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_avx512;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_avx512;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_avx512;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_avx512;
            ptr_quantize_q2_K = quantize_q2_K_amd_avx512;
            ptr_quantize_q3_K = quantize_q3_K_amd_avx512;
            ptr_quantize_q4_K = quantize_q4_K_amd_avx512;
            ptr_quantize_q5_K = quantize_q5_K_amd_avx512;
            ptr_quantize_q6_K = quantize_q6_K_amd_avx512;
            ptr_quantize_q4_0 = quantize_q4_0_amd_avx512;
            ptr_quantize_q4_1 = quantize_q4_1_amd_avx512;
            ptr_quantize_q5_0 = quantize_q5_0_amd_avx512;
            ptr_quantize_q5_1 = quantize_q5_1_amd_avx512;
            ptr_quantize_q8_0 = quantize_q8_0_amd_avx512;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_avx512;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_avx512;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_avx512;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_avx512;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_avx512;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(FMA) && X86_HAVE(F16C) && X86_HAVE(AVX) && X86_HAVE(AVX2)) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_avx2;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_avx2;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_avx2;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_avx2;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_avx2;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_avx2;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_avx2;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_avx2;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_avx2;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_avx2;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_avx2;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_avx2;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_avx2;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_avx2;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_avx2;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_avx2;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_avx2;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_avx2;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_avx2;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_avx2;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_avx2;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_avx2;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_avx2;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_avx2;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_avx2;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_avx2;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_avx2;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_avx2;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_avx2;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_avx2;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_avx2;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_avx2;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_avx2;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_avx2;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_avx2;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_avx2;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_avx2;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_avx2;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_avx2;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_avx2;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_avx2;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_avx2;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_avx2;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_avx2;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_avx2;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_avx2;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_avx2;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_avx2;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_avx2;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_avx2;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_avx2;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_avx2;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_avx2;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_avx2;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_avx2;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_avx2;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_avx2;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_avx2;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_avx2;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_avx2;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_avx2;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_avx2;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_avx2;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_avx2;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_avx2;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_avx2;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_avx2;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_avx2;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_avx2;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_avx2;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_avx2;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_avx2;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_avx2;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_avx2;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_avx2;
            ptr_quantize_q2_K = quantize_q2_K_amd_avx2;
            ptr_quantize_q3_K = quantize_q3_K_amd_avx2;
            ptr_quantize_q4_K = quantize_q4_K_amd_avx2;
            ptr_quantize_q5_K = quantize_q5_K_amd_avx2;
            ptr_quantize_q6_K = quantize_q6_K_amd_avx2;
            ptr_quantize_q4_0 = quantize_q4_0_amd_avx2;
            ptr_quantize_q4_1 = quantize_q4_1_amd_avx2;
            ptr_quantize_q5_0 = quantize_q5_0_amd_avx2;
            ptr_quantize_q5_1 = quantize_q5_1_amd_avx2;
            ptr_quantize_q8_0 = quantize_q8_0_amd_avx2;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_avx2;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_avx2;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_avx2;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_avx2;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_avx2;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(AVX)) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_avx;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_avx;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_avx;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_avx;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_avx;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_avx;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_avx;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_avx;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_avx;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_avx;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_avx;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_avx;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_avx;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_avx;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_avx;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_avx;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_avx;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_avx;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_avx;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_avx;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_avx;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_avx;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_avx;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_avx;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_avx;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_avx;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_avx;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_avx;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_avx;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_avx;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_avx;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_avx;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_avx;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_avx;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_avx;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_avx;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_avx;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_avx;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_avx;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_avx;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_avx;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_avx;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_avx;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_avx;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_avx;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_avx;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_avx;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_avx;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_avx;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_avx;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_avx;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_avx;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_avx;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_avx;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_avx;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_avx;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_avx;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_avx;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_avx;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_avx;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_avx;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_avx;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_avx;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_avx;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_avx;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_avx;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_avx;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_avx;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_avx;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_avx;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_avx;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_avx;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_avx;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_avx;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_avx;
            ptr_quantize_q2_K = quantize_q2_K_amd_avx;
            ptr_quantize_q3_K = quantize_q3_K_amd_avx;
            ptr_quantize_q4_K = quantize_q4_K_amd_avx;
            ptr_quantize_q5_K = quantize_q5_K_amd_avx;
            ptr_quantize_q6_K = quantize_q6_K_amd_avx;
            ptr_quantize_q4_0 = quantize_q4_0_amd_avx;
            ptr_quantize_q4_1 = quantize_q4_1_amd_avx;
            ptr_quantize_q5_0 = quantize_q5_0_amd_avx;
            ptr_quantize_q5_1 = quantize_q5_1_amd_avx;
            ptr_quantize_q8_0 = quantize_q8_0_amd_avx;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_avx;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_avx;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_avx;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_avx;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_avx;
            return;
        }
#endif
#ifdef __x86_64__
        if (X86_HAVE(SSSE3)) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_ssse3;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_ssse3;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_ssse3;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_ssse3;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_ssse3;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_ssse3;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_ssse3;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_ssse3;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_ssse3;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_ssse3;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_ssse3;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_ssse3;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_ssse3;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_ssse3;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_ssse3;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_ssse3;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_ssse3;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_ssse3;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_ssse3;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_ssse3;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_ssse3;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_ssse3;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_ssse3;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_ssse3;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_ssse3;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_ssse3;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_ssse3;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_ssse3;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_ssse3;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_ssse3;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_ssse3;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_ssse3;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_ssse3;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_ssse3;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_ssse3;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_ssse3;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_ssse3;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_ssse3;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_ssse3;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_ssse3;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_ssse3;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_ssse3;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_ssse3;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_ssse3;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_ssse3;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_ssse3;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_ssse3;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_ssse3;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_ssse3;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_ssse3;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_ssse3;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_ssse3;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_ssse3;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_ssse3;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_ssse3;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_ssse3;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_ssse3;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_ssse3;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_ssse3;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_ssse3;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_ssse3;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_ssse3;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_ssse3;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_ssse3;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_ssse3;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_ssse3;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_ssse3;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_ssse3;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_ssse3;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_ssse3;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_ssse3;
            ptr_quantize_q2_K = quantize_q2_K_amd_ssse3;
            ptr_quantize_q3_K = quantize_q3_K_amd_ssse3;
            ptr_quantize_q4_K = quantize_q4_K_amd_ssse3;
            ptr_quantize_q5_K = quantize_q5_K_amd_ssse3;
            ptr_quantize_q6_K = quantize_q6_K_amd_ssse3;
            ptr_quantize_q4_0 = quantize_q4_0_amd_ssse3;
            ptr_quantize_q4_1 = quantize_q4_1_amd_ssse3;
            ptr_quantize_q5_0 = quantize_q5_0_amd_ssse3;
            ptr_quantize_q5_1 = quantize_q5_1_amd_ssse3;
            ptr_quantize_q8_0 = quantize_q8_0_amd_ssse3;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_ssse3;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_ssse3;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_ssse3;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_ssse3;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_ssse3;
            return;
        }
#endif
#ifdef __x86_64__
        if (1) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_amd_k8;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_amd_k8;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_amd_k8;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_amd_k8;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_amd_k8;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_amd_k8;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_amd_k8;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_amd_k8;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_amd_k8;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_amd_k8;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_amd_k8;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_amd_k8;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_amd_k8;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_amd_k8;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_amd_k8;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_amd_k8;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_amd_k8;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_amd_k8;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_amd_k8;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_amd_k8;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_amd_k8;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_amd_k8;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_amd_k8;
            ptr_quantize_row_q2_K = quantize_row_q2_K_amd_k8;
            ptr_quantize_row_q3_K = quantize_row_q3_K_amd_k8;
            ptr_quantize_row_q4_K = quantize_row_q4_K_amd_k8;
            ptr_quantize_row_q5_K = quantize_row_q5_K_amd_k8;
            ptr_quantize_row_q6_K = quantize_row_q6_K_amd_k8;
            ptr_quantize_row_q8_K = quantize_row_q8_K_amd_k8;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_amd_k8;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_amd_k8;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_amd_k8;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_amd_k8;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_amd_k8;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_amd_k8;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_amd_k8;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_amd_k8;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_amd_k8;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_amd_k8;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_amd_k8;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_amd_k8;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_amd_k8;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_amd_k8;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_amd_k8;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_amd_k8;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_amd_k8;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_amd_k8;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_amd_k8;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_amd_k8;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_amd_k8;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_amd_k8;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_amd_k8;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_amd_k8;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_amd_k8;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_amd_k8;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_amd_k8;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_amd_k8;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_amd_k8;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_amd_k8;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_amd_k8;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_amd_k8;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_amd_k8;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_amd_k8;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_amd_k8;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_amd_k8;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_amd_k8;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_amd_k8;
            ptr_quantize_iq2_xs = quantize_iq2_xs_amd_k8;
            ptr_quantize_iq2_s = quantize_iq2_s_amd_k8;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_amd_k8;
            ptr_quantize_iq1_s = quantize_iq1_s_amd_k8;
            ptr_quantize_iq1_m = quantize_iq1_m_amd_k8;
            ptr_quantize_iq4_nl = quantize_iq4_nl_amd_k8;
            ptr_quantize_iq4_xs = quantize_iq4_xs_amd_k8;
            ptr_quantize_iq3_s = quantize_iq3_s_amd_k8;
            ptr_quantize_q2_K = quantize_q2_K_amd_k8;
            ptr_quantize_q3_K = quantize_q3_K_amd_k8;
            ptr_quantize_q4_K = quantize_q4_K_amd_k8;
            ptr_quantize_q5_K = quantize_q5_K_amd_k8;
            ptr_quantize_q6_K = quantize_q6_K_amd_k8;
            ptr_quantize_q4_0 = quantize_q4_0_amd_k8;
            ptr_quantize_q4_1 = quantize_q4_1_amd_k8;
            ptr_quantize_q5_0 = quantize_q5_0_amd_k8;
            ptr_quantize_q5_1 = quantize_q5_1_amd_k8;
            ptr_quantize_q8_0 = quantize_q8_0_amd_k8;
            ptr_iq2xs_init_impl = iq2xs_init_impl_amd_k8;
            ptr_iq2xs_free_impl = iq2xs_free_impl_amd_k8;
            ptr_iq3xs_init_impl = iq3xs_init_impl_amd_k8;
            ptr_iq3xs_free_impl = iq3xs_free_impl_amd_k8;
            ptr_ggml_validate_row_data = ggml_validate_row_data_amd_k8;
            return;
        }
#endif
#ifdef __aarch64__
        if (1) {
            ptr_quantize_row_q4_0_ref = quantize_row_q4_0_ref_arm80;
            ptr_quantize_row_q4_1_ref = quantize_row_q4_1_ref_arm80;
            ptr_quantize_row_q5_0_ref = quantize_row_q5_0_ref_arm80;
            ptr_quantize_row_q5_1_ref = quantize_row_q5_1_ref_arm80;
            ptr_quantize_row_q8_0_ref = quantize_row_q8_0_ref_arm80;
            ptr_quantize_row_q8_1_ref = quantize_row_q8_1_ref_arm80;
            ptr_quantize_row_q2_K_ref = quantize_row_q2_K_ref_arm80;
            ptr_quantize_row_q3_K_ref = quantize_row_q3_K_ref_arm80;
            ptr_quantize_row_q4_K_ref = quantize_row_q4_K_ref_arm80;
            ptr_quantize_row_q5_K_ref = quantize_row_q5_K_ref_arm80;
            ptr_quantize_row_q6_K_ref = quantize_row_q6_K_ref_arm80;
            ptr_quantize_row_q8_K_ref = quantize_row_q8_K_ref_arm80;
            ptr_quantize_row_iq3_xxs_ref = quantize_row_iq3_xxs_ref_arm80;
            ptr_quantize_row_iq4_nl_ref = quantize_row_iq4_nl_ref_arm80;
            ptr_quantize_row_iq4_xs_ref = quantize_row_iq4_xs_ref_arm80;
            ptr_quantize_row_iq3_s_ref = quantize_row_iq3_s_ref_arm80;
            ptr_quantize_row_iq2_s_ref = quantize_row_iq2_s_ref_arm80;
            ptr_quantize_row_q4_0 = quantize_row_q4_0_arm80;
            ptr_quantize_row_q4_1 = quantize_row_q4_1_arm80;
            ptr_quantize_row_q5_0 = quantize_row_q5_0_arm80;
            ptr_quantize_row_q5_1 = quantize_row_q5_1_arm80;
            ptr_quantize_row_q8_0 = quantize_row_q8_0_arm80;
            ptr_quantize_row_q8_1 = quantize_row_q8_1_arm80;
            ptr_quantize_row_q2_K = quantize_row_q2_K_arm80;
            ptr_quantize_row_q3_K = quantize_row_q3_K_arm80;
            ptr_quantize_row_q4_K = quantize_row_q4_K_arm80;
            ptr_quantize_row_q5_K = quantize_row_q5_K_arm80;
            ptr_quantize_row_q6_K = quantize_row_q6_K_arm80;
            ptr_quantize_row_q8_K = quantize_row_q8_K_arm80;
            ptr_quantize_row_iq3_xxs = quantize_row_iq3_xxs_arm80;
            ptr_quantize_row_iq4_nl = quantize_row_iq4_nl_arm80;
            ptr_quantize_row_iq4_xs = quantize_row_iq4_xs_arm80;
            ptr_quantize_row_iq3_s = quantize_row_iq3_s_arm80;
            ptr_quantize_row_iq2_s = quantize_row_iq2_s_arm80;
            ptr_dequantize_row_q4_0 = dequantize_row_q4_0_arm80;
            ptr_dequantize_row_q4_1 = dequantize_row_q4_1_arm80;
            ptr_dequantize_row_q5_0 = dequantize_row_q5_0_arm80;
            ptr_dequantize_row_q5_1 = dequantize_row_q5_1_arm80;
            ptr_dequantize_row_q8_0 = dequantize_row_q8_0_arm80;
            ptr_dequantize_row_q2_K = dequantize_row_q2_K_arm80;
            ptr_dequantize_row_q3_K = dequantize_row_q3_K_arm80;
            ptr_dequantize_row_q4_K = dequantize_row_q4_K_arm80;
            ptr_dequantize_row_q5_K = dequantize_row_q5_K_arm80;
            ptr_dequantize_row_q6_K = dequantize_row_q6_K_arm80;
            ptr_dequantize_row_q8_K = dequantize_row_q8_K_arm80;
            ptr_dequantize_row_iq2_xxs = dequantize_row_iq2_xxs_arm80;
            ptr_dequantize_row_iq2_xs = dequantize_row_iq2_xs_arm80;
            ptr_dequantize_row_iq2_s = dequantize_row_iq2_s_arm80;
            ptr_dequantize_row_iq3_xxs = dequantize_row_iq3_xxs_arm80;
            ptr_dequantize_row_iq1_s = dequantize_row_iq1_s_arm80;
            ptr_dequantize_row_iq1_m = dequantize_row_iq1_m_arm80;
            ptr_dequantize_row_iq4_nl = dequantize_row_iq4_nl_arm80;
            ptr_dequantize_row_iq4_xs = dequantize_row_iq4_xs_arm80;
            ptr_dequantize_row_iq3_s = dequantize_row_iq3_s_arm80;
            ptr_ggml_vec_dot_q4_0_q8_0 = ggml_vec_dot_q4_0_q8_0_arm80;
            ptr_ggml_vec_dot_q4_1_q8_1 = ggml_vec_dot_q4_1_q8_1_arm80;
            ptr_ggml_vec_dot_q5_0_q8_0 = ggml_vec_dot_q5_0_q8_0_arm80;
            ptr_ggml_vec_dot_q5_1_q8_1 = ggml_vec_dot_q5_1_q8_1_arm80;
            ptr_ggml_vec_dot_q8_0_q8_0 = ggml_vec_dot_q8_0_q8_0_arm80;
            ptr_ggml_vec_dot_q2_K_q8_K = ggml_vec_dot_q2_K_q8_K_arm80;
            ptr_ggml_vec_dot_q3_K_q8_K = ggml_vec_dot_q3_K_q8_K_arm80;
            ptr_ggml_vec_dot_q4_K_q8_K = ggml_vec_dot_q4_K_q8_K_arm80;
            ptr_ggml_vec_dot_q5_K_q8_K = ggml_vec_dot_q5_K_q8_K_arm80;
            ptr_ggml_vec_dot_q6_K_q8_K = ggml_vec_dot_q6_K_q8_K_arm80;
            ptr_ggml_vec_dot_iq2_xxs_q8_K = ggml_vec_dot_iq2_xxs_q8_K_arm80;
            ptr_ggml_vec_dot_iq2_xs_q8_K = ggml_vec_dot_iq2_xs_q8_K_arm80;
            ptr_ggml_vec_dot_iq2_s_q8_K = ggml_vec_dot_iq2_s_q8_K_arm80;
            ptr_ggml_vec_dot_iq3_xxs_q8_K = ggml_vec_dot_iq3_xxs_q8_K_arm80;
            ptr_ggml_vec_dot_iq1_s_q8_K = ggml_vec_dot_iq1_s_q8_K_arm80;
            ptr_ggml_vec_dot_iq1_m_q8_K = ggml_vec_dot_iq1_m_q8_K_arm80;
            ptr_ggml_vec_dot_iq4_nl_q8_0 = ggml_vec_dot_iq4_nl_q8_0_arm80;
            ptr_ggml_vec_dot_iq4_xs_q8_K = ggml_vec_dot_iq4_xs_q8_K_arm80;
            ptr_ggml_vec_dot_iq3_s_q8_K = ggml_vec_dot_iq3_s_q8_K_arm80;
            ptr_quantize_iq2_xxs = quantize_iq2_xxs_arm80;
            ptr_quantize_iq2_xs = quantize_iq2_xs_arm80;
            ptr_quantize_iq2_s = quantize_iq2_s_arm80;
            ptr_quantize_iq3_xxs = quantize_iq3_xxs_arm80;
            ptr_quantize_iq1_s = quantize_iq1_s_arm80;
            ptr_quantize_iq1_m = quantize_iq1_m_arm80;
            ptr_quantize_iq4_nl = quantize_iq4_nl_arm80;
            ptr_quantize_iq4_xs = quantize_iq4_xs_arm80;
            ptr_quantize_iq3_s = quantize_iq3_s_arm80;
            ptr_quantize_q2_K = quantize_q2_K_arm80;
            ptr_quantize_q3_K = quantize_q3_K_arm80;
            ptr_quantize_q4_K = quantize_q4_K_arm80;
            ptr_quantize_q5_K = quantize_q5_K_arm80;
            ptr_quantize_q6_K = quantize_q6_K_arm80;
            ptr_quantize_q4_0 = quantize_q4_0_arm80;
            ptr_quantize_q4_1 = quantize_q4_1_arm80;
            ptr_quantize_q5_0 = quantize_q5_0_arm80;
            ptr_quantize_q5_1 = quantize_q5_1_arm80;
            ptr_quantize_q8_0 = quantize_q8_0_arm80;
            ptr_iq2xs_init_impl = iq2xs_init_impl_arm80;
            ptr_iq2xs_free_impl = iq2xs_free_impl_arm80;
            ptr_iq3xs_init_impl = iq3xs_init_impl_arm80;
            ptr_iq3xs_free_impl = iq3xs_free_impl_arm80;
            ptr_ggml_validate_row_data = ggml_validate_row_data_arm80;
            return;
        }
#endif
    }
} funcs;

void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_1_ref(x, y, k);
}

void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q2_K_ref(x, y, k);
}

void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q3_K_ref(x, y, k);
}

void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_K_ref(x, y, k);
}

void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_K_ref(x, y, k);
}

void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q6_K_ref(x, y, k);
}

void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_K_ref(x, y, k);
}

void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq3_xxs_ref(x, y, k);
}

void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq4_xs_ref(x, y, k);
}

void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq3_s_ref(x, y, k);
}

void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq2_s_ref(x, y, k);
}

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_0(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_1(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_0(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_1(x, y, k);
}

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_0(x, y, k);
}

void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_1(x, y, k);
}

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q2_K(x, y, k);
}

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q3_K(x, y, k);
}

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q4_K(x, y, k);
}

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q5_K(x, y, k);
}

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q6_K(x, y, k);
}

void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_q8_K(x, y, k);
}

void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq3_xxs(x, y, k);
}

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq4_nl(x, y, k);
}

void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq4_xs(x, y, k);
}

void quantize_row_iq3_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq3_s(x, y, k);
}

void quantize_row_iq2_s  (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_quantize_row_iq2_s(x, y, k);
}

void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q4_0(x, y, k);
}

void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q4_1(x, y, k);
}

void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q5_0(x, y, k);
}

void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q5_1(x, y, k);
}

void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q8_0(x, y, k);
}

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q2_K(x, y, k);
}

void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q3_K(x, y, k);
}

void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q4_K(x, y, k);
}

void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q5_K(x, y, k);
}

void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q6_K(x, y, k);
}

void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_q8_K(x, y, k);
}

void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq2_xxs(x, y, k);
}

void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq2_xs(x, y, k);
}

void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq2_s(x, y, k);
}

void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq3_xxs(x, y, k);
}

void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq1_s(x, y, k);
}

void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq1_m(x, y, k);
}

void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq4_nl(x, y, k);
}

void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq4_xs(x, y, k);
}

void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
  return funcs.ptr_dequantize_row_iq3_s(x, y, k);
}

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q4_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q4_1_q8_1(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q5_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q5_1_q8_1(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q8_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q2_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q3_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q4_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q5_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_q6_K_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq2_xxs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq2_xs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq2_s_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq3_xxs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq1_s_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq1_m_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq4_nl_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq4_xs_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
  return funcs.ptr_ggml_vec_dot_iq3_s_q8_K(n, s, bs, vx, bx, vy, by, nrc);
}

size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq2_xxs(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq2_xs(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq2_s(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq3_xxs(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq1_s(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq1_m(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq4_nl(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq4_xs(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_iq3_s(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q2_K(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q3_K(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q4_K(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q5_K(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q6_K(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q4_0(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q4_1(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q5_0(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q5_1(src, dst, nrows, n_per_row, imatrix);
}

size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
  return funcs.ptr_quantize_q8_0(src, dst, nrows, n_per_row, imatrix);
}

void iq2xs_init_impl(enum ggml_type type) {
  return funcs.ptr_iq2xs_init_impl(type);
}

void iq2xs_free_impl(enum ggml_type type) {
  return funcs.ptr_iq2xs_free_impl(type);
}

void iq3xs_init_impl(int grid_size) {
  return funcs.ptr_iq3xs_init_impl(grid_size);
}

void iq3xs_free_impl(int grid_size) {
  return funcs.ptr_iq3xs_free_impl(grid_size);
}

bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
  return funcs.ptr_ggml_validate_row_data(type, data, nbytes);
}

