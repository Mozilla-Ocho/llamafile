// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const C_KEYWORDS = new Set([
  '_Alignas',
  '_Alignof',
  '_Atomic',
  '_BitInt',
  '_Bool',
  '_Complex',
  '_Decimal128',
  '_Decimal32',
  '_Decimal64',
  '_Float128',
  '_Float16',
  '_Float32',
  '_Float64',
  '_Generic',
  '_Imaginary',
  '_Noreturn',
  '_Static_assert',
  '_Thread_local',
  '__access__',
  '__alias__',
  '__aligned__',
  '__alignof',
  '__alignof__',
  '__alloc_align__',
  '__alloc_size__',
  '__always_inline__',
  '__artificial__',
  '__asm',
  '__asm__',
  '__assume_aligned__',
  '__attribute',
  '__attribute__',
  '__auto_type',
  '__avx2',
  '__bf16',
  '__builtin___',
  '__builtin_offsetof',
  '__builtin_va_arg',
  '__byte__',
  '__cmn_err__',
  '__cold__',
  '__complex',
  '__complex__',
  '__const',
  '__const__',
  '__constructor__',
  '__copy__',
  '__deprecated__',
  '__destructor__',
  '__error__',
  '__extension__',
  '__externally_visible__',
  '__fd_arg',
  '__fd_arg__',
  '__fentry__',
  '__flatten__',
  '__float80',
  '__force_align_arg_pointer__',
  '__format__',
  '__format_arg__',
  '__funline',
  '__gnu_format__',
  '__gnu_inline__',
  '__gnu_printf__',
  '__gnu_scanf__',
  '__hardbool__',
  '__hot__',
  '__ifunc__',
  '__imag',
  '__imag__',
  '__imag__ ',
  '__inline',
  '__inline__',
  '__interrupt__',
  '__interrupt_handler__',
  '__label__',
  '__leaf__',
  '__malloc__',
  '__may_alias__',
  '__mcarch__',
  '__mcfarch__',
  '__mcffpu__',
  '__mcfhwdiv__',
  '__mode__',
  '__ms_abi__',
  '__msabi',
  '__muarch__',
  '__no_address_safety_analysis__',
  '__no_caller_saved_registers__',
  '__no_icf__',
  '__no_instrument_function__',
  '__no_profile_instrument_function__',
  '__no_reorder__',
  '__no_sanitize__',
  '__no_sanitize_address__',
  '__no_sanitize_thread__',
  '__no_sanitize_undefined__',
  '__no_split_stack__',
  '__no_stack_limit__',
  '__no_stack_protector__',
  '__noclone__',
  '__noinline__',
  '__noipa__',
  '__nonnull__',
  '__noplt__',
  '__noreturn__',
  '__nothrow__',
  '__null',
  '__optimize__',
  '__packed__',
  '__params_nonnull__',
  '__patchable_function_entry__',
  '__pic__',
  '__pie__',
  '__pointer__',
  '__printf__',
  '__pure__',
  '__read_only',
  '__read_only__',
  '__read_write',
  '__read_write__',
  '__real',
  '__real__',
  '__real__ ',
  '__restrict',
  '__restrict__',
  '__retain__',
  '__return__',
  '__returns_nonnull__',
  '__returns_twice__',
  '__scanf__',
  '__section__',
  '__seg_fs',
  '__seg_gs',
  '__sentinel__',
  '__signed',
  '__signed__',
  '__simd__',
  '__strfmon__',
  '__strftime__',
  '__strong__',
  '__symver__',
  '__sysv_abi__',
  '__tainted_args__',
  '__target__',
  '__target_clones',
  '__target_clones__',
  '__thread',
  '__transparent_union__',
  '__typeof',
  '__typeof__',
  '__typeof_unqual__',
  '__unix__',
  '__unused__',
  '__used__',
  '__vax__',
  '__vector_size__',
  '__vex',
  '__visibility__',
  '__volatile',
  '__volatile__',
  '__volatile__ ',
  '__warn_if_not_aligned__',
  '__warn_unused_result__',
  '__warning__',
  '__weak__',
  '__word__',
  '__write_only',
  '__write_only__',
  '__wur',
  '__zero_call_used_regs__',
  'alignas',
  'alignof',
  'asm',
  'auto',
  'autotype',
  'bool',
  'break',
  'case',
  'char',
  'const',
  'constexpr',
  'continue',
  'default',
  'do',
  'dontcallback',
  'dontthrow',
  'double',
  'else',
  'enum',
  'extern',
  'float',
  'for',
  'forcealign',
  'forcealignargpointer',
  'forceinline',
  'goto',
  'hasatleast',
  'if',
  'inline',
  'int',
  'interruptfn',
  'libcesque',
  'long',
  'mallocesque',
  'memcpyesque',
  'nocallersavedregisters',
  'nosideeffect',
  'nullptr',
  'nullterminated',
  'paramsnonnull',
  'printfesque',
  'privileged',
  'pureconst',
  'reallocesque',
  'register',
  'relegated',
  'restrict',
  'return',
  'returnsaligned',
  'returnsnonnull',
  'returnspointerwithnoaliases',
  'returnstwice',
  'scanfesque',
  'short',
  'signed',
  'sizeof',
  'static',
  'strftimeesque',
  'strlenesque',
  'struct',
  'switch',
  'textwindows',
  'thatispacked',
  'thread_local',
  'typedef',
  'typeof',
  'typeof_unqual',
  'union',
  'unsigned',
  'vallocesque',
  'void',
  'volatile',
  'while',
  'wontreturn',
]);

const C_BUILTINS = new Set([
  'COSMOPOLITAN_CXX_END_',
  'COSMOPOLITAN_CXX_START_',
  'COSMOPOLITAN_CXX_USING_',
  'COSMOPOLITAN_C_END_',
  'COSMOPOLITAN_C_START_',
  'STRINGIFY',
  '__ATOMIC_ACQUIRE',
  '__ATOMIC_ACQ_REL',
  '__ATOMIC_CONSUME',
  '__ATOMIC_RELAXED',
  '__ATOMIC_RELEASE',
  '__ATOMIC_SEQ_CST',
  '__atomic_add_fetch',
  '__atomic_always_lock_free',
  '__atomic_and_fetch',
  '__atomic_clear',
  '__atomic_compare_exchange',
  '__atomic_compare_exchange_n',
  '__atomic_exchange',
  '__atomic_exchange_n',
  '__atomic_fetch_add',
  '__atomic_fetch_and',
  '__atomic_fetch_nand',
  '__atomic_fetch_or',
  '__atomic_fetch_sub',
  '__atomic_fetch_xor',
  '__atomic_is_lock_free',
  '__atomic_load',
  '__atomic_load_n',
  '__atomic_nand_fetch',
  '__atomic_or_fetch',
  '__atomic_signal_fence',
  '__atomic_store',
  '__atomic_store_n',
  '__atomic_sub_fetch',
  '__atomic_test_and_set',
  '__atomic_thread_fence',
  '__atomic_xor_fetch',
  '__builtin_FILE',
  '__builtin_FUNCTION',
  '__builtin_LINE',
  '__builtin_abs',
  '__builtin_add_overflow',
  '__builtin_add_overflow_p',
  '__builtin_addc',
  '__builtin_addcl',
  '__builtin_addcll',
  '__builtin_addf128_round_to_odd',
  '__builtin_addg6s',
  '__builtin_alloc',
  '__builtin_alloc_with_align',
  '__builtin_alloca',
  '__builtin_alloca_with_align',
  '__builtin_alloca_with_align_and_max',
  '__builtin_apply',
  '__builtin_apply_args',
  '__builtin_assoc_barrier',
  '__builtin_assume',
  '__builtin_assume_aligned',
  '__builtin_bit_cast',
  '__builtin_bswap128',
  '__builtin_bswap16',
  '__builtin_bswap32',
  '__builtin_bswap64',
  '__builtin_btf_type_id',
  '__builtin_byte_in_either_range',
  '__builtin_byte_in_range',
  '__builtin_byte_in_set',
  '__builtin_call_with_static_chain',
  '__builtin_calloc',
  '__builtin_cbcdtd',
  '__builtin_cdtbcd',
  '__builtin_cfuged',
  '__builtin_choose_expr',
  '__builtin_classify_type',
  '__builtin_clear_padding',
  '__builtin_clrsb',
  '__builtin_clrsbg',
  '__builtin_clrsbl',
  '__builtin_clrsbll',
  '__builtin_clz',
  '__builtin_clzg',
  '__builtin_clzl',
  '__builtin_clzll',
  '__builtin_cmpb',
  '__builtin_cntlzdm',
  '__builtin_cnttzdm',
  '__builtin_complex',
  '__builtin_constant_p',
  '__builtin_convertvector',
  '__builtin_copysignfn',
  '__builtin_copysignfnx',
  '__builtin_copysignq',
  '__builtin_cpu_init',
  '__builtin_cpu_is',
  '__builtin_cpu_supports',
  '__builtin_ctz',
  '__builtin_ctzg',
  '__builtin_ctzl',
  '__builtin_ctzll',
  '__builtin_dynamic_object_size',
  '__builtin_eni',
  '__builtin_expect',
  '__builtin_expect_with_probability',
  '__builtin_extend_pointer',
  '__builtin_extract_return_addr',
  '__builtin_fabsfn',
  '__builtin_fabsfnx',
  '__builtin_fabsq',
  '__builtin_ffs',
  '__builtin_ffsg',
  '__builtin_ffsl',
  '__builtin_ffsll',
  '__builtin_flushd',
  '__builtin_flushda',
  '__builtin_fma',
  '__builtin_fmaf128',
  '__builtin_fmaf128_round_to_odd',
  '__builtin_fpclassify',
  '__builtin_fprintf',
  '__builtin_fprintf_unlocked',
  '__builtin_fputc',
  '__builtin_fputc_unlocked',
  '__builtin_fputs',
  '__builtin_fputs_unlocked',
  '__builtin_frame_address',
  '__builtin_free',
  '__builtin_frob_return_addr',
  '__builtin_fwrite',
  '__builtin_fwrite_unlocked',
  '__builtin_get_texasr',
  '__builtin_get_texasru',
  '__builtin_get_tfhar',
  '__builtin_get_tfiar',
  '__builtin_goacc_parlevel_id',
  '__builtin_goacc_parlevel_size',
  '__builtin_has_attribute',
  '__builtin_huge_val',
  '__builtin_huge_valf',
  '__builtin_huge_valfn',
  '__builtin_huge_valfnx',
  '__builtin_huge_vall',
  '__builtin_huge_valq',
  '__builtin_inf',
  '__builtin_infd128',
  '__builtin_infd32',
  '__builtin_infd64',
  '__builtin_inff',
  '__builtin_inffn',
  '__builtin_inffnx',
  '__builtin_infl',
  '__builtin_infq',
  '__builtin_is_constant_evaluated',
  '__builtin_iseqsig',
  '__builtin_isfinite',
  '__builtin_isgreater',
  '__builtin_isgreaterequal',
  '__builtin_isinf_sign',
  '__builtin_isnan',
  '__builtin_isnormal',
  '__builtin_issignaling',
  '__builtin_isunordered',
  '__builtin_ldbio',
  '__builtin_ldbuio',
  '__builtin_ldex',
  '__builtin_ldhio',
  '__builtin_ldhuio',
  '__builtin_ldsex',
  '__builtin_ldwio',
  '__builtin_longjmp',
  '__builtin_lroundf',
  '__builtin_malloc',
  '__builtin_memcpy',
  '__builtin_memcpy_chk',
  '__builtin_memmove',
  '__builtin_memset',
  '__builtin_memset_chk',
  '__builtin_mffs',
  '__builtin_mffsl',
  '__builtin_mul_overflow',
  '__builtin_mul_overflow_p',
  '__builtin_mulf128_round_to_odd',
  '__builtin_nan',
  '__builtin_nand128',
  '__builtin_nand32',
  '__builtin_nand64',
  '__builtin_nanf',
  '__builtin_nanfn',
  '__builtin_nanfnx',
  '__builtin_nanl',
  '__builtin_nanq',
  '__builtin_nans',
  '__builtin_nansd128',
  '__builtin_nansd32',
  '__builtin_nansd64',
  '__builtin_nansf',
  '__builtin_nansfn',
  '__builtin_nansfnx',
  '__builtin_nansl',
  '__builtin_nansq',
  '__builtin_next_arg',
  '__builtin_non_tx_store',
  '__builtin_nvptx_brev',
  '__builtin_nvptx_brevll',
  '__builtin_object_size',
  '__builtin_offsetof',
  '__builtin_pack_dec128',
  '__builtin_pack_ibm128',
  '__builtin_pack_longdouble',
  '__builtin_pack_vector_int128',
  '__builtin_parity',
  '__builtin_parityg',
  '__builtin_parityl',
  '__builtin_parityll',
  '__builtin_pdepd',
  '__builtin_pextd',
  '__builtin_popcount',
  '__builtin_popcountg',
  '__builtin_popcountl',
  '__builtin_popcountll',
  '__builtin_powi',
  '__builtin_powif',
  '__builtin_powil',
  '__builtin_ppc_get_timebase',
  '__builtin_ppc_mftb',
  '__builtin_prefetch',
  '__builtin_preserve_access_index',
  '__builtin_preserve_enum_value',
  '__builtin_preserve_field_info',
  '__builtin_preserve_type_info',
  '__builtin_printf',
  '__builtin_printf_unlocked',
  '__builtin_putc',
  '__builtin_putc_unlocked',
  '__builtin_putchar',
  '__builtin_putchar_unlocked',
  '__builtin_puts',
  '__builtin_puts_unlocked',
  '__builtin_rdctl',
  '__builtin_rdprs',
  '__builtin_read16',
  '__builtin_read32',
  '__builtin_read64',
  '__builtin_read8',
  '__builtin_realloc',
  '__builtin_recipdiv',
  '__builtin_recipdivf',
  '__builtin_return',
  '__builtin_return_address',
  '__builtin_rs6000_speculation_barrier',
  '__builtin_rsqrt',
  '__builtin_rsqrtf',
  '__builtin_sadd_overflow',
  '__builtin_saddl_overflow',
  '__builtin_saddll_overflow',
  '__builtin_set_fpscr_drn',
  '__builtin_set_fpscr_rn',
  '__builtin_set_texasr',
  '__builtin_set_texasru',
  '__builtin_set_tfhar',
  '__builtin_set_tfiar',
  '__builtin_set_thread_pointer',
  '__builtin_setjmp',
  '__builtin_sh_get_fpscr',
  '__builtin_sh_set_fpscr',
  '__builtin_shuffle',
  '__builtin_shufflevector',
  '__builtin_signbit',
  '__builtin_signbitf',
  '__builtin_signbitl',
  '__builtin_smul_overflow',
  '__builtin_smull_overflow',
  '__builtin_smulll_overflow',
  '__builtin_speculation_safe_copy',
  '__builtin_speculation_safe_value',
  '__builtin_sqrtf128',
  '__builtin_sqrtf128_round_to_odd',
  '__builtin_ssub_overflow',
  '__builtin_ssubl_overflow',
  '__builtin_ssubll_overflow',
  '__builtin_stack_address',
  '__builtin_stack_restore',
  '__builtin_stack_save',
  '__builtin_stbio',
  '__builtin_stdc_bit_ceil',
  '__builtin_stdc_bit_floor',
  '__builtin_stdc_bit_width',
  '__builtin_stdc_count_ones',
  '__builtin_stdc_count_zeros',
  '__builtin_stdc_first_leading_one',
  '__builtin_stdc_first_leading_zero',
  '__builtin_stdc_first_trailing_one',
  '__builtin_stdc_first_trailing_zero',
  '__builtin_stdc_has_single_bit',
  '__builtin_stdc_leading_ones',
  '__builtin_stdc_leading_zeros',
  '__builtin_stdc_trailing_ones',
  '__builtin_stdc_trailing_zeros',
  '__builtin_stex',
  '__builtin_sthio',
  '__builtin_strchr',
  '__builtin_strcpy',
  '__builtin_strcpy_chk',
  '__builtin_strlen',
  '__builtin_stsex',
  '__builtin_stwio',
  '__builtin_sub_overflow',
  '__builtin_sub_overflow_p',
  '__builtin_subc',
  '__builtin_subcl',
  '__builtin_subcll',
  '__builtin_subf128_round_to_odd',
  '__builtin_sync',
  '__builtin_tabort',
  '__builtin_tabortdc',
  '__builtin_tabortdci',
  '__builtin_tabortwc',
  '__builtin_tabortwci',
  '__builtin_tbegin',
  '__builtin_tbegin_nofloat',
  '__builtin_tbegin_retry',
  '__builtin_tbegin_retry_nofloat',
  '__builtin_tbeginc',
  '__builtin_tcheck',
  '__builtin_tend',
  '__builtin_tendall',
  '__builtin_tgmath',
  '__builtin_thread_pointer',
  '__builtin_trap',
  '__builtin_trechkpt',
  '__builtin_treclaim',
  '__builtin_tresume',
  '__builtin_truncf128_round_to_odd',
  '__builtin_tsr',
  '__builtin_tsuspend',
  '__builtin_ttest',
  '__builtin_tx_assist',
  '__builtin_tx_nesting_depth',
  '__builtin_types_compatible_p',
  '__builtin_uadd_overflow',
  '__builtin_uaddl_overflow',
  '__builtin_uaddll_overflow',
  '__builtin_umul_overflow',
  '__builtin_umull_overflow',
  '__builtin_umulll_overflow',
  '__builtin_unpack_dec128',
  '__builtin_unpack_ibm128',
  '__builtin_unpack_longdouble',
  '__builtin_unpack_vector_int128',
  '__builtin_unreachable',
  '__builtin_usub_overflow',
  '__builtin_usubl_overflow',
  '__builtin_usubll_overflow',
  '__builtin_va_arg',
  '__builtin_va_arg_pack',
  '__builtin_va_arg_pack_len',
  '__builtin_va_copy',
  '__builtin_va_end',
  '__builtin_va_list',
  '__builtin_va_start',
  '__builtin_vfprintf',
  '__builtin_vprintf',
  '__builtin_wrctl',
  '__builtin_write16',
  '__builtin_write32',
  '__builtin_write64',
  '__builtin_write8',
  '__builtin_wrpie',
  '__conceal',
  '__dll_import',
  '__expropriate',
  '__has_feature',
  '__has_include',
  '__has_nothrow_assign',
  '__has_nothrow_constructor',
  '__has_nothrow_copy',
  '__has_trivial_assign',
  '__has_trivial_constructor',
  '__has_trivial_copy',
  '__has_trivial_destructor',
  '__has_virtual_destructor',
  '__integer_pack',
  '__is_abstract',
  '__is_base_of',
  '__is_class',
  '__is_empty',
  '__is_enum',
  '__is_literal_type',
  '__is_pod',
  '__is_polymorphic',
  '__is_same',
  '__is_standard_layout',
  '__is_trivial',
  '__is_union',
  '__notice',
  '__shfl',
  '__shfl_down',
  '__shfl_down_sync',
  '__shfl_sync',
  '__shfl_up_sync',
  '__shfl_xor',
  '__shfl_xor_sync',
  '__static_yoink',
  '__strong_reference',
  '__target_clones',
  '__underlying_type',
  '__veil',
  '__vex',
  '__weak_reference',
  '__yoink',
  'static_assert',
]);

const C_CONSTANTS = new Set([
  'ARG_MAX',
  'ATEXIT_MAX',
  'ATOMIC_FLAG_INIT',
  'BC_BASE_MAX',
  'BC_DIM_MAX',
  'BC_SCALE_MAX',
  'BC_STRING_MAX',
  'BUFSIZ',
  'CHARCLASS_NAME_MAX',
  'CHAR_BIT',
  'CHAR_MAX',
  'CHAR_MIN',
  'CHILD_MAX',
  'CLK_TCK',
  'COLL_WEIGHTS_MAX',
  'DBL_DECIMAL_DIG',
  'DBL_DIG',
  'DBL_EPSILON',
  'DBL_HAS_SUBNORM',
  'DBL_IS_IEC_60559',
  'DBL_MANT_DIG',
  'DBL_MAX',
  'DBL_MAX_10_EXP',
  'DBL_MAX_EXP',
  'DBL_MIN',
  'DBL_MIN_10_EXP',
  'DBL_MIN_EXP',
  'DBL_NORM_MAX',
  'DBL_TRUE_MIN',
  'DECIMAL_DIG',
  'DELAYTIMER_MAX',
  'EOF',
  'EXPR_NEST_MAX',
  'FILENAME_MAX',
  'FILESIZEBITS',
  'FLT_DECIMAL_DIG',
  'FLT_DIG',
  'FLT_EPSILON',
  'FLT_HAS_SUBNORM',
  'FLT_IS_IEC_60559',
  'FLT_MANT_DIG',
  'FLT_MAX',
  'FLT_MAX_10_EXP',
  'FLT_MAX_EXP',
  'FLT_MIN',
  'FLT_MIN_10_EXP',
  'FLT_MIN_EXP',
  'FLT_NORM_MAX',
  'FLT_RADIX',
  'FLT_ROUNDS',
  'FLT_TRUE_MIN',
  'FOPEN_MAX',
  'FP_FAST_FMA',
  'FP_FAST_FMAF',
  'FP_FAST_FMAL',
  'FP_ILOGB0',
  'FP_ILOGBNAN',
  'FP_INFINITE',
  'FP_NAN',
  'FP_NORMAL',
  'FP_SUBNORMAL',
  'FP_ZERO',
  'HLF_MAX',
  'HLF_MIN',
  'HOST_NAME_MAX',
  'HUGE_VAL',
  'HUGE_VALF',
  'HUGE_VALL',
  'INFINITY',
  'INT128_MAX',
  'INT128_MIN',
  'INT16_MAX',
  'INT16_MIN',
  'INT32_MAX',
  'INT32_MIN',
  'INT64_MAX',
  'INT64_MIN',
  'INT8_MAX',
  'INT8_MIN',
  'INTMAX_MAX',
  'INTMAX_MIN',
  'INTPTR_MAX',
  'INTPTR_MIN',
  'INT_FAST16_MAX',
  'INT_FAST16_MIN',
  'INT_FAST32_MAX',
  'INT_FAST32_MIN',
  'INT_FAST64_MAX',
  'INT_FAST64_MIN',
  'INT_FAST8_MAX',
  'INT_FAST8_MIN',
  'INT_LEAST16_MAX',
  'INT_LEAST16_MIN',
  'INT_LEAST32_MAX',
  'INT_LEAST32_MIN',
  'INT_LEAST64_MAX',
  'INT_LEAST64_MIN',
  'INT_LEAST8_MAX',
  'INT_LEAST8_MIN',
  'INT_MAX',
  'INT_MIN',
  'LDBL_DECIMAL_DIG',
  'LDBL_DIG',
  'LDBL_EPSILON',
  'LDBL_HAS_SUBNORM',
  'LDBL_IS_IEC_60559',
  'LDBL_MANT_DIG',
  'LDBL_MAX',
  'LDBL_MAX_10_EXP',
  'LDBL_MAX_EXP',
  'LDBL_MIN',
  'LDBL_MIN_10_EXP',
  'LDBL_MIN_EXP',
  'LDBL_NORM_MAX',
  'LDBL_TRUE_MIN',
  'LINE_MAX',
  'LLONG_MAX',
  'LLONG_MIN',
  'LOGIN_NAME_MAX',
  'LONG_BIT',
  'LONG_LONG_MAX',
  'LONG_LONG_MIN',
  'LONG_MAX',
  'LONG_MIN',
  'L_ctermid',
  'L_tmpnam',
  'MATH_ERREXCEPT',
  'MATH_ERRNO',
  'MB_CUR_MAX',
  'MB_LEN_MAX',
  'MQ_PRIO_MAX',
  'M_1_PI',
  'M_1_PIf',
  'M_1_PIl',
  'M_2_PI',
  'M_2_PIf',
  'M_2_PIl',
  'M_2_SQRTPI',
  'M_2_SQRTPIf',
  'M_2_SQRTPIl',
  'M_E',
  'M_Ef',
  'M_El',
  'M_LN10',
  'M_LN10f',
  'M_LN10l',
  'M_LN2',
  'M_LN2f',
  'M_LN2l',
  'M_LOG10E',
  'M_LOG10Ef',
  'M_LOG10El',
  'M_LOG2E',
  'M_LOG2Ef',
  'M_LOG2El',
  'M_PI',
  'M_PI_2',
  'M_PI_2f',
  'M_PI_2l',
  'M_PI_4',
  'M_PI_4f',
  'M_PI_4l',
  'M_PIf',
  'M_PIl',
  'M_SQRT1_2',
  'M_SQRT1_2f',
  'M_SQRT1_2l',
  'M_SQRT2',
  'M_SQRT2f',
  'M_SQRT2l',
  'NAME_MAX',
  'NAN',
  'NDEBUG',
  'NL_ARGMAX',
  'NL_LANGMAX',
  'NL_MSGMAX',
  'NL_SETMAX',
  'NL_TEXTMAX',
  'NSIG',
  'NULL',
  'NZERO',
  'OPEN_MAX',
  'PATH_MAX',
  'PIPE_MAX',
  'PRIB128',
  'PRIB16',
  'PRIB32',
  'PRIB64',
  'PRIB8',
  'PRIBFAST128',
  'PRIBFAST16',
  'PRIBFAST32',
  'PRIBFAST64',
  'PRIBFAST8',
  'PRIBLEAST128',
  'PRIBLEAST16',
  'PRIBLEAST32',
  'PRIBLEAST64',
  'PRIBLEAST8',
  'PRIX128',
  'PRIX16',
  'PRIX32',
  'PRIX64',
  'PRIX8',
  'PRIXFAST128',
  'PRIXFAST16',
  'PRIXFAST32',
  'PRIXFAST64',
  'PRIXFAST8',
  'PRIXLEAST128',
  'PRIXLEAST16',
  'PRIXLEAST32',
  'PRIXLEAST64',
  'PRIXLEAST8',
  'PRIXMAX',
  'PRIXPTR',
  'PRIb128',
  'PRIb16',
  'PRIb32',
  'PRIb64',
  'PRIb8',
  'PRIbFAST128',
  'PRIbFAST16',
  'PRIbFAST32',
  'PRIbFAST64',
  'PRIbFAST8',
  'PRIbLEAST128',
  'PRIbLEAST16',
  'PRIbLEAST32',
  'PRIbLEAST64',
  'PRIbLEAST8',
  'PRId128',
  'PRId16',
  'PRId32',
  'PRId64',
  'PRId8',
  'PRIdFAST128',
  'PRIdFAST16',
  'PRIdFAST32',
  'PRIdFAST64',
  'PRIdFAST8',
  'PRIdLEAST128',
  'PRIdLEAST16',
  'PRIdLEAST32',
  'PRIdLEAST64',
  'PRIdLEAST8',
  'PRIdMAX',
  'PRIdPTR',
  'PRIi128',
  'PRIi16',
  'PRIi32',
  'PRIi64',
  'PRIi8',
  'PRIiFAST128',
  'PRIiFAST16',
  'PRIiFAST32',
  'PRIiFAST64',
  'PRIiFAST8',
  'PRIiLEAST128',
  'PRIiLEAST16',
  'PRIiLEAST32',
  'PRIiLEAST64',
  'PRIiLEAST8',
  'PRIiMAX',
  'PRIiPTR',
  'PRIo128',
  'PRIo16',
  'PRIo32',
  'PRIo64',
  'PRIo8',
  'PRIoFAST128',
  'PRIoFAST16',
  'PRIoFAST32',
  'PRIoFAST64',
  'PRIoFAST8',
  'PRIoLEAST128',
  'PRIoLEAST16',
  'PRIoLEAST32',
  'PRIoLEAST64',
  'PRIoLEAST8',
  'PRIoMAX',
  'PRIoPTR',
  'PRIu128',
  'PRIu16',
  'PRIu32',
  'PRIu64',
  'PRIu8',
  'PRIuFAST128',
  'PRIuFAST16',
  'PRIuFAST32',
  'PRIuFAST64',
  'PRIuFAST8',
  'PRIuLEAST128',
  'PRIuLEAST16',
  'PRIuLEAST32',
  'PRIuLEAST64',
  'PRIuLEAST8',
  'PRIuMAX',
  'PRIuPTR',
  'PRIx128',
  'PRIx16',
  'PRIx32',
  'PRIx64',
  'PRIx8',
  'PRIxFAST128',
  'PRIxFAST16',
  'PRIxFAST32',
  'PRIxFAST64',
  'PRIxFAST8',
  'PRIxLEAST128',
  'PRIxLEAST16',
  'PRIxLEAST32',
  'PRIxLEAST64',
  'PRIxLEAST8',
  'PRIxMAX',
  'PRIxPTR',
  'PTRDIFF_MAX',
  'PTRDIFF_MIN',
  'P_tmpdir',
  'RE_DUP_MAX',
  'SCHAR_MAX',
  'SCHAR_MIN',
  'SCNb128',
  'SCNb16',
  'SCNb32',
  'SCNb64',
  'SCNb8',
  'SCNbFAST128',
  'SCNbFAST16',
  'SCNbFAST32',
  'SCNbFAST64',
  'SCNbFAST8',
  'SCNbLEAST128',
  'SCNbLEAST16',
  'SCNbLEAST32',
  'SCNbLEAST64',
  'SCNbLEAST8',
  'SCNd128',
  'SCNd16',
  'SCNd32',
  'SCNd64',
  'SCNd8',
  'SCNdFAST128',
  'SCNdFAST16',
  'SCNdFAST32',
  'SCNdFAST64',
  'SCNdFAST8',
  'SCNdLEAST128',
  'SCNdLEAST16',
  'SCNdLEAST32',
  'SCNdLEAST64',
  'SCNdLEAST8',
  'SCNdMAX',
  'SCNdPTR',
  'SCNi128',
  'SCNi16',
  'SCNi32',
  'SCNi64',
  'SCNi8',
  'SCNiFAST128',
  'SCNiFAST16',
  'SCNiFAST32',
  'SCNiFAST64',
  'SCNiFAST8',
  'SCNiLEAST128',
  'SCNiLEAST16',
  'SCNiLEAST32',
  'SCNiLEAST64',
  'SCNiLEAST8',
  'SCNiMAX',
  'SCNiPTR',
  'SCNo128',
  'SCNo16',
  'SCNo32',
  'SCNo64',
  'SCNo8',
  'SCNoFAST128',
  'SCNoFAST16',
  'SCNoFAST32',
  'SCNoFAST64',
  'SCNoFAST8',
  'SCNoLEAST128',
  'SCNoLEAST16',
  'SCNoLEAST32',
  'SCNoLEAST64',
  'SCNoLEAST8',
  'SCNoMAX',
  'SCNoPTR',
  'SCNu128',
  'SCNu16',
  'SCNu32',
  'SCNu64',
  'SCNu8',
  'SCNuFAST128',
  'SCNuFAST16',
  'SCNuFAST32',
  'SCNuFAST64',
  'SCNuFAST8',
  'SCNuLEAST128',
  'SCNuLEAST16',
  'SCNuLEAST32',
  'SCNuLEAST64',
  'SCNuLEAST8',
  'SCNuMAX',
  'SCNuPTR',
  'SCNx128',
  'SCNx16',
  'SCNx32',
  'SCNx64',
  'SCNx8',
  'SCNxFAST128',
  'SCNxFAST16',
  'SCNxFAST32',
  'SCNxFAST64',
  'SCNxFAST8',
  'SCNxLEAST128',
  'SCNxLEAST16',
  'SCNxLEAST32',
  'SCNxLEAST64',
  'SCNxLEAST8',
  'SCNxMAX',
  'SCNxPTR',
  'SEM_NSEMS_MAX',
  'SEM_VALUE_MAX',
  'SHRT_MAX',
  'SHRT_MIN',
  'SIG_ATOMIC_MAX',
  'SIG_ATOMIC_MIN',
  'SIZE_MAX',
  'SIZE_MIN',
  'SSIZE_MAX',
  'SYMLOOP_MAX',
  'TMP_MAX',
  'TTY_NAME_MAX',
  'TZNAME_MAX',
  'UCHAR_MAX',
  'UCHAR_MIN',
  'UINT128_MAX',
  'UINT128_MIN',
  'UINT16_MAX',
  'UINT16_MIN',
  'UINT32_MAX',
  'UINT32_MIN',
  'UINT64_MAX',
  'UINT64_MIN',
  'UINT8_MAX',
  'UINT8_MIN',
  'UINTMAX_MAX',
  'UINTMAX_MIN',
  'UINTPTR_MAX',
  'UINTPTR_MIN',
  'UINT_FAST16_MAX',
  'UINT_FAST32_MAX',
  'UINT_FAST64_MAX',
  'UINT_FAST8_MAX',
  'UINT_LEAST16_MAX',
  'UINT_LEAST32_MAX',
  'UINT_LEAST64_MAX',
  'UINT_LEAST8_MAX',
  'UINT_MAX',
  'UINT_MIN',
  'ULLONG_MAX',
  'ULLONG_MIN',
  'ULONG_LONG_MAX',
  'ULONG_LONG_MIN',
  'ULONG_MAX',
  'ULONG_MIN',
  'USHRT_MAX',
  'USHRT_MIN',
  'WCHAR_MAX',
  'WCHAR_MIN',
  'WEOF',
  'WINT_MAX',
  'WINT_MIN',
  'WORD_BIT',
  '_ARCH_PWR5X',
  '_BSD_SOURCE',
  '_COSMO_SOURCE',
  '_GNU_SOURCE',
  '_IOFBF',
  '_IOLBF',
  '_IONBF',
  '_MSC_VER',
  '_WIN32',
  '_XOPEN_SOURCE',
  '__AARCH64EB__',
  '__ABM__',
  '__ADX__',
  '__AES__',
  '__ANDROID__',
  '__APPLE__',
  '__ARM_ARCH',
  '__ARM_FEATURE_ATOMICS',
  '__ARM_FEATURE_CLZ',
  '__ARM_FEATURE_CRC32',
  '__ARM_FEATURE_CRYPTO',
  '__ARM_FEATURE_DOTPROD',
  '__ARM_FEATURE_FMA',
  '__ARM_FEATURE_FP16_FML',
  '__ARM_FEATURE_FP16_VECTOR_ARITHMETIC',
  '__ARM_FEATURE_FRINT',
  '__ARM_FEATURE_MATMUL_INT8',
  '__ARM_FEATURE_NUMERIC_MAXMIN',
  '__ARM_FEATURE_QBIT',
  '__ARM_FEATURE_QRDMX',
  '__ARM_FEATURE_RNG',
  '__ARM_FEATURE_SHA2',
  '__ARM_FEATURE_SHA3',
  '__ARM_FEATURE_SHA512',
  '__ARM_FEATURE_SM3',
  '__ARM_FEATURE_SM4',
  '__ARM_FP16_ARGS',
  '__ARM_FP16_FORMAT_ALTERNATIVE',
  '__ARM_FP16_IEEE',
  '__ARM_FP_FAST',
  '__ARM_NEON',
  '__ASSEMBLER__',
  '__AVX2__',
  '__AVX5124VNNIW__',
  '__AVX512BF16__',
  '__AVX512BW__',
  '__AVX512CD__',
  '__AVX512DQ__',
  '__AVX512FP16__',
  '__AVX512F__',
  '__AVX512IFMA__',
  '__AVX512VBMI__',
  '__AVX512VL__',
  '__AVX512VNNI__',
  '__AVXVNNIINT16__',
  '__AVXVNNIINT8__',
  '__AVXVNNI__',
  '__AVX__',
  '__BASE_FILE__',
  '__BIGGEST_ALIGNMENT__',
  '__BMI2__',
  '__BMI__',
  '__BUILTIN_CPU_SUPPORTS__',
  '__BYTE_ORDER__',
  '__CET__',
  '__CHAR_BIT__',
  '__CLFLUSHOPT__',
  '__CLZERO__',
  '__COSMOCC__',
  '__COSMOPOLITAN__',
  '__COUNTER__',
  '__CRTDLL__',
  '__CUDA_ARCH_LIST__',
  '__CUDA_ARCH__',
  '__CYGWIN__',
  '__DATE__',
  '__ELF__',
  '__EMSCRIPTEN__',
  '__F16C__',
  '__FAST_MATH__',
  '__FATCOSMOCC__',
  '__FILE__',
  '__FINITE_MATH_ONLY__',
  '__FLT16_DECIMAL_DIG__',
  '__FLT16_DENORM_MIN__',
  '__FLT16_DIG__',
  '__FLT16_EPSILON__',
  '__FLT16_HAS_DENORM__',
  '__FLT16_HAS_INFINITY__',
  '__FLT16_HAS_QUIET_NAN__',
  '__FLT16_IS_IEC_60559__',
  '__FLT16_MANT_DIG__',
  '__FLT16_MAX_10_EXP__',
  '__FLT16_MAX_EXP__',
  '__FLT16_MAX__',
  '__FLT16_MIN_10_EXP__',
  '__FLT16_MIN_EXP__',
  '__FLT16_MIN__',
  '__FLT16_NORM_MAX__',
  '__FLT_EVAL_METHOD__',
  '__FMA4__',
  '__FMA__',
  '__FNO_OMIT_FRAME_POINTER__',
  '__FUNCTION__',
  '__FreeBSD__',
  '__Fuchsia__',
  '__GCC_ASM_FLAG_OUTPUTS__',
  '__GLIBC__',
  '__GNUC_GNU_INLINE__',
  '__GNUC_MINOR__',
  '__GNUC_PATCHLEVEL__',
  '__GNUC_STDC_INLINE__',
  '__GNUC__',
  '__GNUG__',
  '__HAIKU__',
  '__HIP_PLATFORM_AMD__',
  '__HIP__',
  '__INCLUDE_LEVEL__',
  '__INT16_MAX__',
  '__INT32_MAX__',
  '__INT64_MAX__',
  '__INT8_MAX__',
  '__INTEL_COMPILER',
  '__INTMAX_WIDTH__',
  '__INTPTR_MAX__',
  '__INTPTR_WIDTH__',
  '__INT_FAST16_MAX__',
  '__INT_FAST16_WIDTH__',
  '__INT_FAST32_MAX__',
  '__INT_FAST32_WIDTH__',
  '__INT_FAST64_MAX__',
  '__INT_FAST64_WIDTH__',
  '__INT_FAST8_MAX__',
  '__INT_FAST8_WIDTH__',
  '__INT_LEAST16_MAX__',
  '__INT_LEAST16_WIDTH__',
  '__INT_LEAST32_MAX__',
  '__INT_LEAST32_WIDTH__',
  '__INT_LEAST64_MAX__',
  '__INT_LEAST64_WIDTH__',
  '__INT_LEAST8_MAX__',
  '__INT_LEAST8_WIDTH__',
  '__INT_WIDTH__',
  '__LINE__',
  '__LINKER__',
  '__LIW__',
  '__LONG_LONG_WIDTH__',
  '__LONG_WIDTH__',
  '__MACH__',
  '__MFENTRY__',
  '__MICROBLAZE__',
  '__MINGW32__',
  '__MNOP_MCOUNT__',
  '__MNO_RED_ZONE__',
  '__MNO_VZEROUPPER__',
  '__MRECORD_MCOUNT__',
  '__MSVCRT_VERSION__',
  '__MSVCRT__',
  '__MWAITX__',
  '__NEXT_RUNTIME__',
  '__NO_LIW__',
  '__NO_MATH_ERRNO__',
  '__NO_SETLB__',
  '__NetBSD_Version__',
  '__NetBSD__',
  '__OPTIMIZE__',
  '__ORDER_BIG_ENDIAN__',
  '__ORDER_LITTLE_ENDIAN__',
  '__ORDER_PDP_ENDIAN__',
  '__OpenBSD__',
  '__PCLMUL__',
  '__PIC__',
  '__PIE__',
  '__POPCNT__',
  '__POWER9_VECTOR__',
  '__POWERPC__',
  '__PRETTY_FUNCTION__',
  '__PTRDIFF_MAX__',
  '__PTRDIFF_WIDTH__',
  '__PTX_ISA_VERSION_MAJOR__',
  '__PTX_ISA_VERSION_MINOR__',
  '__PTX_SM__',
  '__RDPID__',
  '__RDRND__',
  '__RDSEED__',
  '__ROUNDING_MATH__',
  '__RX_ALLOW_STRING_INSNS__',
  '__RX_DISALLOW_STRING_INSNS__',
  '__SCHAR_WIDTH__',
  '__SETLB__',
  '__SET_FPSCR_RN_RETURNS_FPSCR__',
  '__SGX__',
  '__SHA__',
  '__SHRT_WIDTH__',
  '__SIG_ATOMIC_MAX__',
  '__SIG_ATOMIC_MIN__',
  '__SIG_ATOMIC_WIDTH__',
  '__SIZEOF_DOUBLE__',
  '__SIZEOF_FLOAT__',
  '__SIZEOF_INTMAX__',
  '__SIZEOF_INT__',
  '__SIZEOF_LONG_DOUBLE__',
  '__SIZEOF_LONG_LONG__',
  '__SIZEOF_LONG__',
  '__SIZEOF_POINTER__',
  '__SIZEOF_PTRDIFF_T__',
  '__SIZEOF_SHORT__',
  '__SIZEOF_SIZE_T__',
  '__SIZEOF_UINTMAX__',
  '__SIZEOF_WCHAR_T__',
  '__SIZEOF_WINT_T__',
  '__SIZE_MAX__',
  '__SIZE_WIDTH__',
  '__SSE2__',
  '__SSE3__',
  '__SSE4A__',
  '__SSE4_1__',
  '__SSE4_2__',
  '__SSE__',
  '__SSSE3__',
  '__STDC_ANALYZABLE__',
  '__STDC_DEC_FP__',
  '__STDC_HOSTED__',
  '__STDC_IEC_559_COMPLEX__',
  '__STDC_IEC_559__',
  '__STDC_ISO_10646__',
  '__STDC_LIB_EXT1__',
  '__STDC_MB_MIGHT_NEQ_WC__',
  '__STDC_NO_ATOMICS__',
  '__STDC_NO_COMPLEX__',
  '__STDC_NO_THREADS__',
  '__STDC_NO_VLA__',
  '__STDC_UTF_16__',
  '__STDC_UTF_32__',
  '__STDC_VERSION__',
  '__STDC_WANT_LIB_EXT1__',
  '__STDC__',
  '__STRICT_ANSI__',
  '__SUPPORT_SNAN__',
  '__TIMESTAMP__',
  '__TIME__',
  '__TM_FENCE__',
  '__UINT16_MAX__',
  '__UINT32_MAX__',
  '__UINT64_MAX__',
  '__UINT8_MAX__',
  '__UINTMAX_MAX__',
  '__UINTPTR_MAX__',
  '__UINT_FAST16_MAX__',
  '__UINT_FAST32_MAX__',
  '__UINT_FAST64_MAX__',
  '__UINT_FAST8_MAX__',
  '__UINT_LEAST16_MAX__',
  '__UINT_LEAST32_MAX__',
  '__UINT_LEAST64_MAX__',
  '__UINT_LEAST8_MAX__',
  '__VAES__',
  '__VA_ARGS__',
  '__VA_OPT__',
  '__VEC__',
  '__VPCLMULQDQ__',
  '__VSX__',
  '__WCHAR_MAX__',
  '__WCHAR_MIN__',
  '__WCHAR_UNSIGNED__',
  '__WCHAR_WIDTH__',
  '__WINT_MAX__',
  '__WINT_MIN__',
  '__WINT_WIDTH__',
  '__XSAVE__',
  '__XXX__',
  '__aarch64__',
  '__amd64__',
  '__arm__',
  '__chibicc__',
  '__clang__',
  '__cplusplus',
  '__func__',
  '__gun_linux__',
  '__i386__',
  '__i486__',
  '__i586__',
  '__i686__',
  '__ia16__',
  '__linux__',
  '__llvm__',
  '__m68k__',
  '__mips64',
  '__mips__',
  '__powerpc64__',
  '__powerpc__',
  '__ppc__',
  '__riscv',
  '__riscv_flen',
  '__riscv_xlen',
  '__s390__',
  '__s390x__',
  '__wasm_simd128__',
  '__x86_64__',
  'false',
  'true',
]);

const C_PODS = new Set([
  'enum',
  'struct',
  'union',
]);

const C_TYPES = new Set([
  'DIR',
  'FILE',
  '__m128',
  '__m128i',
  '__m256',
  '__m256i',
  '__m512',
  '__m512bh',
  '__m512i',
  'atomic_bool',
  'atomic_bool32',
  'atomic_char',
  'atomic_char16_t',
  'atomic_char32_t',
  'atomic_flag',
  'atomic_int',
  'atomic_int_fast16_t',
  'atomic_int_fast32_t',
  'atomic_int_fast64_t',
  'atomic_int_fast8_t',
  'atomic_int_least16_t',
  'atomic_int_least32_t',
  'atomic_int_least64_t',
  'atomic_int_least8_t',
  'atomic_intptr_t',
  'atomic_llong',
  'atomic_long',
  'atomic_ptrdiff_t',
  'atomic_schar',
  'atomic_short',
  'atomic_size_t',
  'atomic_uchar',
  'atomic_uint',
  'atomic_uint_fast16_t',
  'atomic_uint_fast32_t',
  'atomic_uint_fast64_t',
  'atomic_uint_fast8_t',
  'atomic_uint_least16_t',
  'atomic_uint_least32_t',
  'atomic_uint_least64_t',
  'atomic_uint_least8_t',
  'atomic_uintptr_t',
  'atomic_ullong',
  'atomic_ulong',
  'atomic_ushort',
  'atomic_wchar_t',
  'axdx_t',
  'bfloat16_t',
  'blkcnt_t',
  'blksize_t',
  'bool32',
  'cc_t',
  'char16_t',
  'char32_t',
  'clock_t',
  'clockid_t',
  'cpu_set_t',
  'data_t',
  'dev_t',
  'dim3',
  'div_t',
  'double_t',
  'errno_t',
  'fenv_t',
  'fexcept_t',
  'float128_t',
  'float16_t',
  'float16x8_t',
  'float32_t',
  'float32x4_t',
  'float64_t',
  'float_t',
  'fpos_t',
  'fsblkcnt_t',
  'fsfilcnt_t',
  'gid_t',
  'glob_t',
  'iconv_t',
  'id_t',
  'idtype_t',
  'imaxdiv_t',
  'in_addr_t',
  'in_port_t',
  'ino_t',
  'int128_t',
  'int16_t',
  'int32_t',
  'int64_t',
  'int8_t',
  'intN_t',
  'int_fast16_t',
  'int_fast32_t',
  'int_fast64_t',
  'int_fast8_t',
  'int_least16_t',
  'int_least32_t',
  'int_least64_t',
  'int_least8_t',
  'intmax_t',
  'intptr_t',
  'key_t',
  'ldiv_t',
  'lldiv_t',
  'locale_t',
  'loff_t',
  'max_align_t',
  'mbstate_t',
  'mcontext_t',
  'mode_t',
  'mqd_t',
  'msglen_t',
  'msgqnum_t',
  'nfds_t',
  'nlink_t',
  'off_t',
  'pid_t',
  'posix_spawn_file_actions_t',
  'posix_spawnattr_t',
  'posix_trace_attr_t',
  'pthread_attr_t',
  'pthread_barrier_t',
  'pthread_barrierattr_t',
  'pthread_cond_t',
  'pthread_condattr_t',
  'pthread_key_t',
  'pthread_mutex_t',
  'pthread_mutexattr_t',
  'pthread_once_t',
  'pthread_rwlock_t',
  'pthread_rwlockattr_t',
  'pthread_spinlock_t',
  'pthread_t',
  'ptrdiff_t',
  'regex_t',
  'register_t',
  'regmatch_t',
  'regoff_t',
  'rlim_t',
  'rlimit_t',
  'sa_family_t',
  'sem_t',
  'semaphore_t',
  'shmatt_t',
  'sig_atomic_t',
  'siginfo_t',
  'sigset_t',
  'size_t',
  'socklen_t',
  'speed_t',
  'ssize_t',
  'ssizet_t',
  'stack_t',
  'suseconds_t',
  'syscall_arg_t',
  't_scalar_t',
  't_uscalar_t',
  'tcflag_t',
  'thrd_t',
  'time_t',
  'timed_mutex_t',
  'timed_thread_t',
  'timer_t',
  'trace_attr_t',
  'trace_event_id_t',
  'trace_event_set_t',
  'trace_id_t',
  'ucontext_t',
  'uid_t',
  'uint128_t',
  'uint16_t',
  'uint24_t',
  'uint32_t',
  'uint64_t',
  'uint8_t',
  'uint_fast16_t',
  'uint_fast32_t',
  'uint_fast64_t',
  'uint_fast8_t',
  'uint_least16_t',
  'uint_least32_t',
  'uint_least64_t',
  'uint_least8_t',
  'uintmax_t',
  'uintptr_t',
  'useconds_t',
  'ushort_t',
  'va_list',
  'wchar_t',
  'wctrans_t',
  'wctype_t',
  'wint_t',
  'wordexp_t',
]);

class HighlightC extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static SLASH = 6;
  static SLASH_SLASH = 7;
  static SLASH_STAR = 8;
  static SLASH_SLASH_BACKSLASH = 9;
  static SLASH_STAR_STAR = 10;
  static R = 11;
  static R_DQUOTE = 12;
  static RAW = 13;
  static QUESTION = 14;
  static TRIGRAPH = 15;
  static CPP_LT = 16;
  static BACKSLASH = 17;
  static UNIVERSAL = 18;

  constructor(delegate) {
    super(delegate);
    this.last = 0;
    this.word = '';
    this.current = 0;
    this.t = 0;
    this.i = 0;
    this.is_pod = 0;
    this.is_bol = true;
    this.is_cpp = false;
    this.is_define = false;
    this.is_include = false;
    this.is_cpp_builtin = false;
    this.heredoc = '';
    this.keywords = C_KEYWORDS;
    this.builtins = C_BUILTINS;
    this.constants = C_CONSTANTS;
    this.types = C_TYPES;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      const c = input[i];

      switch (this.state) {

      case HighlightC.NORMAL:
        if (c == 'R') {
          this.state = HighlightC.R;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightC.WORD;
          this.word += c;
        } else if (c == '#' && this.is_bol && !this.is_cpp && !this.is_define) {
          this.is_cpp = true;
          this.is_cpp_builtin = true;
          this.push('span', 'builtin');
          this.append('#');
        } else if (c == '<' && this.is_include) {
          this.push('span', 'string');
          this.append('<');
          this.state = HighlightC.CPP_LT;
        } else if (c == '\\') {
          this.state = HighlightC.BACKSLASH;
        } else if (c == '/') {
          this.state = HighlightC.SLASH;
        } else if (c == '?') {
          this.state = HighlightC.QUESTION;
        } else if (c == '\'') {
          this.state = HighlightC.QUOTE;
          this.push('span', 'string');
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightC.DQUOTE;
          this.push('span', 'string');
          this.append('"');
        } else if (c == '\n') {
          this.append('\n');
          if (this.is_cpp) {
            if (this.is_cpp_builtin) {
              this.pop();
            }
            this.is_include = false;
            this.is_cpp = false;
          }
          this.is_define = false;
          this.is_cpp_builtin = false;
        } else {
          this.append(c);
        }
        break;

      case HighlightC.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '$') {
          this.word += c;
        } else {
          if (this.is_cpp) {
            if (CPP_KEYWORDS.has(this.word)) {
              this.push('span', 'builtin');
              this.append(this.word);
              this.pop();
              if (this.word == "include" || this.word == "include_next")
                this.is_include = true;
              if (this.word == "define") {
                this.is_cpp = false;
                this.is_define = true;
              }
              if (this.is_cpp_builtin) {
                this.pop();
                this.is_cpp_builtin = false;
              }
            } else if (C_CONSTANTS.has(this.word)) {
              this.push('span', 'constant');
              this.append(this.word);
              this.pop();
            } else {
              this.append(this.word);
            }
          } else {
            if (this.keywords.has(this.word)) {
              this.push('span', 'keyword');
              this.append(this.word);
              this.pop();
              if (C_PODS.has(this.word))
                this.is_pod = true;
            } else if (this.is_pod || (this.types && this.types.has(this.word))) {
              this.push('span', 'type');
              this.append(this.word);
              this.pop();
              this.is_pod = false;
            } else if (this.builtin && this.builtins.has(this.word)) {
              this.push('span', 'builtin');
              this.append(this.word);
              this.pop();
              this.is_pod = false;
            } else if (this.constants && this.constants.has(this.word)) {
              this.push('span', 'constant');
              this.append(this.word);
              this.pop();
              this.is_pod = false;
            } else {
              this.append(this.word);
              this.is_pod = false;
            }
          }
          this.word = '';
          this.epsilon(HighlightC.NORMAL);
        }
        break;

      case HighlightC.BACKSLASH:
        if (c == 'u') {
          this.push('span', 'escape');
          this.append("\\u");
          this.state = HighlightC.UNIVERSAL;
          this.i = 4;
        } else if (c == 'U') {
          this.push('span', 'escape');
          this.append("\\U");
          this.state = HighlightC.UNIVERSAL;
          this.i = 8;
        } else {
          this.append('\\');
          this.append(c);
          this.state = HighlightC.NORMAL;
        }
        break;

      case HighlightC.UNIVERSAL:
        if (isascii(c) && isxdigit(c)) {
          this.append(c);
          if (!--this.i) {
            this.pop();
            this.state = HighlightC.NORMAL;
          }
        } else {
          this.pop();
          this.epsilon(HighlightC.NORMAL);
        }
        break;

      case HighlightC.QUESTION:
        if (c == '?') {
          this.state = HighlightC.TRIGRAPH;
        } else {
          this.append('?');
          this.epsilon(HighlightC.NORMAL);
        }
        break;

      case HighlightC.TRIGRAPH:
        if (c == '=' || // '#'
            c == '(' || // '['
            c == '/' || // '\\'
            c == ')' || // ']'
            c == '\'' || // '^'
            c == '<' || // '{'
            c == '!' || // '|'
            c == '>' || // '}'
            c == '-') { // '~'
          this.push('span', 'escape');
          this.append("??");
          this.append(c);
          this.pop();
          this.state = HighlightC.NORMAL;
        } else {
          this.append("??");
          this.epsilon(HighlightC.NORMAL);
        }
        break;

      case HighlightC.SLASH:
        if (c == '/') {
          this.push('span', 'comment');
          this.append("//");
          this.state = HighlightC.SLASH_SLASH;
        } else if (c == '*') {
          this.push('span', 'comment');
          this.append("/*");
          this.state = HighlightC.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightC.NORMAL);
        }
        break;

      case HighlightC.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightC.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightC.SLASH_SLASH_BACKSLASH;
        }
        break;

      case HighlightC.SLASH_SLASH_BACKSLASH:
        this.append(c);
        this.state = HighlightC.SLASH_SLASH;
        break;

      case HighlightC.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightC.SLASH_STAR_STAR;
        break;

      case HighlightC.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightC.NORMAL;
        } else if (c != '*') {
          this.state = HighlightC.SLASH_STAR;
        }
        break;

      case HighlightC.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightC.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightC.QUOTE_BACKSLASH;
        }
        break;

      case HighlightC.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightC.QUOTE;
        break;

      case HighlightC.CPP_LT:
        this.append(c);
        if (c == '>') {
          this.pop();
          this.state = HighlightC.NORMAL;
        }
        break;

      case HighlightC.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightC.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightC.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightC.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightC.DQUOTE;
        break;

      case HighlightC.R:
        if (c == '"') {
          this.state = HighlightC.R_DQUOTE;
          this.append('R');
          this.push('span', 'string');
          this.append('"');
          this.heredoc = ")";
        } else {
          this.word += 'R';
          this.epsilon(HighlightC.WORD);
        }
        break;

      case HighlightC.R_DQUOTE:
        this.append(c);
        if (c == '(') {
          this.state = HighlightC.RAW;
          this.i = 0;
          this.heredoc += '"';
        } else {
          this.heredoc += c;
        }
        break;

      case HighlightC.RAW:
        this.append(c);
        if (this.heredoc[this.i] == c) {
          if (++this.i == this.heredoc.length) {
            this.state = HighlightC.NORMAL;
            this.pop();
          }
        } else {
          this.i = 0;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
      if (this.is_bol) {
        if (!isspace(c))
          this.is_bol = false;
      } else {
        if (c == '\n')
          this.is_bol = true;
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightC.WORD:
      if (this.is_cpp) {
        if (CPP_KEYWORDS.has(this.word)) {
          this.push('span', 'builtin');
          this.append(this.word);
          this.pop();
        } else if (C_CONSTANTS.has(this.word)) {
          this.push('span', 'constant');
          this.append(this.word);
          this.pop();
        } else {
          this.append(this.word);
        }
      } else {
        if (this.keywords.has(this.word)) {
          this.push('span', 'keyword');
          this.append(this.word);
          this.pop();
        } else if (this.is_pod || (this.types && this.types.has(this.word))) {
          this.push('span', 'type');
          this.append(this.word);
          this.pop();
        } else if (this.builtins && this.builtins.has(this.word)) {
          this.push('span', 'builtin');
          this.append(this.word);
          this.pop();
        } else if (this.constants && this.constants.has(this.word)) {
          this.push('span', 'constant');
          this.append(this.word);
          this.pop();
        } else {
          this.append(this.word);
        }
      }
      this.word = '';
      break;
    case HighlightC.SLASH:
      this.append('/');
      break;
    case HighlightC.QUESTION:
      this.append('?');
      break;
    case HighlightC.TRIGRAPH:
      this.append("??");
      break;
    case HighlightC.R:
      this.append('R');
      break;
    case HighlightC.BACKSLASH:
      this.append('\\');
      break;
    case HighlightC.QUOTE:
    case HighlightC.QUOTE_BACKSLASH:
    case HighlightC.DQUOTE:
    case HighlightC.DQUOTE_BACKSLASH:
    case HighlightC.SLASH_SLASH:
    case HighlightC.SLASH_SLASH_BACKSLASH:
    case HighlightC.SLASH_STAR:
    case HighlightC.SLASH_STAR_STAR:
    case HighlightC.R_DQUOTE:
    case HighlightC.RAW:
      this.pop();
      break;
    default:
      break;
    }
    if (this.is_cpp) {
      if (this.is_cpp_builtin) {
        this.pop();
      }
      this.is_cpp = false;
    }
    this.is_cpp_builtin = false;
    this.is_include = false;
    this.is_define = false;
    this.is_pod = false;
    this.is_bol = true;
    this.state = HighlightC.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['c'] = HighlightC;
Highlighter.REGISTRY['h'] = HighlightC;
Highlighter.REGISTRY['m'] = HighlightC;
