#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ECP_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ECP_H_
#include "third_party/mbedtls/bignum.h"
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_ECP_BAD_INPUT_DATA                    -0x4F80  /*< Bad input parameters to function. */
#define MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL                  -0x4F00  /*< The buffer is too small to write to. */
#define MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE               -0x4E80  /*< The requested feature is not available, for example, the requested curve is not supported. */
#define MBEDTLS_ERR_ECP_VERIFY_FAILED                     -0x4E00  /*< The signature is not valid. */
#define MBEDTLS_ERR_ECP_ALLOC_FAILED                      -0x4D80  /*< Memory allocation failed. */
#define MBEDTLS_ERR_ECP_RANDOM_FAILED                     -0x4D00  /*< Generation of random value, such as ephemeral key, failed. */
#define MBEDTLS_ERR_ECP_INVALID_KEY                       -0x4C80  /*< Invalid private or public key. */
#define MBEDTLS_ERR_ECP_SIG_LEN_MISMATCH                  -0x4C00  /*< The buffer contains a valid signature followed by more data. */
#define MBEDTLS_ERR_ECP_HW_ACCEL_FAILED                   -0x4B80  /*< The ECP hardware accelerator failed. */
#define MBEDTLS_ERR_ECP_IN_PROGRESS                       -0x4B00  /*< Operation in progress, call again with the same parameters to continue. */

/**
 * Domain-parameter identifiers: curve, subgroup, and generator.
 *
 * \note Only curves over prime fields are supported.
 *
 * \warning This library does not support validation of arbitrary domain
 * parameters. Therefore, only standardized domain parameters from trusted
 * sources should be used. See mbedtls_ecp_group_load().
 */
typedef enum
{
    /* Note: when adding a new curve:
     * - Add it at the end of this enum, otherwise you'll break the ABI by
     *   changing the numerical value for existing curves.
     * - Increment MBEDTLS_ECP_DP_MAX below if needed.
     * - Add the corresponding MBEDTLS_ECP_DP_xxx_ENABLED macro definition to
     *   config.h.
     * - List the curve as a dependency of MBEDTLS_ECP_C and
     *   MBEDTLS_ECDSA_C if supported in check.h.
     * - Add the curve to the appropriate curve type macro
     *   MBEDTLS_ECP_yyy_ENABLED above.
     * - Add the necessary definitions to ecp_curves.c.
     * - Add the curve to the ecp_supported_curves array in ecp.c.
     * - Add the curve to applicable profiles in x509_crt.c if applicable.
     */
    MBEDTLS_ECP_DP_NONE = 0,       /*!< Curve not defined. */
    MBEDTLS_ECP_DP_SECP192R1,      /*!< Domain parameters for the 192-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP224R1,      /*!< Domain parameters for the 224-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP256R1,      /*!< Domain parameters for the 256-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP384R1,      /*!< Domain parameters for the 384-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_SECP521R1,      /*!< Domain parameters for the 521-bit curve defined by FIPS 186-4 and SEC1. */
    MBEDTLS_ECP_DP_BP256R1,        /*!< Domain parameters for 256-bit Brainpool curve. */
    MBEDTLS_ECP_DP_BP384R1,        /*!< Domain parameters for 384-bit Brainpool curve. */
    MBEDTLS_ECP_DP_BP512R1,        /*!< Domain parameters for 512-bit Brainpool curve. */
    MBEDTLS_ECP_DP_CURVE25519,     /*!< Domain parameters for Curve25519. */
    MBEDTLS_ECP_DP_SECP192K1,      /*!< Domain parameters for 192-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_SECP224K1,      /*!< Domain parameters for 224-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_SECP256K1,      /*!< Domain parameters for 256-bit "Koblitz" curve. */
    MBEDTLS_ECP_DP_CURVE448,       /*!< Domain parameters for Curve448. */
} mbedtls_ecp_group_id;

/**
 * The number of supported curves, plus one for #MBEDTLS_ECP_DP_NONE.
 *
 * \note Montgomery curves are currently excluded.
 */
#define MBEDTLS_ECP_DP_MAX     12

#define MBEDTLS_ECP_PF_UNCOMPRESSED    0   /*< Uncompressed point format (RFC4492) */
#define MBEDTLS_ECP_PF_COMPRESSED      1   /*< Compressed point format (RFC4492) */
#define MBEDTLS_ECP_TLS_NAMED_CURVE    3   /*< The named_curve of ECCurveType (RFC4492) */

/*
 * Curve types
 */
typedef enum
{
    MBEDTLS_ECP_TYPE_NONE = 0,
    MBEDTLS_ECP_TYPE_SHORT_WEIERSTRASS,    /* y^2 = x^3 + a x + b      */
    MBEDTLS_ECP_TYPE_MONTGOMERY,           /* y^2 = x^3 + a x^2 + x    */
} mbedtls_ecp_curve_type;

/**
 * Curve information, for use by other modules.
 */
typedef struct mbedtls_ecp_curve_info
{
    mbedtls_ecp_group_id grp_id;    /*!< An internal identifier. */
    uint16_t tls_id;                /*!< The TLS NamedCurve identifier. */
    uint16_t bit_size;              /*!< The curve size in bits. */
    const char *name;               /*!< A human-friendly name. */
} mbedtls_ecp_curve_info;

/**
 * \brief           The ECP point structure, in Jacobian coordinates.
 *
 * \note            All functions expect and return points satisfying
 *                  the following condition: <code>Z == 0</code> or
 *                  <code>Z == 1</code>. Other values of \p Z are
 *                  used only by internal functions.
 *                  The point is zero, or "at infinity", if <code>Z == 0</code>.
 *                  Otherwise, \p X and \p Y are its standard (affine)
 *                  coordinates.
 */
typedef struct mbedtls_ecp_point
{
    mbedtls_mpi X;          /*!< The X coordinate of the ECP point. */
    mbedtls_mpi Y;          /*!< The Y coordinate of the ECP point. */
    mbedtls_mpi Z;          /*!< The Z coordinate of the ECP point. */
}
mbedtls_ecp_point;

#if !defined(MBEDTLS_ECP_ALT)
/*
 * default mbed TLS elliptic curve arithmetic implementation
 *
 * (in case MBEDTLS_ECP_ALT is defined then the developer has to provide an
 * alternative implementation for the whole module and it will replace this
 * one.)
 */

/**
 * \brief           The ECP group structure.
 *
 * We consider two types of curve equations:
 * <ul><li>Short Weierstrass: <code>y^2 = x^3 + A x + B mod P</code>
 * (SEC1 + RFC-4492)</li>
 * <li>Montgomery: <code>y^2 = x^3 + A x^2 + x mod P</code> (Curve25519,
 * Curve448)</li></ul>
 * In both cases, the generator (\p G) for a prime-order subgroup is fixed.
 *
 * For Short Weierstrass, this subgroup is the whole curve, and its
 * cardinality is denoted by \p N. Our code requires that \p N is an
 * odd prime as mbedtls_ecp_mul() requires an odd number, and
 * mbedtls_ecdsa_sign() requires that it is prime for blinding purposes.
 *
 * For Montgomery curves, we do not store \p A, but <code>(A + 2) / 4</code>,
 * which is the quantity used in the formulas. Additionally, \p nbits is
 * not the size of \p N but the required size for private keys.
 *
 * If \p modp is NULL, reduction modulo \p P is done using a generic algorithm.
 * Otherwise, \p modp must point to a function that takes an \p mbedtls_mpi in the
 * range of <code>0..2^(2*pbits)-1</code>, and transforms it in-place to an integer
 * which is congruent mod \p P to the given MPI, and is close enough to \p pbits
 * in size, so that it may be efficiently brought in the 0..P-1 range by a few
 * additions or subtractions. Therefore, it is only an approximative modular
 * reduction. It must return 0 on success and non-zero on failure.
 *
 * \note        Alternative implementations must keep the group IDs distinct. If
 *              two group structures have the same ID, then they must be
 *              identical.
 *
 */
typedef struct mbedtls_ecp_group
{
    mbedtls_ecp_group_id id;    /*!< An internal group identifier. */
    mbedtls_mpi P;              /*!< The prime modulus of the base field. */
    mbedtls_mpi A;              /*!< For Short Weierstrass: \p A in the equation. For
                                     Montgomery curves: <code>(A + 2) / 4</code>. */
    mbedtls_mpi B;              /*!< For Short Weierstrass: \p B in the equation.
                                     For Montgomery curves: unused. */
    mbedtls_ecp_point G;        /*!< The generator of the subgroup used. */
    mbedtls_mpi N;              /*!< The order of \p G. */
    size_t pbits;               /*!< The number of bits in \p P.*/
    size_t nbits;               /*!< For Short Weierstrass: The number of bits in \p P.
                                     For Montgomery curves: the number of bits in the
                                     private keys. */
    unsigned int h;             /*!< \internal 1 if the constants are static. */
    int (*modp)(mbedtls_mpi *); /*!< The function for fast pseudo-reduction
                                     mod \p P (see above).*/
    int (*t_pre)(mbedtls_ecp_point *, void *);  /*!< Unused. */
    int (*t_post)(mbedtls_ecp_point *, void *); /*!< Unused. */
    void *t_data;               /*!< Unused. */
    mbedtls_ecp_point *T;       /*!< Pre-computed points for ecp_mul_comb(). */
    size_t T_size;              /*!< The number of pre-computed points. */
}
mbedtls_ecp_group;

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h, or define them using the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_ECP_MAX_BITS)
/**
 * The maximum size of the groups, that is, of \c N and \c P.
 */
#define MBEDTLS_ECP_MAX_BITS     521   /*< The maximum size of groups, in bits. */
#endif

#define MBEDTLS_ECP_MAX_BYTES    ( ( MBEDTLS_ECP_MAX_BITS + 7 ) / 8 )
#define MBEDTLS_ECP_MAX_PT_LEN   ( 2 * MBEDTLS_ECP_MAX_BYTES + 1 )

#if !defined(MBEDTLS_ECP_WINDOW_SIZE)
/*
 * Maximum "window" size used for point multiplication.
 * Default: 6.
 * Minimum value: 2. Maximum value: 7.
 *
 * Result is an array of at most ( 1 << ( MBEDTLS_ECP_WINDOW_SIZE - 1 ) )
 * points used for point multiplication. This value is directly tied to EC
 * peak memory usage, so decreasing it by one should roughly cut memory usage
 * by two (if large curves are in use).
 *
 * Reduction in size may reduce speed, but larger curves are impacted first.
 * Sample performances (in ECDHE handshakes/s, with FIXED_POINT_OPTIM = 1):
 *      w-size:     6       5       4       3       2
 *      521       145     141     135     120      97
 *      384       214     209     198     177     146
 *      256       320     320     303     262     226
 *      224       475     475     453     398     342
 *      192       640     640     633     587     476
 */
#define MBEDTLS_ECP_WINDOW_SIZE    6   /*< The maximum window size used. */
#endif /* MBEDTLS_ECP_WINDOW_SIZE */

#if !defined(MBEDTLS_ECP_FIXED_POINT_OPTIM)
/*
 * Trade memory for speed on fixed-point multiplication.
 *
 * This speeds up repeated multiplication of the generator (that is, the
 * multiplication in ECDSA signatures, and half of the multiplications in
 * ECDSA verification and ECDHE) by a factor roughly 3 to 4.
 *
 * The cost is increasing EC peak memory usage by a factor roughly 2.
 *
 * Change this value to 0 to reduce peak memory usage.
 */
#define MBEDTLS_ECP_FIXED_POINT_OPTIM  1   /*< Enable fixed-point speed-up. */
#endif /* MBEDTLS_ECP_FIXED_POINT_OPTIM */

/* \} name SECTION: Module settings */

#endif /* MBEDTLS_ECP_ALT */

#if defined(MBEDTLS_ECP_RESTARTABLE)

/**
 * \brief           Internal restart context for multiplication
 *
 * \note            Opaque struct
 */
typedef struct mbedtls_ecp_restart_mul mbedtls_ecp_restart_mul_ctx;

/**
 * \brief           Internal restart context for ecp_muladd()
 *
 * \note            Opaque struct
 */
typedef struct mbedtls_ecp_restart_muladd mbedtls_ecp_restart_muladd_ctx;

/**
 * \brief           General context for resuming ECC operations
 */
typedef struct
{
    unsigned ops_done;                  /*!<  current ops count             */
    unsigned depth;                     /*!<  call depth (0 = top-level)    */
    mbedtls_ecp_restart_mul_ctx *rsm;   /*!<  ecp_mul_comb() sub-context    */
    mbedtls_ecp_restart_muladd_ctx *ma; /*!<  ecp_muladd() sub-context      */
} mbedtls_ecp_restart_ctx;

/*
 * Operation counts for restartable functions
 */
#define MBEDTLS_ECP_OPS_CHK   3 /*!< basic ops count for ecp_check_pubkey()  */
#define MBEDTLS_ECP_OPS_DBL   8 /*!< basic ops count for ecp_double_jac()    */
#define MBEDTLS_ECP_OPS_ADD  11 /*!< basic ops count for see ecp_add_mixed() */
#define MBEDTLS_ECP_OPS_INV 120 /*!< empirical equivalent for mpi_mod_inv()  */

/**
 * \brief           Internal; for restartable functions in other modules.
 *                  Check and update basic ops budget.
 *
 * \param grp       Group structure
 * \param rs_ctx    Restart context
 * \param ops       Number of basic ops to do
 *
 * \return          \c 0 if doing \p ops basic ops is still allowed,
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS otherwise.
 */
int mbedtls_ecp_check_budget( const mbedtls_ecp_group *grp,
                              mbedtls_ecp_restart_ctx *rs_ctx,
                              unsigned ops );

/* Utility macro for checking and updating ops budget */
#define MBEDTLS_ECP_BUDGET( ops )   \
    MBEDTLS_MPI_CHK( mbedtls_ecp_check_budget( grp, rs_ctx, \
                                               (unsigned) (ops) ) );

#else /* MBEDTLS_ECP_RESTARTABLE */

#define MBEDTLS_ECP_BUDGET( ops )   /* no-op; for compatibility */

/* We want to declare restartable versions of existing functions anyway */
typedef void mbedtls_ecp_restart_ctx;

#endif /* MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief    The ECP key-pair structure.
 *
 * A generic key-pair that may be used for ECDSA and fixed ECDH, for example.
 *
 * \note    Members are deliberately in the same order as in the
 *          ::mbedtls_ecdsa_context structure.
 */
typedef struct mbedtls_ecp_keypair
{
    mbedtls_ecp_group grp;      /*!<  Elliptic curve and base point     */
    mbedtls_mpi d;              /*!<  our secret value                  */
    mbedtls_ecp_point Q;        /*!<  our public value                  */
}
mbedtls_ecp_keypair;

const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_grp_id( mbedtls_ecp_group_id );
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_name( const char * );
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_info_from_tls_id( uint16_t );
const mbedtls_ecp_curve_info *mbedtls_ecp_curve_list( void );
const mbedtls_ecp_group_id *mbedtls_ecp_grp_id_list( void );
int mbedtls_ecp_check_privkey( const mbedtls_ecp_group *, const mbedtls_mpi * );
int mbedtls_ecp_check_pub_priv( const mbedtls_ecp_keypair *, const mbedtls_ecp_keypair * );
int mbedtls_ecp_check_pubkey( const mbedtls_ecp_group *, const mbedtls_ecp_point * );
int mbedtls_ecp_copy( mbedtls_ecp_point *, const mbedtls_ecp_point * );
int mbedtls_ecp_gen_key( mbedtls_ecp_group_id, mbedtls_ecp_keypair *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_ecp_gen_keypair( mbedtls_ecp_group *, mbedtls_mpi *, mbedtls_ecp_point *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_ecp_gen_keypair_base( mbedtls_ecp_group *, const mbedtls_ecp_point *, mbedtls_mpi *, mbedtls_ecp_point *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_ecp_gen_privkey( const mbedtls_ecp_group *, mbedtls_mpi *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_ecp_group_copy( mbedtls_ecp_group *, const mbedtls_ecp_group * );
int mbedtls_ecp_group_load( mbedtls_ecp_group *, mbedtls_ecp_group_id );
int mbedtls_ecp_is_zero( mbedtls_ecp_point * );
int mbedtls_ecp_mul( mbedtls_ecp_group *, mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point *, int (*)(void *, unsigned char *, size_t), void * );
int mbedtls_ecp_mul_restartable( mbedtls_ecp_group *, mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point *, int (*)(void *, unsigned char *, size_t), void *, mbedtls_ecp_restart_ctx * );
int mbedtls_ecp_muladd( mbedtls_ecp_group *, mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point * );
int mbedtls_ecp_muladd_restartable( mbedtls_ecp_group *, mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point *, const mbedtls_mpi *, const mbedtls_ecp_point *, mbedtls_ecp_restart_ctx * );
int mbedtls_ecp_point_cmp( const mbedtls_ecp_point *, const mbedtls_ecp_point * );
int mbedtls_ecp_point_read_binary( const mbedtls_ecp_group *, mbedtls_ecp_point *, const unsigned char *, size_t );
int mbedtls_ecp_point_read_string( mbedtls_ecp_point *, int, const char *, const char * );
int mbedtls_ecp_point_write_binary( const mbedtls_ecp_group *, const mbedtls_ecp_point *, int, size_t *, unsigned char *, size_t );
int mbedtls_ecp_read_key( mbedtls_ecp_group_id, mbedtls_ecp_keypair *, const unsigned char *, size_t );
int mbedtls_ecp_restart_is_enabled( void );
int mbedtls_ecp_self_test( int );
int mbedtls_ecp_set_zero( mbedtls_ecp_point * );
int mbedtls_ecp_tls_read_group( mbedtls_ecp_group *, const unsigned char **, size_t );
int mbedtls_ecp_tls_read_group_id( mbedtls_ecp_group_id *, const unsigned char **, size_t );
int mbedtls_ecp_tls_read_point( const mbedtls_ecp_group *, mbedtls_ecp_point *, const unsigned char **, size_t );
int mbedtls_ecp_tls_write_group( const mbedtls_ecp_group *, size_t *, unsigned char *, size_t );
int mbedtls_ecp_tls_write_point( const mbedtls_ecp_group *, const mbedtls_ecp_point *, int, size_t *, unsigned char *, size_t );
int mbedtls_ecp_write_key( mbedtls_ecp_keypair *, unsigned char *, size_t );
int mbedtls_mpi_shift_l_mod( const mbedtls_ecp_group *, mbedtls_mpi * );
mbedtls_ecp_curve_type mbedtls_ecp_get_type( const mbedtls_ecp_group * );
void mbedtls_ecp_group_free( mbedtls_ecp_group * );
void mbedtls_ecp_group_init( mbedtls_ecp_group * );
void mbedtls_ecp_keypair_free( mbedtls_ecp_keypair * );
void mbedtls_ecp_keypair_init( mbedtls_ecp_keypair * );
void mbedtls_ecp_point_free( mbedtls_ecp_point * );
void mbedtls_ecp_point_init( mbedtls_ecp_point * );
void mbedtls_ecp_restart_free( mbedtls_ecp_restart_ctx * );
void mbedtls_ecp_restart_init( mbedtls_ecp_restart_ctx * );
void mbedtls_ecp_set_max_ops( unsigned );

int ecp_mod_p256(mbedtls_mpi *);
int ecp_mod_p384(mbedtls_mpi *);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ECP_H_ */
