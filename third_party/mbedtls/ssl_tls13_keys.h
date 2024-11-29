#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_TLS13_KEYS_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_TLS13_KEYS_H_
#include "third_party/mbedtls/md.h"
#include "third_party/mbedtls/ssl_internal.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_SSL_TLS1_3_CONTEXT_UNHASHED 0
#define MBEDTLS_SSL_TLS1_3_CONTEXT_HASHED   1

/* The maximum length of HKDF contexts used in the TLS 1.3 standard.
 * Since contexts are always hashes of message transcripts, this can
 * be approximated from above by the maximum hash size. */
#define MBEDTLS_SSL_TLS1_3_KEY_SCHEDULE_MAX_CONTEXT_LEN MBEDTLS_MD_MAX_SIZE

/* Maximum desired length for expanded key material generated
 * by HKDF-Expand-Label.
 *
 * Warning: If this ever needs to be increased, the implementation
 * ssl_tls1_3_hkdf_encode_label() in ssl_tls13_keys.c needs to be
 * adjusted since it currently assumes that HKDF key expansion
 * is never used with more than 255 Bytes of output. */
#define MBEDTLS_SSL_TLS1_3_KEY_SCHEDULE_MAX_EXPANSION_LEN 255

/* This requires MBEDTLS_SSL_TLS1_3_LABEL( idx, name, string ) to be defined at
 * the point of use. See e.g. the definition of mbedtls_ssl_tls1_3_labels_union
 * below. */
#define MBEDTLS_SSL_TLS1_3_LABEL_LIST                    \
  MBEDTLS_SSL_TLS1_3_LABEL(finished, "finished")         \
  MBEDTLS_SSL_TLS1_3_LABEL(resumption, "resumption")     \
  MBEDTLS_SSL_TLS1_3_LABEL(traffic_upd, "traffic upd")   \
  MBEDTLS_SSL_TLS1_3_LABEL(exporter, "exporter")         \
  MBEDTLS_SSL_TLS1_3_LABEL(key, "key")                   \
  MBEDTLS_SSL_TLS1_3_LABEL(iv, "iv")                     \
  MBEDTLS_SSL_TLS1_3_LABEL(c_hs_traffic, "c hs traffic") \
  MBEDTLS_SSL_TLS1_3_LABEL(c_ap_traffic, "c ap traffic") \
  MBEDTLS_SSL_TLS1_3_LABEL(c_e_traffic, "c e traffic")   \
  MBEDTLS_SSL_TLS1_3_LABEL(s_hs_traffic, "s hs traffic") \
  MBEDTLS_SSL_TLS1_3_LABEL(s_ap_traffic, "s ap traffic") \
  MBEDTLS_SSL_TLS1_3_LABEL(s_e_traffic, "s e traffic")   \
  MBEDTLS_SSL_TLS1_3_LABEL(e_exp_master, "e exp master") \
  MBEDTLS_SSL_TLS1_3_LABEL(res_master, "res master")     \
  MBEDTLS_SSL_TLS1_3_LABEL(exp_master, "exp master")     \
  MBEDTLS_SSL_TLS1_3_LABEL(ext_binder, "ext binder")     \
  MBEDTLS_SSL_TLS1_3_LABEL(res_binder, "res binder")     \
  MBEDTLS_SSL_TLS1_3_LABEL(derived, "derived")

#define MBEDTLS_SSL_TLS1_3_LBL_WITH_LEN(LABEL) \
  mbedtls_ssl_tls1_3_labels.LABEL, sizeof(mbedtls_ssl_tls1_3_labels.LABEL)

#define MBEDTLS_SSL_TLS1_3_KEY_SCHEDULE_MAX_LABEL_LEN \
  sizeof(union mbedtls_ssl_tls1_3_labels_union)

#define MBEDTLS_SSL_TLS1_3_LABEL(name, string) \
  const unsigned char name[sizeof(string) - 1];
union mbedtls_ssl_tls1_3_labels_union {
  MBEDTLS_SSL_TLS1_3_LABEL_LIST
};
struct mbedtls_ssl_tls1_3_labels_struct {
  MBEDTLS_SSL_TLS1_3_LABEL_LIST
};
#undef MBEDTLS_SSL_TLS1_3_LABEL

extern const struct mbedtls_ssl_tls1_3_labels_struct mbedtls_ssl_tls1_3_labels;

int mbedtls_ssl_tls1_3_hkdf_expand_label(mbedtls_md_type_t,
                                         const unsigned char *, size_t,
                                         const unsigned char *, size_t,
                                         const unsigned char *, size_t,
                                         unsigned char *, size_t);
int mbedtls_ssl_tls1_3_make_traffic_keys(mbedtls_md_type_t,
                                         const unsigned char *,
                                         const unsigned char *, size_t, size_t,
                                         size_t, mbedtls_ssl_key_set *);
int mbedtls_ssl_tls1_3_derive_secret(mbedtls_md_type_t, const unsigned char *,
                                     size_t, const unsigned char *, size_t,
                                     const unsigned char *, size_t, int,
                                     unsigned char *, size_t);
int mbedtls_ssl_tls1_3_evolve_secret(mbedtls_md_type_t, const unsigned char *,
                                     const unsigned char *, size_t,
                                     unsigned char *);

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_SSL_TLS13_KEYS_H_ */
