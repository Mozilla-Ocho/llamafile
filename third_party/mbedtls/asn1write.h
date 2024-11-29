#ifndef MBEDTLS_ASN1_WRITE_H_
#define MBEDTLS_ASN1_WRITE_H_
#include "third_party/mbedtls/asn1.h"
#include "third_party/mbedtls/config.h"
COSMOPOLITAN_C_START_

#define MBEDTLS_ASN1_CHK_ADD(g, f)                      \
    do                                                  \
    {                                                   \
        if( ( ret = (f) ) < 0 )                         \
            return( ret );                              \
        else                                            \
            (g) += ret;                                 \
    } while( 0 )

int mbedtls_asn1_write_len( unsigned char **, unsigned char *, size_t );
int mbedtls_asn1_write_tag( unsigned char **, unsigned char *, unsigned char );
int mbedtls_asn1_write_raw_buffer( unsigned char **, unsigned char *, const unsigned char *, size_t );
int mbedtls_asn1_write_mpi( unsigned char **, unsigned char *, const mbedtls_mpi * );
int mbedtls_asn1_write_null( unsigned char **, unsigned char * );
int mbedtls_asn1_write_oid( unsigned char **, unsigned char *, const char *, size_t );
int mbedtls_asn1_write_algorithm_identifier( unsigned char **, unsigned char *, const char *, size_t, size_t );
int mbedtls_asn1_write_bool( unsigned char **, unsigned char *, int );
int mbedtls_asn1_write_int( unsigned char **, unsigned char *, int );
int mbedtls_asn1_write_enum( unsigned char **, unsigned char *, int );
int mbedtls_asn1_write_tagged_string( unsigned char **, unsigned char *, int, const char *, size_t );
int mbedtls_asn1_write_printable_string( unsigned char **, unsigned char *, const char *, size_t );
int mbedtls_asn1_write_utf8_string( unsigned char **, unsigned char *, const char *, size_t );
int mbedtls_asn1_write_ia5_string( unsigned char **, unsigned char *, const char *, size_t );
int mbedtls_asn1_write_bitstring( unsigned char **, unsigned char *, const unsigned char *, size_t );
int mbedtls_asn1_write_named_bitstring( unsigned char **, unsigned char *, const unsigned char *, size_t );
int mbedtls_asn1_write_octet_string( unsigned char **, unsigned char *, const unsigned char *, size_t );
mbedtls_asn1_named_data *mbedtls_asn1_store_named_data( mbedtls_asn1_named_data **, const char *, size_t, const unsigned char *, size_t );

COSMOPOLITAN_C_END_
#endif /* MBEDTLS_ASN1_WRITE_H_ */
