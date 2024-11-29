/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:4;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright The Mbed TLS Contributors                                          │
│                                                                              │
│ Licensed under the Apache License, Version 2.0 (the "License");              │
│ you may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│     http://www.apache.org/licenses/LICENSE-2.0                               │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <libc/str/str.h>
#include "third_party/mbedtls/asn1write.h"
#include "third_party/mbedtls/common.h"
#include "third_party/mbedtls/error.h"
#include "third_party/mbedtls/platform.h"
__static_yoink("mbedtls_notice");

/**
 * @fileoverview ASN.1 buffer writing functionality
 */

#if defined(MBEDTLS_ASN1_WRITE_C)

/**
 * \brief           Write a length field in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param len       The length value to write.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_len( unsigned char **p, unsigned char *start, size_t len )
{
    if( len < 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = (unsigned char) len;
        return( 1 );
    }
    if( len <= 0xFF )
    {
        if( *p - start < 2 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = (unsigned char) len;
        *--(*p) = 0x81;
        return( 2 );
    }
    if( len <= 0xFFFF )
    {
        if( *p - start < 3 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = 0x82;
        return( 3 );
    }
    if( len <= 0xFFFFFF )
    {
        if( *p - start < 4 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = ( len >> 16 ) & 0xFF;
        *--(*p) = 0x83;
        return( 4 );
    }
#if SIZE_MAX > 0xFFFFFFFF
    if( len <= 0xFFFFFFFF )
#endif
    {
        if( *p - start < 5 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = ( len       ) & 0xFF;
        *--(*p) = ( len >>  8 ) & 0xFF;
        *--(*p) = ( len >> 16 ) & 0xFF;
        *--(*p) = ( len >> 24 ) & 0xFF;
        *--(*p) = 0x84;
        return( 5 );
    }
#if SIZE_MAX > 0xFFFFFFFF
    return( MBEDTLS_ERR_ASN1_INVALID_LENGTH );
#endif
}

/**
 * \brief           Write an ASN.1 tag in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param tag       The tag to write.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_tag( unsigned char **p, unsigned char *start, unsigned char tag )
{
    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
    *--(*p) = tag;
    return( 1 );
}

/**
 * \brief           Write raw buffer data.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param buf       The data buffer to write.
 * \param size      The length of the data buffer.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_raw_buffer( unsigned char **p, unsigned char *start,
                           const unsigned char *buf, size_t size )
{
    size_t len = 0;
    if( *p < start || (size_t)( *p - start ) < size )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
    len = size;
    (*p) -= len;
    memcpy( *p, buf, len );
    return( (int) len );
}

#if defined(MBEDTLS_BIGNUM_C)
/**
 * \brief           Write a arbitrary-precision number (#MBEDTLS_ASN1_INTEGER)
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param X         The MPI to write.
 *                  It must be non-negative.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_mpi( unsigned char **p, unsigned char *start, const mbedtls_mpi *X )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    // Write the MPI
    //
    len = mbedtls_mpi_size( X );
    if( *p < start || (size_t)( *p - start ) < len )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
    (*p) -= len;
    MBEDTLS_MPI_CHK( mbedtls_mpi_write_binary( X, *p, len ) );
    // DER format assumes 2s complement for numbers, so the leftmost bit
    // should be 0 for positive numbers and 1 for negative numbers.
    //
    if( X->s ==1 && **p & 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = 0x00;
        len += 1;
    }
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_INTEGER ) );
    ret = (int) len;
cleanup:
    return( ret );
}
#endif /* MBEDTLS_BIGNUM_C */

/**
 * \brief           Write a NULL tag (#MBEDTLS_ASN1_NULL) with zero data
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_null( unsigned char **p, unsigned char *start )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    // Write NULL
    //
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, 0) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_NULL ) );
    return( (int) len );
}

/**
 * \brief           Write an OID tag (#MBEDTLS_ASN1_OID) and data
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param oid       The OID to write.
 * \param oid_len   The length of the OID.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_oid( unsigned char **p, unsigned char *start,
                            const char *oid, size_t oid_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
                                  (const unsigned char *) oid, oid_len ) );
    MBEDTLS_ASN1_CHK_ADD( len , mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len , mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OID ) );
    return( (int) len );
}

/**
 * \brief           Write an AlgorithmIdentifier sequence in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param oid       The OID of the algorithm to write.
 * \param oid_len   The length of the algorithm's OID.
 * \param par_len   The length of the parameters, which must be already written.
 *                  If 0, NULL parameters are added
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_algorithm_identifier( unsigned char **p, unsigned char *start,
                                     const char *oid, size_t oid_len,
                                     size_t par_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    if( par_len == 0 )
        MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_null( p, start ) );
    else
        len += par_len;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_oid( p, start, oid, oid_len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start,
                                       MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE ) );
    return( (int) len );
}

/**
 * \brief           Write a boolean tag (#MBEDTLS_ASN1_BOOLEAN) and value
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param boolean   The boolean value to write, either \c 0 or \c 1.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_bool( unsigned char **p, unsigned char *start, int boolean )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    if( *p - start < 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
    *--(*p) = (boolean) ? 255 : 0;
    len++;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_BOOLEAN ) );
    return( (int) len );
}

static int asn1_write_tagged_int( unsigned char **p, unsigned char *start, int val, int tag )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    do
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        len += 1;
        *--(*p) = val & 0xff;
        val >>= 8;
    }
    while( val > 0 );
    if( **p & 0x80 )
    {
        if( *p - start < 1 )
            return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
        *--(*p) = 0x00;
        len += 1;
    }
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, tag ) );
    return( (int) len );
}

/**
 * \brief           Write an int tag (#MBEDTLS_ASN1_INTEGER) and value
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param val       The integer value to write.
 *                  It must be non-negative.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_int( unsigned char **p, unsigned char *start, int val )
{
    return( asn1_write_tagged_int( p, start, val, MBEDTLS_ASN1_INTEGER ) );
}

/**
 * \brief           Write an enum tag (#MBEDTLS_ASN1_ENUMERATED) and value
 *                  in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param val       The integer value to write.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_asn1_write_enum( unsigned char **p, unsigned char *start, int val )
{
    return( asn1_write_tagged_int( p, start, val, MBEDTLS_ASN1_ENUMERATED ) );
}

/**
 * \brief           Write a string in ASN.1 format using a specific
 *                  string encoding tag.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param tag       The string encoding tag to write, e.g.
 *                  #MBEDTLS_ASN1_UTF8_STRING.
 * \param text      The string to write.
 * \param text_len  The length of \p text in bytes (which might
 *                  be strictly larger than the number of characters).
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_tagged_string( unsigned char **p, unsigned char *start,
                                      int tag, const char *text, size_t text_len )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start,
        (const unsigned char *) text, text_len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, tag ) );
    return( (int) len );
}

/**
 * \brief           Write a UTF8 string in ASN.1 format using the UTF8String
 *                  string encoding tag (#MBEDTLS_ASN1_UTF8_STRING).
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param text      The string to write.
 * \param text_len  The length of \p text in bytes (which might
 *                  be strictly larger than the number of characters).
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_utf8_string( unsigned char **p, unsigned char *start,
                                    const char *text, size_t text_len )
{
    return( mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_UTF8_STRING, text, text_len) );
}

/**
 * \brief           Write a string in ASN.1 format using the PrintableString
 *                  string encoding tag (#MBEDTLS_ASN1_PRINTABLE_STRING).
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param text      The string to write.
 * \param text_len  The length of \p text in bytes (which might
 *                  be strictly larger than the number of characters).
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_printable_string( unsigned char **p, unsigned char *start,
                                         const char *text, size_t text_len )
{
    return( mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_PRINTABLE_STRING, text, text_len) );
}

/**
 * \brief           Write a string in ASN.1 format using the IA5String
 *                  string encoding tag (#MBEDTLS_ASN1_IA5_STRING).
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param text      The string to write.
 * \param text_len  The length of \p text in bytes (which might
 *                  be strictly larger than the number of characters).
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_ia5_string( unsigned char **p, unsigned char *start,
                           const char *text, size_t text_len )
{
    return( mbedtls_asn1_write_tagged_string(p, start, MBEDTLS_ASN1_IA5_STRING, text, text_len) );
}

/**
 * \brief           This function writes a named bitstring tag
 *                  (#MBEDTLS_ASN1_BIT_STRING) and value in ASN.1 format.
 *
 *                  As stated in RFC 5280 Appendix B, trailing zeroes are
 *                  omitted when encoding named bitstrings in DER.
 *
 * \note            This function works backwards within the data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer which is used for bounds-checking.
 * \param buf       The bitstring to write.
 * \param bits      The total number of bits in the bitstring.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_named_bitstring( unsigned char **p,
                                        unsigned char *start,
                                        const unsigned char *buf,
                                        size_t bits )
{
    size_t unused_bits, byte_len;
    const unsigned char *cur_byte;
    unsigned char cur_byte_shifted;
    unsigned char bit;
    byte_len = ( bits + 7 ) / 8;
    unused_bits = ( byte_len * 8 ) - bits;
    /*
     * Named bitstrings require that trailing 0s are excluded in the encoding
     * of the bitstring. Trailing 0s are considered part of the 'unused' bits
     * when encoding this value in the first content octet
     */
    if( bits )
    {
        cur_byte = buf + byte_len - 1;
        cur_byte_shifted = *cur_byte >> unused_bits;
        for( ; ; )
        {
            bit = cur_byte_shifted & 0x1;
            cur_byte_shifted >>= 1;
            if( bit )
                break;
            bits--;
            if( bits == 0 )
                break;
            if( bits % 8 == 0 )
                cur_byte_shifted = *--cur_byte;
        }
    }
    return( mbedtls_asn1_write_bitstring( p, start, buf, bits ) );
}

/**
 * \brief           Write a bitstring tag (#MBEDTLS_ASN1_BIT_STRING) and
 *                  value in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param buf       The bitstring to write.
 * \param bits      The total number of bits in the bitstring.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_bitstring( unsigned char **p, unsigned char *start,
                          const unsigned char *buf, size_t bits )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    size_t unused_bits, byte_len;
    byte_len = ( bits + 7 ) / 8;
    unused_bits = ( byte_len * 8 ) - bits;
    if( *p < start || (size_t)( *p - start ) < byte_len + 1 )
        return( MBEDTLS_ERR_ASN1_BUF_TOO_SMALL );
    len = byte_len + 1;
    /* Write the bitstring. Ensure the unused bits are zeroed */
    if( byte_len > 0 )
    {
        byte_len--;
        *--( *p ) = buf[byte_len] & ~( ( 0x1 << unused_bits ) - 1 );
        ( *p ) -= byte_len;
        memcpy( *p, buf, byte_len );
    }
    /* Write unused bits */
    *--( *p ) = (unsigned char)unused_bits;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_BIT_STRING ) );
    return( (int) len );
}

/**
 * \brief           Write an octet string tag (#MBEDTLS_ASN1_OCTET_STRING)
 *                  and value in ASN.1 format.
 *
 * \note            This function works backwards in data buffer.
 *
 * \param p         The reference to the current position pointer.
 * \param start     The start of the buffer, for bounds-checking.
 * \param buf       The buffer holding the data to write.
 * \param size      The length of the data buffer \p buf.
 *
 * \return          The number of bytes written to \p p on success.
 * \return          A negative error code on failure.
 */
int mbedtls_asn1_write_octet_string( unsigned char **p, unsigned char *start,
                             const unsigned char *buf, size_t size )
{
    int ret = MBEDTLS_ERR_THIS_CORRUPTION;
    size_t len = 0;
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_raw_buffer( p, start, buf, size ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_len( p, start, len ) );
    MBEDTLS_ASN1_CHK_ADD( len, mbedtls_asn1_write_tag( p, start, MBEDTLS_ASN1_OCTET_STRING ) );
    return( (int) len );
}

/* This is a copy of the ASN.1 parsing function mbedtls_asn1_find_named_data(),
 * which is replicated to avoid a dependency ASN1_WRITE_C on ASN1_PARSE_C. */
static mbedtls_asn1_named_data *asn1_find_named_data(
                                               mbedtls_asn1_named_data *list,
                                               const char *oid, size_t len )
{
    while( list )
    {
        if( list->oid.len == len &&
            timingsafe_bcmp( list->oid.p, oid, len ) == 0 )
        {
            break;
        }
        list = list->next;
    }
    return( list );
}

/**
 * \brief           Create or find a specific named_data entry for writing in a
 *                  sequence or list based on the OID. If not already in there,
 *                  a new entry is added to the head of the list.
 *                  Warning: Destructive behaviour for the val data!
 *
 * \param list      The pointer to the location of the head of the list to seek
 *                  through (will be updated in case of a new entry).
 * \param oid       The OID to look for.
 * \param oid_len   The size of the OID.
 * \param val       The associated data to store. If this is \c NULL,
 *                  no data is copied to the new or existing buffer.
 * \param val_len   The minimum length of the data buffer needed.
 *                  If this is 0, do not allocate a buffer for the associated
 *                  data.
 *                  If the OID was already present, enlarge, shrink or free
 *                  the existing buffer to fit \p val_len.
 *
 * \return          A pointer to the new / existing entry on success.
 * \return          \c NULL if if there was a memory allocation error.
 */
mbedtls_asn1_named_data *mbedtls_asn1_store_named_data(mbedtls_asn1_named_data **head,
                                                       const char *oid, size_t oid_len,
                                                       const unsigned char *val,
                                                       size_t val_len )
{
    mbedtls_asn1_named_data *cur;
    if( ( cur = asn1_find_named_data( *head, oid, oid_len ) ) == NULL )
    {
        // Add new entry if not present yet based on OID
        //
        cur = (mbedtls_asn1_named_data*)mbedtls_calloc( 1,
                                            sizeof(mbedtls_asn1_named_data) );
        if( !cur )
            return( NULL );
        cur->oid.len = oid_len;
        cur->oid.p = mbedtls_calloc( 1, oid_len );
        if( !cur->oid.p )
        {
            mbedtls_free( cur );
            return( NULL );
        }
        memcpy( cur->oid.p, oid, oid_len );
        cur->val.len = val_len;
        if( val_len )
        {
            cur->val.p = mbedtls_calloc( 1, val_len );
            if( !cur->val.p )
            {
                mbedtls_free( cur->oid.p );
                mbedtls_free( cur );
                return( NULL );
            }
        }
        cur->next = *head;
        *head = cur;
    }
    else if( val_len == 0 )
    {
        mbedtls_free( cur->val.p );
        cur->val.p = NULL;
    }
    else if( cur->val.len != val_len )
    {
        /*
         * Enlarge existing value buffer if needed
         * Preserve old data until the allocation succeeded, to leave list in
         * a consistent state in case allocation fails.
         */
        void *p = mbedtls_calloc( 1, val_len );
        if( !p )
            return( NULL );
        mbedtls_free( cur->val.p );
        cur->val.p = p;
        cur->val.len = val_len;
    }
    if( val )
        memcpy( cur->val.p, val, val_len );
    return( cur );
}

#endif /* MBEDTLS_ASN1_WRITE_C */
