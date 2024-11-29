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
#include "third_party/mbedtls/ecp.h"
#include "third_party/mbedtls/math.h"

static void mbedtls_mpi_shift_l_mod_p256( const mbedtls_ecp_group *G,
                                          mbedtls_mpi *X )
{
    bool c;
    MBEDTLS_ASSERT( G->P.n == 4 );
    MBEDTLS_ASSERT( mbedtls_mpi_bitlen(     X ) <= 256 );
    MBEDTLS_ASSERT( mbedtls_mpi_bitlen( &G->P ) <= 256 );
    X->p[4] =                X->p[3] >> 63;
    X->p[3] = X->p[3] << 1 | X->p[2] >> 63;
    X->p[2] = X->p[2] << 1 | X->p[1] >> 63;
    X->p[1] = X->p[1] << 1 | X->p[0] >> 63;
    X->p[0] = X->p[0] << 1;
    if( (X->p[4] ||
         X->p[3] > G->P.p[3] ||
         ((X->p[3] == G->P.p[3] &&
           X->p[2] > G->P.p[2]) ||
          ((X->p[2] == G->P.p[2] &&
            X->p[0] > G->P.p[0]) ||
           (X->p[0] == G->P.p[0])))) )
    {
        SBB(X->p[0], X->p[0], G->P.p[0], 0, c);
        SBB(X->p[1], X->p[1], G->P.p[1], c, c);
        SBB(X->p[2], X->p[2], G->P.p[2], c, c);
        SBB(X->p[3], X->p[3], G->P.p[3], c, c);
        SBB(X->p[4], X->p[4], 0,         c, c);
    }
}

static void mbedtls_mpi_shift_l_mod_p384( const mbedtls_ecp_group *G,
                                          mbedtls_mpi *X )
{
    bool c;
    MBEDTLS_ASSERT( G->P.n == 6 );
    MBEDTLS_ASSERT( mbedtls_mpi_bitlen(     X ) <= 384 );
    MBEDTLS_ASSERT( mbedtls_mpi_bitlen( &G->P ) <= 384 );
    X->p[6] =                X->p[5] >> 63;
    X->p[5] = X->p[5] << 1 | X->p[4] >> 63;
    X->p[4] = X->p[4] << 1 | X->p[3] >> 63;
    X->p[3] = X->p[3] << 1 | X->p[2] >> 63;
    X->p[2] = X->p[2] << 1 | X->p[1] >> 63;
    X->p[1] = X->p[1] << 1 | X->p[0] >> 63;
    X->p[0] = X->p[0] << 1;
    if( (X->p[6] ||
         X->p[5] > G->P.p[5] ||
         ((X->p[5] == G->P.p[5] &&
           X->p[4] > G->P.p[4]) ||
          ((X->p[4] == G->P.p[4] &&
            X->p[3] > G->P.p[3]) ||
           ((X->p[3] == G->P.p[3] &&
             X->p[2] > G->P.p[2]) ||
            ((X->p[2] == G->P.p[2] &&
              X->p[0] > G->P.p[0]) ||
             (X->p[0] == G->P.p[0])))))) )
    {
        SBB(X->p[0], X->p[0], G->P.p[0], 0, c);
        SBB(X->p[1], X->p[1], G->P.p[1], c, c);
        SBB(X->p[2], X->p[2], G->P.p[2], c, c);
        SBB(X->p[3], X->p[3], G->P.p[3], c, c);
        SBB(X->p[4], X->p[4], G->P.p[4], c, c);
        SBB(X->p[5], X->p[5], G->P.p[5], c, c);
        SBB(X->p[6], X->p[6], 0,         c, c);
    }
}

int mbedtls_mpi_shift_l_mod( const mbedtls_ecp_group *G, mbedtls_mpi *X )
{
    int ret = 0;
    MBEDTLS_ASSERT( mbedtls_mpi_cmp_int( X, 0 ) >= 0 );
    MBEDTLS_ASSERT( mbedtls_mpi_cmp_mpi( X, &G->P ) < 0 );
    if( X->n == 8 )
        mbedtls_mpi_shift_l_mod_p256( G, X );
    else if( X->n == 12 )
        mbedtls_mpi_shift_l_mod_p384( G, X );
    else
    {
        MBEDTLS_MPI_CHK( mbedtls_mpi_shift_l( X, 1 ) );
        if( mbedtls_mpi_cmp_mpi( X, &G->P ) >= 0 )
            MBEDTLS_MPI_CHK( mbedtls_mpi_sub_abs( X, X, &G->P ) );
    }
    MBEDTLS_ASSERT( mbedtls_mpi_cmp_mpi( X, &G->P ) <  0 );
    MBEDTLS_ASSERT( mbedtls_mpi_cmp_int( X, 0     ) >= 0 );
cleanup:
    return( ret );
}
