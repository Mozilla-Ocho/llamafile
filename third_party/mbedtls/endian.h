#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENDIAN_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENDIAN_H_

#define Read32be(S)                                      \
  ({                                                     \
    const uint8_t *Ptr = (S);                            \
    ((uint32_t)Ptr[0] << 030 | (uint32_t)Ptr[1] << 020 | \
     (uint32_t)Ptr[2] << 010 | (uint32_t)Ptr[3] << 000); \
  })

#define Write32be(P, V)                         \
  ({                                            \
    uint8_t *OuT = (P);                         \
    uint64_t VaL = (V);                         \
    OuT[0] = (0x00000000FF000000 & VaL) >> 030; \
    OuT[1] = (0x0000000000FF0000 & VaL) >> 020; \
    OuT[2] = (0x000000000000FF00 & VaL) >> 010; \
    OuT[3] = (0x00000000000000FF & VaL) >> 000; \
    OuT + 4;                                    \
  })

#define Read64be(S)                                      \
  ({                                                     \
    const uint8_t *Ptr = (S);                            \
    ((uint64_t)Ptr[0] << 070 | (uint64_t)Ptr[1] << 060 | \
     (uint64_t)Ptr[2] << 050 | (uint64_t)Ptr[3] << 040 | \
     (uint64_t)Ptr[4] << 030 | (uint64_t)Ptr[5] << 020 | \
     (uint64_t)Ptr[6] << 010 | (uint64_t)Ptr[7] << 000); \
  })

#define Write64be(P, V)                         \
  ({                                            \
    uint64_t VaL = (V);                         \
    uint8_t *OuT = (P);                         \
    OuT[0] = (0xFF00000000000000 & VaL) >> 070; \
    OuT[1] = (0x00FF000000000000 & VaL) >> 060; \
    OuT[2] = (0x0000FF0000000000 & VaL) >> 050; \
    OuT[3] = (0x000000FF00000000 & VaL) >> 040; \
    OuT[4] = (0x00000000FF000000 & VaL) >> 030; \
    OuT[5] = (0x0000000000FF0000 & VaL) >> 020; \
    OuT[6] = (0x000000000000FF00 & VaL) >> 010; \
    OuT[7] = (0x00000000000000FF & VaL) >> 000; \
    OuT + 8;                                    \
  })

#define Write64le(P, V)                         \
  ({                                            \
    uint64_t VaL = (V);                         \
    uint8_t *OuT = (P);                         \
    OuT[0] = (0x00000000000000FF & VaL) >> 000; \
    OuT[1] = (0x000000000000FF00 & VaL) >> 010; \
    OuT[2] = (0x0000000000FF0000 & VaL) >> 020; \
    OuT[3] = (0x00000000FF000000 & VaL) >> 030; \
    OuT[4] = (0x000000FF00000000 & VaL) >> 040; \
    OuT[5] = (0x0000FF0000000000 & VaL) >> 050; \
    OuT[6] = (0x00FF000000000000 & VaL) >> 060; \
    OuT[7] = (0xFF00000000000000 & VaL) >> 070; \
    OuT + 8;                                    \
  })

#define GET_UINT32_BE(n, b, i) (n) = Read32be((b) + (i))
#define PUT_UINT32_BE(n, b, i) Write32be((b) + (i), n)
#define GET_UINT64_BE(n, b, i) (n) = Read64be((b) + (i))
#define PUT_UINT64_BE(n, b, i) Write64be((b) + (i), n)

#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_ENDIAN_H_ */
