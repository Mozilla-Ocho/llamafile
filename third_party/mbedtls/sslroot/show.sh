#!/bin/sh
for f in usr/share/ssl/root/*.pem; do
  echo
  echo File: $f
  echo
  openssl crl2pkcs7 -nocrl -certfile $f |
    openssl pkcs7 -print_certs -text -noout
done
