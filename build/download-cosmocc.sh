#!/bin/sh
# cosmocc downloader script
# https://justine.lol/cosmo3/#install
# https://github.com/jart/cosmopolitan/blob/master/tool/cosmocc/README.md

# collect arguments
OUTPUT_DIR=${1:?OUTPUT_DIR}
COSMOCC_VERSION=${2:?COSMOCC_VERSION}
COSMOCC_SHA256SUM=${3:?COSMOCC_SHA256SUM}
URL1="https://github.com/jart/cosmopolitan/releases/download/${COSMOCC_VERSION}/cosmocc-${COSMOCC_VERSION}.zip"
URL2="https://cosmo.zip/pub/cosmocc/cosmocc-${COSMOCC_VERSION}.zip"

# helper function
abort() {
  printf '%s\n' "download terminated." >&2
  exit 1
}

# exit if already downloaded
# we need it because directory timestamps work wierdly
OUTPUT_DIR=${OUTPUT_DIR%/}
if [ -d "${OUTPUT_DIR}" ]; then
  exit 0
fi

# find commands we need to securely download cosmocc
if ! UNZIP=$(command -v unzip 2>/dev/null); then
  printf '%s\n' "$0: fatal error: you need the unzip command" >&2
  printf '%s\n' "please download https://cosmo.zip/pub/cosmos/bin/unzip and put it on the system path" >&2
  abort
fi
if command -v sha256sum >/dev/null 2>&1; then
  # can use system sha256sum
  true
elif command -v shasum >/dev/null 2>&1; then
  sha256sum() {
    shasum -a 256 "$@"
  }
else
  if [ ! -f build/sha256sum.c ]; then
    printf '%s\n' "$0: fatal error: you need to install sha256sum" >&2
    printf '%s\n' "please download https://cosmo.zip/pub/cosmos/bin/sha256sum and put it on the system path" >&2
    abort
  fi
  if ! SHA256SUM=$(command -v "$PWD/o/build/sha256sum" 2>/dev/null); then
    if ! CC=$(command -v "$CC" 2>/dev/null); then
      if ! CC=$(command -v cc 2>/dev/null); then
        if ! CC=$(command -v cosmocc 2>/dev/null); then
          printf '%s\n' "$0: fatal error: you need to install either sha256sum, cc, or cosmocc" >&2
          printf '%s\n' "please download https://cosmo.zip/pub/cosmos/bin/sha256sum and put it on the system path" >&2
          abort
        fi
      fi
    fi
    mkdir -p o/build || abort
    SHA256SUM="$PWD/o/build/sha256sum"
    printf '%s\n' "${CC} -w -O2 -o ${SHA256SUM} build/sha256sum.c" >&2
    "${CC}" -w -O2 -o "${SHA256SUM}.$$" build/sha256sum.c || abort
    mv -f "${SHA256SUM}.$$" "${SHA256SUM}" || abort
  fi
  sha256sum() {
    "${SHA256SUM}" "$@"
  }
fi
if WGET=$(command -v wget 2>/dev/null); then
  DOWNLOAD=$WGET
  DOWNLOAD_ARGS=-O
elif CURL=$(command -v curl 2>/dev/null); then
  DOWNLOAD=$CURL
  DOWNLOAD_ARGS=-fLo
else
  printf '%s\n' "$0: fatal error: you need to install either wget or curl" >&2
  printf '%s\n' "please download https://cosmo.zip/pub/cosmos/bin/wget and put it on the system path" >&2
  abort
fi

# create temporary output directory
OLDPWD=$PWD
OUTPUT_TMP="${OUTPUT_DIR}.tmp.$$/"
mkdir -p "${OUTPUT_TMP}" || abort
cd "${OUTPUT_TMP}"
die() {
  cd "${OLDPWD}"
  rm -rf "${OUTPUT_TMP}"
  abort
}

# download cosmocc toolchain
# multiple urls avoids outages and national firewalls
if ! "${DOWNLOAD}" ${DOWNLOAD_ARGS} cosmocc.zip "${URL1}"; then
  rm -f cosmocc.zip
  "${DOWNLOAD}" ${DOWNLOAD_ARGS} cosmocc.zip "${URL2}" || die
fi
printf '%s\n' "${COSMOCC_SHA256SUM} *cosmocc.zip" >cosmocc.zip.sha256sum
sha256sum -c cosmocc.zip.sha256sum || die
"${UNZIP}" cosmocc.zip || die
rm -f cosmocc.zip cosmocc.zip.sha256sum

# commit output directory
cd "${OLDPWD}" || die
mv "${OUTPUT_TMP}" "${OUTPUT_DIR}" || die
