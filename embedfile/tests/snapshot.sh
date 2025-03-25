#!/bin/bash

TESTS_DIR=$(dirname "$(realpath "$0")")
EMBEDFILE=$(realpath "$TESTS_DIR/../../../o/embedfile/embedfile")

"$EMBEDFILE" sh < $TESTS_DIR/env.sql > $TESTS_DIR/__snapshots__/env.out
