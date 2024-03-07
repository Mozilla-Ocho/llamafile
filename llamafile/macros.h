// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define ROUNDUP(X, K) (((X) + (K) - 1) & -(K))
#define ARRAYLEN(A) ((sizeof(A) / sizeof(*(A))) / ((unsigned)!(sizeof(A) % sizeof(*(A)))))
