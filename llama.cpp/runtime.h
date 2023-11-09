// -*-mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8-*-
// vi: set net ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#include <string>

void ThrowRuntimeError(std::string) __attribute__((__noreturn__));
void ThrowInvalidArgument(std::string) __attribute__((__noreturn__));
