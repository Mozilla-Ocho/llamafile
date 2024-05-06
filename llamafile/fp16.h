// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
#pragma once
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"

class llamafile_fp16 {
  public:
    llamafile_fp16() = default;

    llamafile_fp16(float x) : x_(GGML_FP32_TO_FP16(x)) {
    }

    operator float() const {
        return GGML_FP16_TO_FP32(x_);
    };

  private:
    ggml_fp16_t x_;
};
