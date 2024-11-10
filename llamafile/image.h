// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <string>

namespace lf {

enum ImageType {
    IMAGE_TYPE_UNKNOWN,
    IMAGE_TYPE_PNG, // supported by stb
    IMAGE_TYPE_JPG, // supported by stb
    IMAGE_TYPE_GIF, // supported by stb
    IMAGE_TYPE_BMP, // supported by stb
    IMAGE_TYPE_TGA, // supported by stb
    IMAGE_TYPE_HDR, // supported by stb
    IMAGE_TYPE_PGM, // supported by stb
    IMAGE_TYPE_PPM, // supported by stb
    IMAGE_TYPE_PIC, // supported by stb
    IMAGE_TYPE_PSD, // partially supported by stb
    IMAGE_TYPE_WEBM,
    IMAGE_TYPE_WEBP,
    IMAGE_TYPE_AVIF,
    IMAGE_TYPE_HEIF,
    IMAGE_TYPE_HEIC,
    IMAGE_TYPE_TIFF,
    IMAGE_TYPE_ICO,
    IMAGE_TYPE_CUR,
    IMAGE_TYPE_DDS,
    IMAGE_TYPE_JXR,
    IMAGE_TYPE_EXR,
    IMAGE_TYPE_PCX,
    IMAGE_TYPE_XCF
};

ImageType get_image_type(const std::string_view &);
bool is_image(const std::string_view &);
const char *get_image_mime(ImageType);
int print_image(int, const std::string_view &, int);
void convert_image_to_uri(std::string *, const std::string_view &);

} // namespace lf
