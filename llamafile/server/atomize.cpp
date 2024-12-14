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

#include "atom.h"
#include "llama.cpp/base64.h"
#include "llamafile/chatbot.h"
#include "llamafile/datauri.h"
#include "llamafile/image.h"
#include "llamafile/llama.h"
#include "llamafile/server/image.h"
#include "llamafile/string.h"
#include <string>
#include <vector>

namespace lf {
namespace server {

static void
append_tokens(const llama_model* model,
              std::vector<Atom>* result,
              const std::string_view& s,
              bool parse_special)
{
    std::vector<int> tokens = llamafile_tokenize(
      model, std::string(s), DONT_ADD_SPECIAL, parse_special);
    for (int token : tokens)
        result->emplace_back(token);
}

void
atomize(const llama_model* model,
        std::vector<Atom>* result,
        std::string_view s,
        bool parse_special)
{
    size_t i = 0;
    for (;;) {
        size_t pos = s.find("data:", i);
        if (pos == std::string_view::npos) {
            append_tokens(model, result, s, parse_special);
            return;
        }
        i = pos + 5;
        DataUri uri;
        size_t end = uri.parse(s.substr(i));
        if (end == std::string_view::npos)
            continue;
        if (!startscasewith(uri.mime, "image/"))
            continue;
        std::string image;
        try {
            image = uri.decode();
        } catch (const base64_error& e) {
            continue;
        }
        if (!get_image_type(image))
            continue;
        append_tokens(model, result, s.substr(0, pos), parse_special);
        result->emplace_back(new Image(image, -1));
        s = s.substr(i + end);
        i = 0;
    }
}

// having multiple images in the context window is janky right now, so
// let's erase old images from the chat history until we find out more
std::vector<Atom>
remove_old_image_atoms(const std::vector<Atom>& atoms)
{
    int last_image_idx = -1;
    for (int i = 0; i < atoms.size(); ++i)
        if (atoms[i].is_image())
            last_image_idx = i;
    std::vector<Atom> result;
    for (int i = 0; i < atoms.size(); i++)
        if (!atoms[i].is_image() || i == last_image_idx)
            result.emplace_back(atoms[i]);
    return result;
}

int
count_tokens(const std::vector<Atom>& atoms)
{
    int n = 0;
    for (const Atom& atom : atoms)
        n += atom.ctx_used();
    return n;
}

} // namespace server
} // namespace lf
