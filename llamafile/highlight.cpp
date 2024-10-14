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

#include "highlight.h"

Highlight *Highlight::create(const std::string_view &lang) {
    if (lang == "markdown" || lang == "md") {
        return new HighlightMarkdown;
    } else if (lang == "c") {
        return new HighlightC(is_keyword_c);
    } else if (lang == "c++" || lang == "cxx" || lang == "cpp") {
        return new HighlightC(is_keyword_cxx);
    } else if (lang == "js" || lang == "javascript") {
        return new HighlightC(is_keyword_js);
    } else if (lang == "java") {
        return new HighlightC(is_keyword_java);
    } else if (lang == "py" || lang == "python") {
        return new HighlightPython();
    } else if (lang == "rs" || lang == "rust") {
        return new HighlightRust();
    } else if (lang == "f" || lang == "fortran") {
        return new HighlightFortran();
    } else if (lang == "cob" || lang == "cbl" || lang == "cobol") {
        return new HighlightCobol();
    } else if (lang == "pas" || lang == "pascal") {
        return new HighlightPascal();
    } else {
        return new HighlightPlain;
    }
}
