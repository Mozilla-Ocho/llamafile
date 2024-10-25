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

    if (lang == "md" || //
        lang == "markdown")
        return new HighlightMarkdown;

    if (lang == "c" || //
        lang == "h" || //
        lang == "m")
        return new HighlightC(is_keyword_c, //
                              is_keyword_c_type, //
                              is_keyword_c_builtin, //
                              is_keyword_c_constant);

    if (lang == "c++" || //
        lang == "cxx" || //
        lang == "cpp" || //
        lang == "hpp" || //
        lang == "cc" || //
        lang == "cu")
        return new HighlightC(is_keyword_cxx, //
                              is_keyword_c_type, //
                              is_keyword_c_builtin, //
                              is_keyword_c_constant);

    if (lang == "js" || //
        lang == "json" || //
        lang == "javascript")
        return new HighlightJs(is_keyword_js, nullptr);

    if (lang == "ts" || //
        lang == "typescript")
        return new HighlightJs(is_keyword_typescript, is_keyword_typescript_type);

    if (lang == "java")
        return new HighlightJava(is_keyword_java);

    if (lang == "py" || //
        lang == "python")
        return new HighlightPython;

    if (lang == "rs" || //
        lang == "rust")
        return new HighlightRust;

    if (lang == "f" || //
        lang == "fortran")
        return new HighlightFortran;

    if (lang == "cob" || //
        lang == "cbl" || //
        lang == "cobol")
        return new HighlightCobol;

    if (lang == "pas" || //
        lang == "pascal")
        return new HighlightPascal;

    if (lang == "go")
        return new HighlightGo;

    if (lang == "sql")
        return new HighlightSql;

    if (lang == "css")
        return new HighlightCss;

    if (lang == "html" || //
        lang == "xml")
        return new HighlightHtml;

    if (lang == "php")
        return new HighlightHtml; // sic

    if (lang == "php!")
        return new HighlightPhp;

    if (lang == "csharp" || //
        lang == "cs" || //
        lang == "c#")
        return new HighlightCsharp;

    if (lang == "kotlin")
        return new HighlightJava(is_keyword_kotlin);

    if (lang == "lua")
        return new HighlightLua;

    if (lang == "lisp" || //
        lang == "el" || //
        lang == "elisp" || //
        lang == "cl" || //
        lang == "clisp" || //
        lang == "scheme" || //
        lang == "clojure")
        return new HighlightLisp;

    if (lang == "ada")
        return new HighlightAda;

    if (lang == "haskell" || //
        lang == "hs")
        return new HighlightHaskell;

    if (lang == "perl" || //
        lang == "pl")
        return new HighlightPerl;

    if (lang == "shell" || //
        lang == "bash" || //
        lang == "sh" || //
        lang == "ksh")
        return new HighlightShell;

    if (lang == "swift")
        return new HighlightC(is_keyword_swift, is_keyword_swift_type);

    if (lang == "d")
        return new HighlightC(is_keyword_d);

    if (lang == "zig")
        return new HighlightZig;

    if (lang == "tcl")
        return new HighlightTcl;

    if (lang == "m4" || //
        lang == "ac")
        return new HighlightM4;

    if (lang == "ruby" || //
        lang == "rb")
        return new HighlightRuby;

    if (lang == "fs" || //
        lang == "4th" || //
        lang == "frt" || //
        lang == "fth" || //
        lang == "forth")
        return new HighlightForth;

    return new HighlightPlain;
}
