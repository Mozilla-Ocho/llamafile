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

struct HighlightEntry
{
    char* name;
    Highlight* (*ctor)(void);
};

extern "C" Highlight*
highlight_create_ada_(void)
{
    return new HighlightAda;
}

extern "C" Highlight*
highlight_create_asm_(void)
{
    return new HighlightAsm;
}

extern "C" Highlight*
highlight_create_basic_(void)
{
    return new HighlightBasic;
}

extern "C" Highlight*
highlight_create_bnf_(void)
{
    return new HighlightBnf;
}

extern "C" Highlight*
highlight_create_c_(void)
{
    return new HighlightC(is_keyword_c, //
                          is_keyword_c_type, //
                          is_keyword_c_builtin, //
                          is_keyword_c_constant);
}

extern "C" Highlight*
highlight_create_cxx_(void)
{
    return new HighlightC(is_keyword_cxx, //
                          is_keyword_c_type, //
                          is_keyword_c_builtin, //
                          is_keyword_c_constant);
}

extern "C" Highlight*
highlight_create_cmake_(void)
{
    return new HighlightCmake;
}

extern "C" Highlight*
highlight_create_cobol_(void)
{
    return new HighlightCobol;
}

extern "C" Highlight*
highlight_create_csharp_(void)
{
    return new HighlightCsharp;
}

extern "C" Highlight*
highlight_create_css_(void)
{
    return new HighlightCss;
}

extern "C" Highlight*
highlight_create_d_(void)
{
    return new HighlightD;
}

extern "C" Highlight*
highlight_create_forth_(void)
{
    return new HighlightForth;
}

extern "C" Highlight*
highlight_create_fortran_(void)
{
    return new HighlightFortran;
}

extern "C" Highlight*
highlight_create_go_(void)
{
    return new HighlightGo;
}

extern "C" Highlight*
highlight_create_haskell_(void)
{
    return new HighlightHaskell;
}

extern "C" Highlight*
highlight_create_html_(void)
{
    return new HighlightHtml;
}

extern "C" Highlight*
highlight_create_java_(void)
{
    return new HighlightJava;
}

extern "C" Highlight*
highlight_create_js_(void)
{
    return new HighlightJs;
}

extern "C" Highlight*
highlight_create_julia_(void)
{
    return new HighlightJulia;
}

extern "C" Highlight*
highlight_create_kotlin_(void)
{
    return new HighlightKotlin;
}

extern "C" Highlight*
highlight_create_ld_(void)
{
    return new HighlightLd;
}

extern "C" Highlight*
highlight_create_lisp_(void)
{
    return new HighlightLisp;
}

extern "C" Highlight*
highlight_create_lua_(void)
{
    return new HighlightLua;
}

extern "C" Highlight*
highlight_create_m4_(void)
{
    return new HighlightM4;
}

extern "C" Highlight*
highlight_create_make_(void)
{
    return new HighlightMake;
}

extern "C" Highlight*
highlight_create_markdown_(void)
{
    return new HighlightMarkdown;
}

extern "C" Highlight*
highlight_create_matlab_(void)
{
    return new HighlightMatlab;
}

extern "C" Highlight*
highlight_create_ocaml_(void)
{
    return new HighlightOcaml;
}

extern "C" Highlight*
highlight_create_pascal_(void)
{
    return new HighlightPascal;
}

extern "C" Highlight*
highlight_create_perl_(void)
{
    return new HighlightPerl;
}

extern "C" Highlight*
highlight_create_php_(void)
{
    return new HighlightPhp;
}

extern "C" Highlight*
highlight_create_python_(void)
{
    return new HighlightPython;
}

extern "C" Highlight*
highlight_create_r_(void)
{
    return new HighlightR;
}

extern "C" Highlight*
highlight_create_ruby_(void)
{
    return new HighlightRuby;
}

extern "C" Highlight*
highlight_create_rust_(void)
{
    return new HighlightRust;
}

extern "C" Highlight*
highlight_create_scala_(void)
{
    return new HighlightScala;
}

extern "C" Highlight*
highlight_create_shell_(void)
{
    return new HighlightShell;
}

extern "C" Highlight*
highlight_create_sql_(void)
{
    return new HighlightSql;
}

extern "C" Highlight*
highlight_create_swift_(void)
{
    return new HighlightSwift;
}

extern "C" Highlight*
highlight_create_tcl_(void)
{
    return new HighlightTcl;
}

extern "C" Highlight*
highlight_create_tex_(void)
{
    return new HighlightTex;
}

extern "C" Highlight*
highlight_create_txt_(void)
{
    return new HighlightTxt;
}

extern "C" Highlight*
highlight_create_typescript_(void)
{
    return new HighlightTypescript;
}

extern "C" Highlight*
highlight_create_zig_(void)
{
    return new HighlightZig;
}

extern "C" const HighlightEntry*
highlight_lookup_(const char* str, size_t len);

Highlight*
Highlight::create(const std::string_view& lang)
{
    const HighlightEntry* slot;
    if ((slot = highlight_lookup_(lang.data(), lang.size())))
        return slot->ctor();
    return nullptr;
}
