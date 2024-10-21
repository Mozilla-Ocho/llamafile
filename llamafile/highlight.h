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
#include <string_view>

#define HI_RESET "\033[0m"
#define HI_BOLD "\033[1m"
#define HI_KEYWORD "\033[1;34m" // bold blue
#define HI_STRING "\033[32m" // green
#define HI_COMMENT "\033[31m" // red
#define HI_VAR "\033[1;35m" // magenta
#define HI_MACRO "\033[35m" // magenta
#define HI_ATTRIB "\033[35m" // magenta
#define HI_LINENO "\033[2m" // fade
#define HI_CONTIN "\033[33m" // yellow
#define HI_LABEL "\033[33m" // yellow
#define HI_TYPE "\033[36m" // cyan
#define HI_CLASS "\033[1;36m" // bold cyan
#define HI_SELECTOR "\033[33m" // yellow
#define HI_PROPERTY "\033[36m" // cyan
#define HI_TAG "\033[33m" // yellow
#define HI_INCODE "\033[1;35m" // magenta
#define HI_BUILTIN "\033[35m" // magenta
#define HI_CONSTANT "\033[1;35m" // bold magenta
#define HI_LISPKW "\033[35m" // magenta
#define HI_ENTITY "\033[36m" // cyan
#define HI_OPERATOR "\033[36m" // cyan

typedef const char *is_keyword_f(const char *, size_t);

extern "C" {
is_keyword_f is_keyword_c;
is_keyword_f is_keyword_c_type;
is_keyword_f is_keyword_c_builtin;
is_keyword_f is_keyword_c_constant;
is_keyword_f is_keyword_cxx;
is_keyword_f is_keyword_js;
is_keyword_f is_keyword_java;
is_keyword_f is_keyword_python;
is_keyword_f is_keyword_rust;
is_keyword_f is_keyword_fortran;
is_keyword_f is_keyword_fortran_type;
is_keyword_f is_keyword_fortran_builtin;
is_keyword_f is_keyword_cobol;
is_keyword_f is_keyword_pascal;
is_keyword_f is_keyword_pascal_type;
is_keyword_f is_keyword_go;
is_keyword_f is_keyword_go_type;
is_keyword_f is_keyword_sql;
is_keyword_f is_keyword_php;
is_keyword_f is_keyword_php_constant;
is_keyword_f is_keyword_csharp;
is_keyword_f is_keyword_csharp_constant;
is_keyword_f is_keyword_kotlin;
is_keyword_f is_keyword_lua;
is_keyword_f is_keyword_lua_builtin;
is_keyword_f is_keyword_lua_constant;
is_keyword_f is_keyword_lisp;
is_keyword_f is_keyword_ada;
is_keyword_f is_keyword_haskell;
is_keyword_f is_keyword_perl;
is_keyword_f is_keyword_shell;
is_keyword_f is_keyword_shell_builtin;
is_keyword_f is_keyword_swift;
is_keyword_f is_keyword_swift_type;
is_keyword_f is_keyword_d;
is_keyword_f is_keyword_zig;
is_keyword_f is_keyword_zig_type;
is_keyword_f is_keyword_zig_builtin;
is_keyword_f is_keyword_zig_constant;
is_keyword_f is_keyword_tcl;
is_keyword_f is_keyword_tcl_type;
is_keyword_f is_keyword_tcl_builtin;
is_keyword_f is_keyword_ruby;
is_keyword_f is_keyword_ruby_builtin;
is_keyword_f is_keyword_ruby_constant;
}

class Highlight {
  public:
    static Highlight *create(const std::string_view &lang);
    virtual ~Highlight() = default;
    virtual void feed(std::string *result, std::string_view input) = 0;
    virtual void flush(std::string *result) = 0;
};

class HighlightPlain : public Highlight {
  public:
    HighlightPlain();
    ~HighlightPlain() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;
};

class HighlightC : public Highlight {
  public:
    HighlightC(is_keyword_f is_keyword = is_keyword_c, is_keyword_f is_type = nullptr,
               is_keyword_f is_builtin = nullptr, is_keyword_f is_constant = nullptr);
    ~HighlightC() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int i_;
    std::string word_;
    std::string heredoc_;
    is_keyword_f *is_type_;
    is_keyword_f *is_keyword_;
    is_keyword_f *is_builtin_;
    is_keyword_f *is_constant_;
};

class HighlightPython : public Highlight {
  public:
    HighlightPython();
    ~HighlightPython() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightMarkdown : public Highlight {
  public:
    HighlightMarkdown();
    ~HighlightMarkdown() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string lang_;
    Highlight *highlighter_ = nullptr;
};

class HighlightRust : public Highlight {
  public:
    HighlightRust();
    ~HighlightRust() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int nest_ = 0;
    std::string word_;
};

class HighlightFortran : public Highlight {
  public:
    HighlightFortran();
    ~HighlightFortran() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int col_ = -1;
    std::string word_;
};

class HighlightCobol : public Highlight {
  public:
    HighlightCobol();
    ~HighlightCobol() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int col_ = -1;
    std::string word_;
};

class HighlightPascal : public Highlight {
  public:
    HighlightPascal();
    ~HighlightPascal() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightSql : public Highlight {
  public:
    HighlightSql();
    ~HighlightSql() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightCss : public Highlight {
  public:
    HighlightCss();
    ~HighlightCss() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
};

class HighlightHtml : public Highlight {
  public:
    HighlightHtml();
    ~HighlightHtml() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int i_ = 0;
    std::string name_;
    std::string closer_;
    std::string pending_;
    Highlight *highlighter_ = nullptr;
};

class HighlightPhp : public Highlight {
  public:
    HighlightPhp();
    ~HighlightPhp() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightLua : public Highlight {
  public:
    HighlightLua();
    ~HighlightLua() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int level1_;
    int level2_;
    std::string word_;
};

class HighlightLisp : public Highlight {
  public:
    HighlightLisp();
    ~HighlightLisp() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    bool is_first_ = false;
    std::string symbol_;
};

class HighlightAda : public Highlight {
  public:
    HighlightAda();
    ~HighlightAda() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string symbol_;
};

class HighlightHaskell : public Highlight {
  public:
    HighlightHaskell();
    ~HighlightHaskell() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string symbol_;
};

class HighlightPerl : public Highlight {
  public:
    HighlightPerl();
    ~HighlightPerl() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int i_ = 0;
    bool pending_heredoc_ = false;
    bool indented_heredoc_ = false;
    std::string word_;
    std::string heredoc_;
};

class HighlightShell : public Highlight {
  public:
    HighlightShell();
    ~HighlightShell() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int i_ = 0;
    bool pending_heredoc_ = false;
    bool indented_heredoc_ = false;
    std::string word_;
    std::string heredoc_;
};

class HighlightZig : public Highlight {
  public:
    HighlightZig();
    ~HighlightZig() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightTcl : public Highlight {
  public:
    HighlightTcl();
    ~HighlightTcl() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
};

class HighlightCsharp : public Highlight {
  public:
    HighlightCsharp();
    ~HighlightCsharp() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int trips1_;
    int trips2_;
    std::string word_;
};

class HighlightRuby : public Highlight {
  public:
    HighlightRuby();
    ~HighlightRuby() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    int i_ = 0;
    int level_ = 0;
    unsigned char q_ = 0;
    unsigned char last_ = 0;
    unsigned char opener_ = 0;
    unsigned char closer_ = 0;
    bool pending_heredoc_ = false;
    bool indented_heredoc_ = false;
    std::string word_;
    std::string heredoc_;
};
