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

#define RESET "\033[0m"
#define BOLDE "\033[1m"
#define KEYWORD "\033[1;34m" // bold blue
#define STRING "\033[0;32m" // green
#define COMMENT "\033[0;31m" // red

typedef const char *is_keyword_f(const char *, size_t);

extern "C" {
is_keyword_f is_keyword_c;
is_keyword_f is_keyword_cxx;
is_keyword_f is_keyword_js;
is_keyword_f is_keyword_java;
is_keyword_f is_keyword_python;
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
    explicit HighlightC(is_keyword_f is_keyword = is_keyword_c);
    ~HighlightC() override;
    void feed(std::string *result, std::string_view input) override;
    void flush(std::string *result) override;

  private:
    int t_ = 0;
    std::string word_;
    is_keyword_f *is_keyword_;
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
