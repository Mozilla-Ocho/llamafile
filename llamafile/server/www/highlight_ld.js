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

const LD_KEYWORDS = new Set([
  'AFTER',
  'ASSERT',
  'AS_NEEDED',
  'AT',
  'BEFORE',
  'BYTE',
  'COMMON',
  'CONSTRUCTORS',
  'COPY',
  'CREATE_OBJECT_SYMBOLS',
  'DSECT',
  'ENTRY',
  'EXTERN',
  'FILEHDR',
  'FILL',
  'FLAGS',
  'FORCE_COMMON_ALLOCATION',
  'GROUP',
  'HIDDEN',
  'INCLUDE',
  'INFO',
  'INHIBIT_COMMON_ALLOCATION',
  'INPUT',
  'INSERT',
  'KEEP',
  'LD_FEATURE',
  'LONG',
  'MEMORY',
  'NOCROSSREFS',
  'NOCROSSREFS_TO',
  'NOLOAD',
  'ONLY_IF_RO',
  'ONLY_IF_RW',
  'OUTPUT',
  'OUTPUT_ARCH',
  'OUTPUT_FORMAT',
  'OVERLAY',
  'PHDR',
  'PHDRS',
  'PROVIDE',
  'PROVIDE_HIDDEN',
  'PT_DYNAMIC',
  'PT_GNU_STACK',
  'PT_INTERP',
  'PT_LOAD',
  'PT_NOTE',
  'PT_NULL',
  'PT_PHDR',
  'PT_SHLIB',
  'PT_TLS',
  'QUAD',
  'REGION_ALIAS',
  'SEARCH_DIR',
  'SECTIONS',
  'SHORT',
  'SORT',
  'SORT_BY_ALIGNMENT',
  'SORT_BY_INIT_PRIORITY',
  'SORT_BY_NAME',
  'SORT_NONE',
  'SQUAD',
  'STARTUP',
  'SUBALIGN',
  'TARGET',
  'VERSION',
  '__CTOR_END__',
  '__CTOR_LIST__',
  '__DTOR_END__',
  '__DTOR_LIST__',
]);

const LD_BUILTINS = new Set([
  'ABSOLUTE',
  'ADDR',
  'ALIGN',
  'ALIGNOF',
  'BLOCK',
  'COMMONPAGESIZE',
  'CONSTANT',
  'DATA_SEGMENT_ALIGN',
  'DATA_SEGMENT_END',
  'DATA_SEGMENT_RELRO_END',
  'DEFINED',
  'LENGTH',
  'LOADADDR',
  'LOG2CEIL',
  'MAX',
  'MAXPAGESIZE',
  'MIN',
  'NEXT',
  'ORIGIN',
  'SEGMENT_START',
  'SIZEOF',
  'SIZEOF_HEADERS',
  'l',
  'len',
  'o',
  'org',
  'sizeof_headers',
]);

const LD_WARNINGS = new Set([
  '/DISCARD/',
  ':NONE',
  'EXCLUDE_FILE',
]);

class HighlightLd extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static DQUOTE = 2;
  static DQUOTE_BACKSLASH = 3;
  static SLASH = 4;
  static SLASH_SLASH = 5;
  static SLASH_STAR = 6;
  static SLASH_STAR_STAR = 7;

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.is_bol = true;
    this.is_cpp = false;
    this.is_cpp_builtin = false;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightLd.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '.' || c == ':') {
          this.state = HighlightLd.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightLd.SLASH;
        } else if (c == '"') {
          this.state = HighlightLd.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '#' && this.is_bol) {
          this.is_cpp = true;
          this.push("span", "builtin");
          this.append(c);
        } else if (c == '\n') {
          this.append(c);
          if (this.is_cpp) {
            if (this.is_cpp_builtin) {
              this.pop();
            }
            this.is_cpp = false;
          }
          this.is_cpp_builtin = false;
        } else {
          this.append(c);
        }
        break;

      case HighlightLd.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '.' || c == '/' || c == ':') {
          this.word += c;
        } else {
          if (this.is_cpp) {
            if (CPP_KEYWORDS.has(this.word)) {
              this.push("span", "builtin");
              this.append(this.word);
              this.pop();
              if (this.is_cpp_builtin) {
                this.pop();
                this.is_cpp_builtin = false;
              }
            } else if (C_CONSTANTS.has(this.word)) {
              this.push("span", "constant");
              this.append(this.word);
              this.pop();
            } else {
              this.append(this.word);
            }
          } else {
            if (LD_KEYWORDS.has(this.word)) {
              this.push("span", "keyword");
              this.append(this.word);
              this.pop();
            } else if (LD_BUILTINS.has(this.word)) {
              this.push("span", "builtin");
              this.append(this.word);
              this.pop();
            } else if (LD_WARNINGS.has(this.word)) {
              this.push("span", "warning");
              this.append(this.word);
              this.pop();
            } else {
              this.append(this.word);
            }
          }
          this.word = '';
          this.epsilon(HighlightLd.NORMAL);
        }
        break;

      case HighlightLd.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightLd.SLASH_SLASH;
        } else if (c == 'D') {
          // for /DISCARD/ warning keyword
          this.word += "/D";
          this.state = HighlightLd.WORD;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightLd.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightLd.NORMAL);
        }
        break;

      case HighlightLd.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightLd.NORMAL;
          this.is_cpp = false;
        }
        break;

      case HighlightLd.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightLd.SLASH_STAR_STAR;
        break;

      case HighlightLd.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightLd.NORMAL;
        } else if (c != '*') {
          this.state = HighlightLd.SLASH_STAR;
        }
        break;

      case HighlightLd.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightLd.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightLd.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightLd.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightLd.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
      if (this.is_bol) {
        if (!isspace(c))
          this.is_bol = false;
      } else {
        if (c == '\n')
          this.is_bol = true;
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightLd.WORD:
      if (this.is_cpp) {
        if (CPP_KEYWORDS.has(this.word)) {
          this.push("span", "builtin");
          this.append(this.word);
          this.pop();
        } else if (C_CONSTANTS.has(this.word)) {
          this.push("span", "constant");
          this.append(this.word);
          this.pop();
        } else {
          this.append(this.word);
          this.pop();
        }
      } else {
        if (LD_KEYWORDS.has(this.word)) {
          this.push("span", "keyword");
          this.append(this.word);
          this.pop();
        } else if (LD_BUILTINS.has(this.word)) {
          this.push("span", "builtin");
          this.append(this.word);
          this.pop();
        } else if (LD_WARNINGS.has(this.word)) {
          this.push("span", "warning");
          this.append(this.word);
          this.pop();
        } else {
          this.append(this.word);
        }
      }
      this.word = '';
      break;
    case HighlightLd.SLASH:
      this.append('/');
      if (this.is_cpp)
        this.pop();
      break;
    case HighlightLd.DQUOTE:
    case HighlightLd.DQUOTE_BACKSLASH:
    case HighlightLd.SLASH_SLASH:
    case HighlightLd.SLASH_STAR:
    case HighlightLd.SLASH_STAR_STAR:
      this.pop();
      break;
    default:
      if (this.is_cpp)
        this.pop();
      break;
    }
    if (this.is_cpp) {
      if (this.is_cpp_builtin) {
        this.pop();
      }
      this.is_cpp = false;
    }
    this.is_cpp_builtin = false;
    this.is_bol = true;
    this.state = HighlightLd.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['ld-script'] = HighlightLd;
Highlighter.REGISTRY['lds'] = HighlightLd;
Highlighter.REGISTRY['ld'] = HighlightLd;
