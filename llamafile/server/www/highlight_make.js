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

const MAKE_KEYWORDS = new Set([
  '-include',
  '-sinclude',
  'define',
  'else',
  'endef',
  'endif',
  'export',
  'if',
  'ifdef',
  'ifeq',
  'ifndef',
  'ifneq',
  'include',
  'override',
  'private',
  'sinclude',
  'undefine',
  'unexport',
  'vpath',
]);

const MAKE_BUILTINS = new Set([
  'abspath',
  'addprefix',
  'addsuffix',
  'and',
  'basename',
  'call',
  'dir',
  'error',
  'eval',
  'file',
  'filter',
  'filter-out',
  'findstring',
  'firstword',
  'flavor',
  'foreach',
  'if',
  'join',
  'lastword',
  'notdir',
  'or',
  'origin',
  'patsubst',
  'realpath',
  'shell',
  'sort',
  'strip',
  'subst',
  'suffix',
  'value',
  'warning',
  'wildcard',
  'word',
  'wordlist',
  'words',
]);

class HighlightMake extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static COMMENT_BACKSLASH = 3;
  static DOLLAR = 4;
  static DOLLAR2 = 5;
  static VARIABLE = 6;
  static BACKSLASH = 7;

  static is_automatic_variable(c) {
    switch (c) {
    case '@':
    case '%':
    case '<':
    case '?':
    case '^':
    case '+':
    case '|':
    case '*':
      return true;
    default:
      return false;
    }
  }

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightMake.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '-' || c == '.') {
          this.state = HighlightMake.WORD;
          this.word += c;
          break;
        } else if (c == '#') {
          this.state = HighlightMake.COMMENT;
          this.push("span", "comment");
          this.append('#');
        } else if (c == '$') {
          this.state = HighlightMake.DOLLAR;
          this.append('$');
        } else if (c == '\\') {
          this.state = HighlightMake.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
        } else {
          this.append(c);
        }
        break;

      case HighlightMake.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightMake.NORMAL;
        break;

      case HighlightMake.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '-' || c == '.') {
          this.word += c;
        } else {
          if (MAKE_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightMake.NORMAL);
        }
        break;

      case HighlightMake.DOLLAR:
        if (isdigit(c) || HighlightMake.is_automatic_variable(c)) {
          this.push("span", "var");
          this.append(c);
          this.pop();
          this.state = HighlightMake.NORMAL;
        } else if (c == '$') {
          this.state = HighlightMake.DOLLAR2;
          this.append('$');
        } else if (c == '(') {
          this.state = HighlightMake.VARIABLE;
          this.append('(');
        } else {
          this.epsilon(HighlightMake.NORMAL);
        }
        break;

      case HighlightMake.DOLLAR2:
        if (c == '(') {
          this.state = HighlightMake.VARIABLE;
          this.append('(');
        } else {
          this.epsilon(HighlightMake.NORMAL);
        }
        break;

      case HighlightMake.VARIABLE:
        if (isalnum(c) || //
            c == '%' || //
            c == '*' || //
            c == '+' || //
            c == '-' || //
            c == '.' || //
            c == '<' || //
            c == '?' || //
            c == '@' || //
            c == '_') {
          this.word += c;
        } else if (c == '$' && !this.word) {
          this.state = HighlightMake.DOLLAR;
          this.append('$');
        } else if (c == ')' || //
                   c == ':') {
          this.push("span", "var");
          this.append(this.word);
          this.pop();
          this.word = '';
          this.append(c);
          this.state = HighlightMake.NORMAL;
        } else {
          if (MAKE_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightMake.NORMAL);
        }
        break;

      case HighlightMake.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightMake.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightMake.COMMENT_BACKSLASH;
        }
        break;

      case HighlightMake.COMMENT_BACKSLASH:
        this.append(c);
        this.state = HighlightMake.COMMENT;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightMake.WORD:
      if (MAKE_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightMake.VARIABLE:
      if (MAKE_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightMake.COMMENT:
    case HighlightMake.COMMENT_BACKSLASH:
    case HighlightMake.BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightMake.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['make'] = HighlightMake;
Highlighter.REGISTRY['mk'] = HighlightMake;
Highlighter.REGISTRY['gmake'] = HighlightMake;
Highlighter.REGISTRY['makefile'] = HighlightMake;
Highlighter.REGISTRY['gmakefile'] = HighlightMake;
