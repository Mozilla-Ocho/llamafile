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

const JULIA_KEYWORDS = new Set([
  'baremodule',
  'begin',
  'break',
  'catch',
  'const',
  'continue',
  'do',
  'else',
  'elseif',
  'end',
  'export',
  'false',
  'finally',
  'for',
  'function',
  'global',
  'if',
  'import',
  'let',
  'local',
  'macro',
  'module',
  'quote',
  'return',
  'struct',
  'true',
  'try',
  'using',
  'while',
]);

class HighlightJulia extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static QUOTE = 3;
  static QUOTE_BACKSLASH = 4;
  static ANNOTATION = 5;
  static ANNOTATION2 = 6;
  static DQUOTE = 7;
  static DQUOTE_VAR = 8;
  static DQUOTESTR = 9;
  static DQUOTESTR_BACKSLASH = 10;
  static DQUOTE2 = 11;
  static DQUOTE3 = 12;
  static DQUOTE3_BACKSLASH = 13;
  static DQUOTE31 = 14;
  static DQUOTE32 = 15;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightJulia.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightJulia.WORD;
          this.word += c;
        } else if (c == '#') {
          this.state = HighlightJulia.COMMENT;
          this.push("span", "comment");
          this.append('#');
        } else if (c == '\'') {
          this.state = HighlightJulia.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightJulia.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '@') {
          this.state = HighlightJulia.ANNOTATION;
        } else {
          this.append(c);
        }
        break;

      case HighlightJulia.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (JULIA_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightJulia.NORMAL);
        }
        break;

      case HighlightJulia.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightJulia.NORMAL;
        }
        break;

      case HighlightJulia.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightJulia.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJulia.QUOTE_BACKSLASH;
        }
        break;

      case HighlightJulia.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightJulia.QUOTE;
        break;

      case HighlightJulia.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJulia.DQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightJulia.DQUOTESTR_BACKSLASH;
        } else {
          this.state = HighlightJulia.DQUOTESTR;
        }
        break;

      case HighlightJulia.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightJulia.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJulia.DQUOTESTR_BACKSLASH;
        }
        break;

      case HighlightJulia.DQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightJulia.DQUOTESTR;
        break;

      case HighlightJulia.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightJulia.DQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightJulia.NORMAL);
        }
        break;

      case HighlightJulia.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJulia.DQUOTE31;
        } else if (c == '\\') {
          this.state = HighlightJulia.DQUOTE3_BACKSLASH;
        }
        break;

      case HighlightJulia.DQUOTE31:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJulia.DQUOTE32;
        } else if (c == '\\') {
          this.state = HighlightJulia.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightJulia.DQUOTE3;
        }
        break;

      case HighlightJulia.DQUOTE32:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightJulia.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJulia.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightJulia.DQUOTE3;
        }
        break;

      case HighlightJulia.DQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightJulia.DQUOTE3;
        break;

      case HighlightJulia.ANNOTATION:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "attrib");
          this.append('@');
          this.append(c);
          this.state = HighlightJulia.ANNOTATION2;
        } else {
          this.append('@');
          this.epsilon(HighlightJulia.NORMAL);
        }
        break;

      case HighlightJulia.ANNOTATION2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightJulia.NORMAL);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightJulia.WORD:
      if (JULIA_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightJulia.ANNOTATION:
      this.append('@');
      break;
    case HighlightJulia.COMMENT:
    case HighlightJulia.QUOTE:
    case HighlightJulia.QUOTE_BACKSLASH:
    case HighlightJulia.DQUOTE:
    case HighlightJulia.DQUOTESTR:
    case HighlightJulia.DQUOTESTR_BACKSLASH:
    case HighlightJulia.DQUOTE2:
    case HighlightJulia.DQUOTE3:
    case HighlightJulia.DQUOTE3_BACKSLASH:
    case HighlightJulia.DQUOTE31:
    case HighlightJulia.DQUOTE32:
    case HighlightJulia.ANNOTATION2:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightJulia.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['julia'] = HighlightJulia;
