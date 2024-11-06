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

const KOTLIN_KEYWORDS = new Set([
  'abstract',
  'actual',
  'annotation',
  'as',
  'break',
  'by',
  'catch',
  'class',
  'companion',
  'const',
  'constructor',
  'continue',
  'crossinline',
  'data',
  'delegate',
  'do',
  'dynamic',
  'else',
  'enum',
  'expect',
  'external',
  'field',
  'file',
  'final',
  'finally',
  'for',
  'fun',
  'get',
  'if',
  'import',
  'in',
  'infix',
  'init',
  'inline',
  'inner',
  'interface',
  'internal',
  'is',
  'it',
  'lateinit',
  'noinline',
  'object',
  'open',
  'operator',
  'out',
  'override',
  'package',
  'param',
  'private',
  'property',
  'protected',
  'public',
  'receiver',
  'reified',
  'return',
  'sealed',
  'set',
  'setparam',
  'super',
  'suspend',
  'tailrec',
  'this',
  'throw',
  'try',
  'typealias',
  'typeof',
  'val',
  'var',
  'vararg',
  'when',
  'where',
  'while',
]);

class HighlightKotlin extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static ANNOTATION = 4;
  static ANNOTATION2 = 5;
  static SLASH = 6;
  static SLASH_SLASH = 7;
  static SLASH_STAR = 8;
  static SLASH_STAR_STAR = 9;
  static DQUOTE = 10;
  static DQUOTE_DOLLAR = 11;
  static DQUOTE_VAR = 12;
  static DQUOTESTR = 13;
  static DQUOTESTR_BACKSLASH = 14;
  static DQUOTESTR_DOLLAR = 15;
  static DQUOTESTR_VAR = 16;
  static DQUOTE2 = 17;
  static DQUOTE3 = 18;
  static DQUOTE3_BACKSLASH = 19;
  static DQUOTE3_DOLLAR = 20;
  static DQUOTE3_VAR = 21;
  static DQUOTE31 = 22;
  static DQUOTE32 = 23;

  constructor(delegate) {
    super(delegate);
    this.nest = [];
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightKotlin.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightKotlin.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightKotlin.SLASH;
        } else if (c == '\'') {
          this.state = HighlightKotlin.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightKotlin.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '@') {
          this.state = HighlightKotlin.ANNOTATION;
        } else if (c == '{' && this.nest) {
          this.nest.push(HighlightKotlin.NORMAL);
          this.append('{');
        } else if (c == '}' && this.nest) {
          if ((this.state = this.nest.pop()) != HighlightKotlin.NORMAL)
            this.push("span", "string");
          this.append('}');
        } else {
          this.append(c);
        }
        break;

      case HighlightKotlin.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (KOTLIN_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (JAVA_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else if (this.word && this.word[0] == this.word[0].toUpperCase()) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightKotlin.NORMAL);
        }
        break;

      case HighlightKotlin.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightKotlin.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightKotlin.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightKotlin.NORMAL);
        }
        break;

      case HighlightKotlin.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightKotlin.NORMAL;
        }
        break;

      case HighlightKotlin.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightKotlin.SLASH_STAR_STAR;
        break;

      case HighlightKotlin.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightKotlin.NORMAL;
        } else if (c != '*') {
          this.state = HighlightKotlin.SLASH_STAR;
        }
        break;

      case HighlightKotlin.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightKotlin.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightKotlin.QUOTE_BACKSLASH;
        }
        break;

      case HighlightKotlin.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightKotlin.QUOTE;
        break;

      case HighlightKotlin.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightKotlin.DQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightKotlin.DQUOTESTR_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightKotlin.DQUOTESTR_DOLLAR;
        } else {
          this.state = HighlightKotlin.DQUOTESTR;
        }
        break;

      case HighlightKotlin.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightKotlin.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightKotlin.DQUOTESTR_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightKotlin.DQUOTESTR_DOLLAR;
        }
        break;

      case HighlightKotlin.DQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightKotlin.DQUOTESTR;
        break;

      case HighlightKotlin.DQUOTESTR_DOLLAR:
        if (c == '{') {
          this.append(c);
          this.pop();
          this.nest.push(HighlightKotlin.DQUOTESTR);
          this.state = HighlightKotlin.NORMAL;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "bold");
          this.append(c);
          this.state = HighlightKotlin.DQUOTESTR_VAR;
        } else {
          this.epsilon(HighlightKotlin.DQUOTESTR);
        }
        break;

      case HighlightKotlin.DQUOTESTR_VAR:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightKotlin.DQUOTESTR);
        }
        break;

      case HighlightKotlin.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightKotlin.DQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightKotlin.NORMAL);
        }
        break;

      case HighlightKotlin.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightKotlin.DQUOTE31;
        } else if (c == '$') {
          this.state = HighlightKotlin.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightKotlin.DQUOTE3_BACKSLASH;
        }
        break;

      case HighlightKotlin.DQUOTE31:
        this.append(c);
        if (c == '"') {
          this.state = HighlightKotlin.DQUOTE32;
        } else if (c == '$') {
          this.state = HighlightKotlin.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightKotlin.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightKotlin.DQUOTE3;
        }
        break;

      case HighlightKotlin.DQUOTE32:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightKotlin.NORMAL;
        } else if (c == '$') {
          this.state = HighlightKotlin.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightKotlin.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightKotlin.DQUOTE3;
        }
        break;

      case HighlightKotlin.DQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightKotlin.DQUOTE3;
        break;

      case HighlightKotlin.DQUOTE3_DOLLAR:
        if (c == '{') {
          this.append(c);
          this.pop();
          this.nest.push(HighlightKotlin.DQUOTE3);
          this.state = HighlightKotlin.NORMAL;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "bold");
          this.append(c);
          this.state = HighlightKotlin.DQUOTE3_VAR;
        } else {
          this.append(c);
          this.state = HighlightKotlin.DQUOTE3;
        }
        break;

      case HighlightKotlin.DQUOTE3_VAR:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightKotlin.DQUOTE3);
        }
        break;

      case HighlightKotlin.ANNOTATION:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "attrib");
          this.append('@');
          this.append(c);
          this.state = HighlightKotlin.ANNOTATION2;
        } else {
          this.append('@');
          this.epsilon(HighlightKotlin.NORMAL);
        }
        break;

      case HighlightKotlin.ANNOTATION2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightKotlin.NORMAL);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightKotlin.WORD:
      if (KOTLIN_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (JAVA_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else if (this.word && isupper(this.word[0])) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightKotlin.SLASH:
      this.append('/');
      break;
    case HighlightKotlin.ANNOTATION:
      this.append('@');
      break;
    case HighlightKotlin.QUOTE:
    case HighlightKotlin.QUOTE_BACKSLASH:
    case HighlightKotlin.SLASH_SLASH:
    case HighlightKotlin.SLASH_STAR:
    case HighlightKotlin.SLASH_STAR_STAR:
    case HighlightKotlin.DQUOTE:
    case HighlightKotlin.DQUOTESTR:
    case HighlightKotlin.DQUOTESTR_BACKSLASH:
    case HighlightKotlin.DQUOTESTR_DOLLAR:
    case HighlightKotlin.DQUOTESTR_VAR:
    case HighlightKotlin.DQUOTE2:
    case HighlightKotlin.DQUOTE3:
    case HighlightKotlin.DQUOTE3_BACKSLASH:
    case HighlightKotlin.DQUOTE31:
    case HighlightKotlin.DQUOTE32:
    case HighlightKotlin.ANNOTATION2:
    case HighlightKotlin.DQUOTE3_DOLLAR:
    case HighlightKotlin.DQUOTE3_VAR:
      this.pop();
      break;
    default:
      break;
    }
    this.nest = [];
    this.state = HighlightKotlin.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['kotlin'] = HighlightKotlin;
Highlighter.REGISTRY['kts'] = HighlightKotlin;
Highlighter.REGISTRY['kt'] = HighlightKotlin;
