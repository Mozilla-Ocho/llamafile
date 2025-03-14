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

const TYPESCRIPT_KEYWORDS = new Set([
  'abstract',
  'as',
  'async',
  'await',
  'break',
  'case',
  'catch',
  'class',
  'const',
  'continue',
  'debugger',
  'declare',
  'default',
  'delete',
  'do',
  'else',
  'enum',
  'export',
  'extends',
  'finally',
  'for',
  'from',
  'function',
  'get',
  'if',
  'implements',
  'import',
  'in',
  'infer',
  'instanceof',
  'interface',
  'is',
  'keyof',
  'let',
  'namespace',
  'new',
  'of',
  'private',
  'protected',
  'public',
  'readonly',
  'return',
  'satisfies',
  'set',
  'static',
  'switch',
  'target',
  'this',
  'throw',
  'try',
  'type',
  'typeof',
  'var',
  'while',
  'with',
  'yield',
]);

const TYPESCRIPT_TYPES = new Set([
  'any',
  'bigint',
  'boolean',
  'never',
  'number',
  'object',
  'string',
  'symbol',
  'unknown',
  'void',
]);

class HighlightTypescript extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static SLASH = 6;
  static SLASH_SLASH = 7;
  static SLASH_STAR = 8;
  static SLASH_STAR_STAR = 9;
  static TICK = 10;
  static TICK_BACKSLASH = 11;
  static TICK_DOLLAR = 12;
  static REGEX = 13;
  static REGEX_BACKSLASH = 14;
  static REGEX_SQUARE = 15;
  static REGEX_SQUARE_BACKSLASH = 16;

  static EXPECT_VALUE = 0;
  static EXPECT_OPERATOR = 1;

  constructor(delegate) {
    super(delegate);
    this.expect = 0;
    this.word = '';
    this.nest = [];
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightTypescript.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightTypescript.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightTypescript.SLASH;
        } else if (c == '\'') {
          this.state = HighlightTypescript.QUOTE;
          this.push("span", "string");
          this.append('\'');
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
        } else if (c == '"') {
          this.state = HighlightTypescript.DQUOTE;
          this.push("span", "string");
          this.append('"');
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
        } else if (c == '`') {
          this.state = HighlightTypescript.TICK;
          this.push("span", "string");
          this.append('`');
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
        } else if (c == '{' && this.nest.length) {
          this.expect = HighlightTypescript.EXPECT_VALUE;
          this.append('{');
          this.nest.push(HighlightTypescript.NORMAL);
        } else if (c == '}' && this.nest.length) {
          if ((this.state = this.nest.pop()) != HighlightTypescript.NORMAL)
            this.push("span", "string");
          this.append('}');
        } else if (c == ')' || c == '}' || c == ']') {
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
          this.append(c);
        } else if (ispunct(c)) {
          this.expect = HighlightTypescript.EXPECT_VALUE;
          this.append(c);
        } else if (isdigit(c) || c == '.') {
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightTypescript.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (TYPESCRIPT_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
            this.expect = HighlightTypescript.EXPECT_VALUE;
          } else if (TYPESCRIPT_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
            this.expect = HighlightTypescript.EXPECT_VALUE;
          } else if (JS_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
            this.expect = HighlightTypescript.EXPECT_OPERATOR;
          } else if (JS_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
            this.expect = HighlightTypescript.EXPECT_OPERATOR;
          } else {
            this.append(this.word);
            this.expect = HighlightTypescript.EXPECT_OPERATOR;
          }
          this.word = '';
          this.epsilon(HighlightTypescript.NORMAL);
        }
        break;

      case HighlightTypescript.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightTypescript.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightTypescript.SLASH_STAR;
        } else if (this.expect == HighlightTypescript.EXPECT_VALUE) {
          this.expect = HighlightTypescript.EXPECT_OPERATOR;
          this.push("span", "string");
          this.append('/');
          this.append(c);
          if (c == '\\') {
            this.state = HighlightTypescript.REGEX_BACKSLASH;
          } else if (c == '[') {
            this.state = HighlightTypescript.REGEX_SQUARE;
          } else {
            this.state = HighlightTypescript.REGEX;
          }
        } else {
          this.append('/');
          this.epsilon(HighlightTypescript.NORMAL);
        }
        break;

      case HighlightTypescript.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        }
        break;

      case HighlightTypescript.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightTypescript.SLASH_STAR_STAR;
        break;

      case HighlightTypescript.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        } else if (c != '*') {
          this.state = HighlightTypescript.SLASH_STAR;
        }
        break;

      case HighlightTypescript.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightTypescript.QUOTE_BACKSLASH;
        }
        break;

      case HighlightTypescript.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightTypescript.QUOTE;
        break;

      case HighlightTypescript.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightTypescript.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightTypescript.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightTypescript.DQUOTE;
        break;

      case HighlightTypescript.TICK:
        if (c == '`') {
          this.append('`');
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        } else if (c == '$') {
          this.state = HighlightTypescript.TICK_DOLLAR;
        } else if (c == '\\') {
          this.append('\\');
          this.state = HighlightTypescript.TICK_BACKSLASH;
        } else {
          this.append(c);
        }
        break;

      case HighlightTypescript.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightTypescript.TICK;
        break;

      case HighlightTypescript.TICK_DOLLAR:
        if (c == '{') {
          this.push("span", "bold");
          this.append('$');
          this.pop();
          this.append('{');
          this.pop();
          this.expect = HighlightTypescript.EXPECT_VALUE;
          this.nest.push(HighlightTypescript.TICK);
          this.state = HighlightTypescript.NORMAL;
        } else {
          this.push("span", "warning");
          this.append('$');
          this.pop();
          this.epsilon(HighlightTypescript.TICK);
        }
        break;

      case HighlightTypescript.REGEX:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightTypescript.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightTypescript.REGEX_BACKSLASH;
        } else if (c == '[') {
          this.state = HighlightTypescript.REGEX_SQUARE;
        }
        break;

      case HighlightTypescript.REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightTypescript.REGEX;
        break;

      case HighlightTypescript.REGEX_SQUARE:
        this.append(c);
        if (c == '\\') {
          this.state = HighlightTypescript.REGEX_SQUARE_BACKSLASH;
        } else if (c == ']') {
          this.state = HighlightTypescript.REGEX;
        }
        break;

      case HighlightTypescript.REGEX_SQUARE_BACKSLASH:
        this.append(c);
        this.state = HighlightTypescript.REGEX_SQUARE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightTypescript.WORD:
      if (TYPESCRIPT_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (TYPESCRIPT_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (JS_BUILTINS.has(this.word)) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (JS_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightTypescript.SLASH:
      this.append('/');
      break;
    case HighlightTypescript.TICK_DOLLAR:
      this.append('$');
      this.pop();
      break;
    case HighlightTypescript.TICK:
    case HighlightTypescript.TICK_BACKSLASH:
    case HighlightTypescript.QUOTE:
    case HighlightTypescript.QUOTE_BACKSLASH:
    case HighlightTypescript.DQUOTE:
    case HighlightTypescript.DQUOTE_BACKSLASH:
    case HighlightTypescript.SLASH_SLASH:
    case HighlightTypescript.SLASH_STAR:
    case HighlightTypescript.SLASH_STAR_STAR:
    case HighlightTypescript.REGEX:
    case HighlightTypescript.REGEX_BACKSLASH:
    case HighlightTypescript.REGEX_SQUARE:
    case HighlightTypescript.REGEX_SQUARE_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightTypescript.NORMAL;
    this.nest = [];
    this.delegate.flush();
    this.delta = 0;
  }
}

Highlighter.REGISTRY['typescript'] = HighlightTypescript;
Highlighter.REGISTRY['ts'] = HighlightTypescript;
