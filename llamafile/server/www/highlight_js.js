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

const JS_KEYWORDS = new Set([
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
  'if',
  'implements',
  'import',
  'in',
  'instanceof',
  'interface',
  'let',
  'native',
  'new',
  'of',
  'package',
  'private',
  'protected',
  'public',
  'return',
  'static',
  'super',
  'switch',
  'synchronized',
  'this',
  'throw',
  'throws',
  'transient',
  'try',
  'typeof',
  'var',
  'void',
  'volatile',
  'while',
  'with',
  'yield',
]);

const JS_BUILTINS = new Set([
  'AggregateError',
  'Array',
  'ArrayBuffer',
  'AsyncFunction',
  'AsyncGenerator',
  'AsyncGeneratorFunction',
  'AsyncIterator',
  'Atomics',
  'BigInt',
  'BigInt64Array',
  'BigUint64Array',
  'Boolean',
  'DataView',
  'Date',
  'Error',
  'EvalError',
  'FinalizationRegistry',
  'Float16Array',
  'Float32Array',
  'Float64Array',
  'Function',
  'Generator',
  'GeneratorFunction',
  'Int16Array',
  'Int32Array',
  'Int8Array',
  'InternalError',
  'Intl',
  'Iterator',
  'JSON',
  'Map',
  'Math',
  'Number',
  'Object',
  'Promise',
  'Proxy',
  'RangeError',
  'ReferenceError',
  'Reflect',
  'RegExp',
  'Set',
  'SharedArrayBuffer',
  'String',
  'Symbol',
  'SyntaxError',
  'TypeError',
  'URIError',
  'Uint16Array',
  'Uint32Array',
  'Uint8Array',
  'Uint8ClampedArray',
  'WeakMap',
  'WeakRef',
  'WeakSet',
  'console',
  'decodeURI',
  'decodeURIComponent',
  'encodeURI',
  'encodeURIComponent',
  'escape',
  'eval',
  'isFinite',
  'isNaN',
  'parseFloat',
  'parseInt',
  'unescape',
]);

const JS_CONSTANTS = new Set([
  'Infinity',
  'NaN',
  'arguments',
  'false',
  'globalThis',
  'null',
  'true',
  'undefined',
]);

class HighlightJs extends Highlighter {

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

  // https://262.ecma-international.org/12.0/#sec-line-terminators
  static is_line_terminator(c) {
    switch (c) {
    case '\r':
    case '\n':
    case '\u2028': // LINE SEPARATOR
    case '\u2029': // PARAGRAPH SEPARATOR
      return true;
    default:
      return false;
    }
  }

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.expect = HighlightJs.EXPECT_VALUE;
    this.nest = [];
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      if (c == '\r')
        continue;
      if (c == 0xFEFF)
        continue; // utf-8 bom
      switch (this.state) {

      case HighlightJs.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightJs.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightJs.SLASH;
        } else if (c == '\'') {
          this.state = HighlightJs.QUOTE;
          this.push("span", "string");
          this.append('\'');
          this.expect = HighlightJs.EXPECT_OPERATOR;
        } else if (c == '"') {
          this.state = HighlightJs.DQUOTE;
          this.push("span", "string");
          this.append('"');
          this.expect = HighlightJs.EXPECT_OPERATOR;
        } else if (c == '`') {
          this.state = HighlightJs.TICK;
          this.push("span", "string");
          this.append('`');
          this.expect = HighlightJs.EXPECT_OPERATOR;
        } else if (c == '{' && this.nest.length) {
          this.expect = HighlightJs.EXPECT_VALUE;
          this.append('{');
          this.nest.push(HighlightJs.NORMAL);
        } else if (c == '}' && this.nest.length) {
          if ((this.state = this.nest.pop()) != HighlightJs.NORMAL)
            this.push("span", "string");
          this.append('}');
        } else if (c == ')' || c == '}' || c == ']') {
          this.expect = HighlightJs.EXPECT_OPERATOR;
          this.append(c);
        } else if (isdigit(c) || c == '.') {
          this.expect = HighlightJs.EXPECT_OPERATOR;
          this.append(c);
        } else if (ispunct(c)) {
          this.expect = HighlightJs.EXPECT_VALUE;
          this.append(c);
        } else if (isdigit(c)) {
          this.expect = HighlightJs.EXPECT_OPERATOR;
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightJs.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (JS_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
            this.expect = HighlightJs.EXPECT_VALUE;
          } else if (JS_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
            this.expect = HighlightJs.EXPECT_OPERATOR;
          } else if (JS_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
            this.expect = HighlightJs.EXPECT_OPERATOR;
          } else {
            this.append(this.word);
            this.expect = HighlightJs.EXPECT_OPERATOR;
          }
          this.word = '';
          this.epsilon(HighlightJs.NORMAL);
        }
        break;

      case HighlightJs.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightJs.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightJs.SLASH_STAR;
        } else if (this.expect == HighlightJs.EXPECT_VALUE) {
          this.expect = HighlightJs.EXPECT_OPERATOR;
          this.push("span", "string");
          this.append('/');
          this.append(c);
          if (c == '\\') {
            this.state = HighlightJs.REGEX_BACKSLASH;
          } else if (c == '[') {
            this.state = HighlightJs.REGEX_SQUARE;
          } else {
            this.state = HighlightJs.REGEX;
          }
        } else {
          this.append('/');
          this.epsilon(HighlightJs.NORMAL);
        }
        break;

      case HighlightJs.SLASH_SLASH:
        this.append(c);
        if (HighlightJs.is_line_terminator(c)) {
          this.pop();
          this.state = HighlightJs.NORMAL;
        }
        break;

      case HighlightJs.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightJs.SLASH_STAR_STAR;
        break;

      case HighlightJs.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightJs.NORMAL;
        } else if (c != '*') {
          this.state = HighlightJs.SLASH_STAR;
        }
        break;

      case HighlightJs.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightJs.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJs.QUOTE_BACKSLASH;
        }
        break;

      case HighlightJs.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightJs.QUOTE;
        break;

      case HighlightJs.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightJs.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJs.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightJs.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightJs.DQUOTE;
        break;

      case HighlightJs.TICK:
        if (c == '`') {
          this.append('`');
          this.pop();
          this.state = HighlightJs.NORMAL;
        } else if (c == '$') {
          this.state = HighlightJs.TICK_DOLLAR;
        } else if (c == '\\') {
          this.append('\\');
          this.state = HighlightJs.TICK_BACKSLASH;
        } else {
          this.append(c);
        }
        break;

      case HighlightJs.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightJs.TICK;
        break;

      case HighlightJs.TICK_DOLLAR:
        if (c == '{') {
          this.push("span", "bold");
          this.append('$');
          this.pop();
          this.append('{');
          this.pop();
          this.expect = HighlightJs.EXPECT_VALUE;
          this.nest.push(HighlightJs.TICK);
          this.state = HighlightJs.NORMAL;
        } else {
          this.push("span", "warning");
          this.append('$');
          this.pop();
          this.epsilon(HighlightJs.TICK);
        }
        break;

      case HighlightJs.REGEX:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightJs.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJs.REGEX_BACKSLASH;
        } else if (c == '[') {
          this.state = HighlightJs.REGEX_SQUARE;
        }
        break;

      case HighlightJs.REGEX_BACKSLASH:
        this.append(c);
        this.state = HighlightJs.REGEX;
        break;

      case HighlightJs.REGEX_SQUARE:
        // because /[/]/g is valid code
        this.append(c);
        if (c == '\\') {
          this.state = HighlightJs.REGEX_SQUARE_BACKSLASH;
        } else if (c == ']') {
          this.state = HighlightJs.REGEX;
        }
        break;

      case HighlightJs.REGEX_SQUARE_BACKSLASH:
        this.append(c);
        this.state = HighlightJs.REGEX_SQUARE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightJs.WORD:
      if (JS_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
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
    case HighlightJs.SLASH:
      this.append('/');
      break;
    case HighlightJs.TICK_DOLLAR:
      this.append('$');
      this.pop();
      break;
    case HighlightJs.TICK:
    case HighlightJs.TICK_BACKSLASH:
    case HighlightJs.QUOTE:
    case HighlightJs.QUOTE_BACKSLASH:
    case HighlightJs.DQUOTE:
    case HighlightJs.DQUOTE_BACKSLASH:
    case HighlightJs.SLASH_SLASH:
    case HighlightJs.SLASH_STAR:
    case HighlightJs.SLASH_STAR_STAR:
    case HighlightJs.REGEX:
    case HighlightJs.REGEX_BACKSLASH:
    case HighlightJs.REGEX_SQUARE:
    case HighlightJs.REGEX_SQUARE_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.nest = [];
    this.state = HighlightJs.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['javascript'] = HighlightJs;
Highlighter.REGISTRY['json'] = HighlightJs;
Highlighter.REGISTRY['js'] = HighlightJs;
