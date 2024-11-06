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

const RUST_KEYWORDS = new Set([
  'Self',
  'abstract',
  'as',
  'async',
  'await',
  'become',
  'box',
  'break',
  'const',
  'continue',
  'crate',
  'do',
  'dyn',
  'else',
  'enum',
  'extern',
  'final',
  'fn',
  'for',
  'if',
  'impl',
  'in',
  'let',
  'loop',
  'macro',
  'match',
  'mod',
  'move',
  'mut',
  'override',
  'priv',
  'pub',
  'ref',
  'return',
  'self',
  'static',
  'struct',
  'super',
  'trait',
  'try',
  'type',
  'typeof',
  'union',
  'unsafe',
  'unsized',
  'use',
  'virtual',
  'where',
  'while',
  'yield',
]);

const RUST_CONSTANTS = new Set([
  'false',
  'true',
]);

const RUST_TYPES = new Set([
  '!',
  'bool',
  'char',
  'f32',
  'f64',
  'i128',
  'i16',
  'i32',
  'i64',
  'i8',
  'isize',
  'str',
  'u128',
  'u16',
  'u32',
  'u64',
  'u8',
  'usize',
]);

class HighlightRust extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static QUOTE2 = 4;
  static DQUOTE = 5;
  static DQUOTE_BACKSLASH = 6;
  static SLASH = 7;
  static SLASH_SLASH = 8;
  static SLASH_STAR = 9;
  static SLASH_STAR_STAR = 10;
  static HASH = 11;
  static HASH_EXCLAIM = 12;
  static ATTRIB = 13;

  constructor(delegate) {
    super(delegate);
    this.nest = 0;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightRust.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '!') {
          this.epsilon(HighlightRust.WORD);
        } else if (c == '/') {
          this.state = HighlightRust.SLASH;
        } else if (c == '#') {
          this.state = HighlightRust.HASH;
        } else if (c == '\'') {
          this.state = HighlightRust.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightRust.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightRust.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '!') {
          this.word += c;
        } else {
          if (RUST_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (this.word.length >= 2 && this.word[this.word.length - 1] == '!') {
            this.push("span", "macro");
            this.append(this.word);
            this.pop();
          } else if (RUST_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (RUST_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightRust.NORMAL);
        }
        break;

      case HighlightRust.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightRust.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightRust.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightRust.NORMAL);
        }
        break;

      case HighlightRust.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightRust.NORMAL;
        }
        break;

      case HighlightRust.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightRust.SLASH_STAR_STAR;
        break;

      case HighlightRust.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightRust.NORMAL;
        } else if (c != '*') {
          this.state = HighlightRust.SLASH_STAR;
        }
        break;

      case HighlightRust.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightRust.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRust.QUOTE_BACKSLASH;
        } else {
          this.state = HighlightRust.QUOTE2;
        }
        break;

      case HighlightRust.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightRust.QUOTE2;
        break;

      case HighlightRust.QUOTE2:
        if (c == '\'') {
          this.append(c);
          this.pop();
          this.state = HighlightRust.NORMAL;
        } else {
          this.pop();
          this.append(c);
          this.state = HighlightRust.NORMAL;
        }
        break;

      case HighlightRust.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightRust.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightRust.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightRust.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightRust.DQUOTE;
        break;

      case HighlightRust.HASH:
        if (c == '!') {
          this.state = HighlightRust.HASH_EXCLAIM;
        } else if (c == '[') {
          this.push("span", "attrib");
          this.append("#[");
          this.state = HighlightRust.ATTRIB;
        } else {
          this.append('#');
          this.epsilon(HighlightRust.NORMAL);
        }
        break;

      case HighlightRust.HASH_EXCLAIM:
        if (c == '[') {
          this.push("span", "attrib");
          this.append("#![");
          this.state = HighlightRust.ATTRIB;
        } else {
          this.append("#!");
          this.epsilon(HighlightRust.NORMAL);
        }
        break;

      case HighlightRust.ATTRIB:
        this.append(c);
        if (c == '[') {
          ++this.nest;
        } else if (c == ']') {
          if (this.nest) {
            --this.nest;
          } else {
            this.pop();
            this.state = HighlightRust.NORMAL;
          }
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightRust.WORD:
      if (RUST_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (this.word.length >= 2 && this.word[this.word.length - 1] == '!') {
        this.push("span", "macro");
        this.append(this.word);
        this.pop();
      } else if (RUST_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (RUST_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightRust.SLASH:
      this.append('/');
      break;
    case HighlightRust.QUOTE:
    case HighlightRust.QUOTE_BACKSLASH:
    case HighlightRust.QUOTE2:
    case HighlightRust.DQUOTE:
    case HighlightRust.DQUOTE_BACKSLASH:
    case HighlightRust.ATTRIB:
    case HighlightRust.SLASH_SLASH:
    case HighlightRust.SLASH_STAR:
    case HighlightRust.SLASH_STAR_STAR:
      this.pop();
      break;
    case HighlightRust.HASH:
      this.append('#');
      break;
    case HighlightRust.HASH_EXCLAIM:
      this.append("#!");
      break;
    default:
      break;
    }
    this.nest = 0;
    this.state = HighlightRust.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['rust'] = HighlightRust;
Highlighter.REGISTRY['rs'] = HighlightRust;
