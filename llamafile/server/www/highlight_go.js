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

const GO_KEYWORDS = new Set([
  'break',
  'case',
  'chan',
  'const',
  'continue',
  'default',
  'defer',
  'else',
  'fallthrough',
  'for',
  'func',
  'go',
  'goto',
  'if',
  'import',
  'interface',
  'map',
  'package',
  'range',
  'return',
  'select',
  'struct',
  'switch',
  'type',
  'var',
]);

const GO_TYPES = new Set([
  'byte',
  'complex128',
  'complex64',
  'float32',
  'float64',
  'int',
  'int16',
  'int32',
  'int64',
  'int8',
  'rune',
  'string',
  'uint',
  'uint16',
  'uint32',
  'uint64',
  'uint8',
  'uintptr',
]);

class HighlightGo extends Highlighter {

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

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightGo.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightGo.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightGo.SLASH;
        } else if (c == '\'') {
          this.state = HighlightGo.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightGo.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '`') {
          this.state = HighlightGo.TICK;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightGo.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (GO_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (GO_TYPES.has(this.word)) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightGo.NORMAL);
        }
        break;

      case HighlightGo.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightGo.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightGo.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightGo.NORMAL);
        }
        break;

      case HighlightGo.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightGo.NORMAL;
        }
        break;

      case HighlightGo.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightGo.SLASH_STAR_STAR;
        break;

      case HighlightGo.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightGo.NORMAL;
        } else if (c != '*') {
          this.state = HighlightGo.SLASH_STAR;
        }
        break;

      case HighlightGo.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightGo.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightGo.QUOTE_BACKSLASH;
        }
        break;

      case HighlightGo.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightGo.QUOTE;
        break;

      case HighlightGo.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightGo.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightGo.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightGo.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightGo.DQUOTE;
        break;

      case HighlightGo.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightGo.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightGo.TICK_BACKSLASH;
        }
        break;

      case HighlightGo.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightGo.TICK;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightGo.WORD:
      if (GO_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (GO_TYPES.has(this.word)) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightGo.SLASH:
      this.append('/');
      break;
    case HighlightGo.QUOTE:
    case HighlightGo.QUOTE_BACKSLASH:
    case HighlightGo.DQUOTE:
    case HighlightGo.DQUOTE_BACKSLASH:
    case HighlightGo.SLASH_SLASH:
    case HighlightGo.SLASH_STAR:
    case HighlightGo.SLASH_STAR_STAR:
    case HighlightGo.TICK:
    case HighlightGo.TICK_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightGo.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['go'] = HighlightGo;
