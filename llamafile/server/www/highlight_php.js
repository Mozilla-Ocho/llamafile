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

const PHP_KEYWORDS = new Set([
  '__class__',
  '__dir__',
  '__file__',
  '__function__',
  '__line__',
  '__method__',
  '__namespace__',
  '__trait__',
  '__halt_compiler',
  'abstract',
  'and',
  'array',
  'as',
  'break',
  'callable',
  'case',
  'catch',
  'class',
  'clone',
  'const',
  'continue',
  'declare',
  'default',
  'die',
  'do',
  'echo',
  'else',
  'elseif',
  'empty',
  'enddeclare',
  'endfor',
  'endforeach',
  'endif',
  'endswitch',
  'endwhile',
  'eval',
  'exit',
  'extends',
  'final',
  'finally',
  'fn',
  'for',
  'foreach',
  'function',
  'global',
  'goto',
  'if',
  'implements',
  'include',
  'include_once',
  'instanceof',
  'insteadof',
  'interface',
  'isset',
  'list',
  'match',
  'namespace',
  'new',
  'or',
  'print',
  'private',
  'protected',
  'public',
  'readonly',
  'require',
  'require_once',
  'return',
  'static',
  'switch',
  'throw',
  'trait',
  'try',
  'unset',
  'use',
  'var',
  'while',
  'xor',
  'yield',
]);

const PHP_CONSTANTS = new Set([
  'true',
  'false',
]);

class HighlightPhp extends Highlighter {

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
  static VAR = 12;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightPhp.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightPhp.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightPhp.SLASH;
        } else if (c == '\'') {
          this.state = HighlightPhp.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightPhp.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '`') {
          this.state = HighlightPhp.TICK;
          this.push("span", "string");
          this.append(c);
        } else if (c == '$') {
          this.append(c);
          this.state = HighlightPhp.VAR;
          this.push("span", "var");
        } else {
          this.append(c);
        }
        break;

      case HighlightPhp.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (PHP_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (PHP_CONSTANTS.has(this.word.toLowerCase())) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightPhp.NORMAL);
        }
        break;

      case HighlightPhp.VAR:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightPhp.NORMAL);
        }
        break;

      case HighlightPhp.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightPhp.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightPhp.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightPhp.NORMAL);
        }
        break;

      case HighlightPhp.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightPhp.NORMAL;
        }
        break;

      case HighlightPhp.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightPhp.SLASH_STAR_STAR;
        break;

      case HighlightPhp.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightPhp.NORMAL;
        } else if (c != '*') {
          this.state = HighlightPhp.SLASH_STAR;
        }
        break;

      case HighlightPhp.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightPhp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPhp.QUOTE_BACKSLASH;
        }
        break;

      case HighlightPhp.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightPhp.QUOTE;
        break;

      case HighlightPhp.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightPhp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPhp.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightPhp.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightPhp.DQUOTE;
        break;

      case HighlightPhp.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightPhp.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightPhp.TICK_BACKSLASH;
        }
        break;

      case HighlightPhp.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightPhp.TICK;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightPhp.WORD:
      if (PHP_KEYWORDS.has(this.word.toLowerCase())) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (PHP_CONSTANTS.has(this.word.toLowerCase())) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightPhp.SLASH:
      this.append('/');
      break;
    case HighlightPhp.VAR:
    case HighlightPhp.TICK:
    case HighlightPhp.TICK_BACKSLASH:
    case HighlightPhp.QUOTE:
    case HighlightPhp.QUOTE_BACKSLASH:
    case HighlightPhp.DQUOTE:
    case HighlightPhp.DQUOTE_BACKSLASH:
    case HighlightPhp.SLASH_SLASH:
    case HighlightPhp.SLASH_STAR:
    case HighlightPhp.SLASH_STAR_STAR:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightPhp.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}
