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

const HASKELL_KEYWORDS = new Set([
  'as',
  'case',
  'class',
  'data',
  'default',
  'deriving',
  'do',
  'else',
  'family',
  'forall',
  'foreign',
  'hiding',
  'if',
  'import',
  'in',
  'infix',
  'infixl',
  'infixr',
  'instance',
  'let',
  'mdo',
  'module',
  'newtype',
  'of',
  'proc',
  'qualified',
  'rec',
  'then',
  'type',
  'where',
]);

class HighlightHaskell extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static DQUOTE = 2;
  static DQUOTE_BACKSLASH = 3;
  static TICK = 4;
  static TICK_BACKSLASH = 5;
  static CURL = 6;
  static CURL_HYPHEN = 7;
  static CURL_HYPHEN_HYPHEN = 8;
  static CURL_HYPHEN_CURL = 9;
  static HYPHEN = 10;
  static HYPHEN_HYPHEN = 11;
  static HYPHEN_LT = 12;
  static EQUAL = 13;
  static COLON = 14;
  static LT = 15;

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.level = 0;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightHaskell.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.epsilon(HighlightHaskell.WORD);
        } else if (c == '-') {
          this.state = HighlightHaskell.HYPHEN;
        } else if (c == '{') {
          this.state = HighlightHaskell.CURL;
        } else if (c == '"') {
          this.state = HighlightHaskell.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '`') {
          this.state = HighlightHaskell.TICK;
          this.push("span", "operator");
          this.append(c);
        } else if (c == '!' || //
                   c == '#' || //
                   c == '$' || //
                   c == '*' || //
                   c == ',' || //
                   c == '>' || //
                   c == '?' || //
                   c == '@' || //
                   c == '|' || //
                   c == '~') {
          this.push("span", "operator");
          this.append(c);
          this.pop();
        } else if (c == '=') {
          this.state = HighlightHaskell.EQUAL;
        } else if (c == ':') {
          this.state = HighlightHaskell.COLON;
        } else if (c == '<') {
          this.state = HighlightHaskell.LT;
        } else {
          this.append(c);
        }
        break;

      case HighlightHaskell.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (HASKELL_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.LT:
        if (c == '-') {
          this.push("span", "operator");
          this.append("<-");
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else {
          this.append('<');
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.COLON:
        if (c == ':') {
          this.push("span", "operator");
          this.append("::");
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else {
          this.append(':');
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.EQUAL:
        if (c == '>') {
          this.push("span", "operator");
          this.append("=>");
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else {
          this.push("span", "operator");
          this.append('=');
          this.pop();
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.HYPHEN:
        if (c == '-') {
          this.push("span", "comment");
          this.append("--");
          this.state = HighlightHaskell.HYPHEN_HYPHEN;
        } else if (c == '<') {
          this.state = HighlightHaskell.HYPHEN_LT;
        } else if (c == '>') {
          this.push("span", "operator");
          this.append("->");
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else {
          this.append('-');
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.HYPHEN_LT:
        if (c == '<') {
          this.push("span", "operator");
          this.append("-<<");
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else {
          this.push("span", "operator");
          this.append("-<");
          this.pop();
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.HYPHEN_HYPHEN:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        }
        break;

      case HighlightHaskell.CURL:
        if (c == '-') {
          this.push("span", "comment");
          this.append("{-");
          this.state = HighlightHaskell.CURL_HYPHEN;
          this.level = 1;
        } else {
          this.append('{');
          this.epsilon(HighlightHaskell.NORMAL);
        }
        break;

      case HighlightHaskell.CURL_HYPHEN:
        this.append(c);
        if (c == '-') {
          this.state = HighlightHaskell.CURL_HYPHEN_HYPHEN;
        } else if (c == '{') {
          this.state = HighlightHaskell.CURL_HYPHEN_CURL;
        }
        break;

      case HighlightHaskell.CURL_HYPHEN_CURL:
        this.append(c);
        if (c == '-') {
          this.state = HighlightHaskell.CURL_HYPHEN;
          ++this.level;
        } else if (c != '{') {
          this.state = HighlightHaskell.CURL_HYPHEN;
        }
        break;

      case HighlightHaskell.CURL_HYPHEN_HYPHEN:
        this.append(c);
        if (c == '}') {
          if (!--this.level) {
            this.pop();
            this.state = HighlightHaskell.NORMAL;
          }
        } else if (c != '-') {
          this.state = HighlightHaskell.CURL_HYPHEN;
        }
        break;

      case HighlightHaskell.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightHaskell.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightHaskell.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightHaskell.DQUOTE;
        break;

      case HighlightHaskell.TICK:
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightHaskell.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightHaskell.TICK_BACKSLASH;
        }
        break;

      case HighlightHaskell.TICK_BACKSLASH:
        this.append(c);
        this.state = HighlightHaskell.TICK;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightHaskell.WORD:
      if (HASKELL_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightHaskell.CURL:
      this.append('{');
      break;
    case HighlightHaskell.HYPHEN:
      this.append('-');
      break;
    case HighlightHaskell.EQUAL:
      this.append('=');
      break;
    case HighlightHaskell.COLON:
      this.append(':');
      break;
    case HighlightHaskell.LT:
      this.append('<');
      break;
    case HighlightHaskell.HYPHEN_LT:
      this.push("span", "operator");
      this.append("-<");
      this.pop();
      break;
    case HighlightHaskell.TICK:
    case HighlightHaskell.TICK_BACKSLASH:
    case HighlightHaskell.DQUOTE:
    case HighlightHaskell.DQUOTE_BACKSLASH:
    case HighlightHaskell.HYPHEN_HYPHEN:
    case HighlightHaskell.CURL_HYPHEN:
    case HighlightHaskell.CURL_HYPHEN_HYPHEN:
    case HighlightHaskell.CURL_HYPHEN_CURL:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightHaskell.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['haskell'] = HighlightHaskell;
Highlighter.REGISTRY['hs'] = HighlightHaskell;
