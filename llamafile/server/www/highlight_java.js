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

const JAVA_KEYWORDS = new Set([
  'abstract',
  'assert',
  'boolean',
  'break',
  'byte',
  'case',
  'catch',
  'char',
  'class',
  'const',
  'continue',
  'default',
  'do',
  'double',
  'else',
  'enum',
  'extends',
  'final',
  'finally',
  'float',
  'for',
  'goto',
  'if',
  'implements',
  'import',
  'instanceof',
  'int',
  'interface',
  'long',
  'native',
  'new',
  'package',
  'private',
  'protected',
  'public',
  'return',
  'short',
  'static',
  'strictfp',
  'super',
  'switch',
  'synchronized',
  'this',
  'throw',
  'throws',
  'transient',
  'try',
  'void',
  'volatile',
  'while',
]);

const JAVA_CONSTANTS = new Set([
  'true',
  'false',
  'null',
]);

class HighlightJava extends Highlighter {

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
  static DQUOTESTR = 11;
  static DQUOTESTR_BACKSLASH = 12;
  static DQUOTE2 = 13;
  static DQUOTE3 = 14;
  static DQUOTE3_BACKSLASH = 15;
  static DQUOTE31 = 16;
  static DQUOTE32 = 17;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightJava.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightJava.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightJava.SLASH;
        } else if (c == '\'') {
          this.state = HighlightJava.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightJava.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '@') {
          this.state = HighlightJava.ANNOTATION;
        } else {
          this.append(c);
        }
        break;

      case HighlightJava.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (JAVA_KEYWORDS.has(this.word)) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (JAVA_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightJava.NORMAL);
        }
        break;

      case HighlightJava.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightJava.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightJava.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightJava.NORMAL);
        }
        break;

      case HighlightJava.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightJava.NORMAL;
        }
        break;

      case HighlightJava.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightJava.SLASH_STAR_STAR;
        break;

      case HighlightJava.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightJava.NORMAL;
        } else if (c != '*') {
          this.state = HighlightJava.SLASH_STAR;
        }
        break;

      case HighlightJava.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightJava.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJava.QUOTE_BACKSLASH;
        }
        break;

      case HighlightJava.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightJava.QUOTE;
        break;

        // handle "string"
      case HighlightJava.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJava.DQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightJava.DQUOTESTR_BACKSLASH;
        } else {
          this.state = HighlightJava.DQUOTESTR;
        }
        break;

      case HighlightJava.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightJava.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJava.DQUOTESTR_BACKSLASH;
        }
        break;

      case HighlightJava.DQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightJava.DQUOTESTR;
        break;

        // handle """string""" from java 15+
      case HighlightJava.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightJava.DQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightJava.NORMAL);
        }
        break;

      case HighlightJava.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJava.DQUOTE31;
        } else if (c == '\\') {
          this.state = HighlightJava.DQUOTE3_BACKSLASH;
        }
        break;

      case HighlightJava.DQUOTE31:
        this.append(c);
        if (c == '"') {
          this.state = HighlightJava.DQUOTE32;
        } else if (c == '\\') {
          this.state = HighlightJava.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightJava.DQUOTE3;
        }
        break;

      case HighlightJava.DQUOTE32:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightJava.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightJava.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightJava.DQUOTE3;
        }
        break;

      case HighlightJava.DQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightJava.DQUOTE3;
        break;

      case HighlightJava.ANNOTATION:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "attrib");
          this.append('@');
          this.append(c);
          this.state = HighlightJava.ANNOTATION2;
        } else {
          this.append('@');
          this.epsilon(HighlightJava.NORMAL);
        }
        break;

      case HighlightJava.ANNOTATION2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightJava.NORMAL);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightJava.WORD:
      if (JAVA_KEYWORDS.has(this.word)) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (JAVA_CONSTANTS.has(this.word)) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightJava.SLASH:
      this.append('/');
      break;
    case HighlightJava.ANNOTATION:
      this.append('@');
      break;
    case HighlightJava.QUOTE:
    case HighlightJava.QUOTE_BACKSLASH:
    case HighlightJava.SLASH_SLASH:
    case HighlightJava.SLASH_STAR:
    case HighlightJava.SLASH_STAR_STAR:
    case HighlightJava.DQUOTE:
    case HighlightJava.DQUOTESTR:
    case HighlightJava.DQUOTESTR_BACKSLASH:
    case HighlightJava.DQUOTE2:
    case HighlightJava.DQUOTE3:
    case HighlightJava.DQUOTE3_BACKSLASH:
    case HighlightJava.DQUOTE31:
    case HighlightJava.DQUOTE32:
    case HighlightJava.ANNOTATION2:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightJava.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['java'] = HighlightJava;
