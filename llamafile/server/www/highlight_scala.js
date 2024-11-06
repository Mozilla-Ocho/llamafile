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

const SCALA_KEYWORDS = new Set([
  'abstract',
  'case',
  'catch',
  'class',
  'def',
  'do',
  'else',
  'extends',
  'final',
  'finally',
  'for',
  'forSome',
  'if',
  'implicit',
  'import',
  'lazy',
  'macro',
  'match',
  'new',
  'object',
  'override',
  'package',
  'private',
  'protected',
  'return',
  'sealed',
  'super',
  'this',
  'throw',
  'trait',
  'try',
  'type',
  'val',
  'var',
  'while',
  'with',
  'yield',
]);

class HighlightScala extends Highlighter {

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
    this.word = '';
    this.nest = [];
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightScala.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightScala.WORD;
          this.word += c;
        } else if (c == '/') {
          this.state = HighlightScala.SLASH;
        } else if (c == '\'') {
          this.state = HighlightScala.QUOTE;
          this.push("span", "string");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightScala.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '@') {
          this.state = HighlightScala.ANNOTATION;
        } else if (c == '{' && this.nest.length) {
          this.nest.push(HighlightScala.NORMAL);
          this.append('{');
        } else if (c == '}' && this.nest.length) {
          if ((this.state = this.nest.pop()) != HighlightScala.NORMAL)
            this.push("span", "string");
          this.append('}');
        } else {
          this.append(c);
        }
        break;

      case HighlightScala.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (SCALA_KEYWORDS.has(this.word)) {
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
          this.epsilon(HighlightScala.NORMAL);
        }
        break;

      case HighlightScala.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightScala.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightScala.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightScala.NORMAL);
        }
        break;

      case HighlightScala.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightScala.NORMAL;
        }
        break;

      case HighlightScala.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightScala.SLASH_STAR_STAR;
        break;

      case HighlightScala.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightScala.NORMAL;
        } else if (c != '*') {
          this.state = HighlightScala.SLASH_STAR;
        }
        break;

      case HighlightScala.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightScala.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightScala.QUOTE_BACKSLASH;
        }
        break;

      case HighlightScala.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightScala.QUOTE;
        break;

      case HighlightScala.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.state = HighlightScala.DQUOTE2;
        } else if (c == '\\') {
          this.state = HighlightScala.DQUOTESTR_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightScala.DQUOTESTR_DOLLAR;
        } else {
          this.state = HighlightScala.DQUOTESTR;
        }
        break;

      case HighlightScala.DQUOTESTR:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightScala.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightScala.DQUOTESTR_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightScala.DQUOTESTR_DOLLAR;
        }
        break;

      case HighlightScala.DQUOTESTR_BACKSLASH:
        this.append(c);
        this.state = HighlightScala.DQUOTESTR;
        break;

      case HighlightScala.DQUOTESTR_DOLLAR:
        if (c == '{') {
          this.append(c);
          this.pop();
          this.nest.push(HighlightScala.DQUOTESTR);
          this.state = HighlightScala.NORMAL;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "bold");
          this.append(c);
          this.state = HighlightScala.DQUOTESTR_VAR;
        } else {
          this.epsilon(HighlightScala.DQUOTESTR);
        }
        break;

      case HighlightScala.DQUOTESTR_VAR:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightScala.DQUOTESTR);
        }
        break;

      case HighlightScala.DQUOTE2:
        if (c == '"') {
          this.append('"');
          this.state = HighlightScala.DQUOTE3;
        } else {
          this.pop();
          this.epsilon(HighlightScala.NORMAL);
        }
        break;

      case HighlightScala.DQUOTE3:
        this.append(c);
        if (c == '"') {
          this.state = HighlightScala.DQUOTE31;
        } else if (c == '$') {
          this.state = HighlightScala.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightScala.DQUOTE3_BACKSLASH;
        }
        break;

      case HighlightScala.DQUOTE31:
        this.append(c);
        if (c == '"') {
          this.state = HighlightScala.DQUOTE32;
        } else if (c == '$') {
          this.state = HighlightScala.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightScala.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightScala.DQUOTE3;
        }
        break;

      case HighlightScala.DQUOTE32:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightScala.NORMAL;
        } else if (c == '$') {
          this.state = HighlightScala.DQUOTE3_DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightScala.DQUOTE3_BACKSLASH;
        } else {
          this.state = HighlightScala.DQUOTE3;
        }
        break;

      case HighlightScala.DQUOTE3_BACKSLASH:
        this.append(c);
        this.state = HighlightScala.DQUOTE3;
        break;

      case HighlightScala.DQUOTE3_DOLLAR:
        if (c == '{') {
          this.append(c);
          this.pop();
          this.nest.push(HighlightScala.DQUOTE3);
          this.state = HighlightScala.NORMAL;
        } else if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "bold");
          this.append(c);
          this.state = HighlightScala.DQUOTE3_VAR;
        } else {
          this.append(c);
          this.state = HighlightScala.DQUOTE3;
        }
        break;

      case HighlightScala.DQUOTE3_VAR:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightScala.DQUOTE3);
        }
        break;

      case HighlightScala.ANNOTATION:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.push("span", "attrib");
          this.append('@');
          this.append(c);
          this.state = HighlightScala.ANNOTATION2;
        } else {
          this.append('@');
          this.epsilon(HighlightScala.NORMAL);
        }
        break;

      case HighlightScala.ANNOTATION2:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightScala.NORMAL);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightScala.WORD:
      if (SCALA_KEYWORDS.has(this.word)) {
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
      break;
    case HighlightScala.SLASH:
      this.append('/');
      break;
    case HighlightScala.ANNOTATION:
      this.append('@');
      break;
    case HighlightScala.QUOTE:
    case HighlightScala.QUOTE_BACKSLASH:
    case HighlightScala.SLASH_SLASH:
    case HighlightScala.SLASH_STAR:
    case HighlightScala.SLASH_STAR_STAR:
    case HighlightScala.DQUOTE:
    case HighlightScala.DQUOTESTR:
    case HighlightScala.DQUOTESTR_BACKSLASH:
    case HighlightScala.DQUOTESTR_DOLLAR:
    case HighlightScala.DQUOTESTR_VAR:
    case HighlightScala.DQUOTE2:
    case HighlightScala.DQUOTE3:
    case HighlightScala.DQUOTE3_BACKSLASH:
    case HighlightScala.DQUOTE31:
    case HighlightScala.DQUOTE32:
    case HighlightScala.ANNOTATION2:
    case HighlightScala.DQUOTE3_DOLLAR:
    case HighlightScala.DQUOTE3_VAR:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightScala.NORMAL;
    this.nest = [];
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['scala'] = HighlightScala;
Highlighter.REGISTRY['sbt'] = HighlightScala;
Highlighter.REGISTRY['sc'] = HighlightScala;
