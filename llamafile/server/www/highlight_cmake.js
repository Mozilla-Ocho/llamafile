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

const CMAKE_KEYWORDS = new Set([
  'block',
  'else',
  'elseif',
  'endblock',
  'endforeach',
  'endfunction',
  'endif',
  'endmacro',
  'endwhile',
  'foreach',
  'function',
  'if',
  'macro',
  'while',
]);

class HighlightCmake extends Highlighter {

  static NORMAL = 0;
  static BACKSLASH = 1;
  static WORD = 2;
  static WORD_SPACE = 3;
  static DOLLAR = 4;
  static VAR = 5;
  static COMMENT = 6;
  static DQUOTE = 7;
  static DQUOTE_BACKSLASH = 8;
  static DQUOTE_DOLLAR = 9;
  static DQUOTE_VAR = 10;

  constructor(delegate) {
    super(delegate);
    this.spaces = 0;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightCmake.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightCmake.WORD;
          this.word += c;
        } else if (c == '#') {
          this.state = HighlightCmake.COMMENT;
          this.push("span", "comment");
          this.append('#');
        } else if (c == '"') {
          this.state = HighlightCmake.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '$') {
          this.state = HighlightCmake.DOLLAR;
        } else if (c == '\\') {
          this.state = HighlightCmake.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
        } else {
          this.append(c);
        }
        break;

      case HighlightCmake.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightCmake.NORMAL;
        break;

      case HighlightCmake.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          this.spaces = 0;
          this.epsilon(HighlightCmake.WORD_SPACE);
        }
        break;

      case HighlightCmake.WORD_SPACE:
        if (c == ' ') {
          ++this.spaces;
        } else if (c == '(') {
          if (CMAKE_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          }
          this.word = '';
          for (let i = 0; i < this.spaces; ++i)
            this.append(' ');
          this.epsilon(HighlightCmake.NORMAL);
        } else {
          this.append(this.word);
          this.word = '';
          for (let i = 0; i < this.spaces; ++i)
            this.append(' ');
          this.epsilon(HighlightCmake.NORMAL);
        }
        break;

      case HighlightCmake.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightCmake.NORMAL;
        }
        break;

      case HighlightCmake.DOLLAR:
        if (c == '{') {
          this.state = HighlightCmake.VAR;
        } else if (c == '$') {
          this.append('$');
        } else {
          this.append('$');
          this.epsilon(HighlightCmake.NORMAL);
        }
        break;

      case HighlightCmake.VAR:
        if (isalnum(c) || c == '_') {
          this.word += c;
        } else if (c == '}') {
          this.append("${");
          this.push("span", "var");
          this.append(this.word);
          this.pop();
          this.append('}');
          this.word = '';
          this.state = HighlightCmake.NORMAL;
        } else {
          this.append("${");
          this.append(this.word);
          this.word = '';
          this.epsilon(HighlightCmake.NORMAL);
        }
        break;

      case HighlightCmake.DQUOTE:
        if (c == '"') {
          this.append('"');
          this.pop();
          this.state = HighlightCmake.NORMAL;
        } else if (c == '\\') {
          this.append('\\');
          this.state = HighlightCmake.DQUOTE_BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightCmake.DQUOTE_DOLLAR;
        } else {
          this.append(c);
        }
        break;

      case HighlightCmake.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightCmake.DQUOTE;
        break;

      case HighlightCmake.DQUOTE_DOLLAR:
        if (c == '{') {
          this.state = HighlightCmake.DQUOTE_VAR;
        } else if (c == '$') {
          this.append('$');
        } else {
          this.append('$');
          this.epsilon(HighlightCmake.DQUOTE);
        }
        break;

      case HighlightCmake.DQUOTE_VAR:
        if (isalnum(c) || c == '_') {
          this.word += c;
        } else if (c == '}') {
          this.append("${");
          this.push("span", "var");
          this.append(this.word);
          this.pop();
          this.append('}');
          this.word = '';
          this.state = HighlightCmake.DQUOTE;
        } else {
          this.append("${");
          this.append(this.word);
          this.word = '';
          this.epsilon(HighlightCmake.DQUOTE);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightCmake.WORD:
      this.append(this.word);
      this.word = '';
      break;
    case HighlightCmake.WORD_SPACE:
      this.append(this.word);
      this.word = '';
      for (let i = 0; i < this.spaces; ++i)
        this.append(' ');
      break;
    case HighlightCmake.DOLLAR:
      this.append('$');
      break;
    case HighlightCmake.DQUOTE_DOLLAR:
      this.append('$');
      this.pop();
      break;
    case HighlightCmake.DQUOTE_VAR:
      this.append("${");
      this.append(this.word);
      this.pop();
      this.word = '';
      break;
    case HighlightCmake.VAR:
      this.append("${");
      this.append(this.word);
      this.word = '';
      break;
    case HighlightCmake.DQUOTE:
    case HighlightCmake.COMMENT:
    case HighlightCmake.BACKSLASH:
    case HighlightCmake.DQUOTE_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightCmake.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['cmake'] = HighlightCmake;
