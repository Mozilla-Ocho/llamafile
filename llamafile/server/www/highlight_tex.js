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

class HighlightTex extends Highlighter {

  static NORMAL = 0;
  static BACKSLASH = 1;
  static COMMAND = 2;
  static COMMENT = 3;
  static DOLLAR = 4;
  static MATH = 5;
  static MATH_BACKSLASH = 6;
  static BACKTICK = 7;
  static STRING = 8;
  static STRING_QUOTE = 9;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightTex.NORMAL:
        if (c == '\\') {
          this.state = HighlightTex.BACKSLASH;
        } else if (c == '$') {
          this.state = HighlightTex.DOLLAR;
        } else if (c == '`') {
          this.state = HighlightTex.BACKTICK;
        } else if (c == '%') {
          this.state = HighlightTex.COMMENT;
          this.push("span", "comment");
          this.append('%');
        } else {
          this.append(c);
        }
        break;

      case HighlightTex.BACKSLASH:
        if (c == '\\') {
          this.push("span", "warning");
          this.append("\\\\");
          this.pop();
          this.state = HighlightTex.NORMAL;
        } else if (isspace(c)) {
          this.append('\\');
          this.epsilon(HighlightTex.NORMAL);
        } else if (isalpha(c) || c == '@') {
          this.push("span", "keyword");
          this.append('\\');
          this.append(c);
          this.state = HighlightTex.COMMAND;
        } else {
          this.push("span", "escape");
          this.append('\\');
          this.append(c);
          this.pop();
          this.state = HighlightTex.NORMAL;
        }
        break;

      case HighlightTex.COMMAND:
        if (isalpha(c) || c == '@') {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightTex.NORMAL);
        }
        break;

      case HighlightTex.DOLLAR:
        if (c == '$') {
          this.append("$$");
          this.state = HighlightTex.NORMAL;
        } else if (c == '\\') {
          this.push("span", "math");
          this.append("$\\");
          this.state = HighlightTex.MATH_BACKSLASH;
        } else {
          this.push("span", "math");
          this.append("$");
          this.append(c);
          this.state = HighlightTex.MATH;
        }
        break;

      case HighlightTex.MATH:
        if (c == '$') {
          this.append("$");
          this.pop();
          this.state = HighlightTex.NORMAL;
        } else if (c == '\\') {
          this.append('\\');
          this.state = HighlightTex.MATH_BACKSLASH;
        } else {
          this.append(c);
        }
        break;

      case HighlightTex.MATH_BACKSLASH:
        this.append(c);
        this.state = HighlightTex.MATH;
        break;

      case HighlightTex.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightTex.NORMAL;
        }
        break;

      case HighlightTex.BACKTICK:
        if (c == '`') {
          this.push("span", "string");
          this.append("``");
          this.state = HighlightTex.STRING;
        } else {
          this.append('`');
          this.epsilon(HighlightTex.NORMAL);
        }
        break;

      case HighlightTex.STRING:
        this.append(c);
        if (c == '\'')
          this.state = HighlightTex.STRING_QUOTE;
        break;

      case HighlightTex.STRING_QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightTex.NORMAL;
        } else {
          this.state = HighlightTex.STRING;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightTex.BACKTICK:
      this.append('`');
      break;
    case HighlightTex.DOLLAR:
      this.append('$');
      break;
    case HighlightTex.BACKSLASH:
      this.append('\\');
      break;
    case HighlightTex.COMMAND:
    case HighlightTex.COMMENT:
    case HighlightTex.MATH:
    case HighlightTex.MATH_BACKSLASH:
    case HighlightTex.STRING:
    case HighlightTex.STRING_QUOTE:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightTex.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['latex'] = HighlightTex;
Highlighter.REGISTRY['tex'] = HighlightTex;
