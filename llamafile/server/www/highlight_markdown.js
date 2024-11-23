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

class HighlightMarkdown extends Highlighter {

  static NORMAL = 0;
  static TICK = 1;
  static TICK_TICK = 2;
  static CODE = 4;
  static CODE_TICK = 5;
  static CODE_TICK_TICK = 6;
  static STAR = 7;
  static STRONG = 8;
  static STRONG_BACKSLASH = 9;
  static STRONG_STAR = 10;
  static BACKSLASH = 11;
  static INCODE = 12;
  static INCODE2 = 13;
  static INCODE2_TICK = 14;
  static INCODE2_TICK2 = 15;
  static EMPHASIS = 16;
  static EMPHASIS_BACKSLASH = 17;

  constructor(delegate) {
    super(delegate);
    this.bol = true;
    this.tail = false;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      if (c == '\u0000')
        c = '\ufffd';
      switch (this.state) {

      case HighlightMarkdown.NORMAL:
        if (c == '`') {
          this.state = HighlightMarkdown.TICK;
          break;
        } else if (c == '*') {
          this.state = HighlightMarkdown.STAR;
          break;
        } else if (c == '\\') {
          // handle \*\*not bold\*\* etc.
          this.state = HighlightMarkdown.BACKSLASH;
          this.append('\\');
          this.bol = false;
        } else {
          this.append(c);
        }
        if (c == '\n') {
          this.bol = true;
          this.tail = false;
        } else {
          this.tail = true;
          if (!isblank(c))
            this.bol = false;
        }
        break;

      case HighlightMarkdown.BACKSLASH:
        this.append(c);
        this.state = HighlightMarkdown.NORMAL;
        break;

      case HighlightMarkdown.STAR:
        if (c == '*') {
          // handle **strong** text
          this.state = HighlightMarkdown.STRONG;
          this.push("STRONG", "");
          this.append("**");
        } else if (this.bol && isblank(c)) {
          this.append('*');
          this.append(c);
          this.state = HighlightMarkdown.NORMAL;
        } else {
          // handle *emphasized* text
          // inverted because \e[3m has a poorly supported western bias
          this.append('*');
          this.push("EM", "");
          this.append(c);
          this.state = HighlightMarkdown.EMPHASIS;
          if (c == '\\')
            this.state = HighlightMarkdown.EMPHASIS_BACKSLASH;
        }
        this.bol = false;
        break;

      case HighlightMarkdown.EMPHASIS:
        // this is for *emphasized* text
        if (c == '*') {
          this.state = HighlightMarkdown.NORMAL;
          this.pop();
          this.append('*');
        } else if (c == '\\') {
          this.state = HighlightMarkdown.EMPHASIS_BACKSLASH;
          this.append('\\');
        } else {
          this.append(c);
        }
        break;

      case HighlightMarkdown.EMPHASIS_BACKSLASH:
        // so we can say *unbroken \* italic* and have it work
        this.append(c);
        this.state = HighlightMarkdown.EMPHASIS;
        break;

      case HighlightMarkdown.STRONG:
        this.append(c);
        if (c == '*') {
          this.state = HighlightMarkdown.STRONG_STAR;
        } else if (c == '\\') {
          this.state = HighlightMarkdown.STRONG_BACKSLASH;
        }
        break;

      case HighlightMarkdown.STRONG_BACKSLASH:
        // so we can say **unbroken \*\* bold** and have it work
        this.append(c);
        this.state = HighlightMarkdown.STRONG;
        break;

      case HighlightMarkdown.STRONG_STAR:
        this.append(c);
        if (c == '*' || // handle **bold** ending
            (c == '\n' && !this.tail)) { // handle *** line break
          this.state = HighlightMarkdown.NORMAL;
          this.pop();
        } else if (c == '\\') {
          this.state = HighlightMarkdown.STRONG_BACKSLASH;
        } else {
          this.state = HighlightMarkdown.STRONG;
        }
        break;

      case HighlightMarkdown.TICK:
        if (c == '`') {
          if (this.bol) {
            this.state = HighlightMarkdown.TICK_TICK;
          } else {
            this.push("SPAN", "incode");
            this.append("``");
            this.state = HighlightMarkdown.INCODE2;
          }
        } else {
          this.push("SPAN", "incode");
          this.append('`');
          this.append(c);
          this.state = HighlightMarkdown.INCODE;
        }
        this.bol = false;
        break;

      case HighlightMarkdown.INCODE:
        // this is for `inline code` like that
        // no backslash escapes are supported here
        this.append(c);
        if (c == '`') {
          this.pop();
          this.state = HighlightMarkdown.NORMAL;
        }
        break;

      case HighlightMarkdown.INCODE2:
        // this is for ``inline ` code`` like that
        // it lets you put backtick inside the code
        this.append(c);
        if (c == '`') {
          this.state = HighlightMarkdown.INCODE2_TICK;
        }
        break;

      case HighlightMarkdown.INCODE2_TICK:
        this.append(c);
        if (c == '`') {
          this.state = HighlightMarkdown.INCODE2_TICK2;
        } else {
          this.state = HighlightMarkdown.INCODE2;
        }
        break;

      case HighlightMarkdown.INCODE2_TICK2:
        if (c == '`') {
          this.append('`');
        } else {
          this.pop();
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.TICK_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.CODE;
          this.append("```");
        } else {
          this.push("SPAN", "incode");
          this.append("``");
          this.append(c);
          this.state = HighlightMarkdown.INCODE2;
        }
        break;

      case HighlightMarkdown.CODE:
        if (c == '`') {
          this.state = HighlightMarkdown.CODE_TICK;
        } else {
          this.append(c);
        }
        break;

      case HighlightMarkdown.CODE_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.CODE_TICK_TICK;
        } else {
          this.append('`');
          this.epsilon(HighlightMarkdown.CODE);
        }
        break;

      case HighlightMarkdown.CODE_TICK_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.NORMAL;
          this.append("```");
        } else {
          this.append("``");
          this.epsilon(HighlightMarkdown.CODE);
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightMarkdown.STAR:
      this.append('*');
      break;
    case HighlightMarkdown.TICK:
    case HighlightMarkdown.CODE_TICK:
      this.append('`');
      break;
    case HighlightMarkdown.TICK_TICK:
    case HighlightMarkdown.CODE_TICK_TICK:
      this.append("``");
      break;
    case HighlightMarkdown.INCODE:
    case HighlightMarkdown.INCODE2:
    case HighlightMarkdown.INCODE2_TICK:
    case HighlightMarkdown.INCODE2_TICK2:
    case HighlightMarkdown.STRONG:
    case HighlightMarkdown.STRONG_BACKSLASH:
    case HighlightMarkdown.STRONG_STAR:
    case HighlightMarkdown.EMPHASIS:
    case HighlightMarkdown.EMPHASIS_BACKSLASH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightMarkdown.NORMAL;
    this.bol = true;
    this.tail = false;
    this.delegate.flush();
  }
}

Highlighter.REGISTRY['markdown'] = HighlightMarkdown;
Highlighter.REGISTRY['md'] = HighlightMarkdown;
