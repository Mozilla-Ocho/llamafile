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
  static LANG = 3;
  static CODE = 4;
  static CODE_TICK = 5;
  static CODE_TICK_TICK = 6;
  static STAR = 7;
  static EMPHASIS_STAR = 8;
  static STRONG_STAR = 9;
  static BACKSLASH = 10;
  static INCODE = 11;
  static INCODE2 = 12;
  static INCODE2_TICK = 13;
  static LANG2 = 14;
  static NEWLINE = 15;
  static EAT_NEWLINE = 16;
  static INCODE2_TICK2 = 17;
  static LSB = 18;
  static LSB_BACKSLASH = 19;
  static LSB_RSB = 20;
  static LSB_RSB_LPAREN = 21;
  static LSB_RSB_LPAREN_BACKSLASH = 22;
  static LAB = 23;
  static LAB_BACKSLASH = 24;
  static HYPHEN = 25;
  static TILDE = 26;
  static EMPHASIS_STAR2 = 27;
  static OLNAME = 28;
  static OLNAME_DOT = 29;

  static STYLE_SPAN = 16;
  static STYLE_STRONG = 16;
  static STYLE_EMPHASIS = 17;
  static STYLE_STRIKE = 18;

  static STYLE_LIST = 32;
  static STYLE_UL = 32;
  static STYLE_OL = 33;

  static STYLE_DIV = 64;
  static STYLE_LI = 64;

  static get_style_tag(t) {
    switch (t) {
    case HighlightMarkdown.STYLE_UL:
      return 'UL';
    case HighlightMarkdown.STYLE_OL:
      return 'OL';
    default:
      throw new Error('bad style');
    }
  }

  static is_escapable(c) {
    switch (c) {
    case '!':
    case '#':
    case '(':
    case ')':
    case '*':
    case '+':
    case '-':
    case '.':
    case '<':
    case '>':
    case '[':
    case '\\':
    case ']':
    case '_':
    case '`':
    case '{':
    case '|':
    case '}':
      return true;
    default:
      return false;
    }
  }

  static is_sublist(depth, next) {
    if (!depth) {
      return next > 0;
    } else {
      return next >= depth + 4;
    }
  }

  constructor(delegate) {
    super(delegate);
    this.bol = true;
    this.tail = false;
    this.lang = '';
    this.highlighter = null;
    this.newlines = 0;
    this.spaces = 0;
    this.text = '';
    this.href = '';
    this.style = [];
    this.trailing_spaces = 0;
    this.olname = '';
  }

  inside(style) {
    return this.style.length && this.style[this.style.length - 1][0] == style;
  }

  li(t) {
    while (this.style.length &&
           (this.style[this.style.length - 1][0] == HighlightMarkdown.STYLE_LI ||
            (this.style[this.style.length - 1][0] & HighlightMarkdown.STYLE_SPAN) ||
            ((this.style[this.style.length - 1][0] & HighlightMarkdown.STYLE_LIST) &&
             (this.spaces < this.style[this.style.length - 1][1] ||
              (this.style[this.style.length - 1][0] != t &&
               !HighlightMarkdown.is_sublist(this.style[this.style.length - 1][1], this.spaces)))))) {
      this.pop();
      this.style.pop();
    }
    if (!this.style.length ||
        ((this.style[this.style.length - 1][0] & HighlightMarkdown.STYLE_LIST) &&
         HighlightMarkdown.is_sublist(this.style[this.style.length - 1][1], this.spaces))) {
      this.style.push([t, this.spaces]);
      this.push(HighlightMarkdown.get_style_tag(t), '');
    }
    this.style.push([HighlightMarkdown.STYLE_LI, this.spaces]);
    this.push('LI', '');
    this.bol = false;
  }

  got() {
    if (this.bol) {
      this.bol = false;
      while (this.style.length &&
             (this.style[this.style.length - 1][0] == HighlightMarkdown.STYLE_LI ||
              (this.style[this.style.length - 1][0] & HighlightMarkdown.STYLE_LIST)) &&
             this.spaces <= this.style[this.style.length - 1][1]) {
        this.pop();
        this.style.pop();
      }
    }
    this.spaces = 0;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      const c = input[i];
      switch (this.state) {

      case HighlightMarkdown.NORMAL:
        if (c == '`') {
          this.state = HighlightMarkdown.TICK;
          break;
        } else if (c == '*') {
          if (this.inside(HighlightMarkdown.STYLE_EMPHASIS)) {
            this.state = HighlightMarkdown.EMPHASIS_STAR;
          } else if (this.inside(HighlightMarkdown.STYLE_STRONG)) {
            this.state = HighlightMarkdown.STRONG_STAR;
          } else {
            this.state = HighlightMarkdown.STAR;
          }
          break;
        } else if (c == '-' && this.bol) {
          this.state = HighlightMarkdown.HYPHEN;
          break;
        } else if (isdigit(c) && this.bol) {
          this.epsilon(HighlightMarkdown.OLNAME);
        } else if (c == '\n') {
          this.bol = true;
          this.tail = false;
          this.state = HighlightMarkdown.NEWLINE;
          this.newlines = 1;
          this.trailing_spaces = this.spaces;
          this.spaces = 0;
          break;
        } else if (c == '~') {
          this.state = HighlightMarkdown.TILDE;
          this.got();
        } else if (c == '[') {
          this.state = HighlightMarkdown.LSB;
          this.got();
        } else if (c == '<') {
          this.state = HighlightMarkdown.LAB;
          this.got();
        } else if (c == '\\') {
          this.state = HighlightMarkdown.BACKSLASH;
          this.got();
        } else if (c == ' ') {
          ++this.spaces;
          this.append(c);
        } else if (c == '\t') {
          this.spaces += 4;
          this.append(c);
        } else {
          this.got();
          this.append(c);
        }
        this.tail = true;
        break;

      case HighlightMarkdown.NEWLINE:
        if (c == '\n') {
          ++this.newlines;
          this.spaces = 0;
        } else if (c == ' ') {
          ++this.spaces;
        } else if (c == '\t') {
          this.spaces += 4;
        } else {
          if (this.newlines >= 2) {
            this.push("p", "");
            this.pop();
          } else if (this.trailing_spaces >= 2) {
            this.push('br', '');
            this.pop();
          } else {
            this.append('\n');
          }
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.SPACE_SPACE:
        if (c == '\n') {
          this.state = HighlightMarkdown.NORMAL;
          this.push('br', '');
          this.pop();
          this.append('\n');
        } else {
          this.append("  ");
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.HYPHEN:
        if (isblank(c)) {
          // - handle list item
          this.li(HighlightMarkdown.STYLE_UL);
          this.append(c);
          this.state = HighlightMarkdown.NORMAL;
        } else {
          this.append('-');
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.OLNAME:
        if (isdigit(c)) {
          this.olname += c;
        } else if (c == '.') {
          this.olname += '.';
          this.state = HighlightMarkdown.OLNAME_DOT;
        } else {
          this.got();
          this.append(this.olname);
          this.olname = '';
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.OLNAME_DOT:
        if (isspace(c)) {
          this.li(HighlightMarkdown.STYLE_OL);
          this.olname = '';
          this.state = HighlightMarkdown.NORMAL;
        } else {
          this.got();
          this.append(this.olname);
          this.olname = '';
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.BACKSLASH:
        if (HighlightMarkdown.is_escapable(c)) {
          this.append(c);
          this.state = HighlightMarkdown.NORMAL;
        } else {
          this.append('\\');
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.STAR:
        if (c == '*') {
          // handle **strong** text
          // we don't call this.got() in case this is *** bar
          this.got();
          this.push('strong', '');
          this.state = HighlightMarkdown.NORMAL;
          this.style.push([HighlightMarkdown.STYLE_STRONG, 0]);
        } else if (this.bol && isblank(c)) {
          // * handle list item
          this.li(HighlightMarkdown.STYLE_UL);
          this.append(c);
          this.state = HighlightMarkdown.NORMAL;
        } else {
          // handle *emphasis* text
          this.got();
          this.push('em', '');
          this.style.push([HighlightMarkdown.STYLE_EMPHASIS, 0]);
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.EMPHASIS_STAR:
        if (c == '*') {
          this.state = HighlightMarkdown.EMPHASIS_STAR2;
        } else {
          // leave *italic* text
          //               ^
          this.style.pop();
          this.pop();
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.EMPHASIS_STAR2:
        if (c == '*') {
          this.style.pop();
          this.pop();
          if (this.inside(HighlightMarkdown.STYLE_STRONG)) {
            // handle ***strong emphasis*** closing
            //                            ^
            this.style.pop();
            this.pop();
            this.state = HighlightMarkdown.NORMAL;
          } else {
            // handle *emphasis***strong**
            //                   ^
            this.push('strong', '');
            this.state = HighlightMarkdown.NORMAL;
            this.style.push([HighlightMarkdown.STYLE_STRONG, 0]);
            this.state = HighlightMarkdown.NORMAL;
          }
        } else {
          // handle *italic **strong** text*
          //                  ^
          this.push('strong', '');
          this.style.push([HighlightMarkdown.STYLE_STRONG, 0]);
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.STRONG_STAR:
        if (c == '*') {
          this.state = HighlightMarkdown.NORMAL;
          this.style.pop();
          this.pop();
        } else if (c == '\n' && !this.tail) {
          // handle *** line break
          this.got();
          this.style.pop();
          this.pop();
          this.push('hr', '');
          this.pop();
          this.epsilon(HighlightMarkdown.NORMAL);
        } else {
          this.push('em', '');
          this.style.push([HighlightMarkdown.STYLE_EMPHASIS, 0]);
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.TILDE:
        if (c == '~') {
          // handle ~~strikethrough text~~
          if (this.inside(HighlightMarkdown.STYLE_STRIKE)) {
            this.pop();
            this.style.pop();
          } else {
            this.push('s', '');
            this.style.push([HighlightMarkdown.STYLE_STRIKE, 0]);
          }
          this.state = HighlightMarkdown.NORMAL;
        } else {
          this.append('~');
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.TICK:
        if (c == '`') {
          if (this.bol) {
            this.state = HighlightMarkdown.TICK_TICK;
          } else {
            this.push("code", "");
            this.state = HighlightMarkdown.INCODE2;
          }
        } else {
          this.push("code", "");
          this.append(c);
          this.state = HighlightMarkdown.INCODE;
        }
        this.got();
        break;

      case HighlightMarkdown.INCODE:
        // this is for `inline code` like that
        // no backslash escapes are supported here
        if (c == '`') {
          this.pop();
          this.state = HighlightMarkdown.NORMAL;
        } else {
          this.append(c);
        }
        break;

      case HighlightMarkdown.INCODE2:
        // this is for ``inline ` code`` like that
        // it lets you put backtick inside the code
        if (c == '`') {
          this.state = HighlightMarkdown.INCODE2_TICK;
        } else {
          this.append(c);
        }
        break;

      case HighlightMarkdown.INCODE2_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.INCODE2_TICK2;
        } else {
          this.append('`');
          this.append(c);
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
          this.state = HighlightMarkdown.LANG;
        } else {
          this.push('code', '');
          this.append(c);
          this.state = HighlightMarkdown.INCODE2;
        }
        break;

      case HighlightMarkdown.LANG:
        if (!isascii(c) || !isspace(c)) {
          this.lang += c.toLowerCase();
        } else {
          this.epsilon(HighlightMarkdown.LANG2);
        }
        break;

      case HighlightMarkdown.LANG2:
        if (c == "\n") {
          this.flush();
          let pre = this.push('pre', '');
          this.setupCodeBlock(pre);
          let hdom = new HighlightDom(pre);
          if (!(this.highlighter = Highlighter.create(this.lang, hdom)))
            this.highlighter = Highlighter.create('txt', hdom);
          this.state = HighlightMarkdown.CODE;
          this.lang = '';
        }
        break;

      case HighlightMarkdown.CODE:
        if (c == '`') {
          this.state = HighlightMarkdown.CODE_TICK;
        } else {
          this.highlighter.feed(c);
        }
        break;

      case HighlightMarkdown.CODE_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.CODE_TICK_TICK;
        } else {
          this.highlighter.feed("`" + c);
          this.state = HighlightMarkdown.CODE;
        }
        break;

      case HighlightMarkdown.CODE_TICK_TICK:
        if (c == '`') {
          this.state = HighlightMarkdown.EAT_NEWLINE;
          this.highlighter.flush();
          this.highlighter = null;
          this.pop();
        } else {
          this.highlighter.feed("``" + c);
          this.state = HighlightMarkdown.CODE;
        }
        break;

      case HighlightMarkdown.EAT_NEWLINE:
        if (c != '\n') {
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.LAB:
        if (c == '>') {
          let a = this.push('a', '');
          a.innerText = this.href;
          a.href = this.href;
          this.pop();
          this.href = '';
          this.state = HighlightMarkdown.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightMarkdown.LAB_BACKSLASH;
        } else {
          this.href += c;
          if (this.href.length <= 4 && !"http".startsWith(this.href)) {
            this.append('<' + this.href);
            this.href = '';
            this.state = HighlightMarkdown.NORMAL;
          }
        }
        break;

      case HighlightMarkdown.LAB_BACKSLASH:
        if (!HighlightMarkdown.is_escapable(c))
          this.href += '\\';
        this.href += c;
        this.state = HighlightMarkdown.LAB;
        break;

      case HighlightMarkdown.LSB:
        if (c == ']') {
          this.state = HighlightMarkdown.LSB_RSB;
        } else if (c == '\\') {
          this.state = HighlightMarkdown.LSB_BACKSLASH;
        } else {
          this.text += c;
        }
        break;

      case HighlightMarkdown.LSB_BACKSLASH:
        if (!HighlightMarkdown.is_escapable(c))
          this.text += '\\';
        this.text += c;
        this.state = HighlightMarkdown.LSB;
        break;

      case HighlightMarkdown.LSB_RSB:
        if (c == '(') {
          this.state = HighlightMarkdown.LSB_RSB_LPAREN;
        } else {
          this.append('[' + this.text + ']');
          this.text = '';
          this.epsilon(HighlightMarkdown.NORMAL);
        }
        break;

      case HighlightMarkdown.LSB_RSB_LPAREN:
        if (c == ')') {
          let a = this.push('a', '');
          a.innerText = this.text;
          a.href = this.href;
          this.pop();
          this.href = '';
          this.text = '';
          this.state = HighlightMarkdown.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightMarkdown.LSB_RSB_LPAREN_BACKSLASH;
        } else {
          this.href += c;
        }
        break;

      case HighlightMarkdown.LSB_RSB_LPAREN_BACKSLASH:
        if (!HighlightMarkdown.is_escapable(c))
          this.href += '\\';
        this.href += c;
        this.state = HighlightMarkdown.LSB_RSB_LPAREN;
        break;

      default:
        throw new Error('Invalid state: ' + this.state);
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightMarkdown.LANG:
      this.lang = '';
      break;
    case HighlightMarkdown.STAR:
    case HighlightMarkdown.STRONG_STAR:
    case HighlightMarkdown.EMPHASIS_STAR:
      this.append('*');
      break;
    case HighlightMarkdown.EMPHASIS_STAR2:
      this.append("**");
      break;
    case HighlightMarkdown.TICK:
      this.append('`');
      break;
    case HighlightMarkdown.TICK_TICK:
      this.append("``");
      break;
    case HighlightMarkdown.INCODE:
    case HighlightMarkdown.INCODE2:
    case HighlightMarkdown.INCODE2_TICK:
    case HighlightMarkdown.INCODE2_TICK2:
      this.pop();
      break;
    case HighlightMarkdown.CODE:
      this.highlighter.flush();
      this.highlighter = null;
      this.pop();
      break;
    case HighlightMarkdown.CODE_TICK:
      this.append('`');
      this.highlighter.flush();
      this.highlighter = null;
      this.pop();
      break;
    case HighlightMarkdown.CODE_TICK_TICK:
      this.append('``');
      this.highlighter.flush();
      this.highlighter = null;
      this.pop();
      break;
    case HighlightMarkdown.LAB:
    case HighlightMarkdown.LAB_BACKSLASH:
      this.append('<' + this.href);
      this.href = '';
      break;
    case HighlightMarkdown.LSB:
    case HighlightMarkdown.LSB_BACKSLASH:
      this.append('[' + this.text);
      this.text = '';
      break;
    case HighlightMarkdown.LSB_RSB:
      this.append('[' + this.text + ']');
      this.text = '';
      break;
    case HighlightMarkdown.LSB_RSB_LPAREN:
    case HighlightMarkdown.LSB_RSB_LPAREN_BACKSLASH:
      this.append('[' + this.text + '](' + this.href);
      this.text = '';
      this.href = '';
      break;
    case HighlightMarkdown.TILDE:
      this.append('~');
      break;
    case HighlightMarkdown.SPACE:
      for (let i = 0; i < this.spaces; ++i)
        this.append(' ');
      break;
    case HighlightMarkdown.OLNAME:
    case HighlightMarkdown.OLNAME_DOT:
      this.append(this.olname);
      this.olname = '';
      break;
    default:
      break;
    }
    while (this.style.length) {
      this.style.pop();
      this.pop();
    }
    this.state = HighlightMarkdown.NORMAL;
    this.bol = true;
    this.spaces = 0;
    this.tail = false;
    this.newlines = 0;
    this.delegate.flush();
    this.delta = 1;
  }

  setupCodeBlock(pre) {
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-button';
    copyButton.innerHTML =
      `<svg xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24" fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round">
         <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
         <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
       </svg>`;
    copyButton.addEventListener('click', function() {
      try {
        copyTextToClipboard(pre.innerText);
        const originalInnerHTML = copyButton.innerHTML;
        copyButton.innerHTML =
          `<svg xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24" fill="none"
                stroke="green"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round">
             <polyline points="20 6 9 17 4 12"></polyline>
           </svg>`;
        setTimeout(() => {
          copyButton.innerHTML = originalInnerHTML;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy text:', err);
      }
    });
    pre.appendChild(copyButton);
  }
}

Highlighter.REGISTRY['markdown'] = HighlightMarkdown;
Highlighter.REGISTRY['md'] = HighlightMarkdown;
