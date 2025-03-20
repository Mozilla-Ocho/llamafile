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

class RenderMarkdown extends Highlighter {

  static NORMAL = 0;
  static TICK = 1;
  static TICK_TICK = 2;
  static LANG = 3;
  static CODE = 4;
  static CODE_TICK = 5;
  static STAR = 7;
  static EMPHASIS_STAR = 8;
  static STRONG_STAR = 9;
  static BACKSLASH = 10;
  static INCODE = 11;
  static INCODE2 = 12;
  static INCODE2_TICK = 13;
  static LANG2 = 14;
  static NEWLINE = 15;
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
  static HASH = 30;
  static HYPHEN_HYPHEN = 31;
  static HYPHEN_HYPHEN_HYPHEN = 32;
  static OLDCODE = 33;
  static BLOCKQUOTE = 34;
  static TICK_TICK_TICK = 35;
  static EXCLAIM = 36;

  static STYLE_SPAN = 16;
  static STYLE_STRONG = 16;
  static STYLE_EMPHASIS = 17;
  static STYLE_STRIKE = 18;

  static STYLE_LIST = 32;
  static STYLE_UL = 32;
  static STYLE_OL = 33;

  static STYLE_DIV = 64;
  static STYLE_LI = 64;
  static STYLE_HEADER = 65;
  static STYLE_BLOCKQUOTE = 66;

  static get_style_tag(t) {
    switch (t) {
    case RenderMarkdown.STYLE_UL:
      return 'UL';
    case RenderMarkdown.STYLE_OL:
      return 'OL';
    default:
      throw new Error('bad style');
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
    this.dedent = 0;
    this.hashes = 0;
    this.bqline = false;
    this.tick1 = 0;
    this.tick2 = 0;
    this.is_image = false;
  }

  inside(style) {
    return this.style.length && this.style[this.style.length - 1][0] == style;
  }

  blockquoted() {
    for (let i = 0; i < this.style.length; ++i)
      if (this.style[i][0] == RenderMarkdown.STYLE_BLOCKQUOTE)
        return true;
    return false;
  }

  li(t) {
    // plan out <li> insertion
    let need_list = true;
    while (this.style.length) {
      let e = this.style[this.style.length - 1];

      // eliminate <b>, <i>, and <s>
      if (e[0] & RenderMarkdown.STYLE_SPAN) {
        this.pop();
        this.style.pop();
        continue;
      }

      // eliminate <blockquote> if we're beneath it
      if (e[0] == RenderMarkdown.STYLE_BLOCKQUOTE && this.spaces < e[1]) {
        this.pop();
        this.style.pop();
        continue;
      }

      // otherwise it must be <ul><li> or <ol><li>
      if (e[0] != RenderMarkdown.STYLE_LI)
        break;
      if (this.style.length < 2)
        break;
      let list = this.style[this.style.length - 2];
      if (!(list[0] & RenderMarkdown.STYLE_LIST))
        break;

      // pop if new list is beneath the current one, e.g.
      //
      //   - one
      //       - two
      //   - three
      //
      if (this.spaces < list[1]) {
        this.pop();
        this.style.pop();
        this.pop();
        this.style.pop();
        continue;
      }

      // don't pop if we're adding a sublist, e.g.
      //
      //   - one
      //       - two
      //
      if (RenderMarkdown.is_sublist(list[1], this.spaces)) {
        break;
      }

      // pop if list kind is changing and we're at the same level, e.g.
      //
      //   1. one
      //   - two
      //
      if (list[0] != t) {
        this.pop();
        this.style.pop();
        this.pop();
        this.style.pop();
        break;
      }

      // pop <li> if we're adding a sibling, e.g.
      //
      //   - one
      //   - two
      //
      this.pop();
      this.style.pop();
      need_list = false;
      break;
    }

    // create new <li> element
    if (need_list) {
      this.style.push([t, this.spaces]);
      this.push(RenderMarkdown.get_style_tag(t), '');
    }
    this.style.push([RenderMarkdown.STYLE_LI, this.spaces]);
    this.push("LI", "");
    this.bol = false;
  }

  got() {
    if (this.bol) {
      this.bol = false;
      while (this.style.length &&
             ((!this.bqline &&
               this.style[this.style.length - 1][0] == RenderMarkdown.STYLE_BLOCKQUOTE) ||
              this.style[this.style.length - 1][0] == RenderMarkdown.STYLE_LI ||
              (this.style[this.style.length - 1][0] & RenderMarkdown.STYLE_LIST)) &&
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
      if (c == '\r')
        continue;
      if (c == '\u0000')
        c = '\ufffd';
      switch (this.state) {

      case RenderMarkdown.NORMAL:
        if (this.bol &&
            this.spaces >= 4 &&
            this.newlines >= 2 &&
            isgraph(c) &&
            c != '`' &&
            (!this.style.length ||
             this.spaces >= this.style[this.style.length - 1][1] + 4)) {
          this.dedent = this.spaces;
          this.spaces = 0;
          this.bol = false;
          this.push("PRE", "");
          this.epsilon(RenderMarkdown.OLDCODE);
        } else if (c == '`') {
          this.state = RenderMarkdown.TICK;
          break;
        } else if (c == '*') {
          if (this.inside(RenderMarkdown.STYLE_EMPHASIS)) {
            this.state = RenderMarkdown.EMPHASIS_STAR;
          } else if (this.inside(RenderMarkdown.STYLE_STRONG)) {
            this.state = RenderMarkdown.STRONG_STAR;
          } else {
            this.state = RenderMarkdown.STAR;
          }
          break;
        } else if (c == '>' && this.bol) {
          this.state = RenderMarkdown.BLOCKQUOTE;
          break;
        } else if (c == '-' && this.bol) {
          this.state = RenderMarkdown.HYPHEN;
          break;
        } else if (c == '#' && this.bol && !this.spaces) {
          this.state = RenderMarkdown.HASH;
          this.hashes = 1;
        } else if (isdigit(c) && this.bol) {
          this.epsilon(RenderMarkdown.OLNAME);
        } else if (c == '\n') {
          if (this.inside(RenderMarkdown.STYLE_HEADER)) {
            this.pop();
            this.style.pop();
          }
          this.bol = true;
          this.tail = false;
          this.bqline = false;
          this.state = RenderMarkdown.NEWLINE;
          this.newlines = 1;
          this.trailing_spaces = this.spaces;
          this.spaces = 0;
          break;
        } else if (c == '~') {
          this.state = RenderMarkdown.TILDE;
          this.got();
        } else if (c == '!') {
          this.is_image = true;
          this.state = RenderMarkdown.EXCLAIM;
          this.got();
        } else if (c == '[') {
          this.is_image = false;
          this.state = RenderMarkdown.LSB;
          this.got();
        } else if (c == '<') {
          this.state = RenderMarkdown.LAB;
          this.got();
        } else if (c == '\\') {
          this.state = RenderMarkdown.BACKSLASH;
          this.got();
        } else if (c == ' ') {
          ++this.spaces;
          this.append(c);
        } else if (c == '\t') {
          this.spaces = (this.spaces + 4) & -4;
          this.append(c);
        } else {
          this.got();
          this.append(c);
        }
        this.tail = true;
        break;

      case RenderMarkdown.NEWLINE:
        if (c == '\n') {
          ++this.newlines;
          this.spaces = 0;
        } else if (c == ' ') {
          ++this.spaces;
        } else if (c == '\t') {
          this.spaces = (this.spaces + 4) & -4;
        } else {
          if (this.newlines >= 2) {
            if (this.style.length &&
                this.style[this.style.length - 1][0] == RenderMarkdown.STYLE_BLOCKQUOTE) {
              this.pop();
              this.style.pop();
            }
            this.push("P", "");
            this.pop();
          } else if (this.trailing_spaces >= 2) {
            this.push("BR", "");
            this.pop();
          } else {
            this.append('\n');
          }
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.BLOCKQUOTE:
        if (isblank(c)) {
          if (!this.blockquoted()) {
            this.push("BLOCKQUOTE", "");
            this.style.push([RenderMarkdown.STYLE_BLOCKQUOTE, this.spaces]);
          }
          this.state = RenderMarkdown.NORMAL;
          this.bqline = true;
        } else {
          this.got();
          this.append('>');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.HYPHEN:
        if (isblank(c)) {
          // - handle list item
          this.li(RenderMarkdown.STYLE_UL);
          this.append(c);
          this.state = RenderMarkdown.NORMAL;
        } else if (c == '-') {
          this.state = RenderMarkdown.HYPHEN_HYPHEN;
        } else {
          this.got();
          this.append('-');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.HYPHEN_HYPHEN:
        if (c == '-') {
          this.state = RenderMarkdown.HYPHEN_HYPHEN_HYPHEN;
        } else {
          this.got();
          this.append("--");
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.HYPHEN_HYPHEN_HYPHEN:
        if (c == '\n') {
          // handle --- line break
          this.got();
          this.push("HR", "");
          this.pop();
          this.epsilon(RenderMarkdown.NORMAL);
        } else {
          this.got();
          this.append("---");
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.OLNAME:
        if (isdigit(c)) {
          this.olname += c;
        } else if (c == '.') {
          this.olname += '.';
          this.state = RenderMarkdown.OLNAME_DOT;
        } else {
          this.got();
          this.append(this.olname);
          this.olname = '';
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.OLNAME_DOT:
        if (isspace(c)) {
          this.li(RenderMarkdown.STYLE_OL);
          this.olname = '';
          this.state = RenderMarkdown.NORMAL;
        } else {
          this.got();
          this.append(this.olname);
          this.olname = '';
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.BACKSLASH:
        if (ispunct(c)) {
          this.append(c);
          this.state = RenderMarkdown.NORMAL;
        } else if (c == '\n') {
          this.push("BR", "");
          this.pop();
          this.state = RenderMarkdown.NORMAL;
        } else {
          this.append('\\');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.STAR:
        if (c == '*') {
          // handle **strong** text
          // we don't call this.got() in case this is *** bar
          this.got();
          this.push("STRONG", "");
          this.state = RenderMarkdown.NORMAL;
          this.style.push([RenderMarkdown.STYLE_STRONG, 0]);
        } else if (this.bol && isblank(c)) {
          // * handle list item
          this.li(RenderMarkdown.STYLE_UL);
          this.append(c);
          this.state = RenderMarkdown.NORMAL;
        } else {
          // handle *emphasis* text
          this.got();
          this.push("EM", "");
          this.style.push([RenderMarkdown.STYLE_EMPHASIS, 0]);
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.EMPHASIS_STAR:
        if (c == '*') {
          this.state = RenderMarkdown.EMPHASIS_STAR2;
        } else {
          // leave *italic* text
          //               ^
          this.style.pop();
          this.pop();
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.EMPHASIS_STAR2:
        if (c == '*') {
          this.style.pop();
          this.pop();
          if (this.inside(RenderMarkdown.STYLE_STRONG)) {
            // handle ***strong emphasis*** closing
            //                            ^
            this.style.pop();
            this.pop();
            this.state = RenderMarkdown.NORMAL;
          } else {
            // handle *emphasis***strong**
            //                   ^
            this.push("STRONG", "");
            this.state = RenderMarkdown.NORMAL;
            this.style.push([RenderMarkdown.STYLE_STRONG, 0]);
            this.state = RenderMarkdown.NORMAL;
          }
        } else {
          // handle *italic **strong** text*
          //                  ^
          this.push("STRONG", "");
          this.style.push([RenderMarkdown.STYLE_STRONG, 0]);
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.STRONG_STAR:
        if (c == '*') {
          this.state = RenderMarkdown.NORMAL;
          this.style.pop();
          this.pop();
        } else if (c == '\n' && !this.tail) {
          // handle *** line break
          this.got();
          this.style.pop();
          this.pop();
          this.push("HR", "");
          this.pop();
          this.epsilon(RenderMarkdown.NORMAL);
        } else {
          this.push("EM", "");
          this.style.push([RenderMarkdown.STYLE_EMPHASIS, 0]);
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.TILDE:
        if (c == '~') {
          // handle ~~strikethrough text~~
          if (this.inside(RenderMarkdown.STYLE_STRIKE)) {
            this.pop();
            this.style.pop();
          } else {
            this.push("S", "");
            this.style.push([RenderMarkdown.STYLE_STRIKE, 0]);
          }
          this.state = RenderMarkdown.NORMAL;
        } else {
          this.append('~');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.HASH:
        if (c == '#') {
          ++this.hashes;
        } else if (isblank(c)) {
          this.got();
          this.push("H" + this.hashes, "");
          this.state = RenderMarkdown.NORMAL;
          this.style.push([RenderMarkdown.STYLE_HEADER, 0]);
        } else {
          this.got();
          for (let i = 0; i < this.hashes; ++i)
            this.append('#');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.TICK:
        if (c == '`') {
          if (this.bol) {
            this.state = RenderMarkdown.TICK_TICK;
          } else {
            this.got();
            this.push("CODE", "");
            this.state = RenderMarkdown.INCODE2;
          }
        } else {
          this.got();
          this.push("CODE", "");
          this.append(c);
          this.state = RenderMarkdown.INCODE;
        }
        break;

      case RenderMarkdown.INCODE:
        // this is for `inline code` like that
        // no backslash escapes are supported here
        if (c == '`') {
          this.pop();
          this.state = RenderMarkdown.NORMAL;
        } else {
          this.append(c);
        }
        break;

      case RenderMarkdown.INCODE2:
        // this is for ``inline ` code`` like that
        // it lets you put backtick inside the code
        if (c == '`') {
          this.state = RenderMarkdown.INCODE2_TICK;
        } else {
          this.append(c);
        }
        break;

      case RenderMarkdown.INCODE2_TICK:
        if (c == '`') {
          this.state = RenderMarkdown.INCODE2_TICK2;
        } else {
          this.append('`');
          this.append(c);
          this.state = RenderMarkdown.INCODE2;
        }
        break;

      case RenderMarkdown.INCODE2_TICK2:
        if (c == '`') {
          this.append('`');
        } else {
          this.pop();
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.TICK_TICK:
        if (c == '`') {
          this.state = RenderMarkdown.TICK_TICK_TICK;
          this.tick1 = 3;
        } else {
          this.got();
          this.push("CODE", "");
          this.append(c);
          this.state = RenderMarkdown.INCODE2;
        }
        break;

      case RenderMarkdown.TICK_TICK_TICK:
        if (c == '`') {
          ++this.tick1;
        } else {
          this.dedent = this.spaces;
          this.got();
          this.epsilon(RenderMarkdown.LANG);
          this.spaces = 0;
        }
        break;

      case RenderMarkdown.LANG:
        if (!isascii(c) || !isspace(c)) {
          this.lang += c.toLowerCase();
        } else {
          this.epsilon(RenderMarkdown.LANG2);
        }
        break;

      case RenderMarkdown.LANG2:
        if (c == "\n") {
          let pre = this.push("PRE", "");
          this.setupCodeBlock(pre);
          let hdom = new HighlightDom(pre);
          if (!(this.highlighter = Highlighter.create(this.lang, hdom)))
            this.highlighter = Highlighter.create('txt', hdom);
          this.state = RenderMarkdown.CODE;
          this.bol = true;
          this.lang = '';
        }
        break;

      case RenderMarkdown.CODE:
        if (this.bol) {
          if (c == ' ') {
            ++this.spaces;
            if (this.spaces <= this.dedent)
              break;
          } else if (c == '\t') {
            this.spaces = (this.spaces + 4) & -4;
            if (this.spaces <= this.dedent)
              break;
          } else {
            this.bol = false;
          }
        }
        if (c == '\n') {
          this.bol = true;
          this.spaces = 0;
        }
        if (c == '`') {
          this.state = RenderMarkdown.CODE_TICK;
          this.tick2 = 1;
        } else {
          this.highlighter.feed(c);
        }
        break;

      case RenderMarkdown.CODE_TICK:
        if (c == '`') {
          if (++this.tick2 == this.tick1) {
            this.spaces = 0;
            this.state = HighlightMarkdown.NORMAL;
            this.highlighter.flush();
            this.highlighter = null;
            this.pop();
          }
        } else {
          for (let i = 0; i < this.tick2; ++i)
            this.highlighter.feed('`');
          this.epsilon(RenderMarkdown.CODE);
        }
        break;

      case RenderMarkdown.LAB:
        if (c == '>') {
          let a = this.push("A", "");
          a.innerText = this.href;
          a.href = this.href;
          this.pop();
          this.href = '';
          this.state = RenderMarkdown.NORMAL;
        } else if (c == '\\') {
          this.state = RenderMarkdown.LAB_BACKSLASH;
        } else {
          this.href += c;
          if (this.href.length <= 4 && !"http".startsWith(this.href)) {
            this.append('<' + this.href);
            this.href = '';
            this.state = RenderMarkdown.NORMAL;
          }
        }
        break;

      case RenderMarkdown.LAB_BACKSLASH:
        if (!ispunct(c))
          this.href += '\\';
        this.href += c;
        this.state = RenderMarkdown.LAB;
        break;

      case RenderMarkdown.EXCLAIM:
        if (c == '[') {
          this.state = RenderMarkdown.LSB;
        } else {
          this.append('!');
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.LSB:
        if (c == ']') {
          this.state = RenderMarkdown.LSB_RSB;
        } else if (c == '\\') {
          this.state = RenderMarkdown.LSB_BACKSLASH;
        } else {
          this.text += c;
        }
        break;

      case RenderMarkdown.LSB_BACKSLASH:
        if (!ispunct(c))
          this.text += '\\';
        this.text += c;
        this.state = RenderMarkdown.LSB;
        break;

      case RenderMarkdown.LSB_RSB:
        if (c == '(') {
          this.state = RenderMarkdown.LSB_RSB_LPAREN;
        } else {
          this.append('[' + this.text + ']');
          this.text = '';
          this.epsilon(RenderMarkdown.NORMAL);
        }
        break;

      case RenderMarkdown.LSB_RSB_LPAREN:
        if (c == ')') {
          if (this.is_image) {
            let a = this.push("IMG", "");
            a.alt = this.text;
            a.src = this.href;
          } else {
            let a = this.push("A", "");
            a.innerText = this.text;
            a.href = this.href;
          }
          this.pop();
          this.href = '';
          this.text = '';
          this.state = RenderMarkdown.NORMAL;
        } else if (c == '\\') {
          this.state = RenderMarkdown.LSB_RSB_LPAREN_BACKSLASH;
        } else {
          this.href += c;
        }
        break;

      case RenderMarkdown.LSB_RSB_LPAREN_BACKSLASH:
        if (!ispunct(c))
          this.href += '\\';
        this.href += c;
        this.state = RenderMarkdown.LSB_RSB_LPAREN;
        break;

      case RenderMarkdown.OLDCODE:
        if (this.bol) {
          if (c == ' ') {
            ++this.spaces;
            if (this.spaces <= this.dedent)
              break;
          } else if (c == '\t') {
            this.spaces = (this.spaces + 4) & -4;
            if (this.spaces <= this.dedent)
              break;
          } else {
            if (this.spaces < this.dedent) {
              this.pop();
              this.epsilon(RenderMarkdown.NORMAL);
              break;
            }
            this.bol = false;
          }
        }
        if (c == '\n') {
          this.bol = true;
          this.spaces = 0;
        }
        this.append(c);
        break;

      default:
        throw new Error('Invalid state: ' + this.state);
      }
    }
  }

  flush() {
    switch (this.state) {
    case RenderMarkdown.LANG:
      this.lang = '';
      break;
    case RenderMarkdown.STAR:
    case RenderMarkdown.STRONG_STAR:
    case RenderMarkdown.EMPHASIS_STAR:
      this.append('*');
      break;
    case RenderMarkdown.EMPHASIS_STAR2:
      this.append("**");
      break;
    case RenderMarkdown.TICK:
      this.append('`');
      break;
    case RenderMarkdown.TICK_TICK:
      this.append("``");
      break;
    case RenderMarkdown.TICK_TICK_TICK:
      for (let i = 0; i < this.tick1; ++i)
        this.append('`');
      break;
    case RenderMarkdown.HYPHEN:
      this.append('-');
      break;
    case RenderMarkdown.HYPHEN_HYPHEN:
      this.append("--");
      break;
    case RenderMarkdown.HYPHEN_HYPHEN_HYPHEN:
      this.append("---");
      break;
    case RenderMarkdown.INCODE:
    case RenderMarkdown.INCODE2:
    case RenderMarkdown.INCODE2_TICK:
    case RenderMarkdown.INCODE2_TICK2:
      this.pop();
      break;
    case RenderMarkdown.CODE:
      this.highlighter.flush();
      this.highlighter = null;
      this.pop();
      break;
    case RenderMarkdown.CODE_TICK:
      for (let i = 0; i < this.tick2; ++i)
        this.append('`');
      this.highlighter.flush();
      this.highlighter = null;
      this.pop();
      break;
    case RenderMarkdown.LAB:
    case RenderMarkdown.LAB_BACKSLASH:
      this.append('<' + this.href);
      this.href = '';
      break;
    case RenderMarkdown.LSB:
    case RenderMarkdown.LSB_BACKSLASH:
      this.append('[' + this.text);
      this.text = '';
      break;
    case RenderMarkdown.LSB_RSB:
      this.append('[' + this.text + ']');
      this.text = '';
      break;
    case RenderMarkdown.LSB_RSB_LPAREN:
    case RenderMarkdown.LSB_RSB_LPAREN_BACKSLASH:
      this.append('[' + this.text + '](' + this.href);
      this.text = '';
      this.href = '';
      break;
    case RenderMarkdown.TILDE:
      this.append('~');
      break;
    case RenderMarkdown.HASH:
      for (let i = 0; i < this.hashes; ++i)
        this.append('#');
      break;
    case RenderMarkdown.OLNAME:
    case RenderMarkdown.OLNAME_DOT:
      this.append(this.olname);
      this.olname = '';
      break;
    case RenderMarkdown.OLDCODE:
      this.pop();
      break;
    default:
      break;
    }
    while (this.style.length) {
      this.style.pop();
      this.pop();
    }
    this.state = RenderMarkdown.NORMAL;
    this.bol = true;
    this.spaces = 0;
    this.tail = false;
    this.newlines = 0;
    this.delegate.flush();
    this.delta = 1;
    this.dedent = 0;
    this.hashes = 0;
    this.bqline = false;
    this.tick1 = 0;
    this.tick2 = 0;
  }

  setupCodeBlock(pre) {
    pre.appendChild(createCopyButton(() => pre.innerText));
  }
}
