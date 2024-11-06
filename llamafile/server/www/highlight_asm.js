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

const ASM_PREFIXES = new Set([
  'addr32',
  'cs',
  'data16',
  'ds',
  'es',
  'fs',
  'gs',
  'lock',
  'rep',
  'repe',
  'repne',
  'repnz',
  'repz',
  'rex',
  'rex.b',
  'rex.r',
  'rex.rb',
  'rex.rx',
  'rex.rxb',
  'rex.w',
  'rex.wb',
  'rex.wr',
  'rex.wrb',
  'rex.wrx',
  'rex.wrxb',
  'rex.wx',
  'rex.wxb',
  'rex.x',
  'rex.xb',
  'ss',
]);

const ASM_QUALIFIERS = new Set([
  ':req',
  ':vararg',
  '@dtpmod',
  '@dtpoff',
  '@fini_array',
  '@function',
  '@gnu_indirect_function',
  '@got',
  '@gotoff',
  '@gotpcrel',
  '@gottpoff',
  '@init_array',
  '@nobits',
  '@note',
  '@notype',
  '@object',
  '@plt',
  '@pltoff',
  '@progbits',
  '@size',
  '@tlsgd',
  '@tlsld',
  '@tpoff',
  '@unwind',
  'comdat',
]);

class HighlightAsm extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static COMMENT = 2;
  static BACKSLASH = 3;
  static SLASH0 = 4;
  static SLASH = 5;
  static REG0 = 6;
  static REG = 7;
  static SLASH_SLASH = 8;
  static SLASH_STAR = 9;
  static SLASH_STAR_STAR = 10;
  static QUOTE = 11;
  static QUOTE_BACKSLASH = 12;
  static QUOTE_FINISH = 13;
  static DQUOTE = 14;
  static DQUOTE_BACKSLASH = 15;
  static DOLLAR = 16;
  static IMMEDIATE = 17;
  static IMMEDIATE_QUOTE = 18;
  static IMMEDIATE_QUOTE_BACKSLASH = 19;
  static IMMEDIATE_QUOTE_FINISH = 20;
  static HASH = 21;

  constructor(delegate) {
    super(delegate);
    this.c = 0;
    this.last = 0;
    this.col = 0;
    this.is_preprocessor = false;
    this.is_first_thing_on_line = true;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      this.last = this.c;
      this.c = c;
      switch (this.state) {

      case HighlightAsm.NORMAL:
        if (!isascii(c) || //
            isalnum(c) || //
            c == '-' || //
            c == '.' || //
            c == '@' || //
            c == '_' || //
            (c == '#' && this.col == 0)) {
          this.state = HighlightAsm.WORD;
          this.word += c;
          break;
        } else if (c == '#' && this.col && isspace(this.last)) {
          this.state = HighlightAsm.HASH;
        } else if ((c == ';' || c == '!') && (!this.col || isspace(this.last))) {
          this.state = HighlightAsm.COMMENT;
          this.push("span", "comment");
          this.append(c);
        } else if (c == '/' && this.col == 0) {
          // bell system five allowed single slash comments
          // anywhere on the line, but we limit that a bit.
          this.state = HighlightAsm.SLASH0;
          this.push("span", "comment");
          this.append('/');
        } else if (c == '/') {
          this.state = HighlightAsm.SLASH;
          this.is_first_thing_on_line = false;
        } else if (c == '$') {
          this.append('$');
          this.state = HighlightAsm.DOLLAR;
          this.is_first_thing_on_line = false;
        } else if (c == '%') {
          this.state = HighlightAsm.REG0;
          this.is_first_thing_on_line = false;
        } else if (c == '\\') {
          this.state = HighlightAsm.BACKSLASH;
          this.push("span", "escape");
          this.append('\\');
          this.is_first_thing_on_line = false;
        } else if (c == '\'') {
          this.state = HighlightAsm.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightAsm.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          if (c == '\n')
            this.is_preprocessor = false;
          if (!isspace(c))
            this.is_first_thing_on_line = false;
          if (c == ':')
            this.is_first_thing_on_line = true;
          this.append(c);
        }
        break;

      case HighlightAsm.DOLLAR:
        if (is_immediate(c) || c == '\'') {
          this.push("span", "immediate");
          this.state = HighlightAsm.IMMEDIATE;
        } else {
          this.epsilon(HighlightAsm.NORMAL);
        }
        // fallthrough

      case HighlightAsm.IMMEDIATE:
        if (is_immediate(c)) {
          this.append(c);
        } else if (c == '\'') {
          this.append(c);
          this.state = HighlightAsm.IMMEDIATE_QUOTE;
        } else {
          this.pop();
          this.epsilon(HighlightAsm.NORMAL);
        }
        break;

      case HighlightAsm.IMMEDIATE_QUOTE:
        if (c == '\\') {
          this.append(c);
          this.state = HighlightAsm.IMMEDIATE_QUOTE_BACKSLASH;
        } else if (c == '\n') {
          this.pop();
          this.epsilon(HighlightAsm.NORMAL);
        } else {
          this.append(c);
          this.state = HighlightAsm.IMMEDIATE_QUOTE_FINISH;
        }
        break;

      case HighlightAsm.IMMEDIATE_QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightAsm.IMMEDIATE_QUOTE_FINISH;
        break;

      case HighlightAsm.IMMEDIATE_QUOTE_FINISH:
        if (c == '\'') {
          this.append(c);
          this.state = HighlightAsm.IMMEDIATE;
        } else {
          // yes '" means '"' in bell system five
          this.epsilon(HighlightAsm.IMMEDIATE);
        }
        break;

      case HighlightAsm.BACKSLASH:
        this.append(c);
        this.pop();
        this.state = HighlightAsm.NORMAL;
        break;

      case HighlightAsm.HASH:
        if (isspace(c)) {
          this.push("span", "comment");
          this.append('#');
          this.epsilon(HighlightAsm.COMMENT);
        } else {
          this.word += '#';
          this.state = HighlightAsm.WORD;
        }
        // fallthrough

      case HighlightAsm.WORD:
        if (!isascii(c) || isalnum(c) || c == '$' || c == '_' || c == '-' || c == '.') {
          this.word += c;
        } else {
          if (this.is_first_thing_on_line) {
            if (this.word.length > 1 &&
                this.word[0] == '#' &&
                C_BUILTINS.has(this.word)) {
              this.push("span", "builtin");
              this.append(this.word);
              this.pop();
              this.is_first_thing_on_line = false;
              this.is_preprocessor = true;
            } else if (c == ':') {
              this.push("span", "label");
              this.append(this.word);
              this.pop();
            } else if (this.word == "C" || this.word == "dnl" || this.word == "m4_dnl") {
              this.push("span", "comment");
              this.append(this.word);
              this.word = '';
              this.epsilon(HighlightAsm.COMMENT);
            } else {
              this.push("span", "keyword");
              this.append(this.word);
              this.pop();
              if (!ASM_PREFIXES.has(this.word.toLowerCase()))
                this.is_first_thing_on_line = false;
            }
          } else if (this.is_preprocessor && C_BUILTINS.has(this.word)) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (ASM_QUALIFIERS.has(this.word.toLowerCase())) {
            this.push("span", "qualifier");
            this.append(this.word);
            this.pop();
          } else if (C_CONSTANTS.has(this.word)) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightAsm.NORMAL);
        }
        break;

      case HighlightAsm.REG0:
        if (isalpha(c) || c == '(' || c == ')') {
          this.state = HighlightAsm.REG;
          this.push("span", "register");
          this.append('%');
        } else {
          this.append('%');
          this.epsilon(HighlightAsm.NORMAL);
        }
        // fallthrough

      case HighlightAsm.REG:
        if (isalnum(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightAsm.NORMAL);
        }
        break;

      case HighlightAsm.QUOTE:
        this.append(c);
        if (c == '\'' || c == '\n') {
          this.pop();
          this.state = HighlightAsm.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightAsm.QUOTE_BACKSLASH;
        }
        break;

      case HighlightAsm.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightAsm.QUOTE;
        break;

      case HighlightAsm.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightAsm.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightAsm.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightAsm.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightAsm.DQUOTE;
        break;

      case HighlightAsm.SLASH0:
        if (c == '*') {
          this.append('*');
          this.state = HighlightAsm.SLASH_STAR;
        } else {
          this.epsilon(HighlightAsm.COMMENT);
        }
        break;

      case HighlightAsm.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightAsm.SLASH_SLASH;
        } else if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state = HighlightAsm.SLASH_STAR;
        } else {
          this.append('/');
          this.epsilon(HighlightAsm.NORMAL);
        }
        break;

      case HighlightAsm.SLASH_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightAsm.SLASH_STAR_STAR;
        break;

      case HighlightAsm.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state = HighlightAsm.NORMAL;
        } else if (c != '*') {
          this.state = HighlightAsm.SLASH_STAR;
        }
        break;

      case HighlightAsm.COMMENT:
      case HighlightAsm.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightAsm.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
      if (c != '\n') {
        this.col += 1;
      } else {
        this.col = 0;
        this.is_first_thing_on_line = true;
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightAsm.WORD:
      this.append(this.word);
      this.word = '';
      break;
    case HighlightAsm.HASH:
      this.append('#');
      break;
    case HighlightAsm.REG0:
      this.append('%');
      break;
    case HighlightAsm.SLASH:
      this.append('/');
      break;
    case HighlightAsm.REG:
    case HighlightAsm.SLASH0:
    case HighlightAsm.COMMENT:
    case HighlightAsm.BACKSLASH:
    case HighlightAsm.SLASH_SLASH:
    case HighlightAsm.SLASH_STAR:
    case HighlightAsm.SLASH_STAR_STAR:
    case HighlightAsm.QUOTE:
    case HighlightAsm.QUOTE_BACKSLASH:
    case HighlightAsm.DQUOTE:
    case HighlightAsm.DQUOTE_BACKSLASH:
    case HighlightAsm.IMMEDIATE:
    case HighlightAsm.IMMEDIATE_QUOTE:
    case HighlightAsm.IMMEDIATE_QUOTE_BACKSLASH:
    case HighlightAsm.IMMEDIATE_QUOTE_FINISH:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightAsm.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['asm'] = HighlightAsm;
Highlighter.REGISTRY['assembler'] = HighlightAsm;
Highlighter.REGISTRY['assembly'] = HighlightAsm;
Highlighter.REGISTRY['fasm'] = HighlightAsm;
Highlighter.REGISTRY['nasm'] = HighlightAsm;
Highlighter.REGISTRY['s'] = HighlightAsm;
Highlighter.REGISTRY['yasm'] = HighlightAsm;
