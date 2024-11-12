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

class HighlightHtml extends Highlighter {

  static NORMAL = 0;
  static TAG = 1;
  static TAG2 = 2;
  static TAG_EXCLAIM = 3;
  static TAG_EXCLAIM_HYPHEN = 4;
  static KEY = 5;
  static VAL = 6;
  static QUOTE = 7;
  static DQUOTE = 8;
  static COMMENT = 9;
  static COMMENT_HYPHEN = 10;
  static COMMENT_HYPHEN_HYPHEN = 11;
  static RELAY = 12;
  static TAG_QUESTION = 13;
  static TAG_QUESTION_P = 14;
  static TAG_QUESTION_P_H = 15;
  static ENTITY = 16;

  constructor(delegate) {
    super(delegate);
    this.i = 0;
    this.name = '';
    this.closer = '';
    this.pending = '';
    this.highlighter = null;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      switch (this.state) {

      case HighlightHtml.NORMAL:
        if (c == '<') {
          this.state = HighlightHtml.TAG;
          this.name = '';
        } else if (c == '&') {
          this.state = HighlightHtml.ENTITY;
          this.push("span", "entity");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightHtml.ENTITY:
        this.append(c);
        if (c == ';') {
          this.pop();
          this.state = HighlightHtml.NORMAL;
        }
        break;

      case HighlightHtml.TAG:
        if (c == '!') {
          this.state = HighlightHtml.TAG_EXCLAIM;
        } else if (c == '?') {
          this.state = HighlightHtml.TAG_QUESTION;
        } else if (c == '>' || isspace(c)) {
          this.append('<');
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        } else {
          this.append('<');
          this.push("span", "tag");
          this.append(c);
          this.name += c.toLowerCase();
          this.state = HighlightHtml.TAG2;
        }
        break;

      case HighlightHtml.TAG2:
        if (c == '>') {
          this.pop();
          this.append(c);
          this.ontag();
        } else if (isspace(c)) {
          this.append(c);
          this.state = HighlightHtml.KEY;
          this.pop();
          this.push("span", "attrib");
        } else {
          this.append(c);
          this.name += c.toLowerCase();
        }
        break;

      case HighlightHtml.TAG_EXCLAIM:
        if (c == '-') {
          this.state = HighlightHtml.TAG_EXCLAIM_HYPHEN;
        } else {
          this.append("<!");
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        }
        break;

      case HighlightHtml.TAG_EXCLAIM_HYPHEN:
        if (c == '-') {
          this.push("span", "comment");
          this.append("<!--");
          this.state = HighlightHtml.COMMENT;
        } else {
          this.append("<!-");
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        }
        break;

      case HighlightHtml.COMMENT:
        this.append(c);
        if (c == '-')
          this.state = HighlightHtml.COMMENT_HYPHEN;
        break;

      case HighlightHtml.COMMENT_HYPHEN:
        this.append(c);
        if (c == '-') {
          this.state = HighlightHtml.COMMENT_HYPHEN_HYPHEN;
        } else {
          this.state = HighlightHtml.COMMENT;
        }
        break;

      case HighlightHtml.COMMENT_HYPHEN_HYPHEN:
        this.append(c);
        if (c == '>') {
          this.pop();
          this.state = HighlightHtml.NORMAL;
        } else if (c != '-') {
          this.state = HighlightHtml.COMMENT;
        }
        break;

      case HighlightHtml.KEY:
        if (c == '=') {
          this.pop();
          this.append(c);
          this.state = HighlightHtml.VAL;
        } else if (c == '>') {
          this.pop();
          this.append(c);
          this.ontag();
        } else {
          this.append(c);
        }
        break;

      case HighlightHtml.VAL:
        if (isspace(c)) {
          this.append(c);
          this.state = HighlightHtml.KEY;
          this.push("span", "attrib");
        } else if (c == '\'') {
          this.push("span", "string");
          this.append(c);
          this.state = HighlightHtml.QUOTE;
        } else if (c == '"') {
          this.push("span", "string");
          this.append(c);
          this.state = HighlightHtml.DQUOTE;
        } else if (c == '>') {
          this.append(c);
          this.ontag();
        } else {
          this.append(c);
        }
        break;

      case HighlightHtml.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightHtml.VAL;
        }
        break;

      case HighlightHtml.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightHtml.VAL;
        }
        break;

      case HighlightHtml.RELAY:
        if (this.closer[this.i] == c.toLowerCase()) {
          this.pending += c;
          if (++this.i == this.closer.length) {
            this.highlighter.flush();
            delete this.highlighter;
            this.highlighter = null;
            this.pop();
            if (this.closer == "</style>") {
              this.pending = "</style>";
            } else if (this.closer == "</script>") {
              this.pending = "</script>";
            }
            this.append(this.pending);
            this.state = HighlightHtml.NORMAL;
            this.i = 0;
          }
        } else {
          this.pending += c;
          this.highlighter.feed(this.pending);
          this.pending = '';
          this.i = 0;
        }
        break;

      case HighlightHtml.TAG_QUESTION:
        if (c == 'p') {
          this.state = HighlightHtml.TAG_QUESTION_P;
        } else if (c == '=') {
          this.push("span", "tag");
          this.append("<?=");
          this.pop();
          this.pending = '';
          this.closer = "?>";
          let hdom = new HighlightDom(this.push('div', ''));
          this.highlighter = new HighlightPhp(hdom);
          this.state = HighlightHtml.RELAY;
          this.i = 0;
        } else {
          this.append("<?");
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        }
        break;

      case HighlightHtml.TAG_QUESTION_P:
        if (c == 'h') {
          this.state = HighlightHtml.TAG_QUESTION_P_H;
        } else {
          this.append("<?p");
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        }
        break;

      case HighlightHtml.TAG_QUESTION_P_H:
        if (c == 'p') {
          this.append("<");
          this.push("span", "tag");
          this.append("?php");
          this.pop();
          this.pending = '';
          this.closer = "?>";
          let hdom = new HighlightDom(this.push('div', ''));
          this.highlighter = new HighlightPhp(hdom);
          this.state = HighlightHtml.RELAY;
          this.i = 0;
        } else {
          this.append("<?ph");
          this.append(c);
          this.state = HighlightHtml.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightHtml.TAG:
      this.append('<');
      break;
    case HighlightHtml.TAG_EXCLAIM:
      this.append("<!");
      break;
    case HighlightHtml.TAG_EXCLAIM_HYPHEN:
      this.append("<!-");
      break;
    case HighlightHtml.TAG_QUESTION:
      this.append("<?");
      break;
    case HighlightHtml.TAG_QUESTION_P:
      this.append("<?p");
      break;
    case HighlightHtml.TAG_QUESTION_P_H:
      this.append("<?ph");
      break;
    case HighlightHtml.COMMENT_HYPHEN_HYPHEN:
    case HighlightHtml.COMMENT_HYPHEN:
    case HighlightHtml.COMMENT:
    case HighlightHtml.ENTITY:
    case HighlightHtml.DQUOTE:
    case HighlightHtml.QUOTE:
    case HighlightHtml.TAG2:
    case HighlightHtml.KEY:
      this.pop();
      break;
    case HighlightHtml.RELAY:
      this.highlighter.feed(this.pending);
      this.highlighter.flush();
      delete this.highlighter;
      this.highlighter = null;
      this.pop();
      break;
    default:
      break;
    }
    this.pending = '';
    this.closer = '';
    this.name = '';
    this.state = HighlightHtml.NORMAL;
    this.delegate.flush();
  }

  ontag() {
    this.state = HighlightHtml.NORMAL;
    if (this.name == "script") {
      this.pending = '';
      this.closer = "</script>";
      let hdom = new HighlightDom(this.push('div', ''));
      this.highlighter = new HighlightJs(hdom);
      this.state = HighlightHtml.RELAY;
      this.i = 0;
    } else if (this.name == "style") {
      this.pending = '';
      this.closer = "</style>";
      let hdom = new HighlightDom(this.push('div', ''));
      this.highlighter = new HighlightCss(hdom);
      this.state = HighlightHtml.RELAY;
      this.i = 0;
    }
    this.delta = 1;
  }
}

Highlighter.REGISTRY['html'] = HighlightHtml;
Highlighter.REGISTRY['htm'] = HighlightHtml;
Highlighter.REGISTRY['xhtml'] = HighlightHtml;
Highlighter.REGISTRY['xml'] = HighlightHtml;
Highlighter.REGISTRY['php'] = HighlightHtml;
