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

class HighlightCss extends Highlighter {

  static NORMAL = 0;
  static SELECTOR = 1;
  static PROPERTY = 2;
  static VALUE = 3;
  static QUOTE = 4;
  static QUOTE_BACKSLASH = 5;
  static DQUOTE = 6;
  static DQUOTE_BACKSLASH = 7;
  static SLASH = 8;
  static SLASH_STAR = 9;
  static SLASH_STAR_STAR = 10;

  constructor(delegate) {
    super(delegate);
    this.pushed = 0;
  }

  push(tagName, className) {
    while (this.pushed)
      this.pop();
    ++this.pushed;
    return this.delegate.push(tagName, className);
  }

  pop() {
    if (!this.pushed)
      throw new Error('bad pop');
    --this.pushed;
    this.delegate.pop();
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];

      switch (this.state & 255) {

      case HighlightCss.NORMAL:
        this.push("span", "selector");
        this.state = HighlightCss.SELECTOR;
        // fallthrough

      case HighlightCss.SELECTOR:
        if (c == '{') {
          this.state = HighlightCss.PROPERTY;
          this.pop();
          this.append(c);
          this.push("span", "property");
        } else if (c == ',') {
          this.pop();
          this.append(c);
          this.push("span", "selector");
        } else if (c == '/') {
          this.state = HighlightCss.SELECTOR << 8 | HighlightCss.SLASH;
        } else if (c == '\'') {
          this.state = HighlightCss.SELECTOR << 8 | HighlightCss.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightCss.SELECTOR << 8 | HighlightCss.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightCss.PROPERTY:
        if (c == '/') {
          this.state = HighlightCss.PROPERTY << 8 | HighlightCss.SLASH;
        } else if (c == '\'') {
          this.state = HighlightCss.PROPERTY << 8 | HighlightCss.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightCss.PROPERTY << 8 | HighlightCss.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == ':') {
          this.state = HighlightCss.VALUE;
          this.pop();
          this.append(c);
        } else if (c == '}') {
          this.state = HighlightCss.SELECTOR;
          this.pop();
          this.append(c);
          this.push("span", "selector");
        } else {
          this.append(c);
        }
        break;

      case HighlightCss.VALUE:
        if (c == '/') {
          this.state = HighlightCss.VALUE << 8 | HighlightCss.SLASH;
        } else if (c == '\'') {
          this.state = HighlightCss.VALUE << 8 | HighlightCss.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightCss.VALUE << 8 | HighlightCss.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == ';') {
          this.state = HighlightCss.PROPERTY;
          this.append(c);
          this.push("span", "property");
        } else if (c == '}') {
          this.state = HighlightCss.SELECTOR;
          this.append(c);
          this.push("span", "selector");
        } else {
          this.append(c);
        }
        break;

      case HighlightCss.SLASH:
        if (c == '*') {
          this.push("span", "comment");
          this.append("/*");
          this.state &= -256;
          this.state |= HighlightCss.SLASH_STAR;
        } else {
          this.append('/');
          this.state >>= 8;
          this.epsilon(this.state);
        }
        break;

      case HighlightCss.SLASH_STAR:
        this.append(c);
        if (c == '*') {
          this.state &= -256;
          this.state |= HighlightCss.SLASH_STAR_STAR;
        }
        break;

      case HighlightCss.SLASH_STAR_STAR:
        this.append(c);
        if (c == '/') {
          this.pop();
          this.state >>= 8;
          if (this.state == HighlightCss.SELECTOR)
            this.push("span", "selector");
          if (this.state == HighlightCss.PROPERTY)
            this.push("span", "property");
        } else if (c != '*') {
          this.state &= -256;
          this.state |= HighlightCss.SLASH_STAR;
        }
        break;

      case HighlightCss.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state >>= 8;
          if (this.state == HighlightCss.SELECTOR)
            this.push("span", "selector");
          if (this.state == HighlightCss.PROPERTY)
            this.push("span", "property");
        } else if (c == '\\') {
          this.state &= -256;
          this.state |= HighlightCss.QUOTE_BACKSLASH;
        }
        break;

      case HighlightCss.QUOTE_BACKSLASH:
        this.append(c);
        this.state &= -256;
        this.state |= HighlightCss.QUOTE;
        break;

      case HighlightCss.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state >>= 8;
          if (this.state == HighlightCss.SELECTOR)
            this.push("span", "selector");
          if (this.state == HighlightCss.PROPERTY)
            this.push("span", "property");
        } else if (c == '\\') {
          this.state &= -256;
          this.state |= HighlightCss.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightCss.DQUOTE_BACKSLASH:
        this.append(c);
        this.state &= -256;
        this.state |= HighlightCss.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state & 255) {
    case HighlightCss.SLASH:
      this.append('/');
      break;
    default:
      break;
    }
    while (this.pushed)
      this.pop();
    this.state = HighlightCss.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['css'] = HighlightCss;
