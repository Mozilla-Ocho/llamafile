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

class Highlight {

  constructor() {
  }

  feed(text) {
    throw new Error('not implemented');
  }

  push(className) {
    throw new Error('not implemented');
  }

  pop() {
    throw new Error('not implemented');
  }

  flush() {
    throw new Error('not implemented');
  }
}

class HighlightDom extends Highlight {

  constructor(containerElement) {
    super();
    this.currentElement = containerElement;
    this.containerElement = containerElement;
    this.lastElement = containerElement;
    this.text = '';
  }

  feed(s) {
    for (let i = 0; i < s.length; ++i) {
      this.text += s[i];
      if (isspace(s[i]) || this.text.length > 50) {
        this.flushText();
      }
    }
  }

  push(tagName, className) {
    this.flushText();
    const elem = document.createElement(tagName);
    if (className)
      elem.className = className;
    this.currentElement.appendChild(elem);
    this.currentElement = elem;
    this.lastElement = elem;
    return elem;
  }

  pop() {
    if (this.currentElement == this.containerElement) {
      throw Error('bad pop');
    }
    this.flushText();
    this.currentElement = this.currentElement.parentNode;
  }

  flush() {
    this.flushText();
    while (this.currentElement != this.containerElement) {
      this.currentElement = this.currentElement.parentNode;
    }
  }

  flushText() {
    if (this.text) {
      this.currentElement.appendChild(document.createTextNode(this.text));
      this.lastElement = this.currentElement;
      this.text = '';
    }
  }
}

class Highlighter extends Highlight {

  static REGISTRY = {};

  static create(lang, delegate) {
    let clazz = Highlighter.REGISTRY[lang.toLowerCase()];
    if (clazz) {
      return new clazz(delegate);
    } else {
      return null;
    }
  }

  constructor(delegate) {
    super();
    this.delegate = delegate;
    this.state = 0;
  }

  push(tagName, className) {
    return this.delegate.push(tagName, className);
  }

  pop() {
    this.delegate.pop();
  }

  epsilon(t) {
    this.state = t;
    this.delta = 0;
  }

  append(s) {
    this.delegate.feed(s);
  }
}
