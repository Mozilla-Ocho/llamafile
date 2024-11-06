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

const FORTH_KEYWORDS = new Set([
  '+loop',
  '2literal',
  ':noname',
  ';',
  '?do',
  '?of',
  'again',
  'begin',
  'begin-structure',
  'case',
  'do',
  'does>',
  'else',
  'end-structure',
  'endcase',
  'endof',
  'exit',
  'fliteral',
  'if',
  'immediate',
  'is',
  'leave',
  'literal',
  'loop',
  'of',
  'postpone',
  'repeat',
  'sliteral',
  'then',
  'to',
  'unloop',
  'until',
  'while',
]);

const FORTH_DEFS = new Set([
  '+field',
  '2constant',
  '2value',
  '2variable',
  ':',
  'cfield:',
  'code',
  'constant',
  'create',
  'defer',
  'dffield:',
  'fconstant',
  'ffield:',
  'field:',
  'fvalue',
  'fvariable',
  'sffield:',
  'synonym',
  'value',
  'variable',
]);

class HighlightForth extends Highlighter {

  static NORMAL = 0;
  static SYNTAX = 1;

  constructor(delegate) {
    super(delegate);
    this.is_label = false;
    this.closer = '';
    this.word = '';
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
      switch (this.state) {

      case HighlightForth.NORMAL:
        if (!isspace(c)) {
          this.word += c;
        } else if (this.word != "") {
          if (this.is_label) {
            this.append(this.word);
            this.pop();
            this.is_label = false;
          } else if (this.word == "\\") { // line comment
            this.push("span", "comment");
            this.append(this.word);
            this.state = HighlightForth.SYNTAX;
            this.closer = '\n';
          } else if (this.word == "(" || // inline comment, e.g. ( arg1 arg2 -- res1 )
                     this.word == ".(") { // printed comment, e.g. .( compiling... )
            this.push("span", "comment");
            this.append(this.word);
            this.state = HighlightForth.SYNTAX;
            this.closer = ')';
          } else if (this.word == ".\"" || // string
                     this.word == "s\"" || // stack string
                     this.word == "S\"" || // stack string
                     this.word == "c\"" || // counted string
                     this.word == "C\"") { // counted string
            this.push("span", "string");
            this.append(this.word);
            this.state = HighlightForth.SYNTAX;
            this.closer = '"';
          } else if (FORTH_DEFS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.push("span", "def");
            this.is_label = true;
          } else if (FORTH_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.append(c);
          break;
        } else {
          this.append(c);
        }
        break;

      case HighlightForth.SYNTAX:
        this.append(c);
        if (c == this.closer) {
          this.pop();
          this.state = HighlightForth.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightForth.NORMAL:
      if (this.is_label) {
        this.append(this.word);
      } else if (FORTH_KEYWORDS.has(this.word.toLowerCase()) ||
                 FORTH_DEFS.has(this.word.toLowerCase())) {
        this.push("span", "keyword");
        this.append(this.word);
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    default:
      break;
    }
    while (this.pushed)
      this.pop();
    this.is_label = false;
    this.state = HighlightForth.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['forth'] = HighlightForth;
Highlighter.REGISTRY['fth'] = HighlightForth;
Highlighter.REGISTRY['frt'] = HighlightForth;
Highlighter.REGISTRY['4th'] = HighlightForth;
Highlighter.REGISTRY['fs'] = HighlightForth;
