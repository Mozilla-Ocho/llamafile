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

const ADA_CONSTANTS = new Set([
  'true',
  'false',
]);

const ADA_KEYWORDS = new Set([
  'abort',
  'abs',
  'abstract',
  'accept',
  'access',
  'aliased',
  'all',
  'and',
  'array',
  'at',
  'begin',
  'body',
  'case',
  'constant',
  'declare',
  'delay',
  'delta',
  'digits',
  'do',
  'else',
  'elsif',
  'end',
  'entry',
  'exception',
  'exit',
  'for',
  'function',
  'generic',
  'goto',
  'if',
  'in',
  'interface',
  'is',
  'limited',
  'loop',
  'mod',
  'new',
  'not',
  'null',
  'of',
  'or',
  'others',
  'out',
  'overriding',
  'package',
  'parallel',
  'pragma',
  'private',
  'procedure',
  'protected',
  'raise',
  'range',
  'record',
  'rem',
  'renames',
  'requeue',
  'return',
  'reverse',
  'select',
  'separate',
  'some',
  'subtype',
  'synchronized',
  'tagged',
  'task',
  'terminate',
  'then',
  'type',
  'until',
  'use',
  'when',
  'while',
  'with',
  'xor',
]);

class HighlightAda extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static DQUOTE = 3;
  static HYPHEN = 4;
  static COMMENT = 5;

  constructor(delegate) {
    super(delegate);
    this.last = 0;
    this.word = '';
    this.current = 0;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      this.last = this.current;
      const c = input[i];
      this.current = c;

      switch (this.state) {
      case HighlightAda.NORMAL:
        if (!isascii(c) || isalpha(c)) {
          this.epsilon(HighlightAda.WORD);
        } else if (c === '-') {
          this.state = HighlightAda.HYPHEN;
        } else if (c === '\'' && this.last !== ')') {
          this.push('span', 'string');
          this.append(c);
          this.state = HighlightAda.QUOTE;
        } else if (c === '"') {
          this.push('span', 'string');
          this.append(c);
          this.state = HighlightAda.DQUOTE;
        } else {
          this.append(c);
        }
        break;

      case HighlightAda.WORD:
        if (!isascii(c) || isalnum(c) || c === '_' || c === '\'') {
          this.word += c;
        } else {
          if (ADA_KEYWORDS.has(this.word.toLowerCase())) {
            this.push('span', 'keyword');
            this.append(this.word);
            this.pop();
          } else if (ADA_CONSTANTS.has(this.word.toLowerCase())) {
            this.push('span', 'constant');
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightAda.NORMAL);
        }
        break;

      case HighlightAda.HYPHEN:
        if (c === '-') {
          this.push('span', 'comment');
          this.append('--');
          this.state = HighlightAda.COMMENT;
        } else {
          this.append('-');
          this.epsilon(HighlightAda.NORMAL);
        }
        break;

      case HighlightAda.COMMENT:
        this.append(c);
        if (c === '\n') {
          this.pop();
          this.state = HighlightAda.NORMAL;
        }
        break;

      case HighlightAda.QUOTE:
        this.append(c);
        if (c === '\'') {
          this.pop();
          this.state = HighlightAda.NORMAL;
        }
        break;

      case HighlightAda.DQUOTE:
        this.append(c);
        if (c === '"') {
          this.pop();
          this.state = HighlightAda.NORMAL;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightAda.WORD:
      if (ADA_KEYWORDS.has(this.word.toLowerCase())) {
        this.push('span', 'keyword');
        this.append(this.word);
        this.pop();
      } else if (ADA_CONSTANTS.has(this.word.toLowerCase())) {
        this.push('span', 'constant');
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightAda.HYPHEN:
      this.append('-');
      break;
    case HighlightAda.QUOTE:
    case HighlightAda.DQUOTE:
    case HighlightAda.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightAda.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['ada'] = HighlightAda;
Highlighter.REGISTRY['adb'] = HighlightAda;
