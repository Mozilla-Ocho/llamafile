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

const COBOL_KEYWORDS = new Set([
  'accept',
  'access',
  'add',
  'address',
  'advancing',
  'after',
  'all',
  'alphabet',
  'alphabetic',
  'alphabetic-lower',
  'alphabetic-upper',
  'alphanumeric',
  'alphanumeric-edited',
  'also',
  'alter',
  'alternate',
  'and',
  'any',
  'apply',
  'are',
  'area',
  'areas',
  'ascending',
  'assign',
  'at',
  'author',
  'basis',
  'before',
  'beginning',
  'binary',
  'blank',
  'block',
  'bottom',
  'by',
  'call',
  'cancel',
  'cbl',
  'cd',
  'cf',
  'ch',
  'character',
  'characters',
  'class',
  'class-id',
  'clock-units',
  'close',
  'cobol',
  'code',
  'code-set',
  'collating',
  'column',
  'com-reg',
  'comma',
  'common',
  'communication',
  'comp',
  'comp-1',
  'comp-2',
  'comp-3',
  'comp-4',
  'comp-5',
  'computational',
  'computational-1',
  'computational-2',
  'computational-3',
  'computational-4',
  'computational-5',
  'compute',
  'configuration',
  'contains',
  'content',
  'continue',
  'control',
  'controls',
  'converting',
  'copy',
  'corr',
  'corresponding',
  'count',
  'currency',
  'data',
  'date-compiled',
  'date-written',
  'day',
  'day-of-week',
  'dbcs',
  'de',
  'debug-contents',
  'debug-item',
  'debug-line',
  'debug-name',
  'debug-sub-1',
  'debug-sub-2',
  'debug-sub-3',
  'debugging',
  'decimal-point',
  'declaratives',
  'delete',
  'delimited',
  'delimiter',
  'depending',
  'descending',
  'destination',
  'detail',
  'display',
  'display-1',
  'divide',
  'division',
  'down',
  'duplicates',
  'dynamic',
  'egcs',
  'egi',
  'eject',
  'else',
  'emi',
  'enable',
  'end',
  'end-add',
  'end-call',
  'end-compute',
  'end-delete',
  'end-divide',
  'end-evaluate',
  'end-if',
  'end-invoke',
  'end-multiply',
  'end-of-page',
  'end-perform',
  'end-read',
  'end-receive',
  'end-return',
  'end-rewrite',
  'end-search',
  'end-start',
  'end-string',
  'end-subtract',
  'end-unstring',
  'end-write',
  'ending',
  'enter',
  'entry',
  'environment',
  'eop',
  'equal',
  'error',
  'esi',
  'evaluate',
  'every',
  'exception',
  'exit',
  'extend',
  'external',
  'false',
  'fd',
  'file',
  'file-control',
  'filler',
  'final',
  'first',
  'footing',
  'for',
  'from',
  'function',
  'generate',
  'giving',
  'global',
  'go',
  'goback',
  'greater',
  'group',
  'heading',
  'high-value',
  'high-values',
  'i-o',
  'i-o-control',
  'id',
  'identification',
  'if',
  'in',
  'index',
  'indexed',
  'indicate',
  'inherits',
  'initial',
  'initialize',
  'initiate',
  'input',
  'input-output',
  'insert',
  'inspect',
  'installation',
  'into',
  'invalid',
  'invoke',
  'is',
  'just',
  'justified ',
  'kanji',
  'key',
  'label',
  'last',
  'leading',
  'left',
  'length',
  'less',
  'limit',
  'limits',
  'linage',
  'linage-counter',
  'line',
  'line-counter',
  'lines',
  'linkage',
  'local-storage',
  'lock',
  'low-value',
  'low-values',
  'memory',
  'merge',
  'message',
  'metaclass',
  'method',
  'method-id',
  'mode',
  'modules',
  'more-labels',
  'move',
  'multiple',
  'multiply',
  'native',
  'native_binary',
  'negative',
  'next',
  'no',
  'not',
  'null',
  'nulls',
  'number',
  'numeric',
  'numeric-edited',
  'object',
  'object-computer',
  'occurs',
  'of',
  'off',
  'omitted',
  'on',
  'open',
  'optional',
  'or',
  'order',
  'organization',
  'other',
  'output',
  'overflow',
  'override',
  'packed-decimal',
  'padding',
  'page',
  'page-counter',
  'password',
  'perform',
  'pf',
  'ph',
  'pic',
  'picture',
  'plus',
  'pointer',
  'position',
  'positive',
  'printing',
  'procedure',
  'procedure-pointer',
  'procedures',
  'proceed',
  'processing',
  'program',
  'program-id',
  'purge',
  'queue',
  'quote',
  'quotes',
  'random',
  'rd',
  'read',
  'ready',
  'receive',
  'record',
  'recording',
  'records',
  'recursive',
  'redefines',
  'reel',
  'reference',
  'references',
  'relative',
  'release',
  'reload',
  'remainder',
  'remarks',
  'removal',
  'renames',
  'replace',
  'replacing',
  'report',
  'reporting',
  'reports',
  'repository',
  'rerun',
  'reserve',
  'reset',
  'return',
  'return-code',
  'returning',
  'reversed',
  'rewind',
  'rewrite',
  'rf',
  'rh',
  'right',
  'rounded',
  'run',
  'same',
  'sd',
  'search',
  'section',
  'security',
  'segment',
  'segment-limit',
  'select',
  'self',
  'send',
  'sentence',
  'separate',
  'sequence',
  'sequential',
  'service',
  'set',
  'shift-in',
  'shift-out',
  'sign',
  'size',
  'skip1',
  'skip2',
  'skip3',
  'sort',
  'sort-control',
  'sort-core-size',
  'sort-file-size',
  'sort-merge',
  'sort-message',
  'sort-mode-size',
  'sort-return',
  'source',
  'source-computer',
  'space',
  'spaces',
  'special-names',
  'standard',
  'standard-1',
  'standard-2',
  'start',
  'status',
  'stop',
  'string',
  'sub-queue-1',
  'sub-queue-2',
  'sub-queue-3',
  'subtract',
  'sum',
  'super',
  'suppress',
  'symbolic',
  'sync',
  'synchronized',
  'table',
  'tally',
  'tallying',
  'tape',
  'terminal',
  'terminate',
  'test',
  'text',
  'than',
  'then',
  'through',
  'thru',
  'time',
  'times',
  'title',
  'to',
  'top',
  'trace',
  'trailing',
  'true',
  'type',
  'unit',
  'unstring',
  'until',
  'up',
  'upon',
  'usage',
  'use',
  'using',
  'value',
  'values',
  'varying',
  'when',
  'when-compiled',
  'with',
  'words',
  'working-storage',
  'write',
  'write-only',
  'zero',
  'zeroes',
  'zeros',
]);

class HighlightCobol extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static COMMENT = 6;

  constructor(delegate) {
    super(delegate);
    this.col = -1;
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];
      ++this.col;
      if (c == '\n')
        this.col = -1;

      if (this.state == HighlightCobol.NORMAL) {
        if (this.col == 6 && c == '*') {
          this.state = HighlightCobol.COMMENT;
          this.push("span", "comment");
        } else if (this.col == 6 && c == '-') {
          this.push("span", "contin");
          this.append(c);
          this.pop();
          continue;
        } else if (this.col < 6 && isdigit(c)) {
          this.push("span", "lineno");
          this.append(c);
          this.pop();
          continue;
        }
      }

      switch (this.state) {

      case HighlightCobol.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '-') {
          this.word += c;
          this.state = HighlightCobol.WORD;
        } else if (c == '!') {
          this.state = HighlightCobol.COMMENT;
          this.push("span", "comment");
          this.append(c);
        } else if (c == '\'') {
          this.state = HighlightCobol.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightCobol.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightCobol.WORD:
        if (!isascii(c) || isalnum(c) || c == '_' || c == '-') {
          this.word += c;
        } else if (c == '.' && this.word[0] == '.') {
          this.word += c;
          if (COBOL_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.state = HighlightCobol.NORMAL;
        } else {
          if (COBOL_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightCobol.NORMAL);
        }
        break;

      case HighlightCobol.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightCobol.NORMAL;
        }
        break;

      case HighlightCobol.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightCobol.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightCobol.QUOTE_BACKSLASH;
        }
        break;

      case HighlightCobol.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightCobol.QUOTE;
        break;

      case HighlightCobol.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightCobol.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightCobol.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightCobol.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightCobol.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightCobol.WORD:
      if (COBOL_KEYWORDS.has(this.word.toLowerCase())) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightCobol.QUOTE:
    case HighlightCobol.QUOTE_BACKSLASH:
    case HighlightCobol.DQUOTE:
    case HighlightCobol.DQUOTE_BACKSLASH:
    case HighlightCobol.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightBasic.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['cobol'] = HighlightCobol;
Highlighter.REGISTRY['cob'] = HighlightCobol;
Highlighter.REGISTRY['cbl'] = HighlightCobol;
