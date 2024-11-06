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

const FORTRAN_KEYWORDS = new Set([
  '.and.',
  '.eq.',
  '.eqv.',
  '.false.',
  '.ge.',
  '.gt.',
  '.le.',
  '.lt.',
  '.ne.',
  '.neqv.',
  '.not.',
  '.or.',
  '.true.',
  'abstract',
  'all',
  'allocatable',
  'allocate',
  'assign',
  'associate',
  'asynchronous',
  'backspace',
  'bind',
  'block',
  'call',
  'case',
  'class',
  'close',
  'codimension',
  'common',
  'concurrent',
  'contains',
  'contiguous',
  'continue',
  'critical',
  'cycle',
  'data',
  'deallocate',
  'deferred',
  'dimension',
  'do',
  'elemental',
  'else',
  'elseif',
  'elsewhere',
  'end',
  'enddo',
  'endfile',
  'endif',
  'entry',
  'enum',
  'enumerator',
  'equivalence',
  'error',
  'exit',
  'extends',
  'external',
  'final',
  'flush',
  'forall',
  'format',
  'function',
  'generic',
  'go',
  'goto',
  'if',
  'images',
  'implicit',
  'import',
  'impure',
  'include',
  'inquire',
  'intent',
  'interface',
  'intrinsic',
  'lock',
  'memory',
  'module',
  'namelist',
  'non_overridable',
  'non_recursive',
  'nopass',
  'nullify',
  'only',
  'open',
  'operator',
  'optional',
  'parameter',
  'pass',
  'pause',
  'pointer',
  'print',
  'private',
  'procedure',
  'program',
  'protected',
  'public',
  'pure',
  'rank',
  'read',
  'recursive',
  'result',
  'return',
  'rewind',
  'rewrite',
  'save',
  'select',
  'sequence',
  'stop',
  'submodule',
  'subroutine',
  'sync',
  'target',
  'then',
  'to',
  'unlock',
  'use',
  'value',
  'volatile',
  'wait',
  'where',
  'while',
  'write',
]);

const FORTRAN_BUILTINS = new Set([
  'abs',
  'acos',
  'aimag',
  'aint',
  'alog',
  'alog10',
  'amax0',
  'amax1',
  'amin0',
  'amin1',
  'amod',
  'anint',
  'asin',
  'atan',
  'atan2',
  'cabs',
  'ccos',
  'cexp',
  'char',
  'clog',
  'cmplx',
  'conjg',
  'cos',
  'cosh',
  'csin',
  'csqrt',
  'dabs',
  'dacos',
  'dasin',
  'datan',
  'datan2',
  'dble',
  'dcos',
  'dcosh',
  'ddim',
  'dexp',
  'dim',
  'dint',
  'dlog',
  'dlog10',
  'dmax1',
  'dmin1',
  'dmod',
  'dnint',
  'dprod',
  'dsign',
  'dsin',
  'dsinh',
  'dsqrt',
  'dtan',
  'dtanh',
  'exp',
  'float',
  'iabs',
  'ichar',
  'idim',
  'idint',
  'idnint',
  'ifix',
  'index',
  'int',
  'isign',
  'len',
  'lge',
  'lgt',
  'lle',
  'llt',
  'log',
  'log10',
  'max',
  'max0',
  'max1',
  'min',
  'min0',
  'min1',
  'mod',
  'nint',
  'real',
  'sign',
  'sin',
  'sinh',
  'sngl',
  'sqrt',
  'tan',
  'tanh',
]);

const FORTRAN_TYPES = new Set([
  'allocatable',
  'byte',
  'character',
  'common',
  'complex',
  'data',
  'dimension',
  'double',
  'integer',
  'intrinsic',
  'logical',
  'map',
  'none',
  'parameter',
  'pointer',
  'precision',
  'real',
  'record',
  'save',
  'sequence',
  'structure',
  'target',
  'type',
]);

class HighlightFortran extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static QUOTE_BACKSLASH = 3;
  static DQUOTE = 4;
  static DQUOTE_BACKSLASH = 5;
  static COMMENT = 6;

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.col = -1;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];

      ++this.col;
      if (c == '\n')
        this.col = -1;

      if (this.state == HighlightFortran.NORMAL) {
        if (this.col == 0 && (c == '*' || c == 'c' || c == 'C')) {
          this.state = HighlightFortran.COMMENT;
          this.push("span", "comment");
        } else if (this.col == 5 && c != ' ') {
          this.push("span", "contin");
          this.append(c);
          this.pop();
          continue;
        } else if (this.col <= 4 && isdigit(c)) {
          this.push("span", "label");
          this.append(c);
          this.pop();
          continue;
        }
      }

      switch (this.state) {

      case HighlightFortran.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_' || c == '.') {
          this.word += c;
          this.state = HighlightFortran.WORD;
        } else if (c == '!') {
          this.state = HighlightFortran.COMMENT;
          this.push("span", "comment");
          this.append(c);
        } else if (c == '\'') {
          this.state = HighlightFortran.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightFortran.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightFortran.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else if (c == '.' && this.word[0] == '.') {
          this.word += c;
          if (FORTRAN_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.state = HighlightFortran.NORMAL;
        } else {
          if (FORTRAN_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (FORTRAN_TYPES.has(this.word.toLowerCase())) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (FORTRAN_BUILTINS.has(this.word.toLowerCase())) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightFortran.NORMAL);
        }
        break;

      case HighlightFortran.COMMENT:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightFortran.NORMAL;
        }
        break;

      case HighlightFortran.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightFortran.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightFortran.QUOTE_BACKSLASH;
        }
        break;

      case HighlightFortran.QUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightFortran.QUOTE;
        break;

      case HighlightFortran.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightFortran.NORMAL;
        } else if (c == '\\') {
          this.state = HighlightFortran.DQUOTE_BACKSLASH;
        }
        break;

      case HighlightFortran.DQUOTE_BACKSLASH:
        this.append(c);
        this.state = HighlightFortran.DQUOTE;
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightFortran.WORD:
      if (FORTRAN_KEYWORDS.has(this.word.toLowerCase())) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (FORTRAN_TYPES.has(this.word.toLowerCase())) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (FORTRAN_BUILTINS.has(this.word.toLowerCase())) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightFortran.QUOTE:
    case HighlightFortran.QUOTE_BACKSLASH:
    case HighlightFortran.DQUOTE:
    case HighlightFortran.DQUOTE_BACKSLASH:
    case HighlightFortran.COMMENT:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightFortran.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['fortran'] = HighlightFortran;
Highlighter.REGISTRY['f'] = HighlightFortran;
