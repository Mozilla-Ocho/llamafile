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

const BASIC_KEYWORDS = new Set([
  '#const',
  '#else',
  '#elseif',
  '#end',
  '#if',
  'addhandler',
  'addressof',
  'alias',
  'and',
  'andalso',
  'as',
  'attribute',
  'byref',
  'byval',
  'cbool',
  'cbyte',
  'cchar',
  'cdate',
  'cdbl',
  'cdec',
  'cint',
  'clng',
  'cobj',
  'csbyte',
  'cshort',
  'csng',
  'cstr',
  'ctype',
  'cuint',
  'culng',
  'cushort',
  'call',
  'case',
  'catch',
  'class',
  'const',
  'constraint',
  'continue',
  'declare',
  'default',
  'delegate',
  'dim',
  'directcast',
  'do',
  'each',
  'else',
  'elseif',
  'end',
  'endif',
  'enum',
  'erase',
  'error',
  'event',
  'exit',
  'finally',
  'for',
  'friend',
  'function',
  'get',
  'gettype',
  'getxmlnamespace',
  'global',
  'gosub',
  'goto',
  'handles',
  'if',
  'implements',
  'imports',
  'in',
  'inherits',
  'interface',
  'is',
  'isnot',
  'ltrim',
  'let',
  'lib',
  'like',
  'loop',
  'mirr',
  'macid',
  'macscript',
  'me',
  'mod',
  'module',
  'mustinherit',
  'mustoverride',
  'mybase',
  'myclass',
  'nper',
  'nameof',
  'namespace',
  'narrowing',
  'new',
  'next',
  'not',
  'notinheritable',
  'notoverridable',
  'of',
  'on',
  'operator',
  'option',
  'optional',
  'or',
  'orelse',
  'out',
  'overloads',
  'overridable',
  'overrides',
  'paramarray',
  'partial',
  'private',
  'property',
  'protected',
  'public',
  'rem',
  'rtrim',
  'raiseevent',
  'redim',
  'readonly',
  'removehandler',
  'resume',
  'return',
  'select',
  'set',
  'shadows',
  'shared',
  'statement',
  'static',
  'step',
  'stop',
  'structure',
  'sub',
  'synclock',
  'then',
  'throw',
  'to',
  'trim',
  'try',
  'trycast',
  'type',
  'typeof',
  'using',
  'wend',
  'when',
  'while',
  'widening',
  'with',
  'withevents',
  'writeonly',
  'xor',
]);

const BASIC_BUILTINS = new Set([
  'abs',
  'array',
  'asc',
  'atan',
  'atn',
  'callbyname',
  'choose',
  'chr',
  'command',
  'cos',
  'curdir',
  'ddb',
  'date',
  'dateadd',
  'datediff',
  'datepart',
  'dateserial',
  'datevalue',
  'day',
  'dir',
  'doevents',
  'eof',
  'environ',
  'exp',
  'fileattr',
  'fileclose',
  'filecopy',
  'filedatetime',
  'fileget',
  'filegetobject',
  'filelen',
  'fileopen',
  'fileput',
  'fileputobject',
  'filewidth',
  'fix',
  'format',
  'freefile',
  'getattr',
  'getobject',
  'hex',
  'instr',
  'instrrev',
  'input',
  'inputbox',
  'inputstring',
  'int',
  'isarray',
  'isdate',
  'isempty',
  'iserror',
  'ismissing',
  'isnull',
  'isobject',
  'join',
  'kill',
  'lbound',
  'lcase',
  'lof',
  'left',
  'len',
  'lineinput',
  'loc',
  'lock',
  'log',
  'mid',
  'month',
  'monthname',
  'msgbox',
  'now',
  'oct',
  'ppmt',
  'pv',
  'partition',
  'pmt',
  'print',
  'qbcolor',
  'randomize',
  'read',
  'replace',
  'reset',
  'right',
  'rnd',
  'round',
  'spc',
  'second',
  'seek',
  'setattr',
  'shell',
  'sign',
  'sin',
  'split',
  'sqr',
  'sqrt',
  'str',
  'strcomp',
  'strconv',
  'strreverse',
  'switch',
  'tab',
  'tan',
  'time',
  'timer',
  'ubound',
  'ucase',
  'unlock',
  'val',
  'vartype',
  'weekday',
  'weekdayname',
  'write',
  'writeline',
  'year',
]);

const BASIC_CONSTANTS = new Set([
  'false',
  'nothing',
  'true',
  'vbabort',
  'vbabortretryignore',
  'vbapplicationmodal',
  'vbarchive',
  'vbarray',
  'vbback',
  'vbbinarycompare',
  'vbboolean',
  'vbbyte',
  'vbcancel',
  'vbcr',
  'vbcrlf',
  'vbcritical',
  'vbcurrency',
  'vbdate',
  'vbdecimal',
  'vbdefaultbutton1',
  'vbdefaultbutton2',
  'vbdefaultbutton3',
  'vbdirectory',
  'vbdouble',
  'vbempty',
  'vbexclamation',
  'vbfalse',
  'vbfirstfourdays',
  'vbfirstfullweek',
  'vbfirstjan1',
  'vbformfeed',
  'vbfriday',
  'vbgeneraldate',
  'vbget',
  'vbhidden',
  'vbhide',
  'vbhiragana',
  'vbignore',
  'vbinformation',
  'vbinteger',
  'vbkatakana',
  'vblet',
  'vblf',
  'vblinguisticcasing',
  'vblong',
  'vblongdate',
  'vblongtime',
  'vblowercase',
  'vbmaximizedfocus',
  'vbmethod',
  'vbminimizedfocus',
  'vbminimizednofocus',
  'vbmonday',
  'vbmsgboxhelp',
  'vbmsgboxright',
  'vbmsgboxrtlreading',
  'vbmsgboxsetforeground',
  'vbnarrow',
  'vbnewline',
  'vbno',
  'vbnormal',
  'vbnormalfocus',
  'vbnormalnofocus',
  'vbnull',
  'vbnullchar',
  'vbnullstring',
  'vbok',
  'vbokcancel',
  'vbokonly',
  'vbobject',
  'vbobjecterror',
  'vbpropercase',
  'vbquestion',
  'vbreadonly',
  'vbretry',
  'vbretrycancel',
  'vbsaturday',
  'vbset',
  'vbshortdate',
  'vbshorttime',
  'vbsimplifiedchinese',
  'vbsingle',
  'vbstring',
  'vbsunday',
  'vbsystem',
  'vbsystemmodal',
  'vbtab',
  'vbtextcompare',
  'vbthursday',
  'vbtraditionalchinese',
  'vbtrue',
  'vbtuesday',
  'vbuppercase',
  'vbusedefault',
  'vbusesystem',
  'vbusesystemdayofweek',
  'vbuserdefinedtype',
  'vbvariant',
  'vbverticaltab',
  'vbvolume',
  'vbwednesday',
  'vbwide',
  'vbyes',
  'vbyesno',
  'vbyesnocancel',
]);

const BASIC_TYPES = new Set([
  'boolean',
  'byte',
  'char',
  'date',
  'decimal',
  'double',
  'integer',
  'long',
  'object',
  'sbyte',
  'short',
  'single',
  'string',
  'uinteger',
  'ulong',
  'ushort',
  'variant',
]);

class HighlightBasic extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static LINENO = 2;
  static DQUOTE = 3;
  static COMMENT = 4;
  static DIRECTIVE = 5;

  constructor(delegate) {
    super(delegate);
    this.word = '';
    this.is_bol = true;
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      this.last = this.current;
      const c = input[i];
      this.current = c;

      switch (this.state) {
      case HighlightBasic.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.state = HighlightBasic.WORD;
          this.word += c;
        } else if (c == '\'') {
          this.state = HighlightBasic.COMMENT;
          this.push("span", "comment");
          this.append('\'');
        } else if (c == '"') {
          this.state = HighlightBasic.DQUOTE;
          this.push("span", "string");
          this.append('"');
        } else if (c == '#' && this.is_bol) {
          this.state = HighlightBasic.DIRECTIVE;
          this.push("span", "directive");
          this.append('#');
        } else if (isdigit(c) && this.is_bol) {
          this.push("span", "lineno");
          this.append(c);
          this.state = HighlightBasic.LINENO;
        } else {
          this.append(c);
        }
        break;

      case HighlightBasic.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (BASIC_KEYWORDS.has(this.word.toLowerCase())) {
            if (this.word.toLowerCase() == "rem") {
              this.push("span", "comment");
              this.append(this.word);
              this.epsilon(HighlightBasic.COMMENT);
              this.word = '';
            } else {
              this.push("span", "keyword");
              this.append(this.word);
              this.pop();
            }
          } else if (BASIC_TYPES.has(this.word.toLowerCase())) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (BASIC_BUILTINS.has(this.word.toLowerCase())) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else if (BASIC_CONSTANTS.has(this.word.toLowerCase())) {
            this.push("span", "constant");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightBasic.NORMAL);
        }
        break;

      case HighlightBasic.DQUOTE:
        this.append(c);
        if (c == '"' || c == '\n') {
          this.pop();
          this.state = HighlightBasic.NORMAL;
        }
        break;

      case HighlightBasic.LINENO:
        if (isdigit(c)) {
          this.append(c);
        } else {
          this.pop();
          this.epsilon(HighlightBasic.NORMAL);
        }
        break;

      case HighlightBasic.COMMENT:
      case HighlightBasic.DIRECTIVE:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightBasic.NORMAL;
        } else {
        }
        break;

      default:
        throw new Error('Invalid state');
      }
      if (this.is_bol) {
        if (!isspace(c))
          this.is_bol = false;
      } else {
        if (c == '\n')
          this.is_bol = true;
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightBasic.WORD:
      if (BASIC_KEYWORDS.has(this.word.toLowerCase())) {
        if (this.word.toLowerCase() == "rem") {
          this.push("span", "keyword");
          this.append(this.word);
          this.pop();
        } else {
          this.push("span", "keyword");
          this.append(this.word);
          this.pop();
        }
      } else if (BASIC_TYPES.has(this.word.toLowerCase())) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (BASIC_BUILTINS.has(this.word.toLowerCase())) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else if (BASIC_CONSTANTS.has(this.word.toLowerCase())) {
        this.push("span", "constant");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightBasic.LINENO:
    case HighlightBasic.DQUOTE:
    case HighlightBasic.COMMENT:
    case HighlightBasic.DIRECTIVE:
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

Highlighter.REGISTRY['basic'] = HighlightBasic;
Highlighter.REGISTRY['vb'] = HighlightBasic;
Highlighter.REGISTRY['vba'] = HighlightBasic;
Highlighter.REGISTRY['vbs'] = HighlightBasic;
Highlighter.REGISTRY['bas'] = HighlightBasic;
Highlighter.REGISTRY['vb.net'] = HighlightBasic;
Highlighter.REGISTRY['qbasic'] = HighlightBasic;
Highlighter.REGISTRY['freebasic'] = HighlightBasic;
