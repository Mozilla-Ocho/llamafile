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

const PASCAL_KEYWORDS = new Set([
  'abstract',
  'alias',
  'and',
  'as',
  'asm',
  'assembler',
  'begin',
  'break',
  'case',
  'cdecl',
  'class',
  'const',
  'constructor',
  'continue',
  'destructor',
  'div',
  'do',
  'downto',
  'dynamic',
  'else',
  'end',
  'except',
  'export',
  'exports',
  'external',
  'finalization',
  'finally',
  'for',
  'function',
  'get',
  'goto',
  'if',
  'implementation',
  'in',
  'inherited',
  'initialization',
  'inline',
  'input',
  'interface',
  'is',
  'label',
  'library',
  'mod',
  'nil',
  'nostackframe',
  'not',
  'of',
  'on',
  'operator',
  'or',
  'output',
  'override',
  'packed',
  'pascal',
  'private',
  'procedure',
  'program',
  'property',
  'protected',
  'public',
  'published',
  'put',
  'raise',
  'record',
  'register',
  'reintroduce',
  'repeat',
  'reset',
  'rewrite',
  'safecall',
  'set',
  'shl',
  'shr',
  'softfloat',
  'stdcall',
  'then',
  'threadvar',
  'to',
  'try',
  'type',
  'unit',
  'until',
  'uses',
  'var',
  'varargs',
  'virtual',
  'while',
  'with',
  'xor',
]);

const PASCAL_BUILTINS = new Set([
  'abort',
  'abs',
  'acquireexceptionobject',
  'activateclassgroup',
  'addexitproc',
  'addterminateproc',
  'adjustlinebreaks',
  'allocatehwnd',
  'allocmem',
  'ansicomparefilename',
  'ansicomparestr',
  'ansicomparetext',
  'ansidequotedstr',
  'ansiextractquotedstr',
  'ansilastchar',
  'ansilowercase',
  'ansilowercasefilename',
  'ansipos',
  'ansiquotedstr',
  'ansisamestr',
  'ansisametext',
  'ansistrcomp',
  'ansistricomp',
  'ansistrlastchar',
  'ansistrlcomp',
  'ansistrlicomp',
  'ansistrlower',
  'ansistrpos',
  'ansistrrscan',
  'ansistrscan',
  'ansistrupper',
  'ansitoutf8',
  'ansiuppercase',
  'ansiuppercasefilename',
  'append',
  'appendstr',
  'arccos',
  'arccosh',
  'arccot',
  'arccoth',
  'arccsc',
  'arccsch',
  'arcsec',
  'arcsech',
  'arcsin',
  'arcsinh',
  'arctan',
  'arctan2',
  'arctanh',
  'assert',
  'assigned',
  'assignfile',
  'assignstr',
  'beep',
  'beginthread',
  'bintohex',
  'blockread',
  'blockwrite',
  'booltostr',
  'bytetocharindex',
  'bytetocharlen',
  'bytetype',
  'callterminateprocs',
  'ceil',
  'changefileext',
  'charlength',
  'chartobyteindex',
  'chartobytelen',
  'chdir',
  'checksynchronize',
  'chr',
  'close',
  'closefile',
  'collectionsequal',
  'comparemem',
  'comparestr',
  'comparetext',
  'comparevalue',
  'comptocurrency',
  'comptodouble',
  'concat',
  'continue',
  'copy',
  'cos',
  'cosecant',
  'cosh',
  'cot',
  'cotan',
  'coth',
  'countgenerations',
  'createdir',
  'createguid',
  'csc',
  'csch',
  'currentyear',
  'currtostr',
  'currtostrf',
  'cycletodeg',
  'cycletograd',
  'cycletorad',
  'date',
  'datetimetofiledate',
  'datetimetostr',
  'datetimetostring',
  'datetimetosystemtime',
  'datetimetotimestamp',
  'datetostr',
  'dayofweek',
  'deallocatehwnd',
  'dec',
  'decodedate',
  'decodedatefully',
  'decodetime',
  'degtocycle',
  'degtograd',
  'degtorad',
  'delete',
  'deletefile',
  'directoryexists',
  'diskfree',
  'disksize',
  'dispose',
  'disposestr',
  'divmod',
  'doubledecliningbalance',
  'doubletocomp',
  'encodedate',
  'encodetime',
  'endthread',
  'ensurerange',
  'enummodules',
  'enumresourcemodules',
  'eof',
  'eoln',
  'equalrect',
  'erase',
  'exceptaddr',
  'exceptionerrormessage',
  'exceptobject',
  'exclude',
  'excludetrailingbackslash',
  'excludetrailingpathdelimiter',
  'exit',
  'exp',
  'expandfilename',
  'expandfilenamecase',
  'expanduncfilename',
  'extractfiledir',
  'extractfiledrive',
  'extractfileext',
  'extractfilename',
  'extractfilepath',
  'extractrelativepath',
  'extractshortpathname',
  'extractstrings',
  'fileage',
  'fileclose',
  'filecreate',
  'filedatetodatetime',
  'fileexists',
  'filegetattr',
  'filegetdate',
  'fileisreadonly',
  'fileopen',
  'filepos',
  'fileread',
  'filesearch',
  'fileseek',
  'filesetattr',
  'filesetdate',
  'filesetreadonly',
  'filesize',
  'filewrite',
  'fillchar',
  'finalize',
  'finalizepackage',
  'findclass',
  'findclasshinstance',
  'findclose',
  'findcmdlineswitch',
  'findfirst',
  'findglobalcomponent',
  'findhinstance',
  'findnext',
  'findresourcehinstance',
  'floattocurr',
  'floattodatetime',
  'floattodecimal',
  'floattostr',
  'floattostrf',
  'floattotext',
  'floattotextfmt',
  'floor',
  'flush',
  'fmtloadstr',
  'fmtstr',
  'forcedirectories',
  'format',
  'formatbuf',
  'formatcurr',
  'formatdatetime',
  'formatfloat',
  'frac',
  'freeandnil',
  'freemem',
  'frexp',
  'futurevalue',
  'get8087cw',
  'getclass',
  'getcurrentdir',
  'getdir',
  'getenvironmentvariable',
  'getexceptionmask',
  'getfileversion',
  'getformatsettings',
  'getlasterror',
  'getlocaleformatsettings',
  'getmem',
  'getmemorymanager',
  'getmodulefilename',
  'getmodulename',
  'getpackagedescription',
  'getpackageinfo',
  'getprecisionmode',
  'getroundmode',
  'gettime',
  'getvariantmanager',
  'gradtocycle',
  'gradtodeg',
  'gradtorad',
  'groupdescendantswith',
  'guidtostring',
  'halt',
  'hextobin',
  'high',
  'hypot',
  'identtoint',
  'inc',
  'incamonth',
  'include',
  'includetrailingbackslash',
  'includetrailingpathdelimiter',
  'incmonth',
  'initialize',
  'initializepackage',
  'initinheritedcomponent',
  'inrange',
  'insert',
  'int',
  'interestpayment',
  'interestrate',
  'interlockeddecrement',
  'interlockedexchange',
  'interlockedexchangeadd',
  'interlockedincrement',
  'internalrateofreturn',
  'intpower',
  'inttohex',
  'inttoident',
  'inttostr',
  'invalidpoint',
  'ioresult',
  'isdelimiter',
  'isequalguid',
  'isinfinite',
  'isleapyear',
  'ismemorymanagerset',
  'isnan',
  'ispathdelimiter',
  'isuniqueglobalcomponentname',
  'isvalidident',
  'isvariantmanagerset',
  'iszero',
  'languages',
  'lastdelimiter',
  'ldexp',
  'linestart',
  'lnxp1',
  'loadpackage',
  'loadstr',
  'log10',
  'log2',
  'logn',
  'low',
  'lowercase',
  'max',
  'maxintvalue',
  'maxvalue',
  'mean',
  'meanandstddev',
  'min',
  'minintvalue',
  'minvalue',
  'mkdir',
  'momentskewkurtosis',
  'move',
  'msecstotimestamp',
  'netpresentvalue',
  'new',
  'newstr',
  'nextcharindex',
  'norm',
  'now',
  'numberofperiods',
  'objectbinarytotext',
  'objectresourcetotext',
  'objecttexttobinary',
  'objecttexttoresource',
  'odd',
  'olestrtostring',
  'olestrtostrvar',
  'ord',
  'outofmemoryerror',
  'paramcount',
  'paramstr',
  'payment',
  'periodpayment',
  'pi',
  'pointsequal',
  'poly',
  'popnstddev',
  'popnvariance',
  'power',
  'pred',
  'presentvalue',
  'pucs4chars',
  'quotedstr',
  'radtocycle',
  'radtodeg',
  'radtograd',
  'raiselastoserror',
  'raiselastwin32error',
  'randg',
  'random',
  'randomize',
  'randomrange',
  'read',
  'readcomponentres',
  'readcomponentresex',
  'readcomponentresfile',
  'readln',
  'reallocmem',
  'rect',
  'registerclass',
  'registerclassalias',
  'registerclasses',
  'registercomponents',
  'registerintegerconsts',
  'registernoicon',
  'registernonactivex',
  'releaseexceptionobject',
  'removedir',
  'rename',
  'renamefile',
  'replacedate',
  'replacetime',
  'reset',
  'rewrite',
  'rmdir',
  'round',
  'roundto',
  'runerror',
  'safeloadlibrary',
  'samefilename',
  'sametext',
  'samevalue',
  'sec',
  'secant',
  'sech',
  'seek',
  'seekeof',
  'seekeoln',
  'set8087cw',
  'setcurrentdir',
  'setexceptionmask',
  'setlength',
  'setlinebreakstyle',
  'setmemorymanager',
  'setprecisionmode',
  'setroundmode',
  'setstring',
  'settextbuf',
  'setvariantmanager',
  'showexception',
  'sign',
  'simpleroundto',
  'sin',
  'sincos',
  'sinh',
  'sizeof',
  'sleep',
  'slice',
  'slndepreciation',
  'smallpoint',
  'sqr',
  'sqrt',
  'startclassgroup',
  'stddev',
  'stralloc',
  'strbufsize',
  'strbytetype',
  'strcat',
  'strcharlength',
  'strcomp',
  'strcopy',
  'strdispose',
  'strecopy',
  'strend',
  'strfmt',
  'stricomp',
  'stringofchar',
  'stringreplace',
  'stringtoguid',
  'stringtoolestr',
  'stringtowidechar',
  'strlcat',
  'strlcomp',
  'strlcopy',
  'strlen',
  'strlfmt',
  'strlicomp',
  'strlower',
  'strmove',
  'strnew',
  'strnextchar',
  'strpas',
  'strpcopy',
  'strplcopy',
  'strpos',
  'strrscan',
  'strscan',
  'strtobool',
  'strtobooldef',
  'strtocurr',
  'strtocurrdef',
  'strtodate',
  'strtodatedef',
  'strtodatetime',
  'strtodatetimedef',
  'strtofloat',
  'strtofloatdef',
  'strtoint',
  'strtoint64',
  'strtoint64def',
  'strtointdef',
  'strtotime',
  'strtotimedef',
  'strupper',
  'succ',
  'sum',
  'sumint',
  'sumofsquares',
  'sumsandsquares',
  'supports',
  'swap',
  'syddepreciation',
  'syserrormessage',
  'systemtimetodatetime',
  'tan',
  'tanh',
  'teststreamformat',
  'texttofloat',
  'timestamptodatetime',
  'timestamptomsecs',
  'timetostr',
  'totalvariance',
  'trim',
  'trimleft',
  'trimright',
  'trunc',
  'truncate',
  'tryencodedate',
  'tryencodetime',
  'tryfloattocurr',
  'tryfloattodatetime',
  'trystrtobool',
  'trystrtocurr',
  'trystrtodate',
  'trystrtodatetime',
  'trystrtofloat',
  'trystrtoint',
  'trystrtoint64',
  'trystrtotime',
  'typeinfo',
  'ucs4stringtowidestring',
  'unicodetoutf8',
  'uniquestring',
  'unloadpackage',
  'unregisterclass',
  'unregisterclasses',
  'unregisterintegerconsts',
  'unregistermoduleclasses',
  'upcase',
  'uppercase',
  'utf8decode',
  'utf8encode',
  'utf8toansi',
  'utf8tounicode',
  'vararrayredim',
  'varclear',
  'variance',
  'widecharlentostring',
  'widecharlentostrvar',
  'widechartostring',
  'widechartostrvar',
  'widecomparestr',
  'widecomparetext',
  'widefmtstr',
  'wideformat',
  'wideformatbuf',
  'widelowercase',
  'widesamestr',
  'widesametext',
  'widestringtoucs4string',
  'wideuppercase',
  'win32check',
  'wraptext',
  'write',
  'writecomponentresfile',
  'writeln',
]);

const PASCAL_TYPES = new Set([
  'ansichar',
  'ansistring',
  'array',
  'bool',
  'boolean',
  'byte',
  'bytebool',
  'cardinal',
  'char',
  'comp',
  'currency',
  'double',
  'dword',
  'extended',
  'file',
  'fixedint',
  'int16',
  'int32',
  'int64',
  'integer',
  'iunknown',
  'longbool',
  'longint',
  'longword',
  'nativeint',
  'pansichar',
  'pansistring',
  'pbool',
  'pboolean',
  'pbyte',
  'pbytearray',
  'pcardinal',
  'pchar',
  'pcomp',
  'pcurrency',
  'pdate',
  'pdatetime',
  'pdouble',
  'pdword',
  'pextended',
  'phandle',
  'pint64',
  'pinteger',
  'plongint',
  'plongword',
  'pointer',
  'ppointer',
  'pshortint',
  'pshortstring',
  'psingle',
  'psmallint',
  'pstring',
  'pvariant',
  'pwidechar',
  'pwidestring',
  'pword',
  'pwordarray',
  'pwordbool',
  'real',
  'real48',
  'record',
  'set',
  'shortint',
  'shortstring',
  'single',
  'smallint',
  'string',
  'tclass',
  'tdate',
  'tdatetime',
  'text',
  'textfile',
  'thandle',
  'tobject',
  'ttime',
  'ucs2char',
  'ucs4char',
  'uint16',
  'uint32',
  'uint64',
  'uint8',
  'variant',
  'widechar',
  'widestring',
  'word',
  'wordbool',
]);

class HighlightPascal extends Highlighter {

  static NORMAL = 0;
  static WORD = 1;
  static QUOTE = 2;
  static DQUOTE = 3;
  static SLASH = 4;
  static SLASH_SLASH = 5;
  static CURLY = 6;
  static PAREN = 7;
  static PAREN_STAR = 8;
  static PAREN_STAR_STAR = 9;

  constructor(delegate) {
    super(delegate);
    this.word = '';
  }

  feed(input) {
    for (let i = 0; i < input.length; i += this.delta) {
      this.delta = 1;
      let c = input[i];

      switch (this.state) {

      case HighlightPascal.NORMAL:
        if (!isascii(c) || isalpha(c) || c == '_') {
          this.epsilon(HighlightPascal.WORD);
        } else if (c == '/') {
          this.state = HighlightPascal.SLASH;
        } else if (c == '(') {
          this.state = HighlightPascal.PAREN;
        } else if (c == '\'') {
          this.state = HighlightPascal.QUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '"') {
          this.state = HighlightPascal.DQUOTE;
          this.push("span", "string");
          this.append(c);
        } else if (c == '{') {
          this.state = HighlightPascal.CURLY;
          this.push("span", "comment");
          this.append(c);
        } else {
          this.append(c);
        }
        break;

      case HighlightPascal.WORD:
        if (!isascii(c) || isalnum(c) || c == '_') {
          this.word += c;
        } else {
          if (PASCAL_KEYWORDS.has(this.word.toLowerCase())) {
            this.push("span", "keyword");
            this.append(this.word);
            this.pop();
          } else if (PASCAL_TYPES.has(this.word.toLowerCase())) {
            this.push("span", "type");
            this.append(this.word);
            this.pop();
          } else if (PASCAL_BUILTINS.has(this.word.toLowerCase())) {
            this.push("span", "builtin");
            this.append(this.word);
            this.pop();
          } else {
            this.append(this.word);
          }
          this.word = '';
          this.epsilon(HighlightPascal.NORMAL);
        }
        break;

      case HighlightPascal.SLASH:
        if (c == '/') {
          this.push("span", "comment");
          this.append("//");
          this.state = HighlightPascal.SLASH_SLASH;
        } else {
          this.append('/');
          this.epsilon(HighlightPascal.NORMAL);
        }
        break;

      case HighlightPascal.SLASH_SLASH:
        this.append(c);
        if (c == '\n') {
          this.pop();
          this.state = HighlightPascal.NORMAL;
        }
        break;

      case HighlightPascal.QUOTE:
        this.append(c);
        if (c == '\'') {
          this.pop();
          this.state = HighlightPascal.NORMAL;
        }
        break;

      case HighlightPascal.DQUOTE:
        this.append(c);
        if (c == '"') {
          this.pop();
          this.state = HighlightPascal.NORMAL;
        }
        break;

      case HighlightPascal.CURLY:
        this.append(c);
        if (c == '}') {
          this.pop();
          this.state = HighlightPascal.NORMAL;
        }
        break;

      case HighlightPascal.PAREN:
        if (c == '*') {
          this.push("span", "comment");
          this.append("(*");
          this.state = HighlightPascal.PAREN_STAR;
        } else {
          this.append('(');
          this.epsilon(HighlightPascal.NORMAL);
        }
        break;

      case HighlightPascal.PAREN_STAR:
        this.append(c);
        if (c == '*')
          this.state = HighlightPascal.PAREN_STAR_STAR;
        break;

      case HighlightPascal.PAREN_STAR_STAR:
        this.append(c);
        if (c == ')') {
          this.pop();
          this.state = HighlightPascal.NORMAL;
        } else if (c != '*') {
          this.state = HighlightPascal.PAREN_STAR;
        }
        break;

      default:
        throw new Error('Invalid state');
      }
    }
  }

  flush() {
    switch (this.state) {
    case HighlightPascal.WORD:
      if (PASCAL_KEYWORDS.has(this.word.toLowerCase())) {
        this.push("span", "keyword");
        this.append(this.word);
        this.pop();
      } else if (PASCAL_TYPES.has(this.word.toLowerCase())) {
        this.push("span", "type");
        this.append(this.word);
        this.pop();
      } else if (PASCAL_BUILTINS.has(this.word.toLowerCase())) {
        this.push("span", "builtin");
        this.append(this.word);
        this.pop();
      } else {
        this.append(this.word);
      }
      this.word = '';
      break;
    case HighlightPascal.SLASH:
      this.append('/');
      break;
    case HighlightPascal.PAREN:
      this.append('(');
      break;
    case HighlightPascal.QUOTE:
    case HighlightPascal.DQUOTE:
    case HighlightPascal.SLASH_SLASH:
    case HighlightPascal.CURLY:
    case HighlightPascal.PAREN_STAR:
    case HighlightPascal.PAREN_STAR_STAR:
      this.pop();
      break;
    default:
      break;
    }
    this.state = HighlightPascal.NORMAL;
    this.delegate.flush();
    this.delta = 1;
  }
}

Highlighter.REGISTRY['pascal'] = HighlightPascal;
Highlighter.REGISTRY['pas'] = HighlightPascal;
Highlighter.REGISTRY['delphi'] = HighlightPascal;
