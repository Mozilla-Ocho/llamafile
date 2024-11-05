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

const TCL_KEYWORDS = new Set([
  'body',
  'break',
  'case',
  'chain',
  'class',
  'configbody',
  'constructor',
  'continue',
  'default',
  'destructor',
  'else',
  'elseif',
  'error',
  'eval',
  'exit',
  'for',
  'for_array_keys',
  'for_file',
  'for_recursive_glob',
  'foreach',
  'if',
  'in',
  'itcl_class',
  'loop',
  'method',
  'proc',
  'return',
  'switch',
  'then',
  'uplevel',
  'while',
]);

const TCL_BUILTINS = new Set([
  'after',
  'append',
  'array',
  'bgerror',
  'binary',
  'catch',
  'cd',
  'clock',
  'close',
  'concat',
  'console',
  'dde',
  'encoding',
  'eof',
  'exec',
  'expr',
  'fblocked',
  'fconfigure',
  'fcopy',
  'file',
  'fileevent',
  'flush',
  'format',
  'gets',
  'glob',
  'history',
  'incr',
  'info',
  'interp',
  'join',
  'lappend',
  'lindex',
  'linsert',
  'list',
  'llength',
  'load',
  'lrange',
  'lreplace',
  'lsort',
  'namespace',
  'open',
  'package',
  'pid',
  'puts',
  'pwd',
  'read',
  'regexp',
  'registry',
  'regsub',
  'rename',
  'scan',
  'seek',
  'set',
  'socket',
  'source',
  'split',
  'string',
  'subst',
  'tell',
  'time',
  'trace',
  'unknown',
  'unset',
  'vwait',
]);

const TCL_TYPES = new Set([
  'common',
  'global',
  'inherit',
  'itk_option',
  'private',
  'protected',
  'public',
  'upvar',
  'variable',
]);

class HighlightTcl extends Highlighter {

  constructor(delegate) {
    super(delegate);
  }

  feed(input) {
    this.append(input);
  }

  flush() {
    this.delegate.flush();
  }
}

Highlighter.REGISTRY['tcl'] = HighlightTcl;
