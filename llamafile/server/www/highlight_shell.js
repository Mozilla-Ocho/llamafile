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

const SHELL_KEYWORDS = new Set([
  'break',
  'case',
  'coproc',
  'do',
  'done',
  'elif',
  'else',
  'esac',
  'exec',
  'exit',
  'expr',
  'fi',
  'for',
  'function',
  'if',
  'in',
  'return',
  'select',
  'then',
  'time',
  'trap',
  'until',
  'while',
]);

const SHELL_BUILTINS = new Set([
  'alias',
  'bg',
  'bind',
  'builtin',
  'caller',
  'cd',
  'chdir',
  'command',
  'declare',
  'echo',
  'enable',
  'eval',
  'false',
  'fg',
  'getopts',
  'hash',
  'help',
  'jobs',
  'kill',
  'let',
  'local',
  'logout',
  'mapfile',
  'printf',
  'read',
  'readarray',
  'set',
  'shift',
  'source',
  'stop',
  'suspend',
  'test',
  'times',
  'true',
  'type',
  'typeset',
  'ulimit',
  'unalias',
  'unset',
  'wait',
]);

class HighlightShell extends Highlighter {

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

Highlighter.REGISTRY['shell'] = HighlightShell;
Highlighter.REGISTRY['bash'] = HighlightShell;
Highlighter.REGISTRY['ksh'] = HighlightShell;
Highlighter.REGISTRY['sh'] = HighlightShell;
