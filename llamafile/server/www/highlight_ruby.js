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

const RUBY_KEYWORDS = new Set([
  'BEGIN',
  'END',
  'alias',
  'and',
  'begin',
  'break',
  'case',
  'class',
  'def',
  'defined?',
  'do',
  'else',
  'elsif',
  'end',
  'ensure',
  'fail',
  'for',
  'if',
  'in',
  'module',
  'next',
  'not',
  'or',
  'redo',
  'rescue',
  'retry',
  'return',
  'self',
  'super',
  'then',
  'undef',
  'unless',
  'until',
  'when',
  'while',
  'yield',
]);

const RUBY_BUILTINS = new Set([
  '__callee__',
  '__dir__',
  '__method__',
  'abort',
  'alias_method',
  'at_exit',
  'attr',
  'attr_accessor',
  'attr_reader',
  'attr_writer',
  'autoload',
  'autoload?',
  'binding',
  'block_given?',
  'callcc',
  'caller',
  'catch',
  'define_method',
  'eval',
  'exec',
  'exit',
  'exit!',
  'extend',
  'fail',
  'fork',
  'format',
  'global_variables',
  'include',
  'lambda',
  'load',
  'local_variables',
  'loop',
  'module_function',
  'open',
  'p',
  'prepend',
  'print',
  'printf',
  'private',
  'private_class_method',
  'private_constant',
  'proc',
  'protected',
  'public',
  'public_class_method',
  'public_constant',
  'putc',
  'puts',
  'raise',
  'rand',
  'readline',
  'readlines',
  'refine',
  'require',
  'require_relative',
  'sleep',
  'spawn',
  'sprintf',
  'srand',
  'syscall',
  'system',
  'throw',
  'trace_var',
  'trap',
  'untrace_var',
  'using',
  'warn',
]);

const RUBY_CONSTANTS = new Set([
  '__ENCODING__',
  '__FILE__',
  '__LINE__',
  'false',
  'nil',
  'true',
]);

class HighlightRuby extends Highlighter {

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

Highlighter.REGISTRY['ruby'] = HighlightRuby;
Highlighter.REGISTRY['rb'] = HighlightRuby;
