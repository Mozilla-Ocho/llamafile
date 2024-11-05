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

const CPP_KEYWORDS = new Set([
  '__has_attribute',
  '__has_builtin',
  '__has_cpp_attribute',
  '__has_extension',
  'define',
  'defined',
  'elif',
  'elifdef',
  'elifndef',
  'else',
  'embed',
  'endif',
  'error',
  'if',
  'ifdef',
  'ifndef',
  'import',
  'include',
  'include_next',
  'line',
  'pragma',
  'undef',
  'warning',
]);
