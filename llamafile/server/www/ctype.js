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

function isascii(c) {
  let i = c.charCodeAt(0);
  return 0 <= i && i <= 127;
}

function isgraph(c) {
  let i = c.charCodeAt(0);
  return 0x21 <= i && i <= 0x7E;
}

function isdigit(c) {
  let i = c.charCodeAt(0);
  return (48 <= i && i <= 57);
}

function isalpha(c) {
  let i = c.charCodeAt(0);
  return (65 <= i && i <= 90) ||
         (97 <= i && i <= 122);
}

function isalnum(c) {
  let i = c.charCodeAt(0);
  return (48 <= i && i <= 57) ||
         (65 <= i && i <= 90) ||
         (97 <= i && i <= 122);
}

function isspace(c) {
  switch (c) {
  case ' ':
  case '\t':
  case '\r':
  case '\n':
  case '\v':
    return true;
  default:
    return false;
  }
}

function isblank(c) {
  switch (c) {
  case ' ':
  case '\t':
    return true;
  default:
    return false;
  }
}

function ispunct(c) {
  switch (c) {
  case '!':
  case '"':
  case '#':
  case '$':
  case '%':
  case '&':
  case '\'':
  case '(':
  case ')':
  case '*':
  case '+':
  case ',':
  case '-':
  case '.':
  case '/':
  case ':':
  case ';':
  case '<':
  case '=':
  case '>':
  case '?':
  case '@':
  case '[':
  case '\\':
  case ']':
  case '^':
  case '_':
  case '`':
  case '{':
  case '|':
  case '}':
  case '~':
    return true;
  default:
    return false;
  }
}

function isxdigit(c) {
  switch (c.toLowerCase()) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
    return true;
  default:
    return false;
  }
}
