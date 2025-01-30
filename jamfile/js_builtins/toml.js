/**
 * Original source: https://jsr.io/@std/toml
 *
 * Generated from jsr:@std/toml@1.0.2 using:
 * 
 * ```bash
 * echo 'export * from "@std/toml"' \
 *  | ./node_modules/.bin/esbuild --bundle --format=esm \
 *  > ../jamfile/js_builtins/toml.js
 * ```
 *
 */

// node_modules/@std/toml/stringify.js
function joinKeys(keys) {
  return keys.map((str) => {
    return str.length === 0 || str.match(/[^A-Za-z0-9_-]/) ? JSON.stringify(str) : str;
  }).join(".");
}
var Dumper = class {
  maxPad = 0;
  srcObject;
  output = [];
  #arrayTypeCache = /* @__PURE__ */ new Map();
  constructor(srcObjc) {
    this.srcObject = srcObjc;
  }
  dump(fmtOptions = {}) {
    this.output = this.#printObject(this.srcObject);
    this.output = this.#format(fmtOptions);
    return this.output;
  }
  #printObject(obj, keys = []) {
    const out = [];
    const props = Object.keys(obj);
    const inlineProps = [];
    const multilineProps = [];
    for (const prop of props) {
      if (this.#isSimplySerializable(obj[prop])) {
        inlineProps.push(prop);
      } else {
        multilineProps.push(prop);
      }
    }
    const sortedProps = inlineProps.concat(multilineProps);
    for (const prop of sortedProps) {
      const value2 = obj[prop];
      if (value2 instanceof Date) {
        out.push(this.#dateDeclaration([
          prop
        ], value2));
      } else if (typeof value2 === "string" || value2 instanceof RegExp) {
        out.push(this.#strDeclaration([
          prop
        ], value2.toString()));
      } else if (typeof value2 === "number") {
        out.push(this.#numberDeclaration([
          prop
        ], value2));
      } else if (typeof value2 === "boolean") {
        out.push(this.#boolDeclaration([
          prop
        ], value2));
      } else if (value2 instanceof Array) {
        const arrayType = this.#getTypeOfArray(value2);
        if (arrayType === "ONLY_PRIMITIVE") {
          out.push(this.#arrayDeclaration([
            prop
          ], value2));
        } else if (arrayType === "ONLY_OBJECT_EXCLUDING_ARRAY") {
          for (let i = 0; i < value2.length; i++) {
            out.push("");
            out.push(this.#headerGroup([
              ...keys,
              prop
            ]));
            out.push(...this.#printObject(value2[i], [
              ...keys,
              prop
            ]));
          }
        } else {
          const str = value2.map((x) => this.#printAsInlineValue(x)).join(",");
          out.push(`${this.#declaration([
            prop
          ])}[${str}]`);
        }
      } else if (typeof value2 === "object") {
        out.push("");
        out.push(this.#header([
          ...keys,
          prop
        ]));
        if (value2) {
          const toParse = value2;
          out.push(...this.#printObject(toParse, [
            ...keys,
            prop
          ]));
        }
      }
    }
    out.push("");
    return out;
  }
  #isPrimitive(value2) {
    return value2 instanceof Date || value2 instanceof RegExp || [
      "string",
      "number",
      "boolean"
    ].includes(typeof value2);
  }
  #getTypeOfArray(arr) {
    if (this.#arrayTypeCache.has(arr)) {
      return this.#arrayTypeCache.get(arr);
    }
    const type = this.#doGetTypeOfArray(arr);
    this.#arrayTypeCache.set(arr, type);
    return type;
  }
  #doGetTypeOfArray(arr) {
    if (!arr.length) {
      return "ONLY_PRIMITIVE";
    }
    const onlyPrimitive = this.#isPrimitive(arr[0]);
    if (arr[0] instanceof Array) {
      return "MIXED";
    }
    for (let i = 1; i < arr.length; i++) {
      if (onlyPrimitive !== this.#isPrimitive(arr[i]) || arr[i] instanceof Array) {
        return "MIXED";
      }
    }
    return onlyPrimitive ? "ONLY_PRIMITIVE" : "ONLY_OBJECT_EXCLUDING_ARRAY";
  }
  #printAsInlineValue(value2) {
    if (value2 instanceof Date) {
      return `"${this.#printDate(value2)}"`;
    } else if (typeof value2 === "string" || value2 instanceof RegExp) {
      return JSON.stringify(value2.toString());
    } else if (typeof value2 === "number") {
      return value2;
    } else if (typeof value2 === "boolean") {
      return value2.toString();
    } else if (value2 instanceof Array) {
      const str = value2.map((x) => this.#printAsInlineValue(x)).join(",");
      return `[${str}]`;
    } else if (typeof value2 === "object") {
      if (!value2) {
        throw new Error("should never reach");
      }
      const str = Object.keys(value2).map((key) => {
        return `${joinKeys([
          key
        ])} = ${// deno-lint-ignore no-explicit-any
        this.#printAsInlineValue(value2[key])}`;
      }).join(",");
      return `{${str}}`;
    }
    throw new Error("should never reach");
  }
  #isSimplySerializable(value2) {
    return typeof value2 === "string" || typeof value2 === "number" || typeof value2 === "boolean" || value2 instanceof RegExp || value2 instanceof Date || value2 instanceof Array && this.#getTypeOfArray(value2) !== "ONLY_OBJECT_EXCLUDING_ARRAY";
  }
  #header(keys) {
    return `[${joinKeys(keys)}]`;
  }
  #headerGroup(keys) {
    return `[[${joinKeys(keys)}]]`;
  }
  #declaration(keys) {
    const title = joinKeys(keys);
    if (title.length > this.maxPad) {
      this.maxPad = title.length;
    }
    return `${title} = `;
  }
  #arrayDeclaration(keys, value2) {
    return `${this.#declaration(keys)}${JSON.stringify(value2)}`;
  }
  #strDeclaration(keys, value2) {
    return `${this.#declaration(keys)}${JSON.stringify(value2)}`;
  }
  #numberDeclaration(keys, value2) {
    switch (value2) {
      case Infinity:
        return `${this.#declaration(keys)}inf`;
      case -Infinity:
        return `${this.#declaration(keys)}-inf`;
      default:
        return `${this.#declaration(keys)}${value2}`;
    }
  }
  #boolDeclaration(keys, value2) {
    return `${this.#declaration(keys)}${value2}`;
  }
  #printDate(value2) {
    function dtPad(v, lPad = 2) {
      return v.padStart(lPad, "0");
    }
    const m = dtPad((value2.getUTCMonth() + 1).toString());
    const d = dtPad(value2.getUTCDate().toString());
    const h = dtPad(value2.getUTCHours().toString());
    const min = dtPad(value2.getUTCMinutes().toString());
    const s = dtPad(value2.getUTCSeconds().toString());
    const ms = dtPad(value2.getUTCMilliseconds().toString(), 3);
    const fData = `${value2.getUTCFullYear()}-${m}-${d}T${h}:${min}:${s}.${ms}`;
    return fData;
  }
  #dateDeclaration(keys, value2) {
    return `${this.#declaration(keys)}${this.#printDate(value2)}`;
  }
  #format(options = {}) {
    const { keyAlignment = false } = options;
    const rDeclaration = /^(\".*\"|[^=]*)\s=/;
    const out = [];
    for (let i = 0; i < this.output.length; i++) {
      const l = this.output[i];
      if (l[0] === "[" && l[1] !== "[") {
        if (this.output[i + 1] === "" && this.output[i + 2]?.slice(0, l.length) === l.slice(0, -1) + ".") {
          i += 1;
          continue;
        }
        out.push(l);
      } else {
        if (keyAlignment) {
          const m = rDeclaration.exec(l);
          if (m && m[1]) {
            out.push(l.replace(m[1], m[1].padEnd(this.maxPad)));
          } else {
            out.push(l);
          }
        } else {
          out.push(l);
        }
      }
    }
    const cleanedOutput = [];
    for (let i = 0; i < out.length; i++) {
      const l = out[i];
      if (!(l === "" && out[i + 1] === "")) {
        cleanedOutput.push(l);
      }
    }
    return cleanedOutput;
  }
};
function stringify(obj, options) {
  return new Dumper(obj).dump(options).join("\n");
}

// node_modules/@jsr/std__collections/_utils.js
function filterInPlace(array, predicate) {
  let outputIndex = 0;
  for (const cur of array) {
    if (!predicate(cur)) {
      continue;
    }
    array[outputIndex] = cur;
    outputIndex += 1;
  }
  array.splice(outputIndex);
  return array;
}

// node_modules/@jsr/std__collections/deep_merge.js
function deepMerge(record, other, options) {
  return deepMergeInternal(record, other, /* @__PURE__ */ new Set(), options);
}
function deepMergeInternal(record, other, seen, options) {
  const result = {};
  const keys = /* @__PURE__ */ new Set([
    ...getKeys(record),
    ...getKeys(other)
  ]);
  for (const key of keys) {
    if (key === "__proto__") {
      continue;
    }
    const a = record[key];
    if (!Object.hasOwn(other, key)) {
      result[key] = a;
      continue;
    }
    const b = other[key];
    if (isNonNullObject(a) && isNonNullObject(b) && !seen.has(a) && !seen.has(b)) {
      seen.add(a);
      seen.add(b);
      result[key] = mergeObjects(a, b, seen, options);
      continue;
    }
    result[key] = b;
  }
  return result;
}
function mergeObjects(left, right, seen, options = {
  arrays: "merge",
  sets: "merge",
  maps: "merge"
}) {
  if (isMergeable(left) && isMergeable(right)) {
    return deepMergeInternal(left, right, seen, options);
  }
  if (isIterable(left) && isIterable(right)) {
    if (Array.isArray(left) && Array.isArray(right)) {
      if (options.arrays === "merge") {
        return left.concat(right);
      }
      return right;
    }
    if (left instanceof Map && right instanceof Map) {
      if (options.maps === "merge") {
        return new Map([
          ...left,
          ...right
        ]);
      }
      return right;
    }
    if (left instanceof Set && right instanceof Set) {
      if (options.sets === "merge") {
        return /* @__PURE__ */ new Set([
          ...left,
          ...right
        ]);
      }
      return right;
    }
  }
  return right;
}
function isMergeable(value2) {
  return Object.getPrototypeOf(value2) === Object.prototype;
}
function isIterable(value2) {
  return typeof value2[Symbol.iterator] === "function";
}
function isNonNullObject(value2) {
  return value2 !== null && typeof value2 === "object";
}
function getKeys(record) {
  const result = Object.getOwnPropertySymbols(record);
  filterInPlace(result, (key) => Object.prototype.propertyIsEnumerable.call(record, key));
  result.push(...Object.keys(record));
  return result;
}

// node_modules/@std/toml/_parser.js
var Scanner = class {
  #whitespace = /[ \t]/;
  #position = 0;
  #source;
  constructor(source) {
    this.#source = source;
  }
  /**
   * Get current character
   * @param index - relative index from current position
   */
  char(index = 0) {
    return this.#source[this.#position + index] ?? "";
  }
  /**
   * Get sliced string
   * @param start - start position relative from current position
   * @param end - end position relative from current position
   */
  slice(start, end) {
    return this.#source.slice(this.#position + start, this.#position + end);
  }
  /**
   * Move position to next
   */
  next(count) {
    if (typeof count === "number") {
      for (let i = 0; i < count; i++) {
        this.#position++;
      }
    } else {
      this.#position++;
    }
  }
  /**
   * Move position until current char is not a whitespace, EOL, or comment.
   * @param options.inline - skip only whitespaces
   */
  nextUntilChar(options = {
    comment: true
  }) {
    if (options.inline) {
      while (this.#whitespace.test(this.char()) && !this.eof()) {
        this.next();
      }
    } else {
      while (!this.eof()) {
        const char = this.char();
        if (this.#whitespace.test(char) || this.isCurrentCharEOL()) {
          this.next();
        } else if (options.comment && this.char() === "#") {
          while (!this.isCurrentCharEOL() && !this.eof()) {
            this.next();
          }
        } else {
          break;
        }
      }
    }
    if (!this.isCurrentCharEOL() && /\s/.test(this.char())) {
      const escaped = "\\u" + this.char().charCodeAt(0).toString(16);
      const position = this.#position;
      throw new SyntaxError(`Cannot parse the TOML: It contains invalid whitespace at position '${position}': \`${escaped}\``);
    }
  }
  /**
   * Position reached EOF or not
   */
  eof() {
    return this.position() >= this.#source.length;
  }
  /**
   * Get current position
   */
  position() {
    return this.#position;
  }
  isCurrentCharEOL() {
    return this.char() === "\n" || this.slice(0, 2) === "\r\n";
  }
};
function success(body) {
  return {
    ok: true,
    body
  };
}
function failure() {
  return {
    ok: false
  };
}
function unflat(keys, values = {}, cObj) {
  const out = {};
  if (keys.length === 0) {
    return cObj;
  }
  if (!cObj) cObj = values;
  const key = keys[keys.length - 1];
  if (typeof key === "string") out[key] = cObj;
  return unflat(keys.slice(0, -1), values, out);
}
function deepAssignWithTable(target, table2) {
  if (table2.key.length === 0 || table2.key[0] == null) {
    throw new Error("Cannot parse the TOML: key length is not a positive number");
  }
  const value2 = target[table2.key[0]];
  if (typeof value2 === "undefined") {
    Object.assign(target, unflat(table2.key, table2.type === "Table" ? table2.value : [
      table2.value
    ]));
  } else if (Array.isArray(value2)) {
    if (table2.type === "TableArray" && table2.key.length === 1) {
      value2.push(table2.value);
    } else {
      const last = value2[value2.length - 1];
      deepAssignWithTable(last, {
        type: table2.type,
        key: table2.key.slice(1),
        value: table2.value
      });
    }
  } else if (typeof value2 === "object" && value2 !== null) {
    deepAssignWithTable(value2, {
      type: table2.type,
      key: table2.key.slice(1),
      value: table2.value
    });
  } else {
    throw new Error("Unexpected assign");
  }
}
function or(parsers) {
  return (scanner) => {
    for (const parse2 of parsers) {
      const result = parse2(scanner);
      if (result.ok) return result;
    }
    return failure();
  };
}
function join(parser, separator) {
  const Separator = character(separator);
  return (scanner) => {
    const first = parser(scanner);
    if (!first.ok) return failure();
    const out = [
      first.body
    ];
    while (!scanner.eof()) {
      if (!Separator(scanner).ok) break;
      const result = parser(scanner);
      if (!result.ok) {
        throw new SyntaxError(`Invalid token after "${separator}"`);
      }
      out.push(result.body);
    }
    return success(out);
  };
}
function kv(keyParser, separator, valueParser) {
  const Separator = character(separator);
  return (scanner) => {
    const key = keyParser(scanner);
    if (!key.ok) return failure();
    const sep = Separator(scanner);
    if (!sep.ok) {
      throw new SyntaxError(`key/value pair doesn't have "${separator}"`);
    }
    const value2 = valueParser(scanner);
    if (!value2.ok) {
      throw new SyntaxError(`Value of key/value pair is invalid data format`);
    }
    return success(unflat(key.body, value2.body));
  };
}
function merge(parser) {
  return (scanner) => {
    const result = parser(scanner);
    if (!result.ok) return failure();
    let body = {};
    for (const record of result.body) {
      if (typeof body === "object" && body !== null) {
        body = deepMerge(body, record);
      }
    }
    return success(body);
  };
}
function repeat(parser) {
  return (scanner) => {
    const body = [];
    while (!scanner.eof()) {
      const result = parser(scanner);
      if (!result.ok) break;
      body.push(result.body);
      scanner.nextUntilChar();
    }
    if (body.length === 0) return failure();
    return success(body);
  };
}
function surround(left, parser, right) {
  const Left = character(left);
  const Right = character(right);
  return (scanner) => {
    if (!Left(scanner).ok) {
      return failure();
    }
    const result = parser(scanner);
    if (!result.ok) {
      throw new SyntaxError(`Invalid token after "${left}"`);
    }
    if (!Right(scanner).ok) {
      throw new SyntaxError(`Not closed by "${right}" after started with "${left}"`);
    }
    return success(result.body);
  };
}
function character(str) {
  return (scanner) => {
    scanner.nextUntilChar({
      inline: true
    });
    if (scanner.slice(0, str.length) !== str) return failure();
    scanner.next(str.length);
    scanner.nextUntilChar({
      inline: true
    });
    return success(void 0);
  };
}
var BARE_KEY_REGEXP = /[A-Za-z0-9_-]/;
var FLOAT_REGEXP = /[0-9_\.e+\-]/i;
var END_OF_VALUE_REGEXP = /[ \t\r\n#,}\]]/;
function bareKey(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (!scanner.char() || !BARE_KEY_REGEXP.test(scanner.char())) {
    return failure();
  }
  const acc = [];
  while (scanner.char() && BARE_KEY_REGEXP.test(scanner.char())) {
    acc.push(scanner.char());
    scanner.next();
  }
  const key = acc.join("");
  return success(key);
}
function escapeSequence(scanner) {
  if (scanner.char() !== "\\") return failure();
  scanner.next();
  switch (scanner.char()) {
    case "b":
      scanner.next();
      return success("\b");
    case "t":
      scanner.next();
      return success("	");
    case "n":
      scanner.next();
      return success("\n");
    case "f":
      scanner.next();
      return success("\f");
    case "r":
      scanner.next();
      return success("\r");
    case "u":
    case "U": {
      const codePointLen = scanner.char() === "u" ? 4 : 6;
      const codePoint = parseInt("0x" + scanner.slice(1, 1 + codePointLen), 16);
      const str = String.fromCodePoint(codePoint);
      scanner.next(codePointLen + 1);
      return success(str);
    }
    case '"':
      scanner.next();
      return success('"');
    case "\\":
      scanner.next();
      return success("\\");
    default:
      throw new SyntaxError(`Invalid escape sequence: \\${scanner.char()}`);
  }
}
function basicString(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (scanner.char() !== '"') return failure();
  scanner.next();
  const acc = [];
  while (scanner.char() !== '"' && !scanner.eof()) {
    if (scanner.char() === "\n") {
      throw new SyntaxError("Single-line string cannot contain EOL");
    }
    const escapedChar = escapeSequence(scanner);
    if (escapedChar.ok) {
      acc.push(escapedChar.body);
    } else {
      acc.push(scanner.char());
      scanner.next();
    }
  }
  if (scanner.eof()) {
    throw new SyntaxError(`Single-line string is not closed:
${acc.join("")}`);
  }
  scanner.next();
  return success(acc.join(""));
}
function literalString(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (scanner.char() !== "'") return failure();
  scanner.next();
  const acc = [];
  while (scanner.char() !== "'" && !scanner.eof()) {
    if (scanner.char() === "\n") {
      throw new SyntaxError("Single-line string cannot contain EOL");
    }
    acc.push(scanner.char());
    scanner.next();
  }
  if (scanner.eof()) {
    throw new SyntaxError(`Single-line string is not closed:
${acc.join("")}`);
  }
  scanner.next();
  return success(acc.join(""));
}
function multilineBasicString(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (scanner.slice(0, 3) !== '"""') return failure();
  scanner.next(3);
  if (scanner.char() === "\n") {
    scanner.next();
  } else if (scanner.slice(0, 2) === "\r\n") {
    scanner.next(2);
  }
  const acc = [];
  while (scanner.slice(0, 3) !== '"""' && !scanner.eof()) {
    if (scanner.slice(0, 2) === "\\\n") {
      scanner.next();
      scanner.nextUntilChar({
        comment: false
      });
      continue;
    } else if (scanner.slice(0, 3) === "\\\r\n") {
      scanner.next();
      scanner.nextUntilChar({
        comment: false
      });
      continue;
    }
    const escapedChar = escapeSequence(scanner);
    if (escapedChar.ok) {
      acc.push(escapedChar.body);
    } else {
      acc.push(scanner.char());
      scanner.next();
    }
  }
  if (scanner.eof()) {
    throw new SyntaxError(`Multi-line string is not closed:
${acc.join("")}`);
  }
  if (scanner.char(3) === '"') {
    acc.push('"');
    scanner.next();
  }
  scanner.next(3);
  return success(acc.join(""));
}
function multilineLiteralString(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (scanner.slice(0, 3) !== "'''") return failure();
  scanner.next(3);
  if (scanner.char() === "\n") {
    scanner.next();
  } else if (scanner.slice(0, 2) === "\r\n") {
    scanner.next(2);
  }
  const acc = [];
  while (scanner.slice(0, 3) !== "'''" && !scanner.eof()) {
    acc.push(scanner.char());
    scanner.next();
  }
  if (scanner.eof()) {
    throw new SyntaxError(`Multi-line string is not closed:
${acc.join("")}`);
  }
  if (scanner.char(3) === "'") {
    acc.push("'");
    scanner.next();
  }
  scanner.next(3);
  return success(acc.join(""));
}
var symbolPairs = [
  [
    "true",
    true
  ],
  [
    "false",
    false
  ],
  [
    "inf",
    Infinity
  ],
  [
    "+inf",
    Infinity
  ],
  [
    "-inf",
    -Infinity
  ],
  [
    "nan",
    NaN
  ],
  [
    "+nan",
    NaN
  ],
  [
    "-nan",
    NaN
  ]
];
function symbols(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  const found = symbolPairs.find(([str2]) => scanner.slice(0, str2.length) === str2);
  if (!found) return failure();
  const [str, value2] = found;
  scanner.next(str.length);
  return success(value2);
}
var dottedKey = join(or([
  bareKey,
  basicString,
  literalString
]), ".");
function integer(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  const first2 = scanner.slice(0, 2);
  if (first2.length === 2 && /0(?:x|o|b)/i.test(first2)) {
    scanner.next(2);
    const acc2 = [
      first2
    ];
    while (/[0-9a-f_]/i.test(scanner.char()) && !scanner.eof()) {
      acc2.push(scanner.char());
      scanner.next();
    }
    if (acc2.length === 1) return failure();
    return success(acc2.join(""));
  }
  const acc = [];
  if (/[+-]/.test(scanner.char())) {
    acc.push(scanner.char());
    scanner.next();
  }
  while (/[0-9_]/.test(scanner.char()) && !scanner.eof()) {
    acc.push(scanner.char());
    scanner.next();
  }
  if (acc.length === 0 || acc.length === 1 && /[+-]/.test(acc[0])) {
    return failure();
  }
  const int = parseInt(acc.filter((char) => char !== "_").join(""));
  return success(int);
}
function float(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  let position = 0;
  while (scanner.char(position) && !END_OF_VALUE_REGEXP.test(scanner.char(position))) {
    if (!FLOAT_REGEXP.test(scanner.char(position))) return failure();
    position++;
  }
  const acc = [];
  if (/[+-]/.test(scanner.char())) {
    acc.push(scanner.char());
    scanner.next();
  }
  while (FLOAT_REGEXP.test(scanner.char()) && !scanner.eof()) {
    acc.push(scanner.char());
    scanner.next();
  }
  if (acc.length === 0) return failure();
  const float2 = parseFloat(acc.filter((char) => char !== "_").join(""));
  if (isNaN(float2)) return failure();
  return success(float2);
}
function dateTime(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  let dateStr = scanner.slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}/.test(dateStr)) return failure();
  scanner.next(10);
  const acc = [];
  while (/[ 0-9TZ.:+-]/.test(scanner.char()) && !scanner.eof()) {
    acc.push(scanner.char());
    scanner.next();
  }
  dateStr += acc.join("");
  const date = new Date(dateStr.trim());
  if (isNaN(date.getTime())) {
    throw new SyntaxError(`Invalid date string "${dateStr}"`);
  }
  return success(date);
}
function localTime(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  let timeStr = scanner.slice(0, 8);
  if (!/^(\d{2}):(\d{2}):(\d{2})/.test(timeStr)) return failure();
  scanner.next(8);
  const acc = [];
  if (scanner.char() !== ".") return success(timeStr);
  acc.push(scanner.char());
  scanner.next();
  while (/[0-9]/.test(scanner.char()) && !scanner.eof()) {
    acc.push(scanner.char());
    scanner.next();
  }
  timeStr += acc.join("");
  return success(timeStr);
}
function arrayValue(scanner) {
  scanner.nextUntilChar({
    inline: true
  });
  if (scanner.char() !== "[") return failure();
  scanner.next();
  const array = [];
  while (!scanner.eof()) {
    scanner.nextUntilChar();
    const result = value(scanner);
    if (!result.ok) break;
    array.push(result.body);
    scanner.nextUntilChar({
      inline: true
    });
    if (scanner.char() !== ",") break;
    scanner.next();
  }
  scanner.nextUntilChar();
  if (scanner.char() !== "]") throw new SyntaxError("Array is not closed");
  scanner.next();
  return success(array);
}
function inlineTable(scanner) {
  scanner.nextUntilChar();
  if (scanner.char(1) === "}") {
    scanner.next(2);
    return success({});
  }
  const pairs = surround("{", join(pair, ","), "}")(scanner);
  if (!pairs.ok) return failure();
  let table2 = {};
  for (const pair2 of pairs.body) {
    table2 = deepMerge(table2, pair2);
  }
  return success(table2);
}
var value = or([
  multilineBasicString,
  multilineLiteralString,
  basicString,
  literalString,
  symbols,
  dateTime,
  localTime,
  float,
  integer,
  arrayValue,
  inlineTable
]);
var pair = kv(dottedKey, "=", value);
function block(scanner) {
  scanner.nextUntilChar();
  const result = merge(repeat(pair))(scanner);
  if (result.ok) return success({
    type: "Block",
    value: result.body
  });
  return failure();
}
var tableHeader = surround("[", dottedKey, "]");
function table(scanner) {
  scanner.nextUntilChar();
  const header = tableHeader(scanner);
  if (!header.ok) return failure();
  scanner.nextUntilChar();
  const b = block(scanner);
  return success({
    type: "Table",
    key: header.body,
    value: b.ok ? b.body.value : {}
  });
}
var tableArrayHeader = surround("[[", dottedKey, "]]");
function tableArray(scanner) {
  scanner.nextUntilChar();
  const header = tableArrayHeader(scanner);
  if (!header.ok) return failure();
  scanner.nextUntilChar();
  const b = block(scanner);
  return success({
    type: "TableArray",
    key: header.body,
    value: b.ok ? b.body.value : {}
  });
}
function toml(scanner) {
  const blocks = repeat(or([
    block,
    tableArray,
    table
  ]))(scanner);
  if (!blocks.ok) return failure();
  let body = {};
  for (const block2 of blocks.body) {
    switch (block2.type) {
      case "Block": {
        body = deepMerge(body, block2.value);
        break;
      }
      case "Table": {
        deepAssignWithTable(body, block2);
        break;
      }
      case "TableArray": {
        deepAssignWithTable(body, block2);
        break;
      }
    }
  }
  return success(body);
}
function parserFactory(parser) {
  return (tomlString) => {
    const scanner = new Scanner(tomlString);
    let parsed = null;
    let err = null;
    try {
      parsed = parser(scanner);
    } catch (e) {
      err = e instanceof Error ? e : new Error("Invalid error type caught");
    }
    if (err || !parsed || !parsed.ok || !scanner.eof()) {
      const position = scanner.position();
      const subStr = tomlString.slice(0, position);
      const lines = subStr.split("\n");
      const row = lines.length;
      const column = (() => {
        let count = subStr.length;
        for (const line of lines) {
          if (count <= line.length) break;
          count -= line.length + 1;
        }
        return count;
      })();
      const message = `Parse error on line ${row}, column ${column}: ${err ? err.message : `Unexpected character: "${scanner.char()}"`}`;
      throw new SyntaxError(message);
    }
    return parsed.body;
  };
}

// node_modules/@std/toml/parse.js
function parse(tomlString) {
  return parserFactory(toml)(tomlString);
}
export {
  parse,
  stringify
};
