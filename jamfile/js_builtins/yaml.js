/**
 * Original source: https://jsr.io/@std/yaml
 *
 * Generated from jsr:@std/yaml@1.0.5 using:
 * 
 * ```bash
 * echo 'export * from "@std/toml"' \
 *  | ./node_modules/.bin/esbuild --bundle --format=esm \
 *  > ../jamfile/js_builtins/toml.js
 * ```
 *
 */

// node_modules/@std/yaml/_chars.js
var BOM = 65279;
var TAB = 9;
var LINE_FEED = 10;
var CARRIAGE_RETURN = 13;
var SPACE = 32;
var EXCLAMATION = 33;
var DOUBLE_QUOTE = 34;
var SHARP = 35;
var PERCENT = 37;
var AMPERSAND = 38;
var SINGLE_QUOTE = 39;
var ASTERISK = 42;
var PLUS = 43;
var COMMA = 44;
var MINUS = 45;
var DOT = 46;
var COLON = 58;
var SMALLER_THAN = 60;
var GREATER_THAN = 62;
var QUESTION = 63;
var COMMERCIAL_AT = 64;
var LEFT_SQUARE_BRACKET = 91;
var BACKSLASH = 92;
var RIGHT_SQUARE_BRACKET = 93;
var GRAVE_ACCENT = 96;
var LEFT_CURLY_BRACKET = 123;
var VERTICAL_LINE = 124;
var RIGHT_CURLY_BRACKET = 125;
function isEOL(c) {
  return c === LINE_FEED || c === CARRIAGE_RETURN;
}
function isWhiteSpace(c) {
  return c === TAB || c === SPACE;
}
function isWhiteSpaceOrEOL(c) {
  return isWhiteSpace(c) || isEOL(c);
}
function isFlowIndicator(c) {
  return c === COMMA || c === LEFT_SQUARE_BRACKET || c === RIGHT_SQUARE_BRACKET || c === LEFT_CURLY_BRACKET || c === RIGHT_CURLY_BRACKET;
}

// node_modules/@std/yaml/_type/binary.js
var BASE64_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r";
function resolveYamlBinary(data) {
  if (data === null) return false;
  let code;
  let bitlen = 0;
  const max = data.length;
  const map2 = BASE64_MAP;
  for (let idx = 0; idx < max; idx++) {
    code = map2.indexOf(data.charAt(idx));
    if (code > 64) continue;
    if (code < 0) return false;
    bitlen += 6;
  }
  return bitlen % 8 === 0;
}
function constructYamlBinary(data) {
  const input = data.replace(/[\r\n=]/g, "");
  const max = input.length;
  const map2 = BASE64_MAP;
  const result = [];
  let bits = 0;
  for (let idx = 0; idx < max; idx++) {
    if (idx % 4 === 0 && idx) {
      result.push(bits >> 16 & 255);
      result.push(bits >> 8 & 255);
      result.push(bits & 255);
    }
    bits = bits << 6 | map2.indexOf(input.charAt(idx));
  }
  const tailbits = max % 4 * 6;
  if (tailbits === 0) {
    result.push(bits >> 16 & 255);
    result.push(bits >> 8 & 255);
    result.push(bits & 255);
  } else if (tailbits === 18) {
    result.push(bits >> 10 & 255);
    result.push(bits >> 2 & 255);
  } else if (tailbits === 12) {
    result.push(bits >> 4 & 255);
  }
  return new Uint8Array(result);
}
function representYamlBinary(object) {
  const max = object.length;
  const map2 = BASE64_MAP;
  let result = "";
  let bits = 0;
  for (let idx = 0; idx < max; idx++) {
    if (idx % 3 === 0 && idx) {
      result += map2[bits >> 18 & 63];
      result += map2[bits >> 12 & 63];
      result += map2[bits >> 6 & 63];
      result += map2[bits & 63];
    }
    bits = (bits << 8) + object[idx];
  }
  const tail = max % 3;
  if (tail === 0) {
    result += map2[bits >> 18 & 63];
    result += map2[bits >> 12 & 63];
    result += map2[bits >> 6 & 63];
    result += map2[bits & 63];
  } else if (tail === 2) {
    result += map2[bits >> 10 & 63];
    result += map2[bits >> 4 & 63];
    result += map2[bits << 2 & 63];
    result += map2[64];
  } else if (tail === 1) {
    result += map2[bits >> 2 & 63];
    result += map2[bits << 4 & 63];
    result += map2[64];
    result += map2[64];
  }
  return result;
}
function isBinary(obj) {
  return obj instanceof Uint8Array;
}
var binary = {
  tag: "tag:yaml.org,2002:binary",
  construct: constructYamlBinary,
  kind: "scalar",
  predicate: isBinary,
  represent: representYamlBinary,
  resolve: resolveYamlBinary
};

// node_modules/@std/yaml/_type/bool.js
var YAML_TRUE_BOOLEANS = [
  "true",
  "True",
  "TRUE"
];
var YAML_FALSE_BOOLEANS = [
  "false",
  "False",
  "FALSE"
];
var YAML_BOOLEANS = [
  ...YAML_TRUE_BOOLEANS,
  ...YAML_FALSE_BOOLEANS
];
var bool = {
  tag: "tag:yaml.org,2002:bool",
  kind: "scalar",
  defaultStyle: "lowercase",
  predicate: (value) => typeof value === "boolean" || value instanceof Boolean,
  construct: (data) => YAML_TRUE_BOOLEANS.includes(data),
  resolve: (data) => YAML_BOOLEANS.includes(data),
  represent: {
    // deno-lint-ignore ban-types
    lowercase: (object) => {
      const value = object instanceof Boolean ? object.valueOf() : object;
      return value ? "true" : "false";
    },
    // deno-lint-ignore ban-types
    uppercase: (object) => {
      const value = object instanceof Boolean ? object.valueOf() : object;
      return value ? "TRUE" : "FALSE";
    },
    // deno-lint-ignore ban-types
    camelcase: (object) => {
      const value = object instanceof Boolean ? object.valueOf() : object;
      return value ? "True" : "False";
    }
  }
};

// node_modules/@std/yaml/_utils.js
function isObject(value) {
  return value !== null && typeof value === "object";
}
function isNegativeZero(i) {
  return i === 0 && Number.NEGATIVE_INFINITY === 1 / i;
}
function isPlainObject(object) {
  return Object.prototype.toString.call(object) === "[object Object]";
}

// node_modules/@std/yaml/_type/float.js
var YAML_FLOAT_PATTERN = new RegExp(
  // 2.5e4, 2.5 and integers
  "^(?:[-+]?(?:0|[1-9][0-9_]*)(?:\\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?|\\.[0-9_]+(?:[eE][-+]?[0-9]+)?|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$"
);
function resolveYamlFloat(data) {
  if (!YAML_FLOAT_PATTERN.test(data) || // Quick hack to not allow integers end with `_`
  // Probably should update regexp & check speed
  data[data.length - 1] === "_") {
    return false;
  }
  return true;
}
function constructYamlFloat(data) {
  let value = data.replace(/_/g, "").toLowerCase();
  const sign = value[0] === "-" ? -1 : 1;
  if (value[0] && "+-".includes(value[0])) {
    value = value.slice(1);
  }
  if (value === ".inf") {
    return sign === 1 ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  }
  if (value === ".nan") {
    return NaN;
  }
  return sign * parseFloat(value);
}
var SCIENTIFIC_WITHOUT_DOT = /^[-+]?[0-9]+e/;
function representYamlFloat(object, style) {
  const value = object instanceof Number ? object.valueOf() : object;
  if (isNaN(value)) {
    switch (style) {
      case "lowercase":
        return ".nan";
      case "uppercase":
        return ".NAN";
      case "camelcase":
        return ".NaN";
    }
  } else if (Number.POSITIVE_INFINITY === value) {
    switch (style) {
      case "lowercase":
        return ".inf";
      case "uppercase":
        return ".INF";
      case "camelcase":
        return ".Inf";
    }
  } else if (Number.NEGATIVE_INFINITY === value) {
    switch (style) {
      case "lowercase":
        return "-.inf";
      case "uppercase":
        return "-.INF";
      case "camelcase":
        return "-.Inf";
    }
  } else if (isNegativeZero(value)) {
    return "-0.0";
  }
  const res = value.toString(10);
  return SCIENTIFIC_WITHOUT_DOT.test(res) ? res.replace("e", ".e") : res;
}
function isFloat(object) {
  if (object instanceof Number) object = object.valueOf();
  return typeof object === "number" && (object % 1 !== 0 || isNegativeZero(object));
}
var float = {
  tag: "tag:yaml.org,2002:float",
  construct: constructYamlFloat,
  defaultStyle: "lowercase",
  kind: "scalar",
  predicate: isFloat,
  represent: representYamlFloat,
  resolve: resolveYamlFloat
};

// node_modules/@std/yaml/_type/int.js
function isCharCodeInRange(c, lower, upper) {
  return lower <= c && c <= upper;
}
function isHexCode(c) {
  return isCharCodeInRange(c, 48, 57) || // 0-9
  isCharCodeInRange(c, 65, 70) || // A-F
  isCharCodeInRange(c, 97, 102);
}
function isOctCode(c) {
  return isCharCodeInRange(c, 48, 55);
}
function isDecCode(c) {
  return isCharCodeInRange(c, 48, 57);
}
function resolveYamlInteger(data) {
  const max = data.length;
  let index = 0;
  let hasDigits = false;
  if (!max) return false;
  let ch = data[index];
  if (ch === "-" || ch === "+") {
    ch = data[++index];
  }
  if (ch === "0") {
    if (index + 1 === max) return true;
    ch = data[++index];
    if (ch === "b") {
      index++;
      for (; index < max; index++) {
        ch = data[index];
        if (ch === "_") continue;
        if (ch !== "0" && ch !== "1") return false;
        hasDigits = true;
      }
      return hasDigits && ch !== "_";
    }
    if (ch === "x") {
      index++;
      for (; index < max; index++) {
        ch = data[index];
        if (ch === "_") continue;
        if (!isHexCode(data.charCodeAt(index))) return false;
        hasDigits = true;
      }
      return hasDigits && ch !== "_";
    }
    for (; index < max; index++) {
      ch = data[index];
      if (ch === "_") continue;
      if (!isOctCode(data.charCodeAt(index))) return false;
      hasDigits = true;
    }
    return hasDigits && ch !== "_";
  }
  if (ch === "_") return false;
  for (; index < max; index++) {
    ch = data[index];
    if (ch === "_") continue;
    if (!isDecCode(data.charCodeAt(index))) {
      return false;
    }
    hasDigits = true;
  }
  if (!hasDigits || ch === "_") return false;
  return /^(:[0-5]?[0-9])+$/.test(data.slice(index));
}
function constructYamlInteger(data) {
  let value = data;
  if (value.includes("_")) {
    value = value.replace(/_/g, "");
  }
  let sign = 1;
  let ch = value[0];
  if (ch === "-" || ch === "+") {
    if (ch === "-") sign = -1;
    value = value.slice(1);
    ch = value[0];
  }
  if (value === "0") return 0;
  if (ch === "0") {
    if (value[1] === "b") return sign * parseInt(value.slice(2), 2);
    if (value[1] === "x") return sign * parseInt(value, 16);
    return sign * parseInt(value, 8);
  }
  return sign * parseInt(value, 10);
}
function isInteger(object) {
  if (object instanceof Number) object = object.valueOf();
  return typeof object === "number" && object % 1 === 0 && !isNegativeZero(object);
}
var int = {
  tag: "tag:yaml.org,2002:int",
  construct: constructYamlInteger,
  defaultStyle: "decimal",
  kind: "scalar",
  predicate: isInteger,
  represent: {
    // deno-lint-ignore ban-types
    binary(object) {
      const value = object instanceof Number ? object.valueOf() : object;
      return value >= 0 ? `0b${value.toString(2)}` : `-0b${value.toString(2).slice(1)}`;
    },
    // deno-lint-ignore ban-types
    octal(object) {
      const value = object instanceof Number ? object.valueOf() : object;
      return value >= 0 ? `0${value.toString(8)}` : `-0${value.toString(8).slice(1)}`;
    },
    // deno-lint-ignore ban-types
    decimal(object) {
      const value = object instanceof Number ? object.valueOf() : object;
      return value.toString(10);
    },
    // deno-lint-ignore ban-types
    hexadecimal(object) {
      const value = object instanceof Number ? object.valueOf() : object;
      return value >= 0 ? `0x${value.toString(16).toUpperCase()}` : `-0x${value.toString(16).toUpperCase().slice(1)}`;
    }
  },
  resolve: resolveYamlInteger
};

// node_modules/@std/yaml/_type/map.js
var map = {
  tag: "tag:yaml.org,2002:map",
  resolve() {
    return true;
  },
  construct(data) {
    return data !== null ? data : {};
  },
  kind: "mapping"
};

// node_modules/@std/yaml/_type/merge.js
var merge = {
  tag: "tag:yaml.org,2002:merge",
  kind: "scalar",
  resolve: (data) => data === "<<" || data === null,
  construct: (data) => data
};

// node_modules/@std/yaml/_type/nil.js
var nil = {
  tag: "tag:yaml.org,2002:null",
  kind: "scalar",
  defaultStyle: "lowercase",
  predicate: (object) => object === null,
  construct: () => null,
  resolve: (data) => {
    return data === "~" || data === "null" || data === "Null" || data === "NULL";
  },
  represent: {
    lowercase: () => "null",
    uppercase: () => "NULL",
    camelcase: () => "Null"
  }
};

// node_modules/@std/yaml/_type/omap.js
function resolveYamlOmap(data) {
  const objectKeys = /* @__PURE__ */ new Set();
  for (const object of data) {
    if (!isPlainObject(object)) return false;
    const keys = Object.keys(object);
    if (keys.length !== 1) return false;
    for (const key of keys) {
      if (objectKeys.has(key)) return false;
      objectKeys.add(key);
    }
  }
  return true;
}
var omap = {
  tag: "tag:yaml.org,2002:omap",
  kind: "sequence",
  resolve: resolveYamlOmap,
  construct(data) {
    return data;
  }
};

// node_modules/@std/yaml/_type/pairs.js
function resolveYamlPairs(data) {
  if (data === null) return true;
  return data.every((it) => isPlainObject(it) && Object.keys(it).length === 1);
}
var pairs = {
  tag: "tag:yaml.org,2002:pairs",
  construct(data) {
    return data?.flatMap(Object.entries) ?? [];
  },
  kind: "sequence",
  resolve: resolveYamlPairs
};

// node_modules/@std/yaml/_type/regexp.js
var REGEXP = /^\/(?<regexp>[\s\S]+)\/(?<modifiers>[gismuy]*)$/;
var regexp = {
  tag: "tag:yaml.org,2002:js/regexp",
  kind: "scalar",
  resolve(data) {
    if (data === null || !data.length) return false;
    if (data.charAt(0) === "/") {
      const groups = data.match(REGEXP)?.groups;
      if (!groups) return false;
      const modifiers = groups.modifiers ?? "";
      if (new Set(modifiers).size < modifiers.length) return false;
    }
    return true;
  },
  construct(data) {
    const { regexp: regexp2 = data, modifiers = "" } = data.match(REGEXP)?.groups ?? {};
    return new RegExp(regexp2, modifiers);
  },
  predicate: (object) => object instanceof RegExp,
  represent: (object) => object.toString()
};

// node_modules/@std/yaml/_type/seq.js
var seq = {
  tag: "tag:yaml.org,2002:seq",
  kind: "sequence",
  resolve: () => true,
  construct: (data) => data !== null ? data : []
};

// node_modules/@std/yaml/_type/set.js
var set = {
  tag: "tag:yaml.org,2002:set",
  kind: "mapping",
  construct: (data) => data !== null ? data : {},
  resolve: (data) => {
    if (data === null) return true;
    return Object.values(data).every((it) => it === null);
  }
};

// node_modules/@std/yaml/_type/str.js
var str = {
  tag: "tag:yaml.org,2002:str",
  kind: "scalar",
  resolve: () => true,
  construct: (data) => data !== null ? data : ""
};

// node_modules/@std/yaml/_type/timestamp.js
var YAML_DATE_REGEXP = new RegExp("^([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])$");
var YAML_TIMESTAMP_REGEXP = new RegExp("^([0-9][0-9][0-9][0-9])-([0-9][0-9]?)-([0-9][0-9]?)(?:[Tt]|[ \\t]+)([0-9][0-9]?):([0-9][0-9]):([0-9][0-9])(?:\\.([0-9]*))?(?:[ \\t]*(Z|([-+])([0-9][0-9]?)(?::([0-9][0-9]))?))?$");
function resolveYamlTimestamp(data) {
  if (data === null) return false;
  if (YAML_DATE_REGEXP.exec(data) !== null) return true;
  if (YAML_TIMESTAMP_REGEXP.exec(data) !== null) return true;
  return false;
}
function constructYamlTimestamp(data) {
  let match = YAML_DATE_REGEXP.exec(data);
  if (match === null) match = YAML_TIMESTAMP_REGEXP.exec(data);
  if (match === null) {
    throw new Error("Cannot construct YAML timestamp: date resolve error");
  }
  const year = +match[1];
  const month = +match[2] - 1;
  const day = +match[3];
  if (!match[4]) {
    return new Date(Date.UTC(year, month, day));
  }
  const hour = +match[4];
  const minute = +match[5];
  const second = +match[6];
  let fraction = 0;
  if (match[7]) {
    let partFraction = match[7].slice(0, 3);
    while (partFraction.length < 3) {
      partFraction += "0";
    }
    fraction = +partFraction;
  }
  let delta = null;
  if (match[9] && match[10]) {
    const tzHour = +match[10];
    const tzMinute = +(match[11] || 0);
    delta = (tzHour * 60 + tzMinute) * 6e4;
    if (match[9] === "-") delta = -delta;
  }
  const date = new Date(Date.UTC(year, month, day, hour, minute, second, fraction));
  if (delta) date.setTime(date.getTime() - delta);
  return date;
}
function representYamlTimestamp(date) {
  return date.toISOString();
}
var timestamp = {
  tag: "tag:yaml.org,2002:timestamp",
  construct: constructYamlTimestamp,
  predicate(object) {
    return object instanceof Date;
  },
  kind: "scalar",
  represent: representYamlTimestamp,
  resolve: resolveYamlTimestamp
};

// node_modules/@std/yaml/_type/undefined.js
var undefinedType = {
  tag: "tag:yaml.org,2002:js/undefined",
  kind: "scalar",
  resolve() {
    return true;
  },
  construct() {
    return void 0;
  },
  predicate(object) {
    return typeof object === "undefined";
  },
  represent() {
    return "";
  }
};

// node_modules/@std/yaml/_schema.js
function createTypeMap(implicitTypes, explicitTypes) {
  const result = {
    fallback: /* @__PURE__ */ new Map(),
    mapping: /* @__PURE__ */ new Map(),
    scalar: /* @__PURE__ */ new Map(),
    sequence: /* @__PURE__ */ new Map()
  };
  const fallbackMap = result.fallback;
  for (const type of [
    ...implicitTypes,
    ...explicitTypes
  ]) {
    const map2 = result[type.kind];
    map2.set(type.tag, type);
    fallbackMap.set(type.tag, type);
  }
  return result;
}
function createSchema({ explicitTypes = [], implicitTypes = [], include }) {
  if (include) {
    implicitTypes.push(...include.implicitTypes);
    explicitTypes.push(...include.explicitTypes);
  }
  const typeMap = createTypeMap(implicitTypes, explicitTypes);
  return {
    implicitTypes,
    explicitTypes,
    typeMap
  };
}
var FAILSAFE_SCHEMA = createSchema({
  explicitTypes: [
    str,
    seq,
    map
  ]
});
var JSON_SCHEMA = createSchema({
  implicitTypes: [
    nil,
    bool,
    int,
    float
  ],
  include: FAILSAFE_SCHEMA
});
var CORE_SCHEMA = createSchema({
  include: JSON_SCHEMA
});
var DEFAULT_SCHEMA = createSchema({
  explicitTypes: [
    binary,
    omap,
    pairs,
    set
  ],
  implicitTypes: [
    timestamp,
    merge
  ],
  include: CORE_SCHEMA
});
var EXTENDED_SCHEMA = createSchema({
  explicitTypes: [
    regexp,
    undefinedType
  ],
  include: DEFAULT_SCHEMA
});
var SCHEMA_MAP = /* @__PURE__ */ new Map([
  [
    "core",
    CORE_SCHEMA
  ],
  [
    "default",
    DEFAULT_SCHEMA
  ],
  [
    "failsafe",
    FAILSAFE_SCHEMA
  ],
  [
    "json",
    JSON_SCHEMA
  ],
  [
    "extended",
    EXTENDED_SCHEMA
  ]
]);

// node_modules/@std/yaml/_loader_state.js
var CONTEXT_FLOW_IN = 1;
var CONTEXT_FLOW_OUT = 2;
var CONTEXT_BLOCK_IN = 3;
var CONTEXT_BLOCK_OUT = 4;
var CHOMPING_CLIP = 1;
var CHOMPING_STRIP = 2;
var CHOMPING_KEEP = 3;
var PATTERN_NON_PRINTABLE = (
  // deno-lint-ignore no-control-regex
  /[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x84\x86-\x9F\uFFFE\uFFFF]|[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF]/
);
var PATTERN_NON_ASCII_LINE_BREAKS = /[\x85\u2028\u2029]/;
var PATTERN_FLOW_INDICATORS = /[,\[\]\{\}]/;
var PATTERN_TAG_HANDLE = /^(?:!|!!|![a-z\-]+!)$/i;
var PATTERN_TAG_URI = /^(?:!|[^,\[\]\{\}])(?:%[0-9a-f]{2}|[0-9a-z\-#;\/\?:@&=\+\$,_\.!~\*'\(\)\[\]])*$/i;
var ESCAPED_HEX_LENGTHS = /* @__PURE__ */ new Map([
  [
    120,
    2
  ],
  [
    117,
    4
  ],
  [
    85,
    8
  ]
]);
var SIMPLE_ESCAPE_SEQUENCES = /* @__PURE__ */ new Map([
  [
    48,
    "\0"
  ],
  [
    97,
    "\x07"
  ],
  [
    98,
    "\b"
  ],
  [
    116,
    "	"
  ],
  [
    9,
    "	"
  ],
  [
    110,
    "\n"
  ],
  [
    118,
    "\v"
  ],
  [
    102,
    "\f"
  ],
  [
    114,
    "\r"
  ],
  [
    101,
    "\x1B"
  ],
  [
    32,
    " "
  ],
  [
    34,
    '"'
  ],
  [
    47,
    "/"
  ],
  [
    92,
    "\\"
  ],
  [
    78,
    "\x85"
  ],
  [
    95,
    "\xA0"
  ],
  [
    76,
    "\u2028"
  ],
  [
    80,
    "\u2029"
  ]
]);
function hexCharCodeToNumber(charCode) {
  if (48 <= charCode && charCode <= 57) return charCode - 48;
  const lc = charCode | 32;
  if (97 <= lc && lc <= 102) return lc - 97 + 10;
  return -1;
}
function decimalCharCodeToNumber(charCode) {
  if (48 <= charCode && charCode <= 57) return charCode - 48;
  return -1;
}
function codepointToChar(codepoint) {
  if (codepoint <= 65535) return String.fromCharCode(codepoint);
  return String.fromCharCode((codepoint - 65536 >> 10) + 55296, (codepoint - 65536 & 1023) + 56320);
}
var INDENT = 4;
var MAX_LENGTH = 75;
var DELIMITERS = "\0\r\n\x85\u2028\u2029";
function getSnippet(buffer, position) {
  if (!buffer) return null;
  let start = position;
  let end = position;
  let head = "";
  let tail = "";
  while (start > 0 && !DELIMITERS.includes(buffer.charAt(start - 1))) {
    start--;
    if (position - start > MAX_LENGTH / 2 - 1) {
      head = " ... ";
      start += 5;
      break;
    }
  }
  while (end < buffer.length && !DELIMITERS.includes(buffer.charAt(end))) {
    end++;
    if (end - position > MAX_LENGTH / 2 - 1) {
      tail = " ... ";
      end -= 5;
      break;
    }
  }
  const snippet = buffer.slice(start, end);
  const indent = " ".repeat(INDENT);
  const caretIndent = " ".repeat(INDENT + position - start + head.length);
  return `${indent + head + snippet + tail}
${caretIndent}^`;
}
function markToString(buffer, position, line, column) {
  let where = `at line ${line + 1}, column ${column + 1}`;
  const snippet = getSnippet(buffer, position);
  if (snippet) where += `:
${snippet}`;
  return where;
}
var LoaderState = class {
  input;
  length;
  lineIndent = 0;
  lineStart = 0;
  position = 0;
  line = 0;
  onWarning;
  allowDuplicateKeys;
  implicitTypes;
  typeMap;
  version;
  checkLineBreaks = false;
  tagMap = /* @__PURE__ */ new Map();
  anchorMap = /* @__PURE__ */ new Map();
  tag;
  anchor;
  kind;
  result = "";
  constructor(input, { schema = DEFAULT_SCHEMA, onWarning, allowDuplicateKeys = false }) {
    this.input = input;
    this.onWarning = onWarning;
    this.allowDuplicateKeys = allowDuplicateKeys;
    this.implicitTypes = schema.implicitTypes;
    this.typeMap = schema.typeMap;
    this.length = input.length;
    this.version = null;
    this.readIndent();
  }
  readIndent() {
    let char = this.peek();
    while (char === SPACE) {
      this.lineIndent += 1;
      char = this.next();
    }
  }
  peek(offset = 0) {
    return this.input.charCodeAt(this.position + offset);
  }
  next() {
    this.position += 1;
    return this.peek();
  }
  #createError(message) {
    const mark = markToString(this.input, this.position, this.line, this.position - this.lineStart);
    return new SyntaxError(`${message} ${mark}`);
  }
  throwError(message) {
    throw this.#createError(message);
  }
  dispatchWarning(message) {
    const error = this.#createError(message);
    this.onWarning?.(error);
  }
  yamlDirectiveHandler(...args) {
    if (this.version !== null) {
      return this.throwError("Cannot handle YAML directive: duplication of %YAML directive");
    }
    if (args.length !== 1) {
      return this.throwError("Cannot handle YAML directive: YAML directive accepts exactly one argument");
    }
    const match = /^([0-9]+)\.([0-9]+)$/.exec(args[0]);
    if (match === null) {
      return this.throwError("Cannot handle YAML directive: ill-formed argument");
    }
    const major = parseInt(match[1], 10);
    const minor = parseInt(match[2], 10);
    if (major !== 1) {
      return this.throwError("Cannot handle YAML directive: unacceptable YAML version");
    }
    this.version = args[0] ?? null;
    this.checkLineBreaks = minor < 2;
    if (minor !== 1 && minor !== 2) {
      return this.dispatchWarning("Cannot handle YAML directive: unsupported YAML version");
    }
  }
  tagDirectiveHandler(...args) {
    if (args.length !== 2) {
      return this.throwError(`Cannot handle tag directive: directive accepts exactly two arguments, received ${args.length}`);
    }
    const handle = args[0];
    const prefix = args[1];
    if (!PATTERN_TAG_HANDLE.test(handle)) {
      return this.throwError(`Cannot handle tag directive: ill-formed handle (first argument) in "${handle}"`);
    }
    if (this.tagMap.has(handle)) {
      return this.throwError(`Cannot handle tag directive: previously declared suffix for "${handle}" tag handle`);
    }
    if (!PATTERN_TAG_URI.test(prefix)) {
      return this.throwError("Cannot handle tag directive: ill-formed tag prefix (second argument) of the TAG directive");
    }
    this.tagMap.set(handle, prefix);
  }
  captureSegment(start, end, checkJson) {
    let result;
    if (start < end) {
      result = this.input.slice(start, end);
      if (checkJson) {
        for (let position = 0; position < result.length; position++) {
          const character = result.charCodeAt(position);
          if (!(character === 9 || 32 <= character && character <= 1114111)) {
            return this.throwError(`Expected valid JSON character: received "${character}"`);
          }
        }
      } else if (PATTERN_NON_PRINTABLE.test(result)) {
        return this.throwError("Stream contains non-printable characters");
      }
      this.result += result;
    }
  }
  readBlockSequence(nodeIndent) {
    let line;
    let following;
    let detected = false;
    let ch;
    const tag = this.tag;
    const anchor = this.anchor;
    const result = [];
    if (this.anchor !== null && typeof this.anchor !== "undefined") {
      this.anchorMap.set(this.anchor, result);
    }
    ch = this.peek();
    while (ch !== 0) {
      if (ch !== MINUS) {
        break;
      }
      following = this.peek(1);
      if (!isWhiteSpaceOrEOL(following)) {
        break;
      }
      detected = true;
      this.position++;
      if (this.skipSeparationSpace(true, -1)) {
        if (this.lineIndent <= nodeIndent) {
          result.push(null);
          ch = this.peek();
          continue;
        }
      }
      line = this.line;
      this.composeNode(nodeIndent, CONTEXT_BLOCK_IN, false, true);
      result.push(this.result);
      this.skipSeparationSpace(true, -1);
      ch = this.peek();
      if ((this.line === line || this.lineIndent > nodeIndent) && ch !== 0) {
        return this.throwError("Cannot read block sequence: bad indentation of a sequence entry");
      } else if (this.lineIndent < nodeIndent) {
        break;
      }
    }
    if (detected) {
      this.tag = tag;
      this.anchor = anchor;
      this.kind = "sequence";
      this.result = result;
      return true;
    }
    return false;
  }
  mergeMappings(destination, source, overridableKeys) {
    if (!isObject(source)) {
      return this.throwError("Cannot merge mappings: the provided source object is unacceptable");
    }
    for (const [key, value] of Object.entries(source)) {
      if (Object.hasOwn(destination, key)) continue;
      Object.defineProperty(destination, key, {
        value,
        writable: true,
        enumerable: true,
        configurable: true
      });
      overridableKeys.add(key);
    }
  }
  storeMappingPair(result, overridableKeys, keyTag, keyNode, valueNode, startLine, startPos) {
    if (Array.isArray(keyNode)) {
      keyNode = Array.prototype.slice.call(keyNode);
      for (let index = 0; index < keyNode.length; index++) {
        if (Array.isArray(keyNode[index])) {
          return this.throwError("Cannot store mapping pair: nested arrays are not supported inside keys");
        }
        if (typeof keyNode === "object" && isPlainObject(keyNode[index])) {
          keyNode[index] = "[object Object]";
        }
      }
    }
    if (typeof keyNode === "object" && isPlainObject(keyNode)) {
      keyNode = "[object Object]";
    }
    keyNode = String(keyNode);
    if (keyTag === "tag:yaml.org,2002:merge") {
      if (Array.isArray(valueNode)) {
        for (let index = 0; index < valueNode.length; index++) {
          this.mergeMappings(result, valueNode[index], overridableKeys);
        }
      } else {
        this.mergeMappings(result, valueNode, overridableKeys);
      }
    } else {
      if (!this.allowDuplicateKeys && !overridableKeys.has(keyNode) && Object.hasOwn(result, keyNode)) {
        this.line = startLine || this.line;
        this.position = startPos || this.position;
        return this.throwError("Cannot store mapping pair: duplicated key");
      }
      Object.defineProperty(result, keyNode, {
        value: valueNode,
        writable: true,
        enumerable: true,
        configurable: true
      });
      overridableKeys.delete(keyNode);
    }
    return result;
  }
  readLineBreak() {
    const ch = this.peek();
    if (ch === LINE_FEED) {
      this.position++;
    } else if (ch === CARRIAGE_RETURN) {
      this.position++;
      if (this.peek() === LINE_FEED) {
        this.position++;
      }
    } else {
      return this.throwError("Cannot read line: line break not found");
    }
    this.line += 1;
    this.lineStart = this.position;
  }
  skipSeparationSpace(allowComments, checkIndent) {
    let lineBreaks = 0;
    let ch = this.peek();
    while (ch !== 0) {
      while (isWhiteSpace(ch)) {
        ch = this.next();
      }
      if (allowComments && ch === SHARP) {
        do {
          ch = this.next();
        } while (ch !== LINE_FEED && ch !== CARRIAGE_RETURN && ch !== 0);
      }
      if (isEOL(ch)) {
        this.readLineBreak();
        ch = this.peek();
        lineBreaks++;
        this.lineIndent = 0;
        this.readIndent();
        ch = this.peek();
      } else {
        break;
      }
    }
    if (checkIndent !== -1 && lineBreaks !== 0 && this.lineIndent < checkIndent) {
      this.dispatchWarning("deficient indentation");
    }
    return lineBreaks;
  }
  testDocumentSeparator() {
    let ch = this.peek();
    if ((ch === MINUS || ch === DOT) && ch === this.peek(1) && ch === this.peek(2)) {
      ch = this.peek(3);
      if (ch === 0 || isWhiteSpaceOrEOL(ch)) {
        return true;
      }
    }
    return false;
  }
  writeFoldedLines(count) {
    if (count === 1) {
      this.result += " ";
    } else if (count > 1) {
      this.result += "\n".repeat(count - 1);
    }
  }
  readPlainScalar(nodeIndent, withinFlowCollection) {
    const kind = this.kind;
    const result = this.result;
    let ch = this.peek();
    if (isWhiteSpaceOrEOL(ch) || isFlowIndicator(ch) || ch === SHARP || ch === AMPERSAND || ch === ASTERISK || ch === EXCLAMATION || ch === VERTICAL_LINE || ch === GREATER_THAN || ch === SINGLE_QUOTE || ch === DOUBLE_QUOTE || ch === PERCENT || ch === COMMERCIAL_AT || ch === GRAVE_ACCENT) {
      return false;
    }
    let following;
    if (ch === QUESTION || ch === MINUS) {
      following = this.peek(1);
      if (isWhiteSpaceOrEOL(following) || withinFlowCollection && isFlowIndicator(following)) {
        return false;
      }
    }
    this.kind = "scalar";
    this.result = "";
    let captureEnd = this.position;
    let captureStart = this.position;
    let hasPendingContent = false;
    let line = 0;
    while (ch !== 0) {
      if (ch === COLON) {
        following = this.peek(1);
        if (isWhiteSpaceOrEOL(following) || withinFlowCollection && isFlowIndicator(following)) {
          break;
        }
      } else if (ch === SHARP) {
        const preceding = this.peek(-1);
        if (isWhiteSpaceOrEOL(preceding)) {
          break;
        }
      } else if (this.position === this.lineStart && this.testDocumentSeparator() || withinFlowCollection && isFlowIndicator(ch)) {
        break;
      } else if (isEOL(ch)) {
        line = this.line;
        const lineStart = this.lineStart;
        const lineIndent = this.lineIndent;
        this.skipSeparationSpace(false, -1);
        if (this.lineIndent >= nodeIndent) {
          hasPendingContent = true;
          ch = this.peek();
          continue;
        } else {
          this.position = captureEnd;
          this.line = line;
          this.lineStart = lineStart;
          this.lineIndent = lineIndent;
          break;
        }
      }
      if (hasPendingContent) {
        this.captureSegment(captureStart, captureEnd, false);
        this.writeFoldedLines(this.line - line);
        captureStart = captureEnd = this.position;
        hasPendingContent = false;
      }
      if (!isWhiteSpace(ch)) {
        captureEnd = this.position + 1;
      }
      ch = this.next();
    }
    this.captureSegment(captureStart, captureEnd, false);
    if (this.result) {
      return true;
    }
    this.kind = kind;
    this.result = result;
    return false;
  }
  readSingleQuotedScalar(nodeIndent) {
    let ch;
    let captureStart;
    let captureEnd;
    ch = this.peek();
    if (ch !== SINGLE_QUOTE) {
      return false;
    }
    this.kind = "scalar";
    this.result = "";
    this.position++;
    captureStart = captureEnd = this.position;
    while ((ch = this.peek()) !== 0) {
      if (ch === SINGLE_QUOTE) {
        this.captureSegment(captureStart, this.position, true);
        ch = this.next();
        if (ch === SINGLE_QUOTE) {
          captureStart = this.position;
          this.position++;
          captureEnd = this.position;
        } else {
          return true;
        }
      } else if (isEOL(ch)) {
        this.captureSegment(captureStart, captureEnd, true);
        this.writeFoldedLines(this.skipSeparationSpace(false, nodeIndent));
        captureStart = captureEnd = this.position;
      } else if (this.position === this.lineStart && this.testDocumentSeparator()) {
        return this.throwError("Unexpected end of the document within a single quoted scalar");
      } else {
        this.position++;
        captureEnd = this.position;
      }
    }
    return this.throwError("Unexpected end of the stream within a single quoted scalar");
  }
  readDoubleQuotedScalar(nodeIndent) {
    let ch = this.peek();
    if (ch !== DOUBLE_QUOTE) {
      return false;
    }
    this.kind = "scalar";
    this.result = "";
    this.position++;
    let captureEnd = this.position;
    let captureStart = this.position;
    let tmp;
    while ((ch = this.peek()) !== 0) {
      if (ch === DOUBLE_QUOTE) {
        this.captureSegment(captureStart, this.position, true);
        this.position++;
        return true;
      }
      if (ch === BACKSLASH) {
        this.captureSegment(captureStart, this.position, true);
        ch = this.next();
        if (isEOL(ch)) {
          this.skipSeparationSpace(false, nodeIndent);
        } else if (ch < 256 && SIMPLE_ESCAPE_SEQUENCES.has(ch)) {
          this.result += SIMPLE_ESCAPE_SEQUENCES.get(ch);
          this.position++;
        } else if ((tmp = ESCAPED_HEX_LENGTHS.get(ch) ?? 0) > 0) {
          let hexLength = tmp;
          let hexResult = 0;
          for (; hexLength > 0; hexLength--) {
            ch = this.next();
            if ((tmp = hexCharCodeToNumber(ch)) >= 0) {
              hexResult = (hexResult << 4) + tmp;
            } else {
              return this.throwError("Cannot read double quoted scalar: expected hexadecimal character");
            }
          }
          this.result += codepointToChar(hexResult);
          this.position++;
        } else {
          return this.throwError("Cannot read double quoted scalar: unknown escape sequence");
        }
        captureStart = captureEnd = this.position;
      } else if (isEOL(ch)) {
        this.captureSegment(captureStart, captureEnd, true);
        this.writeFoldedLines(this.skipSeparationSpace(false, nodeIndent));
        captureStart = captureEnd = this.position;
      } else if (this.position === this.lineStart && this.testDocumentSeparator()) {
        return this.throwError("Unexpected end of the document within a double quoted scalar");
      } else {
        this.position++;
        captureEnd = this.position;
      }
    }
    return this.throwError("Unexpected end of the stream within a double quoted scalar");
  }
  readFlowCollection(nodeIndent) {
    let ch = this.peek();
    let terminator;
    let isMapping = true;
    let result = {};
    if (ch === LEFT_SQUARE_BRACKET) {
      terminator = RIGHT_SQUARE_BRACKET;
      isMapping = false;
      result = [];
    } else if (ch === LEFT_CURLY_BRACKET) {
      terminator = RIGHT_CURLY_BRACKET;
    } else {
      return false;
    }
    if (this.anchor !== null && typeof this.anchor !== "undefined") {
      this.anchorMap.set(this.anchor, result);
    }
    ch = this.next();
    const tag = this.tag;
    const anchor = this.anchor;
    let readNext = true;
    let valueNode = null;
    let keyNode = null;
    let keyTag = null;
    let isExplicitPair = false;
    let isPair = false;
    let following = 0;
    let line = 0;
    const overridableKeys = /* @__PURE__ */ new Set();
    while (ch !== 0) {
      this.skipSeparationSpace(true, nodeIndent);
      ch = this.peek();
      if (ch === terminator) {
        this.position++;
        this.tag = tag;
        this.anchor = anchor;
        this.kind = isMapping ? "mapping" : "sequence";
        this.result = result;
        return true;
      }
      if (!readNext) {
        return this.throwError("Cannot read flow collection: missing comma between flow collection entries");
      }
      keyTag = keyNode = valueNode = null;
      isPair = isExplicitPair = false;
      if (ch === QUESTION) {
        following = this.peek(1);
        if (isWhiteSpaceOrEOL(following)) {
          isPair = isExplicitPair = true;
          this.position++;
          this.skipSeparationSpace(true, nodeIndent);
        }
      }
      line = this.line;
      this.composeNode(nodeIndent, CONTEXT_FLOW_IN, false, true);
      keyTag = this.tag || null;
      keyNode = this.result;
      this.skipSeparationSpace(true, nodeIndent);
      ch = this.peek();
      if ((isExplicitPair || this.line === line) && ch === COLON) {
        isPair = true;
        ch = this.next();
        this.skipSeparationSpace(true, nodeIndent);
        this.composeNode(nodeIndent, CONTEXT_FLOW_IN, false, true);
        valueNode = this.result;
      }
      if (isMapping) {
        this.storeMappingPair(result, overridableKeys, keyTag, keyNode, valueNode);
      } else if (isPair) {
        result.push(this.storeMappingPair({}, overridableKeys, keyTag, keyNode, valueNode));
      } else {
        result.push(keyNode);
      }
      this.skipSeparationSpace(true, nodeIndent);
      ch = this.peek();
      if (ch === COMMA) {
        readNext = true;
        ch = this.next();
      } else {
        readNext = false;
      }
    }
    return this.throwError("Cannot read flow collection: unexpected end of the stream within a flow collection");
  }
  // Handles block scaler styles: e.g. '|', '>', '|-' and '>-'.
  // https://yaml.org/spec/1.2.2/#81-block-scalar-styles
  readBlockScalar(nodeIndent) {
    let chomping = CHOMPING_CLIP;
    let didReadContent = false;
    let detectedIndent = false;
    let textIndent = nodeIndent;
    let emptyLines = 0;
    let atMoreIndented = false;
    let ch = this.peek();
    let folding = false;
    if (ch === VERTICAL_LINE) {
      folding = false;
    } else if (ch === GREATER_THAN) {
      folding = true;
    } else {
      return false;
    }
    this.kind = "scalar";
    this.result = "";
    let tmp = 0;
    while (ch !== 0) {
      ch = this.next();
      if (ch === PLUS || ch === MINUS) {
        if (CHOMPING_CLIP === chomping) {
          chomping = ch === PLUS ? CHOMPING_KEEP : CHOMPING_STRIP;
        } else {
          return this.throwError("Cannot read block: chomping mode identifier repeated");
        }
      } else if ((tmp = decimalCharCodeToNumber(ch)) >= 0) {
        if (tmp === 0) {
          return this.throwError("Cannot read block: indentation width must be greater than 0");
        } else if (!detectedIndent) {
          textIndent = nodeIndent + tmp - 1;
          detectedIndent = true;
        } else {
          return this.throwError("Cannot read block: indentation width identifier repeated");
        }
      } else {
        break;
      }
    }
    if (isWhiteSpace(ch)) {
      do {
        ch = this.next();
      } while (isWhiteSpace(ch));
      if (ch === SHARP) {
        do {
          ch = this.next();
        } while (!isEOL(ch) && ch !== 0);
      }
    }
    while (ch !== 0) {
      this.readLineBreak();
      this.lineIndent = 0;
      ch = this.peek();
      while ((!detectedIndent || this.lineIndent < textIndent) && ch === SPACE) {
        this.lineIndent++;
        ch = this.next();
      }
      if (!detectedIndent && this.lineIndent > textIndent) {
        textIndent = this.lineIndent;
      }
      if (isEOL(ch)) {
        emptyLines++;
        continue;
      }
      if (this.lineIndent < textIndent) {
        if (chomping === CHOMPING_KEEP) {
          this.result += "\n".repeat(didReadContent ? 1 + emptyLines : emptyLines);
        } else if (chomping === CHOMPING_CLIP) {
          if (didReadContent) {
            this.result += "\n";
          }
        }
        break;
      }
      if (folding) {
        if (isWhiteSpace(ch)) {
          atMoreIndented = true;
          this.result += "\n".repeat(didReadContent ? 1 + emptyLines : emptyLines);
        } else if (atMoreIndented) {
          atMoreIndented = false;
          this.result += "\n".repeat(emptyLines + 1);
        } else if (emptyLines === 0) {
          if (didReadContent) {
            this.result += " ";
          }
        } else {
          this.result += "\n".repeat(emptyLines);
        }
      } else {
        this.result += "\n".repeat(didReadContent ? 1 + emptyLines : emptyLines);
      }
      didReadContent = true;
      detectedIndent = true;
      emptyLines = 0;
      const captureStart = this.position;
      while (!isEOL(ch) && ch !== 0) {
        ch = this.next();
      }
      this.captureSegment(captureStart, this.position, false);
    }
    return true;
  }
  readBlockMapping(nodeIndent, flowIndent) {
    const tag = this.tag;
    const anchor = this.anchor;
    const result = {};
    const overridableKeys = /* @__PURE__ */ new Set();
    let following;
    let allowCompact = false;
    let line;
    let pos;
    let keyTag = null;
    let keyNode = null;
    let valueNode = null;
    let atExplicitKey = false;
    let detected = false;
    let ch;
    if (this.anchor !== null && typeof this.anchor !== "undefined") {
      this.anchorMap.set(this.anchor, result);
    }
    ch = this.peek();
    while (ch !== 0) {
      following = this.peek(1);
      line = this.line;
      pos = this.position;
      if ((ch === QUESTION || ch === COLON) && isWhiteSpaceOrEOL(following)) {
        if (ch === QUESTION) {
          if (atExplicitKey) {
            this.storeMappingPair(result, overridableKeys, keyTag, keyNode, null);
            keyTag = keyNode = valueNode = null;
          }
          detected = true;
          atExplicitKey = true;
          allowCompact = true;
        } else if (atExplicitKey) {
          atExplicitKey = false;
          allowCompact = true;
        } else {
          return this.throwError("Cannot read block as explicit mapping pair is incomplete: a key node is missed or followed by a non-tabulated empty line");
        }
        this.position += 1;
        ch = following;
      } else if (this.composeNode(flowIndent, CONTEXT_FLOW_OUT, false, true)) {
        if (this.line === line) {
          ch = this.peek();
          while (isWhiteSpace(ch)) {
            ch = this.next();
          }
          if (ch === COLON) {
            ch = this.next();
            if (!isWhiteSpaceOrEOL(ch)) {
              return this.throwError("Cannot read block: a whitespace character is expected after the key-value separator within a block mapping");
            }
            if (atExplicitKey) {
              this.storeMappingPair(result, overridableKeys, keyTag, keyNode, null);
              keyTag = keyNode = valueNode = null;
            }
            detected = true;
            atExplicitKey = false;
            allowCompact = false;
            keyTag = this.tag;
            keyNode = this.result;
          } else if (detected) {
            return this.throwError("Cannot read an implicit mapping pair: missing colon");
          } else {
            this.tag = tag;
            this.anchor = anchor;
            return true;
          }
        } else if (detected) {
          return this.throwError("Cannot read a block mapping entry: a multiline key may not be an implicit key");
        } else {
          this.tag = tag;
          this.anchor = anchor;
          return true;
        }
      } else {
        break;
      }
      if (this.line === line || this.lineIndent > nodeIndent) {
        if (this.composeNode(nodeIndent, CONTEXT_BLOCK_OUT, true, allowCompact)) {
          if (atExplicitKey) {
            keyNode = this.result;
          } else {
            valueNode = this.result;
          }
        }
        if (!atExplicitKey) {
          this.storeMappingPair(result, overridableKeys, keyTag, keyNode, valueNode, line, pos);
          keyTag = keyNode = valueNode = null;
        }
        this.skipSeparationSpace(true, -1);
        ch = this.peek();
      }
      if (this.lineIndent > nodeIndent && ch !== 0) {
        return this.throwError("Cannot read block: bad indentation of a mapping entry");
      } else if (this.lineIndent < nodeIndent) {
        break;
      }
    }
    if (atExplicitKey) {
      this.storeMappingPair(result, overridableKeys, keyTag, keyNode, null);
    }
    if (detected) {
      this.tag = tag;
      this.anchor = anchor;
      this.kind = "mapping";
      this.result = result;
    }
    return detected;
  }
  readTagProperty() {
    let position;
    let isVerbatim = false;
    let isNamed = false;
    let tagHandle = "";
    let tagName;
    let ch;
    ch = this.peek();
    if (ch !== EXCLAMATION) return false;
    if (this.tag !== null) {
      return this.throwError("Cannot read tag property: duplication of a tag property");
    }
    ch = this.next();
    if (ch === SMALLER_THAN) {
      isVerbatim = true;
      ch = this.next();
    } else if (ch === EXCLAMATION) {
      isNamed = true;
      tagHandle = "!!";
      ch = this.next();
    } else {
      tagHandle = "!";
    }
    position = this.position;
    if (isVerbatim) {
      do {
        ch = this.next();
      } while (ch !== 0 && ch !== GREATER_THAN);
      if (this.position < this.length) {
        tagName = this.input.slice(position, this.position);
        ch = this.next();
      } else {
        return this.throwError("Cannot read tag property: unexpected end of stream");
      }
    } else {
      while (ch !== 0 && !isWhiteSpaceOrEOL(ch)) {
        if (ch === EXCLAMATION) {
          if (!isNamed) {
            tagHandle = this.input.slice(position - 1, this.position + 1);
            if (!PATTERN_TAG_HANDLE.test(tagHandle)) {
              return this.throwError("Cannot read tag property: named tag handle contains invalid characters");
            }
            isNamed = true;
            position = this.position + 1;
          } else {
            return this.throwError("Cannot read tag property: tag suffix cannot contain an exclamation mark");
          }
        }
        ch = this.next();
      }
      tagName = this.input.slice(position, this.position);
      if (PATTERN_FLOW_INDICATORS.test(tagName)) {
        return this.throwError("Cannot read tag property: tag suffix cannot contain flow indicator characters");
      }
    }
    if (tagName && !PATTERN_TAG_URI.test(tagName)) {
      return this.throwError(`Cannot read tag property: invalid characters in tag name "${tagName}"`);
    }
    if (isVerbatim) {
      this.tag = tagName;
    } else if (this.tagMap.has(tagHandle)) {
      this.tag = this.tagMap.get(tagHandle) + tagName;
    } else if (tagHandle === "!") {
      this.tag = `!${tagName}`;
    } else if (tagHandle === "!!") {
      this.tag = `tag:yaml.org,2002:${tagName}`;
    } else {
      return this.throwError(`Cannot read tag property: undeclared tag handle "${tagHandle}"`);
    }
    return true;
  }
  readAnchorProperty() {
    let ch = this.peek();
    if (ch !== AMPERSAND) return false;
    if (this.anchor !== null) {
      return this.throwError("Cannot read anchor property: duplicate anchor property");
    }
    ch = this.next();
    const position = this.position;
    while (ch !== 0 && !isWhiteSpaceOrEOL(ch) && !isFlowIndicator(ch)) {
      ch = this.next();
    }
    if (this.position === position) {
      return this.throwError("Cannot read anchor property: name of an anchor node must contain at least one character");
    }
    this.anchor = this.input.slice(position, this.position);
    return true;
  }
  readAlias() {
    if (this.peek() !== ASTERISK) return false;
    let ch = this.next();
    const position = this.position;
    while (ch !== 0 && !isWhiteSpaceOrEOL(ch) && !isFlowIndicator(ch)) {
      ch = this.next();
    }
    if (this.position === position) {
      return this.throwError("Cannot read alias: alias name must contain at least one character");
    }
    const alias = this.input.slice(position, this.position);
    if (!this.anchorMap.has(alias)) {
      return this.throwError(`Cannot read alias: unidentified alias "${alias}"`);
    }
    this.result = this.anchorMap.get(alias);
    this.skipSeparationSpace(true, -1);
    return true;
  }
  composeNode(parentIndent, nodeContext, allowToSeek, allowCompact) {
    let allowBlockScalars;
    let allowBlockCollections;
    let indentStatus = 1;
    let atNewLine = false;
    let hasContent = false;
    let type;
    let flowIndent;
    let blockIndent;
    this.tag = null;
    this.anchor = null;
    this.kind = null;
    this.result = null;
    const allowBlockStyles = allowBlockScalars = allowBlockCollections = CONTEXT_BLOCK_OUT === nodeContext || CONTEXT_BLOCK_IN === nodeContext;
    if (allowToSeek) {
      if (this.skipSeparationSpace(true, -1)) {
        atNewLine = true;
        if (this.lineIndent > parentIndent) {
          indentStatus = 1;
        } else if (this.lineIndent === parentIndent) {
          indentStatus = 0;
        } else if (this.lineIndent < parentIndent) {
          indentStatus = -1;
        }
      }
    }
    if (indentStatus === 1) {
      while (this.readTagProperty() || this.readAnchorProperty()) {
        if (this.skipSeparationSpace(true, -1)) {
          atNewLine = true;
          allowBlockCollections = allowBlockStyles;
          if (this.lineIndent > parentIndent) {
            indentStatus = 1;
          } else if (this.lineIndent === parentIndent) {
            indentStatus = 0;
          } else if (this.lineIndent < parentIndent) {
            indentStatus = -1;
          }
        } else {
          allowBlockCollections = false;
        }
      }
    }
    if (allowBlockCollections) {
      allowBlockCollections = atNewLine || allowCompact;
    }
    if (indentStatus === 1 || CONTEXT_BLOCK_OUT === nodeContext) {
      const cond = CONTEXT_FLOW_IN === nodeContext || CONTEXT_FLOW_OUT === nodeContext;
      flowIndent = cond ? parentIndent : parentIndent + 1;
      blockIndent = this.position - this.lineStart;
      if (indentStatus === 1) {
        if (allowBlockCollections && (this.readBlockSequence(blockIndent) || this.readBlockMapping(blockIndent, flowIndent)) || this.readFlowCollection(flowIndent)) {
          hasContent = true;
        } else {
          if (allowBlockScalars && this.readBlockScalar(flowIndent) || this.readSingleQuotedScalar(flowIndent) || this.readDoubleQuotedScalar(flowIndent)) {
            hasContent = true;
          } else if (this.readAlias()) {
            hasContent = true;
            if (this.tag !== null || this.anchor !== null) {
              return this.throwError("Cannot compose node: alias node should not have any properties");
            }
          } else if (this.readPlainScalar(flowIndent, CONTEXT_FLOW_IN === nodeContext)) {
            hasContent = true;
            if (this.tag === null) {
              this.tag = "?";
            }
          }
          if (this.anchor !== null) {
            this.anchorMap.set(this.anchor, this.result);
          }
        }
      } else if (indentStatus === 0) {
        hasContent = allowBlockCollections && this.readBlockSequence(blockIndent);
      }
    }
    if (this.tag !== null && this.tag !== "!") {
      if (this.tag === "?") {
        for (let typeIndex = 0; typeIndex < this.implicitTypes.length; typeIndex++) {
          type = this.implicitTypes[typeIndex];
          if (type.resolve(this.result)) {
            this.result = type.construct(this.result);
            this.tag = type.tag;
            if (this.anchor !== null) {
              this.anchorMap.set(this.anchor, this.result);
            }
            break;
          }
        }
      } else if (this.typeMap[this.kind ?? "fallback"].has(this.tag)) {
        const map2 = this.typeMap[this.kind ?? "fallback"];
        type = map2.get(this.tag);
        if (this.result !== null && type.kind !== this.kind) {
          return this.throwError(`Unacceptable node kind for !<${this.tag}> tag: it should be "${type.kind}", not "${this.kind}"`);
        }
        if (!type.resolve(this.result)) {
          return this.throwError(`Cannot resolve a node with !<${this.tag}> explicit tag`);
        } else {
          this.result = type.construct(this.result);
          if (this.anchor !== null) {
            this.anchorMap.set(this.anchor, this.result);
          }
        }
      } else {
        return this.throwError(`Cannot resolve unknown tag !<${this.tag}>`);
      }
    }
    return this.tag !== null || this.anchor !== null || hasContent;
  }
  readDocument() {
    const documentStart = this.position;
    let position;
    let directiveName;
    let directiveArgs;
    let hasDirectives = false;
    let ch;
    this.version = null;
    this.checkLineBreaks = false;
    this.tagMap = /* @__PURE__ */ new Map();
    this.anchorMap = /* @__PURE__ */ new Map();
    while ((ch = this.peek()) !== 0) {
      this.skipSeparationSpace(true, -1);
      ch = this.peek();
      if (this.lineIndent > 0 || ch !== PERCENT) {
        break;
      }
      hasDirectives = true;
      ch = this.next();
      position = this.position;
      while (ch !== 0 && !isWhiteSpaceOrEOL(ch)) {
        ch = this.next();
      }
      directiveName = this.input.slice(position, this.position);
      directiveArgs = [];
      if (directiveName.length < 1) {
        return this.throwError("Cannot read document: directive name length must be greater than zero");
      }
      while (ch !== 0) {
        while (isWhiteSpace(ch)) {
          ch = this.next();
        }
        if (ch === SHARP) {
          do {
            ch = this.next();
          } while (ch !== 0 && !isEOL(ch));
          break;
        }
        if (isEOL(ch)) break;
        position = this.position;
        while (ch !== 0 && !isWhiteSpaceOrEOL(ch)) {
          ch = this.next();
        }
        directiveArgs.push(this.input.slice(position, this.position));
      }
      if (ch !== 0) this.readLineBreak();
      switch (directiveName) {
        case "YAML":
          this.yamlDirectiveHandler(...directiveArgs);
          break;
        case "TAG":
          this.tagDirectiveHandler(...directiveArgs);
          break;
        default:
          this.dispatchWarning(`unknown document directive "${directiveName}"`);
          break;
      }
    }
    this.skipSeparationSpace(true, -1);
    if (this.lineIndent === 0 && this.peek() === MINUS && this.peek(1) === MINUS && this.peek(2) === MINUS) {
      this.position += 3;
      this.skipSeparationSpace(true, -1);
    } else if (hasDirectives) {
      return this.throwError("Cannot read document: directives end mark is expected");
    }
    this.composeNode(this.lineIndent - 1, CONTEXT_BLOCK_OUT, false, true);
    this.skipSeparationSpace(true, -1);
    if (this.checkLineBreaks && PATTERN_NON_ASCII_LINE_BREAKS.test(this.input.slice(documentStart, this.position))) {
      this.dispatchWarning("non-ASCII line breaks are interpreted as content");
    }
    if (this.position === this.lineStart && this.testDocumentSeparator()) {
      if (this.peek() === DOT) {
        this.position += 3;
        this.skipSeparationSpace(true, -1);
      }
    } else if (this.position < this.length - 1) {
      return this.throwError("Cannot read document: end of the stream or a document separator is expected");
    }
    return this.result;
  }
  *readDocuments() {
    while (this.position < this.length - 1) {
      yield this.readDocument();
    }
  }
};

// node_modules/@std/yaml/parse.js
function sanitizeInput(input) {
  input = String(input);
  if (input.length > 0) {
    if (!isEOL(input.charCodeAt(input.length - 1))) input += "\n";
    if (input.charCodeAt(0) === 65279) input = input.slice(1);
  }
  input += "\0";
  return input;
}
function parse(content, options = {}) {
  content = sanitizeInput(content);
  const state = new LoaderState(content, {
    ...options,
    schema: SCHEMA_MAP.get(options.schema)
  });
  const documentGenerator = state.readDocuments();
  const document = documentGenerator.next().value;
  if (!documentGenerator.next().done) {
    throw new SyntaxError("Found more than 1 document in the stream: expected a single document");
  }
  return document ?? null;
}
function parseAll(content, options = {}) {
  content = sanitizeInput(content);
  const state = new LoaderState(content, {
    ...options,
    schema: SCHEMA_MAP.get(options.schema)
  });
  return [
    ...state.readDocuments()
  ];
}

// node_modules/@std/yaml/_dumper_state.js
var STYLE_PLAIN = 1;
var STYLE_SINGLE = 2;
var STYLE_LITERAL = 3;
var STYLE_FOLDED = 4;
var STYLE_DOUBLE = 5;
var LEADING_SPACE_REGEXP = /^\n* /;
var ESCAPE_SEQUENCES = /* @__PURE__ */ new Map([
  [
    0,
    "\\0"
  ],
  [
    7,
    "\\a"
  ],
  [
    8,
    "\\b"
  ],
  [
    9,
    "\\t"
  ],
  [
    10,
    "\\n"
  ],
  [
    11,
    "\\v"
  ],
  [
    12,
    "\\f"
  ],
  [
    13,
    "\\r"
  ],
  [
    27,
    "\\e"
  ],
  [
    34,
    '\\"'
  ],
  [
    92,
    "\\\\"
  ],
  [
    133,
    "\\N"
  ],
  [
    160,
    "\\_"
  ],
  [
    8232,
    "\\L"
  ],
  [
    8233,
    "\\P"
  ]
]);
var DEPRECATED_BOOLEANS_SYNTAX = [
  "y",
  "Y",
  "yes",
  "Yes",
  "YES",
  "on",
  "On",
  "ON",
  "n",
  "N",
  "no",
  "No",
  "NO",
  "off",
  "Off",
  "OFF"
];
function charCodeToHexString(charCode) {
  const hexString = charCode.toString(16).toUpperCase();
  if (charCode <= 255) return `\\x${hexString.padStart(2, "0")}`;
  if (charCode <= 65535) return `\\u${hexString.padStart(4, "0")}`;
  if (charCode <= 4294967295) return `\\U${hexString.padStart(8, "0")}`;
  throw new Error("Code point within a string may not be greater than 0xFFFFFFFF");
}
function createStyleMap(map2) {
  const result = /* @__PURE__ */ new Map();
  for (let tag of Object.keys(map2)) {
    const style = String(map2[tag]);
    if (tag.slice(0, 2) === "!!") {
      tag = `tag:yaml.org,2002:${tag.slice(2)}`;
    }
    result.set(tag, style);
  }
  return result;
}
function indentString(string, spaces) {
  const indent = " ".repeat(spaces);
  return string.split("\n").map((line) => line.length ? indent + line : line).join("\n");
}
function generateNextLine(indent, level) {
  return `
${" ".repeat(indent * level)}`;
}
function isPrintable(c) {
  return 32 <= c && c <= 126 || 161 <= c && c <= 55295 && c !== 8232 && c !== 8233 || 57344 <= c && c <= 65533 && c !== BOM || 65536 <= c && c <= 1114111;
}
function isPlainSafe(c) {
  return isPrintable(c) && c !== BOM && c !== COMMA && c !== LEFT_SQUARE_BRACKET && c !== RIGHT_SQUARE_BRACKET && c !== LEFT_CURLY_BRACKET && c !== RIGHT_CURLY_BRACKET && c !== COLON && c !== SHARP;
}
function isPlainSafeFirst(c) {
  return isPlainSafe(c) && !isWhiteSpace(c) && c !== MINUS && c !== QUESTION && c !== AMPERSAND && c !== ASTERISK && c !== EXCLAMATION && c !== VERTICAL_LINE && c !== GREATER_THAN && c !== SINGLE_QUOTE && c !== DOUBLE_QUOTE && c !== PERCENT && c !== COMMERCIAL_AT && c !== GRAVE_ACCENT;
}
function needIndentIndicator(string) {
  return LEADING_SPACE_REGEXP.test(string);
}
function chooseScalarStyle(string, singleLineOnly, indentPerLevel, lineWidth, implicitTypes) {
  const shouldTrackWidth = lineWidth !== -1;
  let hasLineBreak = false;
  let hasFoldableLine = false;
  let previousLineBreak = -1;
  let plain = isPlainSafeFirst(string.charCodeAt(0)) && !isWhiteSpace(string.charCodeAt(string.length - 1));
  let char;
  let i;
  if (singleLineOnly) {
    for (i = 0; i < string.length; i++) {
      char = string.charCodeAt(i);
      if (!isPrintable(char)) {
        return STYLE_DOUBLE;
      }
      plain = plain && isPlainSafe(char);
    }
  } else {
    for (i = 0; i < string.length; i++) {
      char = string.charCodeAt(i);
      if (char === LINE_FEED) {
        hasLineBreak = true;
        if (shouldTrackWidth) {
          hasFoldableLine = hasFoldableLine || // Foldable line = too long, and not more-indented.
          i - previousLineBreak - 1 > lineWidth && string[previousLineBreak + 1] !== " ";
          previousLineBreak = i;
        }
      } else if (!isPrintable(char)) {
        return STYLE_DOUBLE;
      }
      plain = plain && isPlainSafe(char);
    }
    hasFoldableLine = hasFoldableLine || shouldTrackWidth && i - previousLineBreak - 1 > lineWidth && string[previousLineBreak + 1] !== " ";
  }
  if (!hasLineBreak && !hasFoldableLine) {
    return plain && !implicitTypes.some((type) => type.resolve(string)) ? STYLE_PLAIN : STYLE_SINGLE;
  }
  if (indentPerLevel > 9 && needIndentIndicator(string)) {
    return STYLE_DOUBLE;
  }
  return hasFoldableLine ? STYLE_FOLDED : STYLE_LITERAL;
}
function foldLine(line, width) {
  if (line === "" || line[0] === " ") return line;
  const breakRegExp = / [^ ]/g;
  let start = 0;
  let end;
  let curr = 0;
  let next = 0;
  const lines = [];
  for (const match of line.matchAll(breakRegExp)) {
    next = match.index;
    if (next - start > width) {
      end = curr > start ? curr : next;
      lines.push(line.slice(start, end));
      start = end + 1;
    }
    curr = next;
  }
  if (line.length - start > width && curr > start) {
    lines.push(line.slice(start, curr));
    lines.push(line.slice(curr + 1));
  } else {
    lines.push(line.slice(start));
  }
  return lines.join("\n");
}
function trimTrailingNewline(string) {
  return string.at(-1) === "\n" ? string.slice(0, -1) : string;
}
function foldString(string, width) {
  const lineRe = /(\n+)([^\n]*)/g;
  let result = (() => {
    let nextLF = string.indexOf("\n");
    nextLF = nextLF !== -1 ? nextLF : string.length;
    lineRe.lastIndex = nextLF;
    return foldLine(string.slice(0, nextLF), width);
  })();
  let prevMoreIndented = string[0] === "\n" || string[0] === " ";
  let moreIndented;
  let match;
  while (match = lineRe.exec(string)) {
    const prefix = match[1];
    const line = match[2] || "";
    moreIndented = line[0] === " ";
    result += prefix + (!prevMoreIndented && !moreIndented && line !== "" ? "\n" : "") + foldLine(line, width);
    prevMoreIndented = moreIndented;
  }
  return result;
}
function escapeString(string) {
  let result = "";
  let char;
  let nextChar;
  let escapeSeq;
  for (let i = 0; i < string.length; i++) {
    char = string.charCodeAt(i);
    if (char >= 55296 && char <= 56319) {
      nextChar = string.charCodeAt(i + 1);
      if (nextChar >= 56320 && nextChar <= 57343) {
        result += charCodeToHexString((char - 55296) * 1024 + nextChar - 56320 + 65536);
        i++;
        continue;
      }
    }
    escapeSeq = ESCAPE_SEQUENCES.get(char);
    result += !escapeSeq && isPrintable(char) ? string[i] : escapeSeq || charCodeToHexString(char);
  }
  return result;
}
function blockHeader(string, indentPerLevel) {
  const indentIndicator = needIndentIndicator(string) ? String(indentPerLevel) : "";
  const clip = string[string.length - 1] === "\n";
  const keep = clip && (string[string.length - 2] === "\n" || string === "\n");
  const chomp = keep ? "+" : clip ? "" : "-";
  return `${indentIndicator}${chomp}
`;
}
function inspectNode(object, objects, duplicateObjects) {
  if (!isObject(object)) return;
  if (objects.has(object)) {
    duplicateObjects.add(object);
    return;
  }
  objects.add(object);
  const entries = Array.isArray(object) ? object : Object.values(object);
  for (const value of entries) {
    inspectNode(value, objects, duplicateObjects);
  }
}
var DumperState = class {
  indent;
  arrayIndent;
  skipInvalid;
  flowLevel;
  sortKeys;
  lineWidth;
  useAnchors;
  compatMode;
  condenseFlow;
  implicitTypes;
  explicitTypes;
  duplicates = [];
  usedDuplicates = /* @__PURE__ */ new Set();
  styleMap = /* @__PURE__ */ new Map();
  constructor({ schema = DEFAULT_SCHEMA, indent = 2, arrayIndent = true, skipInvalid = false, flowLevel = -1, styles = void 0, sortKeys = false, lineWidth = 80, useAnchors = true, compatMode = true, condenseFlow = false }) {
    this.indent = Math.max(1, indent);
    this.arrayIndent = arrayIndent;
    this.skipInvalid = skipInvalid;
    this.flowLevel = flowLevel;
    if (styles) this.styleMap = createStyleMap(styles);
    this.sortKeys = sortKeys;
    this.lineWidth = lineWidth;
    this.useAnchors = useAnchors;
    this.compatMode = compatMode;
    this.condenseFlow = condenseFlow;
    this.implicitTypes = schema.implicitTypes;
    this.explicitTypes = schema.explicitTypes;
  }
  // Note: line breaking/folding is implemented for only the folded style.
  // NB. We drop the last trailing newline (if any) of a returned block scalar
  //  since the dumper adds its own newline. This always works:
  //    • No ending newline => unaffected; already using strip "-" chomping.
  //    • Ending newline    => removed then restored.
  //  Importantly, this keeps the "+" chomp indicator from gaining an extra line.
  stringifyScalar(string, { level, isKey }) {
    if (string.length === 0) {
      return "''";
    }
    if (this.compatMode && DEPRECATED_BOOLEANS_SYNTAX.includes(string)) {
      return `'${string}'`;
    }
    const indent = this.indent * Math.max(1, level);
    const lineWidth = this.lineWidth === -1 ? -1 : Math.max(Math.min(this.lineWidth, 40), this.lineWidth - indent);
    const singleLineOnly = isKey || // No block styles in flow mode.
    this.flowLevel > -1 && level >= this.flowLevel;
    const scalarStyle = chooseScalarStyle(string, singleLineOnly, this.indent, lineWidth, this.implicitTypes);
    switch (scalarStyle) {
      case STYLE_PLAIN:
        return string;
      case STYLE_SINGLE:
        return `'${string.replace(/'/g, "''")}'`;
      case STYLE_LITERAL:
        return `|${blockHeader(string, this.indent)}${trimTrailingNewline(indentString(string, indent))}`;
      case STYLE_FOLDED:
        return `>${blockHeader(string, this.indent)}${trimTrailingNewline(indentString(foldString(string, lineWidth), indent))}`;
      case STYLE_DOUBLE:
        return `"${escapeString(string)}"`;
      default:
        throw new TypeError("Invalid scalar style should be unreachable: please file a bug report against Deno at https://github.com/denoland/std/issues");
    }
  }
  stringifyFlowSequence(array, { level }) {
    const results = [];
    for (const value of array) {
      const string = this.stringifyNode(value, {
        level,
        block: false,
        compact: false,
        isKey: false
      });
      if (string === null) continue;
      results.push(string);
    }
    const separator = this.condenseFlow ? "," : ", ";
    return `[${results.join(separator)}]`;
  }
  stringifyBlockSequence(array, { level, compact }) {
    const whitespace = generateNextLine(this.indent, level);
    const prefix = compact ? "" : whitespace;
    const results = [];
    for (const value of array) {
      const string = this.stringifyNode(value, {
        level: level + 1,
        block: true,
        compact: true,
        isKey: false
      });
      if (string === null) continue;
      const linePrefix = LINE_FEED === string.charCodeAt(0) ? "-" : "- ";
      results.push(`${linePrefix}${string}`);
    }
    return results.length ? prefix + results.join(whitespace) : "[]";
  }
  stringifyFlowMapping(object, { level }) {
    const quote = this.condenseFlow ? '"' : "";
    const separator = this.condenseFlow ? ":" : ": ";
    const results = [];
    for (const [key, value] of Object.entries(object)) {
      const keyString = this.stringifyNode(key, {
        level,
        block: false,
        compact: false,
        isKey: false
      });
      if (keyString === null) continue;
      const valueString = this.stringifyNode(value, {
        level,
        block: false,
        compact: false,
        isKey: false
      });
      if (valueString === null) continue;
      const keyPrefix = keyString.length > 1024 ? "? " : "";
      results.push(quote + keyPrefix + keyString + quote + separator + valueString);
    }
    return `{${results.join(", ")}}`;
  }
  stringifyBlockMapping(object, { tag, level, compact }) {
    const keys = Object.keys(object);
    if (this.sortKeys === true) {
      keys.sort();
    } else if (typeof this.sortKeys === "function") {
      keys.sort(this.sortKeys);
    } else if (this.sortKeys) {
      throw new TypeError(`"sortKeys" must be a boolean or a function: received ${typeof this.sortKeys}`);
    }
    const separator = generateNextLine(this.indent, level);
    const results = [];
    for (const key of keys) {
      const value = object[key];
      const keyString = this.stringifyNode(key, {
        level: level + 1,
        block: true,
        compact: true,
        isKey: true
      });
      if (keyString === null) continue;
      const explicitPair = tag !== null && tag !== "?" || keyString.length > 1024;
      const valueString = this.stringifyNode(value, {
        level: level + 1,
        block: true,
        compact: explicitPair,
        isKey: false
      });
      if (valueString === null) continue;
      let pairBuffer = "";
      if (explicitPair) {
        pairBuffer += keyString.charCodeAt(0) === LINE_FEED ? "?" : "? ";
      }
      pairBuffer += keyString;
      if (explicitPair) pairBuffer += separator;
      pairBuffer += valueString.charCodeAt(0) === LINE_FEED ? ":" : ": ";
      pairBuffer += valueString;
      results.push(pairBuffer);
    }
    const prefix = compact ? "" : separator;
    return results.length ? prefix + results.join(separator) : "{}";
  }
  getTypeRepresentation(type, value) {
    if (!type.represent) return value;
    const style = this.styleMap.get(type.tag) ?? type.defaultStyle;
    if (typeof type.represent === "function") {
      return type.represent(value, style);
    }
    const represent = type.represent[style];
    if (!represent) {
      throw new TypeError(`!<${type.tag}> tag resolver accepts not "${style}" style`);
    }
    return represent(value, style);
  }
  detectType(value) {
    for (const type of this.implicitTypes) {
      if (type.predicate?.(value)) {
        value = this.getTypeRepresentation(type, value);
        return {
          tag: "?",
          value
        };
      }
    }
    for (const type of this.explicitTypes) {
      if (type.predicate?.(value)) {
        value = this.getTypeRepresentation(type, value);
        return {
          tag: type.tag,
          value
        };
      }
    }
    return {
      tag: null,
      value
    };
  }
  // Serializes `object` and writes it to global `result`.
  // Returns true on success, or false on invalid object.
  stringifyNode(value, { level, block, compact, isKey }) {
    const result = this.detectType(value);
    const tag = result.tag;
    value = result.value;
    if (block) {
      block = this.flowLevel < 0 || this.flowLevel > level;
    }
    if (typeof value === "string" || value instanceof String) {
      value = value instanceof String ? value.valueOf() : value;
      if (tag !== "?") {
        value = this.stringifyScalar(value, {
          level,
          isKey
        });
      }
    } else if (isObject(value)) {
      const duplicateIndex = this.duplicates.indexOf(value);
      const duplicate = duplicateIndex !== -1;
      if (duplicate) {
        if (this.usedDuplicates.has(value)) return `*ref_${duplicateIndex}`;
        this.usedDuplicates.add(value);
      }
      if (tag !== null && tag !== "?" || duplicate || this.indent !== 2 && level > 0) {
        compact = false;
      }
      if (Array.isArray(value)) {
        const arrayLevel = !this.arrayIndent && level > 0 ? level - 1 : level;
        if (block && value.length !== 0) {
          value = this.stringifyBlockSequence(value, {
            level: arrayLevel,
            compact
          });
          if (duplicate) {
            value = `&ref_${duplicateIndex}${value}`;
          }
        } else {
          value = this.stringifyFlowSequence(value, {
            level: arrayLevel
          });
          if (duplicate) {
            value = `&ref_${duplicateIndex} ${value}`;
          }
        }
      } else {
        if (block && Object.keys(value).length !== 0) {
          value = this.stringifyBlockMapping(value, {
            tag,
            level,
            compact
          });
          if (duplicate) {
            value = `&ref_${duplicateIndex}${value}`;
          }
        } else {
          value = this.stringifyFlowMapping(value, {
            level
          });
          if (duplicate) {
            value = `&ref_${duplicateIndex} ${value}`;
          }
        }
      }
    } else {
      if (this.skipInvalid) return null;
      throw new TypeError(`Cannot stringify ${typeof value}`);
    }
    if (tag !== null && tag !== "?") {
      value = `!<${tag}> ${value}`;
    }
    return value;
  }
  stringify(value) {
    if (this.useAnchors) {
      const values = /* @__PURE__ */ new Set();
      const duplicateObjects = /* @__PURE__ */ new Set();
      inspectNode(value, values, duplicateObjects);
      this.duplicates = [
        ...duplicateObjects
      ];
      this.usedDuplicates = /* @__PURE__ */ new Set();
    }
    const string = this.stringifyNode(value, {
      level: 0,
      block: true,
      compact: true,
      isKey: false
    });
    if (string !== null) {
      return `${string}
`;
    }
    return "";
  }
};

// node_modules/@std/yaml/stringify.js
function stringify(data, options = {}) {
  const state = new DumperState({
    ...options,
    schema: SCHEMA_MAP.get(options.schema)
  });
  return state.stringify(data);
}
export {
  parse,
  parseAll,
  stringify
};
