/**
 * Original source: https://jsr.io/@std/fmt
 *
 * Generated from jsr:@std/fmt@1.03 using esbuild --bundle.
 *
 * Only the formatBytes and formatDuration functions are in this module,
 * the color utilities is in the separate colors.js module.
 *
 * Changes made:
 * - in formatDuration(), changed the default value of the  param to true
 */

// node_modules/@std/fmt/bytes.js
function format(num, options = {}) {
  if (!Number.isFinite(num)) {
    throw new TypeError(`Expected a finite number, got ${typeof num}: ${num}`);
  }
  const UNITS_FIRSTLETTER = (options.bits ? "b" : "B") + "kMGTPEZY";
  if (options.signed && num === 0) {
    return ` 0 ${UNITS_FIRSTLETTER[0]}`;
  }
  const prefix = num < 0 ? "-" : options.signed ? "+" : "";
  num = Math.abs(num);
  const localeOptions = getLocaleOptions(options);
  if (num < 1) {
    const numberString2 = toLocaleString(num, options.locale, localeOptions);
    return prefix + numberString2 + " " + UNITS_FIRSTLETTER[0];
  }
  const exponent = Math.min(Math.floor(options.binary ? Math.log(num) / Math.log(1024) : Math.log10(num) / 3), UNITS_FIRSTLETTER.length - 1);
  num /= Math.pow(options.binary ? 1024 : 1e3, exponent);
  if (!localeOptions) {
    num = Number(num.toPrecision(3));
  }
  const numberString = toLocaleString(num, options.locale, localeOptions);
  let unit = UNITS_FIRSTLETTER[exponent];
  if (exponent > 0) {
    unit += options.binary ? "i" : "";
    unit += options.bits ? "bit" : "B";
  }
  return prefix + numberString + " " + unit;
}
function getLocaleOptions({ maximumFractionDigits, minimumFractionDigits }) {
  if (maximumFractionDigits === void 0 && minimumFractionDigits === void 0) {
    return;
  }
  const ret = {};
  if (maximumFractionDigits !== void 0) {
    ret.maximumFractionDigits = maximumFractionDigits;
  }
  if (minimumFractionDigits !== void 0) {
    ret.minimumFractionDigits = minimumFractionDigits;
  }
  return ret;
}
function toLocaleString(num, locale, options) {
  if (typeof locale === "string" || Array.isArray(locale)) {
    return num.toLocaleString(locale, options);
  } else if (locale === true || options !== void 0) {
    return num.toLocaleString(void 0, options);
  }
  return num.toString();
}

// node_modules/@std/fmt/duration.js
function addZero(num, digits) {
  return String(num).padStart(digits, "0");
}
var keyList = {
  d: "days",
  h: "hours",
  m: "minutes",
  s: "seconds",
  ms: "milliseconds",
  us: "microseconds",
  ns: "nanoseconds"
};
function millisecondsToDurationObject(ms) {
  const millis = Math.abs(ms);
  const millisFraction = millis.toFixed(7).slice(-7, -1);
  return {
    d: Math.trunc(millis / 864e5),
    h: Math.trunc(millis / 36e5) % 24,
    m: Math.trunc(millis / 6e4) % 60,
    s: Math.trunc(millis / 1e3) % 60,
    ms: Math.trunc(millis) % 1e3,
    us: +millisFraction.slice(0, 3),
    ns: +millisFraction.slice(3, 6)
  };
}
function durationArray(duration) {
  return [
    {
      type: "d",
      value: duration.d
    },
    {
      type: "h",
      value: duration.h
    },
    {
      type: "m",
      value: duration.m
    },
    {
      type: "s",
      value: duration.s
    },
    {
      type: "ms",
      value: duration.ms
    },
    {
      type: "us",
      value: duration.us
    },
    {
      type: "ns",
      value: duration.ns
    }
  ];
}
function format2(ms, options) {
  const { style = "narrow", ignoreZero = true } = options ?? {};
  const duration = millisecondsToDurationObject(ms);
  const durationArr = durationArray(duration);
  switch (style) {
    case "narrow": {
      if (ignoreZero) {
        return `${durationArr.filter((x) => x.value).map((x) => `${x.value}${x.type === "us" ? "\xB5s" : x.type}`).join(" ")}`;
      }
      return `${durationArr.map((x) => `${x.value}${x.type === "us" ? "\xB5s" : x.type}`).join(" ")}`;
    }
    case "full": {
      if (ignoreZero) {
        return `${durationArr.filter((x) => x.value).map((x) => `${x.value} ${keyList[x.type]}`).join(", ")}`;
      }
      return `${durationArr.map((x) => `${x.value} ${keyList[x.type]}`).join(", ")}`;
    }
    case "digital": {
      const arr = durationArr.map((x) => [
        "ms",
        "us",
        "ns"
      ].includes(x.type) ? addZero(x.value, 3) : addZero(x.value, 2));
      if (ignoreZero) {
        let cont = true;
        while (cont) {
          if (!Number(arr[arr.length - 1])) arr.pop();
          else cont = false;
        }
      }
      return arr.join(":");
    }
    default: {
      throw new TypeError(`style must be "narrow", "full", or "digital"!`);
    }
  }
}
export {
  format as formatBytes,
  format2 as formatDuration
};
