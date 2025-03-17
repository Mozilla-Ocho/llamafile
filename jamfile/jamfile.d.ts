/**
 * This will eventually contain the full types of all the modules this JS runtime offers.
 *
 * Generating this automatically was too complicated, so will unfortunately have to manage
 * manually for now.
 */


declare namespace Jamfile {
  export const version: string;
}
declare module "qjs:std" {
  /**
   * Exit the process.
   * @param n - Exit code.
   */
  export function exit(n: number): void;

  /**
   * Evaluate the string `str` as a script (global eval).
   * @param str - The script to evaluate.
   * @param options - Optional settings for evaluation.
   *  - `backtrace_barrier` (boolean, default = false): If true, error backtraces do not list the stack frames below the evalScript.
   *  - `async` (boolean, default = false): If true, await is accepted in the script and a promise is returned.
   * @returns The result of the evaluated script or a promise if `async` is true.
   */
  export function evalScript(
      str: string,
      options?: { backtrace_barrier?: boolean; async?: boolean }
  ): any;

  /**
   * Evaluate the file `filename` as a script (global eval).
   * @param filename - The path to the file.
   * @returns The result of the evaluated script.
   */
  export function loadScript(filename: string): any;

  /**
   * Load the file `filename` and return it as a string assuming UTF-8 encoding.
   * @param filename - The path to the file.
   * @returns The file content as a string, or null in case of I/O error.
   */
  export function loadFile(filename: string): string | null;

  /**
   * Open a file (wrapper to the libc `fopen()`).
   * @param filename - The path to the file.
   * @param flags - The flags for opening the file.
   * @param errorObj - An optional object where `errno` will be set to the error code, or 0 if no error occurred.
   * @returns A `FILE` object, or null in case of I/O error.
   */
  export function open(
      filename: string,
      flags: string,
      errorObj?: { errno?: number }
  ): FILE | null;

  /**
   * Open a process by creating a pipe (wrapper to the libc `popen()`).
   * @param command - The command to execute.
   * @param flags - The flags for opening the process.
   * @param errorObj - An optional object where `errno` will be set to the error code, or 0 if no error occurred.
   * @returns A `FILE` object, or null in case of I/O error.
   */
  export function popen(
      command: string,
      flags: string,
      errorObj?: { errno?: number }
  ): FILE | null;

  /**
   * Open a file from a file handle (wrapper to the libc `fdopen()`).
   * @param fd - The file descriptor.
   * @param flags - The flags for opening the file.
   * @param errorObj - An optional object where `errno` will be set to the error code, or 0 if no error occurred.
   * @returns A `FILE` object, or null in case of I/O error.
   */
  export function fdopen(
      fd: number,
      flags: string,
      errorObj?: { errno?: number }
  ): FILE | null;

  /**
   * Open a temporary file.
   * @param errorObj - An optional object where `errno` will be set to the error code, or 0 if no error occurred.
   * @returns A `FILE` object, or null in case of I/O error.
   */
  export function tmpfile(errorObj?: { errno?: number }): FILE | null;

  /**
   * Outputs the string `str` to stdout.
   * @param str - The string to output.
   */
  export function puts(str: string): void;

  /**
   * Outputs formatted text to stdout.
   * @param fmt - The format string.
   * @param args - Arguments for formatting.
   */
  export function printf(fmt: string, ...args: any[]): void;

  /**
   * Formats text into a string using the libc `sprintf()`.
   * @param fmt - The format string.
   * @param args - Arguments for formatting.
   * @returns The formatted string.
   */
  export function sprintf(fmt: string, ...args: any[]): string;

  /** Wrapper to the libc `stdin`. */
  // TODO syntax error on the name "in"
  //export const in: FILE;

  /** Wrapper to the libc `stdout`. */
  export const out: FILE;

  /** Wrapper to the libc `stderr`. */
  export const err: FILE;

  /** Constants for seek operations. */
  export const SEEK_SET: number;
  export const SEEK_CUR: number;
  export const SEEK_END: number;

  /** Enumeration object containing common error codes. */
  export const Error: {
      EINVAL: number;
      EIO: number;
      EACCES: number;
      EEXIST: number;
      ENOSPC: number;
      ENOSYS: number;
      EBUSY: number;
      ENOENT: number;
      EPERM: number;
      EPIPE: number;
  };

  /**
   * Return a string that describes the error `errno`.
   * @param errno - The error code.
   * @returns The error description.
   */
  export function strerror(errno: number): string;

  /**
   * Manually invoke the cycle removal algorithm.
   * Useful for specific memory constraints or testing.
   */
  export function gc(): void;

  /**
   * Return the value of the environment variable `name`.
   * @param name - The name of the environment variable.
   * @returns The value of the environment variable, or undefined if not defined.
   */
  export function getenv(name: string): string | undefined;

  /**
   * Set the value of the environment variable `name` to the string `value`.
   * @param name - The name of the environment variable.
   * @param value - The value to set.
   */
  export function setenv(name: string, value: string): void;

  /**
   * Delete the environment variable `name`.
   * @param name - The name of the environment variable.
   */
  export function unsetenv(name: string): void;

  /**
   * Return an object containing the environment variables as key-value pairs.
   * @returns An object with environment variables.
   */
  export function getenviron(): Record<string, string>;

  /**
   * Download a URL using the curl command line utility.
   * @param url - The URL to download.
   * @param options - Optional settings for the download.
   *  - `binary` (boolean, default = false): If true, the response is an ArrayBuffer instead of a string.
   *  - `full` (boolean, default = false): If true, return an object containing properties `response`, `responseHeaders`, and `status`. If `full` is false, only the response is returned if the status is between 200 and 299, otherwise null.
   * @returns The response content or an object with additional properties based on `options`.
   */
  export function urlGet(
      url: string,
      options?: { binary?: boolean; full?: boolean }
  ): string | ArrayBuffer | { response: string | null; responseHeaders: string; status: number };

  /**
   * Parse `str` using a superset of JSON.parse with extended syntax support.
   * Supported extensions:
   * - Single line and multiline comments
   * - Unquoted properties (ASCII-only Javascript identifiers)
   * - Trailing comma in array and object definitions
   * - Single quoted strings
   * - \f and \v as space characters
   * - Leading plus in numbers
   * - Octal (0o prefix) and hexadecimal (0x prefix) numbers
   * @param str - The string to parse.
   * @returns The parsed object.
   */
  export function parseExtJSON(str: string): any;

  /** Represents a file object. */
  export interface FILE {
      /** Close the file. */
      close(): number;

      /** Outputs the string with the UTF-8 encoding. */
      puts(str: string): void;

      /** Formatted printf. */
      printf(fmt: string, ...args: any[]): void;

      /** Flush the buffered file. */
      flush(): void;

      /** Seek to a given file position. */
      seek(offset: number | bigint, whence: number): number;

      /** Return the current file position. */
      tell(): number;

      /** Return the current file position as a bigint. */
      tello(): bigint;

      /** Return true if end of file. */
      eof(): boolean;

      /** Return the associated OS handle. */
      fileno(): number;

      /** Return true if there was an error. */
      error(): boolean;

      /** Clear the error indication. */
      clearerr(): void;

      /** Read bytes from the file to an ArrayBuffer. */
      read(buffer: ArrayBuffer, position: number, length: number): number;

      /** Write bytes to the file from an ArrayBuffer. */
      write(buffer: ArrayBuffer, position: number, length: number): number;

      /** Return the next line from the file as a string. */
      getline(): string | null;

      /** Read bytes as a string up to `max_size`. */
      readAsString(max_size?: number): string | null;

      /** Return the next byte from the file. */
      getByte(): number;

      /** Write one byte to the file. */
      putByte(c: number): void;
  }
}


declare module "qjs:os" {
  /**
   * Open a file. Return a handle or < 0 if error.
   * @param filename The file name.
   * @param flags POSIX open flags.
   * @param mode File mode (default = 0o666).
   */
  export function open(filename: string, flags: number, mode?: number): number;

  /** POSIX open flags. */
  export const O_RDONLY: number;
  export const O_WRONLY: number;
  export const O_RDWR: number;
  export const O_APPEND: number;
  export const O_CREAT: number;
  export const O_EXCL: number;
  export const O_TRUNC: number;

  /** Windows-specific flag to open the file in text mode. Default is binary mode. */
  export const O_TEXT: number;

  /**
   * Close the file handle.
   * @param fd The file handle to close.
   */
  export function close(fd: number): number;

  /**
   * Seek in the file.
   * @param fd The file handle.
   * @param offset The position to seek to (number or bigint).
   * @param whence The reference position (use std.SEEK_* constants).
   */
  export function seek(fd: number, offset: number | bigint, whence: number): number | bigint;

  /**
   * Read bytes from a file handle.
   * @param fd The file handle to read from.
   * @param buffer The ArrayBuffer to store the read data.
   * @param offset The starting position in the buffer.
   * @param length The number of bytes to read.
   */
  export function read(fd: number, buffer: ArrayBuffer, offset: number, length: number): number;

  /**
   * Write bytes to a file handle.
   * @param fd The file handle to write to.
   * @param buffer The ArrayBuffer containing the data to write.
   * @param offset The starting position in the buffer.
   * @param length The number of bytes to write.
   */
  export function write(fd: number, buffer: ArrayBuffer, offset: number, length: number): number;

  /**
   * Check if the file handle is a TTY (terminal).
   * @param fd The file handle to check.
   */
  export function isatty(fd: number): boolean;

  /**
   * Get the size of a TTY.
   * @param fd The TTY file handle.
   * @returns [width, height] or null if not available.
   */
  export function ttyGetWinSize(fd: number): [number, number] | null;

  /**
   * Set the TTY in raw mode.
   * @param fd The TTY file handle.
   */
  export function ttySetRaw(fd: number): void;

  /**
   * Remove a file.
   * @param filename The file name to remove.
   */
  export function remove(filename: string): number;

  /**
   * Rename a file.
   * @param oldname The current file name.
   * @param newname The new file name.
   */
  export function rename(oldname: string, newname: string): number;

  /**
   * Get the canonicalized absolute pathname of a path.
   * @param path The input path.
   * @returns [path, error code].
   */
  export function realpath(path: string): [string, number];

  /**
   * Get the current working directory.
   * @returns [directory, error code].
   */
  export function getcwd(): [string, number];

  /**
   * Change the current working directory.
   * @param path The new working directory.
   */
  export function chdir(path: string): number;

  /**
   * Create a directory.
   * @param path The directory path.
   * @param mode The mode for the directory (default = 0o777).
   */
  export function mkdir(path: string, mode?: number): number;

  /**
   * Get file status for a path.
   * @param path The file path.
   * @returns [status object, error code].
   */
  export function stat(path: string): [FileStatus, number];

  /**
   * Get file status for a path (without following symlinks).
   * @param path The file path.
   * @returns [status object, error code].
   */
  export function lstat(path: string): [FileStatus, number];

  /** File status object returned by `stat` and `lstat`. */
  export interface FileStatus {
    dev: number;
    ino: number;
    mode: number;
    nlink: number;
    uid: number;
    gid: number;
    rdev: number;
    size: number;
    blocks: number;
    atime: number;
    mtime: number;
    ctime: number;
  }

  /** Constants to interpret the mode property returned by `stat`. */
  export const S_IFMT: number;
  export const S_IFIFO: number;
  export const S_IFCHR: number;
  export const S_IFDIR: number;
  export const S_IFBLK: number;
  export const S_IFREG: number;
  export const S_IFSOCK: number;
  export const S_IFLNK: number;
  export const S_ISGID: number;
  export const S_ISUID: number;

  /**
   * Change the access and modification times of a file.
   * @param path The file path.
   * @param atime Access time in milliseconds since 1970.
   * @param mtime Modification time in milliseconds since 1970.
   */
  export function utimes(path: string, atime: number, mtime: number): number;

  /**
   * Create a symbolic link.
   * @param target The target of the link.
   * @param linkpath The link path to create.
   */
  export function symlink(target: string, linkpath: string): number;

  /**
   * Read a symbolic link.
   * @param path The link path.
   * @returns [link target, error code].
   */
  export function readlink(path: string): [string, number];

  /**
   * Read a directory's contents.
   * @param path The directory path.
   * @returns [array of filenames, error code].
   */
  export function readdir(path: string): [string[], number];

  /**
   * Add a read handler to a file handle.
   * @param fd The file handle.
   * @param func The function to call when there is data pending (or `null` to remove the handler).
   */
  export function setReadHandler(fd: number, func: (() => void) | null): void;

  /**
   * Add a write handler to a file handle.
   * @param fd The file handle.
   * @param func The function to call when data can be written (or `null` to remove the handler).
   */
  export function setWriteHandler(fd: number, func: (() => void) | null): void;

  /**
   * Add a signal handler.
   * @param signal The signal number.
   * @param func The function to call when the signal is received (`null` for the default handler, `undefined` to ignore the signal).
   */
  export function signal(signal: number, func: (() => void) | null | undefined): void;

  /** POSIX signal numbers. */
  export const SIGINT: number;
  export const SIGABRT: number;
  export const SIGFPE: number;
  export const SIGILL: number;
  export const SIGSEGV: number;
  export const SIGTERM: number;

  /**
   * Send a signal to a process.
   * @param pid The process ID.
   * @param sig The signal to send.
   */
  export function kill(pid: number, sig: number): number;

  /**
   * Execute a process with the given arguments.
   * @param args The arguments for the process.
   * @param options Optional execution parameters.
   * @returns Exit code, process ID, or negated signal number depending on the `block` option.
   */
  export function exec(
    args: string[],
    options?: {
      block?: boolean;
      usePath?: boolean;
      file?: string;
      cwd?: string;
      stdin?: number;
      stdout?: number;
      stderr?: number;
      env?: Record<string, string>;
      uid?: number;
      gid?: number;
    }
  ): number;

  /**
   * Get the current process ID.
   * @returns The process ID.
   */
  export function getpid(): number;

  /**
   * Wait for a process to change state.
   * @param pid The process ID to wait for.
   * @param options Options for the waitpid call (use `WNOHANG` if non-blocking behavior is desired).
   * @returns [result, status], where `result` is the PID of the child process or -errno, and `status` is the exit status or signal number.
   */
  export function waitpid(pid: number, options?: number): [number, number];

  /** Constant for non-blocking waitpid call. */
  export const WNOHANG: number;

  /**
   * Duplicate a file descriptor.
   * @param fd The file descriptor to duplicate.
   * @returns The new file descriptor, or -errno on failure.
   */
  export function dup(fd: number): number;

  /**
   * Duplicate a file descriptor to a specific target descriptor.
   * @param oldfd The file descriptor to duplicate.
   * @param newfd The target descriptor number.
   * @returns 0 if successful, or -errno on failure.
   */
  export function dup2(oldfd: number, newfd: number): number;

  /**
   * Create a pipe.
   * @returns [read_fd, write_fd] or `null` in case of error.
   */
  export function pipe(): [number, number] | null;

  /**
   * Sleep for the specified duration.
   * @param delay_ms The delay in milliseconds.
   */
  export function sleep(delay_ms: number): void;

  /**
   * Sleep asynchronously for the specified duration.
   * @param delay_ms The delay in milliseconds.
   * @returns A promise that resolves after the specified delay.
   */
  export function sleepAsync(delay_ms: number): Promise<void>;

  /**
   * Get a high-precision timestamp.
   * @returns A timestamp in milliseconds with more precision than `Date.now()`.
   */
  export function now(): number;

  /**
   * Call a function after a specified delay.
   * @param func The function to call.
   * @param delay The delay in milliseconds.
   * @returns A timer handle.
   */
  export function setTimeout(func: () => void, delay: number): number;

  /**
   * Cancel a timer.
   * @param handle The timer handle to cancel.
   */
  export function clearTimeout(handle: number): void;

  /**
   * Get the platform string.
   * @returns A string representing the platform: "linux", "darwin", "win32", or "js".
   */
  export function platform(): "linux" | "darwin" | "win32" | "js";

  /**
   * Worker constructor for creating a new thread.
   * @param module_filename The module filename to execute in the new thread.
   */
  export class Worker {
    constructor(module_filename: string);

    /**
     * Send a message to the corresponding worker.
     * @param msg The message to send.
     */
    postMessage(msg: any): void;

    /**
     * Set a function to handle received messages.
     * @param handler A function that receives a single argument containing the message.
     */
    onmessage: ((msg: { data: any }) => void) | null;

    /** Represents the parent worker in the created worker. */
    static parent: Worker;
  }
}


declare module 'jamfile:yaml' {

  export type SchemaType = "failsafe" | "json" | "core" | "default" | "extended";
  export type StyleVariant =
  | "lowercase"
  | "uppercase"
  | "camelcase"
  | "decimal"
  | "binary"
  | "octal"
  | "hexadecimal";

  export type StringifyOptions = {
    /**
     * Indentation width to use (in spaces).
     *
     * @default {2}
     */
    indent?: number;
    /**
     * When true, adds an indentation level to array elements.
     *
     * @default {true}
     */
    arrayIndent?: boolean;
    /**
     * Do not throw on invalid types (like function in the safe schema) and skip
     * pairs and single values with such types.
     *
     * @default {false}
     */
    skipInvalid?: boolean;
    /**
     * Specifies level of nesting, when to switch from block to flow style for
     * collections. `-1` means block style everywhere.
     *
     * @default {-1}
     */
    flowLevel?: number;
    /** Each tag may have own set of styles.	- "tag" => "style" map. */
    styles?: Record<string, StyleVariant>;
    /**
     * Name of the schema to use.
     *
     * @default {"default"}
     */
    schema?: SchemaType;
    /**
     * If true, sort keys when dumping YAML in ascending, ASCII character order.
     * If a function, use the function to sort the keys.
     * If a function is specified, the function must return a negative value
     * if first argument is less than second argument, zero if they're equal
     * and a positive value otherwise.
     *
     * @default {false}
     */
    sortKeys?: boolean | ((a: string, b: string) => number);
    /**
     * Set max line width.
     *
     * @default {80}
     */
    lineWidth?: number;
    /**
     * If false, don't convert duplicate objects into references.
     *
     * @default {true}
     */
    useAnchors?: boolean;
    /**
     * If false don't try to be compatible with older yaml versions.
     * Currently: don't quote "yes", "no" and so on,
     * as required for YAML 1.1.
     *
     * @default {true}
     */
    compatMode?: boolean;
    /**
     * If true flow sequences will be condensed, omitting the
     * space between `key: value` or `a, b`. Eg. `'[a,b]'` or `{a:{b:c}}`.
     * Can be useful when using yaml for pretty URL query params
     * as spaces are %-encoded.
     *
     * @default {false}
     */
    condenseFlow?: boolean;
  };
  
  export function stringify(data: unknown, options?: StringifyOptions): string;


  export interface ParseOptions {
    /**
     * Name of the schema to use.
     *
     * @default {"default"}
     */
    schema?: SchemaType;
    /**
     * If `true`, duplicate keys will overwrite previous values. Otherwise,
     * duplicate keys will throw a {@linkcode SyntaxError}.
     *
     * @default {false}
     */
    allowDuplicateKeys?: boolean;
    /**
     * If defined, a function to call on warning messages taking an
     * {@linkcode Error} as its only argument.
     */
    onWarning?(error: Error): void;
  }

  export function parse(content: string, options?: ParseOptions): unknown;
  export function parseAll(content: string, options?: ParseOptions): unknown;


}

declare module 'jamfile:toml' {

  export interface StringifyOptions {
    /**
     * Define if the keys should be aligned or not.
     *
     * @default {false}
     */
    keyAlignment?: boolean;
  }

  export function stringify(obj: Record<string, unknown>, options?: StringifyOptions): string;

  export function parse(tomlString: string): Record<string, unknown>;
}

declare module 'jamfile:frontmatter' {

  export type Format = "yaml" | "toml" | "json";

  export function test(str: string, formats?: Format[]): boolean;

  export type Extract<T> = {
    frontMatter: string;
    body: string;
    attrs: T;
  };

  export function extractJson<T>(text: string): Extract<T>;
  export function extractToml<T>(text: string): Extract<T>;
  export function extractYaml<T>(text: string): Extract<T>;
}

declare module 'jamfile:marked' {
  export type MarkedToken = (Tokens.Blockquote | Tokens.Br | Tokens.Code | Tokens.Codespan | Tokens.Def | Tokens.Del | Tokens.Em | Tokens.Escape | Tokens.Heading | Tokens.Hr | Tokens.HTML | Tokens.Image | Tokens.Link | Tokens.List | Tokens.ListItem | Tokens.Paragraph | Tokens.Space | Tokens.Strong | Tokens.Table | Tokens.Tag | Tokens.Text);
  export type Token = (MarkedToken | Tokens.Generic);
  export namespace Tokens {
    interface Blockquote {
      type: "blockquote";
      raw: string;
      text: string;
      tokens: Token[];
    }
    interface Br {
      type: "br";
      raw: string;
    }
    interface Checkbox {
      checked: boolean;
    }
    interface Code {
      type: "code";
      raw: string;
      codeBlockStyle?: "indented";
      lang?: string;
      text: string;
      escaped?: boolean;
    }
    interface Codespan {
      type: "codespan";
      raw: string;
      text: string;
    }
    interface Def {
      type: "def";
      raw: string;
      tag: string;
      href: string;
      title: string;
    }
    interface Del {
      type: "del";
      raw: string;
      text: string;
      tokens: Token[];
    }
    interface Em {
      type: "em";
      raw: string;
      text: string;
      tokens: Token[];
    }
    interface Escape {
      type: "escape";
      raw: string;
      text: string;
    }
    interface Generic {
      [index: string]: any;
      type: string;
      raw: string;
      tokens?: Token[];
    }
    interface Heading {
      type: "heading";
      raw: string;
      depth: number;
      text: string;
      tokens: Token[];
    }
    interface Hr {
      type: "hr";
      raw: string;
    }
    interface HTML {
      type: "html";
      raw: string;
      pre: boolean;
      text: string;
      block: boolean;
    }
    interface Image {
      type: "image";
      raw: string;
      href: string;
      title: string | null;
      text: string;
    }
    interface Link {
      type: "link";
      raw: string;
      href: string;
      title?: string | null;
      text: string;
      tokens: Token[];
    }
    interface List {
      type: "list";
      raw: string;
      ordered: boolean;
      start: number | "";
      loose: boolean;
      items: ListItem[];
    }
    interface ListItem {
      type: "list_item";
      raw: string;
      task: boolean;
      checked?: boolean;
      loose: boolean;
      text: string;
      tokens: Token[];
    }
    interface Paragraph {
      type: "paragraph";
      raw: string;
      pre?: boolean;
      text: string;
      tokens: Token[];
    }
    interface Space {
      type: "space";
      raw: string;
    }
    interface Strong {
      type: "strong";
      raw: string;
      text: string;
      tokens: Token[];
    }
    interface Table {
      type: "table";
      raw: string;
      align: Array<"center" | "left" | "right" | null>;
      header: TableCell[];
      rows: TableCell[][];
    }
    interface TableCell {
      text: string;
      tokens: Token[];
      header: boolean;
      align: "center" | "left" | "right" | null;
    }
    interface TableRow {
      text: string;
    }
    interface Tag {
      type: "html";
      raw: string;
      inLink: boolean;
      inRawBlock: boolean;
      text: string;
      block: boolean;
    }
    interface Text {
      type: "text";
      raw: string;
      text: string;
      tokens?: Token[];
      escaped?: boolean;
    }
  }
  export type Links = Record<string, Pick<Tokens.Link | Tokens.Image, "href" | "title">>;
  export type TokensList = Token[] & {
    links: Links;
  };
  /**
   * Renderer
   */
  class _Renderer {
    options: MarkedOptions;
    parser: _Parser;
    constructor(options?: MarkedOptions);
    space(token: Tokens.Space): string;
    code({ text, lang, escaped }: Tokens.Code): string;
    blockquote({ tokens }: Tokens.Blockquote): string;
    html({ text }: Tokens.HTML | Tokens.Tag): string;
    heading({ tokens, depth }: Tokens.Heading): string;
    hr(token: Tokens.Hr): string;
    list(token: Tokens.List): string;
    listitem(item: Tokens.ListItem): string;
    checkbox({ checked }: Tokens.Checkbox): string;
    paragraph({ tokens }: Tokens.Paragraph): string;
    table(token: Tokens.Table): string;
    tablerow({ text }: Tokens.TableRow): string;
    tablecell(token: Tokens.TableCell): string;
    /**
     * span level renderer
     */
    strong({ tokens }: Tokens.Strong): string;
    em({ tokens }: Tokens.Em): string;
    codespan({ text }: Tokens.Codespan): string;
    br(token: Tokens.Br): string;
    del({ tokens }: Tokens.Del): string;
    link({ href, title, tokens }: Tokens.Link): string;
    image({ href, title, text }: Tokens.Image): string;
    text(token: Tokens.Text | Tokens.Escape): string;
  }
  /**
   * TextRenderer
   * returns only the textual part of the token
   */
  class _TextRenderer {
    strong({ text }: Tokens.Strong): string;
    em({ text }: Tokens.Em): string;
    codespan({ text }: Tokens.Codespan): string;
    del({ text }: Tokens.Del): string;
    html({ text }: Tokens.HTML | Tokens.Tag): string;
    text({ text }: Tokens.Text | Tokens.Escape | Tokens.Tag): string;
    link({ text }: Tokens.Link): string;
    image({ text }: Tokens.Image): string;
    br(): string;
  }
  /**
   * Parsing & Compiling
   */
  class _Parser {
    options: MarkedOptions;
    renderer: _Renderer;
    textRenderer: _TextRenderer;
    constructor(options?: MarkedOptions);
    /**
     * Static Parse Method
     */
    static parse(tokens: Token[], options?: MarkedOptions): string;
    /**
     * Static Parse Inline Method
     */
    static parseInline(tokens: Token[], options?: MarkedOptions): string;
    /**
     * Parse Loop
     */
    parse(tokens: Token[], top?: boolean): string;
    /**
     * Parse Inline Tokens
     */
    parseInline(tokens: Token[], renderer?: _Renderer | _TextRenderer): string;
  }
  const other: {
    codeRemoveIndent: RegExp;
    outputLinkReplace: RegExp;
    indentCodeCompensation: RegExp;
    beginningSpace: RegExp;
    endingHash: RegExp;
    startingSpaceChar: RegExp;
    endingSpaceChar: RegExp;
    nonSpaceChar: RegExp;
    newLineCharGlobal: RegExp;
    tabCharGlobal: RegExp;
    multipleSpaceGlobal: RegExp;
    blankLine: RegExp;
    doubleBlankLine: RegExp;
    blockquoteStart: RegExp;
    blockquoteSetextReplace: RegExp;
    blockquoteSetextReplace2: RegExp;
    listReplaceTabs: RegExp;
    listReplaceNesting: RegExp;
    listIsTask: RegExp;
    listReplaceTask: RegExp;
    anyLine: RegExp;
    hrefBrackets: RegExp;
    tableDelimiter: RegExp;
    tableAlignChars: RegExp;
    tableRowBlankLine: RegExp;
    tableAlignRight: RegExp;
    tableAlignCenter: RegExp;
    tableAlignLeft: RegExp;
    startATag: RegExp;
    endATag: RegExp;
    startPreScriptTag: RegExp;
    endPreScriptTag: RegExp;
    startAngleBracket: RegExp;
    endAngleBracket: RegExp;
    pedanticHrefTitle: RegExp;
    unicodeAlphaNumeric: RegExp;
    escapeTest: RegExp;
    escapeReplace: RegExp;
    escapeTestNoEncode: RegExp;
    escapeReplaceNoEncode: RegExp;
    unescapeTest: RegExp;
    caret: RegExp;
    percentDecode: RegExp;
    findPipe: RegExp;
    splitPipe: RegExp;
    slashPipe: RegExp;
    carriageReturn: RegExp;
    spaceLine: RegExp;
    notSpaceStart: RegExp;
    endingNewline: RegExp;
    listItemRegex: (bull: string) => RegExp;
    nextBulletRegex: (indent: number) => RegExp;
    hrRegex: (indent: number) => RegExp;
    fencesBeginRegex: (indent: number) => RegExp;
    headingBeginRegex: (indent: number) => RegExp;
    htmlBeginRegex: (indent: number) => RegExp;
  };
  const blockNormal: {
    blockquote: RegExp;
    code: RegExp;
    def: RegExp;
    fences: RegExp;
    heading: RegExp;
    hr: RegExp;
    html: RegExp;
    lheading: RegExp;
    list: RegExp;
    newline: RegExp;
    paragraph: RegExp;
    table: RegExp;
    text: RegExp;
  };
  export type BlockKeys = keyof typeof blockNormal;
  const inlineNormal: {
    _backpedal: RegExp;
    anyPunctuation: RegExp;
    autolink: RegExp;
    blockSkip: RegExp;
    br: RegExp;
    code: RegExp;
    del: RegExp;
    emStrongLDelim: RegExp;
    emStrongRDelimAst: RegExp;
    emStrongRDelimUnd: RegExp;
    escape: RegExp;
    link: RegExp;
    nolink: RegExp;
    punctuation: RegExp;
    reflink: RegExp;
    reflinkSearch: RegExp;
    tag: RegExp;
    text: RegExp;
    url: RegExp;
  };
  export type InlineKeys = keyof typeof inlineNormal;
  export interface Rules {
    other: typeof other;
    block: Record<BlockKeys, RegExp>;
    inline: Record<InlineKeys, RegExp>;
  }
  /**
   * Tokenizer
   */
  class _Tokenizer {
    options: MarkedOptions;
    rules: Rules;
    lexer: _Lexer;
    constructor(options?: MarkedOptions);
    space(src: string): Tokens.Space | undefined;
    code(src: string): Tokens.Code | undefined;
    fences(src: string): Tokens.Code | undefined;
    heading(src: string): Tokens.Heading | undefined;
    hr(src: string): Tokens.Hr | undefined;
    blockquote(src: string): Tokens.Blockquote | undefined;
    list(src: string): Tokens.List | undefined;
    html(src: string): Tokens.HTML | undefined;
    def(src: string): Tokens.Def | undefined;
    table(src: string): Tokens.Table | undefined;
    lheading(src: string): Tokens.Heading | undefined;
    paragraph(src: string): Tokens.Paragraph | undefined;
    text(src: string): Tokens.Text | undefined;
    escape(src: string): Tokens.Escape | undefined;
    tag(src: string): Tokens.Tag | undefined;
    link(src: string): Tokens.Link | Tokens.Image | undefined;
    reflink(src: string, links: Links): Tokens.Link | Tokens.Image | Tokens.Text | undefined;
    emStrong(src: string, maskedSrc: string, prevChar?: string): Tokens.Em | Tokens.Strong | undefined;
    codespan(src: string): Tokens.Codespan | undefined;
    br(src: string): Tokens.Br | undefined;
    del(src: string): Tokens.Del | undefined;
    autolink(src: string): Tokens.Link | undefined;
    url(src: string): Tokens.Link | undefined;
    inlineText(src: string): Tokens.Text | undefined;
  }
  class _Hooks {
    options: MarkedOptions;
    block?: boolean;
    constructor(options?: MarkedOptions);
    static passThroughHooks: Set<string>;
    /**
     * Process markdown before marked
     */
    preprocess(markdown: string): string;
    /**
     * Process HTML after marked is finished
     */
    postprocess(html: string): string;
    /**
     * Process all tokens before walk tokens
     */
    processAllTokens(tokens: Token[] | TokensList): Token[] | TokensList;
    /**
     * Provide function to tokenize markdown
     */
    provideLexer(): typeof _Lexer.lexInline;
    /**
     * Provide function to parse tokens
     */
    provideParser(): typeof _Parser.parse;
  }
  export interface TokenizerThis {
    lexer: _Lexer;
  }
  export type TokenizerExtensionFunction = (this: TokenizerThis, src: string, tokens: Token[] | TokensList) => Tokens.Generic | undefined;
  export type TokenizerStartFunction = (this: TokenizerThis, src: string) => number | void;
  export interface TokenizerExtension {
    name: string;
    level: "block" | "inline";
    start?: TokenizerStartFunction;
    tokenizer: TokenizerExtensionFunction;
    childTokens?: string[];
  }
  export interface RendererThis {
    parser: _Parser;
  }
  export type RendererExtensionFunction = (this: RendererThis, token: Tokens.Generic) => string | false | undefined;
  export interface RendererExtension {
    name: string;
    renderer: RendererExtensionFunction;
  }
  export type TokenizerAndRendererExtension = TokenizerExtension | RendererExtension | (TokenizerExtension & RendererExtension);
  export type HooksApi = Omit<_Hooks, "constructor" | "options" | "block">;
  export type HooksObject = {
    [K in keyof HooksApi]?: (this: _Hooks, ...args: Parameters<HooksApi[K]>) => ReturnType<HooksApi[K]> | Promise<ReturnType<HooksApi[K]>>;
  };
  export type RendererApi = Omit<_Renderer, "constructor" | "options" | "parser">;
  export type RendererObject = {
    [K in keyof RendererApi]?: (this: _Renderer, ...args: Parameters<RendererApi[K]>) => ReturnType<RendererApi[K]> | false;
  };
  export type TokenizerApi = Omit<_Tokenizer, "constructor" | "options" | "rules" | "lexer">;
  export type TokenizerObject = {
    [K in keyof TokenizerApi]?: (this: _Tokenizer, ...args: Parameters<TokenizerApi[K]>) => ReturnType<TokenizerApi[K]> | false;
  };
  export interface MarkedExtension {
    /**
     * True will tell marked to await any walkTokens functions before parsing the tokens and returning an HTML string.
     */
    async?: boolean;
    /**
     * Enable GFM line breaks. This option requires the gfm option to be true.
     */
    breaks?: boolean;
    /**
     * Add tokenizers and renderers to marked
     */
    extensions?: TokenizerAndRendererExtension[] | null;
    /**
     * Enable GitHub flavored markdown.
     */
    gfm?: boolean;
    /**
     * Hooks are methods that hook into some part of marked.
     * preprocess is called to process markdown before sending it to marked.
     * processAllTokens is called with the TokensList before walkTokens.
     * postprocess is called to process html after marked has finished parsing.
     * provideLexer is called to provide a function to tokenize markdown.
     * provideParser is called to provide a function to parse tokens.
     */
    hooks?: HooksObject | null;
    /**
     * Conform to obscure parts of markdown.pl as much as possible. Don't fix any of the original markdown bugs or poor behavior.
     */
    pedantic?: boolean;
    /**
     * Type: object Default: new Renderer()
     *
     * An object containing functions to render tokens to HTML.
     */
    renderer?: RendererObject | null;
    /**
     * Shows an HTML error message when rendering fails.
     */
    silent?: boolean;
    /**
     * The tokenizer defines how to turn markdown text into tokens.
     */
    tokenizer?: TokenizerObject | null;
    /**
     * The walkTokens function gets called with every token.
     * Child tokens are called before moving on to sibling tokens.
     * Each token is passed by reference so updates are persisted when passed to the parser.
     * The return value of the function is ignored.
     */
    walkTokens?: ((token: Token) => void | Promise<void>) | null;
  }
  export interface MarkedOptions extends Omit<MarkedExtension, "hooks" | "renderer" | "tokenizer" | "extensions" | "walkTokens"> {
    /**
     * Hooks are methods that hook into some part of marked.
     */
    hooks?: _Hooks | null;
    /**
     * Type: object Default: new Renderer()
     *
     * An object containing functions to render tokens to HTML.
     */
    renderer?: _Renderer | null;
    /**
     * The tokenizer defines how to turn markdown text into tokens.
     */
    tokenizer?: _Tokenizer | null;
    /**
     * Custom extensions
     */
    extensions?: null | {
      renderers: {
        [name: string]: RendererExtensionFunction;
      };
      childTokens: {
        [name: string]: string[];
      };
      inline?: TokenizerExtensionFunction[];
      block?: TokenizerExtensionFunction[];
      startInline?: TokenizerStartFunction[];
      startBlock?: TokenizerStartFunction[];
    };
    /**
     * walkTokens function returns array of values for Promise.all
     */
    walkTokens?: null | ((token: Token) => void | Promise<void> | (void | Promise<void>)[]);
  }
  /**
   * Block Lexer
   */
  class _Lexer {
    tokens: TokensList;
    options: MarkedOptions;
    state: {
      inLink: boolean;
      inRawBlock: boolean;
      top: boolean;
    };
    private tokenizer;
    private inlineQueue;
    constructor(options?: MarkedOptions);
    /**
     * Expose Rules
     */
    static get rules(): {
      block: {
        normal: {
          blockquote: RegExp;
          code: RegExp;
          def: RegExp;
          fences: RegExp;
          heading: RegExp;
          hr: RegExp;
          html: RegExp;
          lheading: RegExp;
          list: RegExp;
          newline: RegExp;
          paragraph: RegExp;
          table: RegExp;
          text: RegExp;
        };
        gfm: Record<"code" | "blockquote" | "hr" | "html" | "table" | "text" | "def" | "heading" | "list" | "paragraph" | "fences" | "lheading" | "newline", RegExp>;
        pedantic: Record<"code" | "blockquote" | "hr" | "html" | "table" | "text" | "def" | "heading" | "list" | "paragraph" | "fences" | "lheading" | "newline", RegExp>;
      };
      inline: {
        normal: {
          _backpedal: RegExp;
          anyPunctuation: RegExp;
          autolink: RegExp;
          blockSkip: RegExp;
          br: RegExp;
          code: RegExp;
          del: RegExp;
          emStrongLDelim: RegExp;
          emStrongRDelimAst: RegExp;
          emStrongRDelimUnd: RegExp;
          escape: RegExp;
          link: RegExp;
          nolink: RegExp;
          punctuation: RegExp;
          reflink: RegExp;
          reflinkSearch: RegExp;
          tag: RegExp;
          text: RegExp;
          url: RegExp;
        };
        gfm: Record<"link" | "code" | "url" | "br" | "del" | "text" | "escape" | "tag" | "reflink" | "nolink" | "_backpedal" | "anyPunctuation" | "autolink" | "blockSkip" | "emStrongLDelim" | "emStrongRDelimAst" | "emStrongRDelimUnd" | "punctuation" | "reflinkSearch", RegExp>;
        breaks: Record<"link" | "code" | "url" | "br" | "del" | "text" | "escape" | "tag" | "reflink" | "nolink" | "_backpedal" | "anyPunctuation" | "autolink" | "blockSkip" | "emStrongLDelim" | "emStrongRDelimAst" | "emStrongRDelimUnd" | "punctuation" | "reflinkSearch", RegExp>;
        pedantic: Record<"link" | "code" | "url" | "br" | "del" | "text" | "escape" | "tag" | "reflink" | "nolink" | "_backpedal" | "anyPunctuation" | "autolink" | "blockSkip" | "emStrongLDelim" | "emStrongRDelimAst" | "emStrongRDelimUnd" | "punctuation" | "reflinkSearch", RegExp>;
      };
    };
    /**
     * Static Lex Method
     */
    static lex(src: string, options?: MarkedOptions): TokensList;
    /**
     * Static Lex Inline Method
     */
    static lexInline(src: string, options?: MarkedOptions): Token[];
    /**
     * Preprocessing
     */
    lex(src: string): TokensList;
    /**
     * Lexing
     */
    blockTokens(src: string, tokens?: Token[], lastParagraphClipped?: boolean): Token[];
    blockTokens(src: string, tokens?: TokensList, lastParagraphClipped?: boolean): TokensList;
    inline(src: string, tokens?: Token[]): Token[];
    /**
     * Lexing/Compiling
     */
    inlineTokens(src: string, tokens?: Token[]): Token[];
  }
  /**
   * Gets the original marked default options.
   */
  function _getDefaults(): MarkedOptions;
  let _defaults: MarkedOptions;
  export type MaybePromise = void | Promise<void>;
  export class Marked {
    defaults: MarkedOptions;
    options: (opt: MarkedOptions) => this;
    parse: {
      (src: string, options: MarkedOptions & {
        async: true;
      }): Promise<string>;
      (src: string, options: MarkedOptions & {
        async: false;
      }): string;
      (src: string, options?: MarkedOptions | null): string | Promise<string>;
    };
    parseInline: {
      (src: string, options: MarkedOptions & {
        async: true;
      }): Promise<string>;
      (src: string, options: MarkedOptions & {
        async: false;
      }): string;
      (src: string, options?: MarkedOptions | null): string | Promise<string>;
    };
    Parser: typeof _Parser;
    Renderer: typeof _Renderer;
    TextRenderer: typeof _TextRenderer;
    Lexer: typeof _Lexer;
    Tokenizer: typeof _Tokenizer;
    Hooks: typeof _Hooks;
    constructor(...args: MarkedExtension[]);
    /**
     * Run callback for every token
     */
    walkTokens(tokens: Token[] | TokensList, callback: (token: Token) => MaybePromise | MaybePromise[]): MaybePromise[];
    use(...args: MarkedExtension[]): this;
    setOptions(opt: MarkedOptions): this;
    lexer(src: string, options?: MarkedOptions): TokensList;
    parser(tokens: Token[], options?: MarkedOptions): string;
    private parseMarkdown;
    private onError;
  }
  /**
   * Compiles markdown to HTML asynchronously.
   *
   * @param src String of markdown source to be compiled
   * @param options Hash of options, having async: true
   * @return Promise of string of compiled HTML
   */
  export function marked(src: string, options: MarkedOptions & {
    async: true;
  }): Promise<string>;
  /**
   * Compiles markdown to HTML.
   *
   * @param src String of markdown source to be compiled
   * @param options Optional hash of options
   * @return String of compiled HTML. Will be a Promise of string if async is set to true by any extensions.
   */
  export function marked(src: string, options: MarkedOptions & {
    async: false;
  }): string;
  export function marked(src: string, options: MarkedOptions & {
    async: true;
  }): Promise<string>;
  export function marked(src: string, options?: MarkedOptions | null): string | Promise<string>;
  export namespace marked {
    var options: (options: MarkedOptions) => typeof marked;
    var setOptions: (options: MarkedOptions) => typeof marked;
    var getDefaults: typeof _getDefaults;
    var defaults: MarkedOptions;
    var use: (...args: MarkedExtension[]) => typeof marked;
    var walkTokens: (tokens: Token[] | TokensList, callback: (token: Token) => MaybePromise | MaybePromise[]) => MaybePromise[];
    var parseInline: {
      (src: string, options: MarkedOptions & {
        async: true;
      }): Promise<string>;
      (src: string, options: MarkedOptions & {
        async: false;
      }): string;
      (src: string, options?: MarkedOptions | null): string | Promise<string>;
    };
    var Parser: typeof _Parser;
    var parser: typeof _Parser.parse;
    var Renderer: typeof _Renderer;
    var TextRenderer: typeof _TextRenderer;
    var Lexer: typeof _Lexer;
    var lexer: typeof _Lexer.lex;
    var Tokenizer: typeof _Tokenizer;
    var Hooks: typeof _Hooks;
    var parse: typeof marked;
  }
  export const options: (options: MarkedOptions) => typeof marked;
  export const setOptions: (options: MarkedOptions) => typeof marked;
  export const use: (...args: MarkedExtension[]) => typeof marked;
  export const walkTokens: (tokens: Token[] | TokensList, callback: (token: Token) => MaybePromise | MaybePromise[]) => MaybePromise[];
  export const parseInline: {
    (src: string, options: MarkedOptions & {
      async: true;
    }): Promise<string>;
    (src: string, options: MarkedOptions & {
      async: false;
    }): string;
    (src: string, options?: MarkedOptions | null): string | Promise<string>;
  };
  export const parse: typeof marked;
  export const parser: typeof _Parser.parse;
  export const lexer: typeof _Lexer.lex;

  export {
    _Hooks as Hooks,
    _Lexer as Lexer,
    _Parser as Parser,
    _Renderer as Renderer,
    _TextRenderer as TextRenderer,
    _Tokenizer as Tokenizer,
    _defaults as defaults,
    _getDefaults as getDefaults,
  };
}
/**
* Read and write data to SQLite databases. 
*
* @module jamfile:sqlite
*/
declare module 'jamfile:sqlite'{

  /**
   *  SQLite library version string
  */
  export const SQLITE_VERSION: string;

  export type SqlValue =
    | number
    | string
    | Uint8Array // BLOB or // sqlite-vec bit vector
    | Float32Array // sqlite-vec float32 vector
    | Int8Array // sqlite-vec int8 vector
    | null;

    export type SqlParameterValue =
    | number
    | string
    | null
    | Uint8Array
    | ArrayBufferView
    | Date;
    
  /**
   * "Escapes" a astring for usage as an identifier in SQL. Must be wrapped by double quotes.
   *
   * Internally uses the "%w" substitution in SQLite's custom printf https://www.sqlite.org/printf.html#percentw
   *
   * @param identifier string identifier to escape.
   */
  export function escapeIdentifier(identifier: string): string;

  export class Database {
      /**
       * Opens a SQLite database.
       *
       * @param path The path of the database.
       */
      constructor(path: string);

      /**
       * Execute a SQL query and return the first value of the first row.
       *
       * Throws an error in the following cases:
       * - Preparing the SQL query fails
       * - The SQL query doesn't return any rows
       * - The SQL query is not read-only
       * @param sql
       * @param params
       */
      queryValue(sql: string, params?: SqlParameterValue[]) : SqlValue;
      queryRow(sql: string, params?: SqlParameterValue[]) : {[key: string]: SqlValue};
      queryAll(sql: string, params?: SqlParameterValue[]) : {[key: string]: SqlValue}[];

      execute(sql: string, params?: SqlParameterValue[]): void;
      executeScript(sql: string): void;

  }
}


declare module 'jamfile:llamafile' {

  export class TextEmbeddingModel {
    constructor(path?:string);

    /**
     * number of dimensions per embedding, ex 386, 768, 1024, etc
     */
    dimensions: number;

    embed(input: string): Float32Array;
    tokenize(input: string): Int32Array;
    detokenize(input: Int32Array): string;
  }

  export type CompletionOptions<T> = {
    schema: JSONSchema7 | string,
    parse: (response:string) => T;
  };
  
  export class CompletionModel {
    /**
     * A string description the "type" of the underyling model.
     * 
     * Result of the `llama_model_desc()` C function.
     * 
     */
    description: string;

    constructor(model_path?: string);

    complete<T>(input: string, options?: CompletionOptions<T>): string | T;
    tokenize(input: string): Int32Array;
    detokenize(input: Int32Array): string;
  }
}

declare module 'jamfile:cli' {
  /** Combines recursively all intersection types and returns a new single type.
   * @internal
   */
  type Id<TRecord> = TRecord extends Record<string, unknown>
  ? TRecord extends infer InferredRecord
    ? { [Key in keyof InferredRecord]: Id<InferredRecord[Key]> }
  : never
  : TRecord;

  /** Converts a union type `A | B | C` into an intersection type `A & B & C`.
  * @internal
  */
  type UnionToIntersection<TValue> =
  (TValue extends unknown ? (args: TValue) => unknown : never) extends
    (args: infer R) => unknown ? R extends Record<string, unknown> ? R : never
    : never;

  /** @internal */
  type BooleanType = boolean | string | undefined;
  /** @internal */
  type StringType = string | undefined;
  /** @internal */
  type ArgType = StringType | BooleanType;

  /** @internal */
  type Collectable = string | undefined;
  /** @internal */
  type Negatable = string | undefined;

  type UseTypes<
  TBooleans extends BooleanType,
  TStrings extends StringType,
  TCollectable extends Collectable,
  > = undefined extends (
  & (false extends TBooleans ? undefined : TBooleans)
  & TCollectable
  & TStrings
  ) ? false
  : true;

  /**
  * Creates a record with all available flags with the corresponding type and
  * default type.
  * @internal
  */
  type Values<
  TBooleans extends BooleanType,
  TStrings extends StringType,
  TCollectable extends Collectable,
  TNegatable extends Negatable,
  TDefault extends Record<string, unknown> | undefined,
  TAliases extends Aliases | undefined,
  > = UseTypes<TBooleans, TStrings, TCollectable> extends true ?
    & Record<string, unknown>
    & AddAliases<
      SpreadDefaults<
        & CollectValues<TStrings, string, TCollectable, TNegatable>
        & RecursiveRequired<CollectValues<TBooleans, boolean, TCollectable>>
        & CollectUnknownValues<
          TBooleans,
          TStrings,
          TCollectable,
          TNegatable
        >,
        DedotRecord<TDefault>
      >,
      TAliases
    >
  // deno-lint-ignore no-explicit-any
  : Record<string, any>;

  /** @internal */
  type Aliases<TArgNames = string, TAliasNames extends string = string> = Partial<
  Record<Extract<TArgNames, string>, TAliasNames | ReadonlyArray<TAliasNames>>
  >;

  type AddAliases<
  TArgs,
  TAliases extends Aliases | undefined,
  > = {
  [TArgName in keyof TArgs as AliasNames<TArgName, TAliases>]: TArgs[TArgName];
  };

  type AliasNames<
  TArgName,
  TAliases extends Aliases | undefined,
  > = TArgName extends keyof TAliases
  ? string extends TAliases[TArgName] ? TArgName
  : TAliases[TArgName] extends string ? TArgName | TAliases[TArgName]
  : TAliases[TArgName] extends Array<string>
    ? TArgName | TAliases[TArgName][number]
  : TArgName
  : TArgName;

  /**
  * Spreads all default values of Record `TDefaults` into Record `TArgs`
  * and makes default values required.
  *
  * **Example:**
  * `SpreadValues<{ foo?: boolean, bar?: number }, { foo: number }>`
  *
  * **Result:** `{ foo: boolean | number, bar?: number }`
  */
  type SpreadDefaults<TArgs, TDefaults> = TDefaults extends undefined ? TArgs
  : TArgs extends Record<string, unknown> ?
      & Omit<TArgs, keyof TDefaults>
      & {
        [Default in keyof TDefaults]: Default extends keyof TArgs
          ? (TArgs[Default] & TDefaults[Default] | TDefaults[Default]) extends
            Record<string, unknown>
            ? NonNullable<SpreadDefaults<TArgs[Default], TDefaults[Default]>>
          : TDefaults[Default] | NonNullable<TArgs[Default]>
          : unknown;
      }
  : never;

  /**
  * Defines the Record for the `default` option to add
  * auto-suggestion support for IDE's.
  * @internal
  */
  type Defaults<TBooleans extends BooleanType, TStrings extends StringType> = Id<
  UnionToIntersection<
    & Record<string, unknown>
    // Dedotted auto suggestions: { foo: { bar: unknown } }
    & MapTypes<TStrings, unknown>
    & MapTypes<TBooleans, unknown>
    // Flat auto suggestions: { "foo.bar": unknown }
    & MapDefaults<TBooleans>
    & MapDefaults<TStrings>
  >
  >;

  type MapDefaults<TArgNames extends ArgType> = Partial<
  Record<TArgNames extends string ? TArgNames : string, unknown>
  >;

  type RecursiveRequired<TRecord> = TRecord extends Record<string, unknown> ? {
    [Key in keyof TRecord]-?: RecursiveRequired<TRecord[Key]>;
  }
  : TRecord;

  /** Same as `MapTypes` but also supports collectable options. */
  type CollectValues<
  TArgNames extends ArgType,
  TType,
  TCollectable extends Collectable,
  TNegatable extends Negatable = undefined,
  > = UnionToIntersection<
  Extract<TArgNames, TCollectable> extends string ?
      & (Exclude<TArgNames, TCollectable> extends never ? Record<never, never>
        : MapTypes<Exclude<TArgNames, TCollectable>, TType, TNegatable>)
      & (Extract<TArgNames, TCollectable> extends never ? Record<never, never>
        : RecursiveRequired<
          MapTypes<Extract<TArgNames, TCollectable>, Array<TType>, TNegatable>
        >)
    : MapTypes<TArgNames, TType, TNegatable>
  >;

  /** Same as `Record` but also supports dotted and negatable options. */
  type MapTypes<
  TArgNames extends ArgType,
  TType,
  TNegatable extends Negatable = undefined,
  > = undefined extends TArgNames ? Record<never, never>
  : TArgNames extends `${infer Name}.${infer Rest}` ? {
      [Key in Name]?: MapTypes<
        Rest,
        TType,
        TNegatable extends `${Name}.${infer Negate}` ? Negate : undefined
      >;
    }
  : TArgNames extends string ? Partial<
      Record<TArgNames, TNegatable extends TArgNames ? TType | false : TType>
    >
  : Record<never, never>;

  type CollectUnknownValues<
  TBooleans extends BooleanType,
  TStrings extends StringType,
  TCollectable extends Collectable,
  TNegatable extends Negatable,
  > = UnionToIntersection<
  TCollectable extends TBooleans & TStrings ? Record<never, never>
    : DedotRecord<
      // Unknown collectable & non-negatable args.
      & Record<
        Exclude<
          Extract<Exclude<TCollectable, TNegatable>, string>,
          Extract<TStrings | TBooleans, string>
        >,
        Array<unknown>
      >
      // Unknown collectable & negatable args.
      & Record<
        Exclude<
          Extract<Extract<TCollectable, TNegatable>, string>,
          Extract<TStrings | TBooleans, string>
        >,
        Array<unknown> | false
      >
    >
  >;

  /** Converts `{ "foo.bar.baz": unknown }` into `{ foo: { bar: { baz: unknown } } }`. */
  type DedotRecord<TRecord> = Record<string, unknown> extends TRecord ? TRecord
  : TRecord extends Record<string, unknown> ? UnionToIntersection<
      ValueOf<
        {
          [Key in keyof TRecord]: Key extends string ? Dedot<Key, TRecord[Key]>
            : never;
        }
      >
    >
  : TRecord;

  type Dedot<TKey extends string, TValue> = TKey extends
  `${infer Name}.${infer Rest}` ? { [Key in Name]: Dedot<Rest, TValue> }
  : { [Key in TKey]: TValue };

  type ValueOf<TValue> = TValue[keyof TValue];

  /** The value returned from {@linkcode parseArgs}. */
  export type Args<
  // deno-lint-ignore no-explicit-any
  TArgs extends Record<string, unknown> = Record<string, any>,
  TDoubleDash extends boolean | undefined = undefined,
  > = Id<
  & TArgs
  & {
    /** Contains all the arguments that didn't have an option associated with
     * them. */
    _: Array<string | number>;
  }
  & (boolean extends TDoubleDash ? DoubleDash
    : true extends TDoubleDash ? Required<DoubleDash>
    : Record<never, never>)
  >;

  /** @internal */
  type DoubleDash = {
  /** Contains all the arguments that appear after the double dash: "--". */
  "--"?: Array<string>;
  };

  /** Options for {@linkcode parseArgs}. */
  export interface ParseOptions<
  TBooleans extends BooleanType = BooleanType,
  TStrings extends StringType = StringType,
  TCollectable extends Collectable = Collectable,
  TNegatable extends Negatable = Negatable,
  TDefault extends Record<string, unknown> | undefined =
    | Record<string, unknown>
    | undefined,
  TAliases extends Aliases | undefined = Aliases | undefined,
  TDoubleDash extends boolean | undefined = boolean | undefined,
  > {
  /**
   * When `true`, populate the result `_` with everything before the `--` and
   * the result `['--']` with everything after the `--`.
   *
   * @default {false}
   *
   *  @example
   * ```ts
   * // $ deno run example.ts -- a arg1
   * import { parseArgs } from "@std/cli/parse-args";
   * console.dir(parseArgs(Deno.args, { "--": false }));
   * // output: { _: [ "a", "arg1" ] }
   * console.dir(parseArgs(Deno.args, { "--": true }));
   * // output: { _: [], --: [ "a", "arg1" ] }
   * ```
   */
  "--"?: TDoubleDash;

  /**
   * An object mapping string names to strings or arrays of string argument
   * names to use as aliases.
   *
   * @default {{}}
   */
  alias?: TAliases;

  /**
   * A boolean, string or array of strings to always treat as booleans. If
   * `true` will treat all double hyphenated arguments without equal signs as
   * `boolean` (e.g. affects `--foo`, not `-f` or `--foo=bar`).
   *  All `boolean` arguments will be set to `false` by default.
   *
   * @default {false}
   */
  boolean?: TBooleans | ReadonlyArray<Extract<TBooleans, string>>;

  /**
   * An object mapping string argument names to default values.
   *
   * @default {{}}
   */
  default?: TDefault & Defaults<TBooleans, TStrings>;

  /**
   * When `true`, populate the result `_` with everything after the first
   * non-option.
   *
   * @default {false}
   */
  stopEarly?: boolean;

  /**
   * A string or array of strings argument names to always treat as strings.
   *
   * @default {[]}
   */
  string?: TStrings | ReadonlyArray<Extract<TStrings, string>>;

  /**
   * A string or array of strings argument names to always treat as arrays.
   * Collectable options can be used multiple times. All values will be
   * collected into one array. If a non-collectable option is used multiple
   * times, the last value is used.
   *
   * @default {[]}
   */
  collect?: TCollectable | ReadonlyArray<Extract<TCollectable, string>>;

  /**
   * A string or array of strings argument names which can be negated
   * by prefixing them with `--no-`, like `--no-config`.
   *
   * @default {[]}
   */
  negatable?: TNegatable | ReadonlyArray<Extract<TNegatable, string>>;

  /**
   * A function which is invoked with a command line parameter not defined in
   * the `options` configuration object. If the function returns `false`, the
   * unknown option is not added to `parsedArgs`.
   *
   * @default {unknown}
   */
  unknown?: (arg: string, key?: string, value?: unknown) => unknown;
  }

  interface NestedMapping {
  [key: string]: NestedMapping | unknown;
  }

  /**
  * Take a set of command line arguments, optionally with a set of options, and
  * return an object representing the flags found in the passed arguments.
  *
  * By default, any arguments starting with `-` or `--` are considered boolean
  * flags. If the argument name is followed by an equal sign (`=`) it is
  * considered a key-value pair. Any arguments which could not be parsed are
  * available in the `_` property of the returned object.
  *
  * By default, this module tries to determine the type of all arguments
  * automatically and the return type of this function will have an index
  * signature with `any` as value (`{ [x: string]: any }`).
  *
  * If the `string`, `boolean` or `collect` option is set, the return value of
  * this function will be fully typed and the index signature of the return
  * type will change to `{ [x: string]: unknown }`.
  *
  * Any arguments after `'--'` will not be parsed and will end up in `parsedArgs._`.
  *
  * Numeric-looking arguments will be returned as numbers unless `options.string`
  * or `options.boolean` is set for that argument name.
  *
  * @param args An array of command line arguments.
  * @param options Options for the parse function.
  *
  * @typeParam TArgs Type of result.
  * @typeParam TDoubleDash Used by `TArgs` for the result.
  * @typeParam TBooleans Used by `TArgs` for the result.
  * @typeParam TStrings Used by `TArgs` for the result.
  * @typeParam TCollectable Used by `TArgs` for the result.
  * @typeParam TNegatable Used by `TArgs` for the result.
  * @typeParam TDefaults Used by `TArgs` for the result.
  * @typeParam TAliases Used by `TArgs` for the result.
  * @typeParam TAliasArgNames Used by `TArgs` for the result.
  * @typeParam TAliasNames Used by `TArgs` for the result.
  *
  * @return The parsed arguments.
  *
  * @example Usage
  * ```ts
  * import { parseArgs } from "@std/cli/parse-args";
  * import { assertEquals } from "@std/assert";
  *
  * // For proper use, one should use `parseArgs(Deno.args)`
  * assertEquals(parseArgs(["--foo", "--bar=baz", "./quux.txt"]), {
  *   foo: true,
  *   bar: "baz",
  *   _: ["./quux.txt"],
  * });
  * ```
  */
  export function parseArgs<
  TArgs extends Values<
    TBooleans,
    TStrings,
    TCollectable,
    TNegatable,
    TDefaults,
    TAliases
  >,
  TDoubleDash extends boolean | undefined = undefined,
  TBooleans extends BooleanType = undefined,
  TStrings extends StringType = undefined,
  TCollectable extends Collectable = undefined,
  TNegatable extends Negatable = undefined,
  TDefaults extends Record<string, unknown> | undefined = undefined,
  TAliases extends Aliases<TAliasArgNames, TAliasNames> | undefined = undefined,
  TAliasArgNames extends string = string,
  TAliasNames extends string = string,
  >(
  args: string[],
  options?: ParseOptions<
    TBooleans,
    TStrings,
    TCollectable,
    TNegatable,
    TDefaults,
    TAliases,
    TDoubleDash
  >,
  ): Args<TArgs, TDoubleDash>;
}

declare module 'jamfile:fmt' {

  /** Options for {@linkcode format}. */
  export interface FormatDurationOptions {
  /**
   * The style for formatting the duration.
   *
   * "narrow" for "0d 0h 0m 0s 0ms..."
   * "digital" for "00:00:00:00:000..."
   * "full" for "0 days, 0 hours, 0 minutes,..."
   *
   * @default {"narrow"}
   */
  style?: "narrow" | "digital" | "full";
  /**
   * Whether to ignore zero values.
   * With style="narrow" | "full", all zero values are ignored.
   * With style="digital", only values in the ends are ignored.
   *
   * @default {false}
   */
  ignoreZero?: boolean;
}

  /**
   * Format milliseconds to time duration.
   *
   * @param ms The milliseconds value to format
   * @param options The options for formatting
   * @returns The formatted string
   */
  export function formatDuration(
    ms: number,
    options?: FormatDurationOptions,
  ): string;


  type LocaleOptions = {
    minimumFractionDigits?: number;
    maximumFractionDigits?: number;
  };

  /** Options for {@linkcode format}. */
  export interface FormatBytesOptions {
    /**
     * Uses bits representation.
     *
     * @default {false}
     */
    bits?: boolean;
    /**
     * Uses binary bytes (e.g. kibibyte).
     *
     * @default {false}
     */
    binary?: boolean;
    /**
     * Include plus sign for positive numbers.
     *
     * @default {false}
     */
    signed?: boolean;
    /**
     * Uses localized number formatting. If it is set to true, uses default
     * locale on the system. If it's set to string, uses that locale. The given
     * string should be a
     * {@link https://en.wikipedia.org/wiki/IETF_language_tag | BCP 47 language tag}.
     * You can also give the list of language tags.
     */
    locale?: boolean | string | string[];
    /**
     * The minimum number of fraction digits to display. If neither
     * {@linkcode minimumFractionDigits} or {@linkcode maximumFractionDigits}
     * are set.
     *
     * @default {3}
     */
    minimumFractionDigits?: number;
    /**
     * The maximum number of fraction digits to display. If neither
     * {@linkcode minimumFractionDigits} or {@linkcode maximumFractionDigits}
     * are set.
     *
     * @default {3}
     */
    maximumFractionDigits?: number;
  }

  /**
 * Convert bytes to a human-readable string: 1337 → 1.34 kB
 *
 * Based on {@link https://github.com/sindresorhus/pretty-bytes | pretty-bytes}.
 * A utility for displaying file sizes for humans.
 *
 * @param num The bytes value to format
 * @param options The options for formatting
 * @returns The formatted string
 *
 * @example Basic usage
 * ```ts
 * import { formatBytes } from "jamfile:fmt";
 *
 * formatBytes(1337); // "1.34 kB"
 * formatBytes(100); // "100 B"
 * ```
 *

 */
export function formatBytes(
  num: number,
  options?: FormatBytesOptions,
): string;
}


declare module 'jamfile:assert' {

type T = any;
/**
 * Make an assertion that `actual` and `expected` are almost equal numbers
through a given tolerance. It can be used to take into account IEEE-754
double-precision floating-point representation limitations. If the values
are not almost equal then throw.

The default tolerance is one hundred thousandth of a percent of the
expected value.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param tolerance The tolerance to consider the values almost equal. The
default is one hundred thousandth of a percent of the expected value.
 * @param msg The optional message to include in the error.
 */
export function assertAlmostEquals(actual: number, expected: number, tolerance: number, msg?: string): void;

/**
 * Make an assertion that `actual` includes the `expected` values. If not then
an error will be thrown.

Type parameter can be specified to ensure values under comparison have the
same type.
 * @param actual The array-like object to check for.
 * @param expected The array-like object to check for.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertArrayIncludes(actual: any[], expected: any[], msg?: string): void;

/**
 * Make an assertion that `actual` and `expected` are equal, deeply. If not
deeply equal, then throw.

Type parameter can be specified to ensure values under comparison have the
same type.

Note: When comparing `Blob` objects, you should first convert them to
`Uint8Array` using the `Blob.bytes()` method and then compare their
contents.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertEquals(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that actual is not null or undefined.
If not then throw.
 * @param actual The actual value to check.
 * @param msg The optional message to include in the error if the assertion fails.
 */
export function assertExists<T>(actual: T, msg?: string): asserts actual is NonNullable<T>;

type Falsy = false
| 0
| 0
| ""
| null
| undefined;

/**
 * Make an assertion, error will be thrown if `expr` have truthy value.
 * @param expr The expression to test.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertFalse(expr: unknown, undefined: undefined): asserts expr is Falsy;

/**
 * Make an assertion that `actual` is greater than or equal to `expected`.
If not then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertGreaterOrEqual(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `actual` is greater than `expected`.
If not then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertGreater(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `obj` is an instance of `type`.
If not then throw.
 * @param actual The object to check.
 * @param expectedType The expected class constructor.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertInstanceOf<T extends abstract new (...args: any) => any>(actual: unknown, expectedType: T, undefined: undefined): asserts actual is InstanceType<T>;

/**
 * Make an assertion that `error` is an `Error`.
If not then an error will be thrown.
An error class and a string that should be included in the
error message can also be asserted.
 * @param error The error to assert.
 * @param ErrorClass The optional error class to assert.
 * @param msgMatches The optional string or RegExp to assert in the error message.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertIsError<E>(error: unknown, ErrorClass: any, msgMatches: string | RegExp, msg?: string): asserts error is E;

/**
 * Make an assertion that `actual` is less than or equal to `expected`.
If not then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertLessOrEqual(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `actual` is less than `expected`.
If not then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertLess(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `actual` match RegExp `expected`. If not
then throw.
 * @param actual The actual value to be matched.
 * @param expected The expected pattern to match.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertMatch(actual: string, expected: RegExp, msg?: string): void;

/**
 * Make an assertion that `actual` and `expected` are not equal, deeply.
If not then throw.

Type parameter can be specified to ensure values under comparison have the same type.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertNotEquals<T>(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `obj` is not an instance of `type`.
If so, then throw.
 * @param actual The object to check.
 * @param unexpectedType The class constructor to check against.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertNotInstanceOf<A, T>(actual: A, unexpectedType: any, msg?: string): asserts actual is Exclude<A, T>;

/**
 * Make an assertion that `actual` not match RegExp `expected`. If match
then throw.
 * @param actual The actual value to match.
 * @param expected The expected value to not match.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertNotMatch(actual: string, expected: RegExp, msg?: string): void;

/**
 * Make an assertion that `actual` and `expected` are not strictly equal, using
{@linkcode Object.is} for equality comparison. If the values are strictly
equal then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertNotStrictEquals<T>(actual: T, expected: T, msg?: string): void;

/**
 * Make an assertion that `expected` object is a subset of `actual` object,
deeply. If not, then throw a diff of the objects, with mismatching
properties highlighted.
 * @param actual The actual value to be matched.
 * @param expected The expected value to match.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertObjectMatch<K extends keyof any,V>(actual: Record<K,V>, expected: Record<K,V>, msg?: string): void;

/**
 * Executes a function which returns a promise, expecting it to reject.

To assert that a synchronous function throws, use {@linkcode assertThrows}.
 * @param fn The function to execute.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertRejects<T>(fn: () =>void, msg?: string): Promise<T>;

/**
 * Executes a function which returns a promise, expecting it to reject.
If it does not, then it throws. An error class and a string that should be
included in the error message can also be asserted.

To assert that a synchronous function throws, use {@linkcode assertThrows}.
 * @param fn The function to execute.
 * @param ErrorClass The error class to assert.
 * @param msgIncludes The string that should be included in the error message.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertRejects<T>(fn: () => void, ErrorClass: any, msgIncludes: string, msg?: string): Promise<T>;


/**
 * Make an assertion that `actual` and `expected` are strictly equal, using
{@linkcode Object.is} for equality comparison. If not, then throw.
 * @param actual The actual value to compare.
 * @param expected The expected value to compare.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertStrictEquals(actual: unknown, expected: T, msg?: string): asserts actual is T;

/**
 * Make an assertion that actual includes expected. If not
then throw.
 * @param actual The actual string to check for inclusion.
 * @param expected The expected string to check for inclusion.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertStringIncludes(actual: string, expected: string, msg?: string): void;

/**
 * Executes a function, expecting it to throw. If it does not, then it
throws.

To assert that an asynchronous function rejects, use
{@linkcode assertRejects}.
 * @param fn The function to execute.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertThrows(fn: () => void, msg?: string): unknown;

/**
 * Executes a function, expecting it to throw. If it does not, then it
throws. An error class and a string that should be included in the
error message can also be asserted.

To assert that an asynchronous function rejects, use
{@linkcode assertRejects}.
 * @param fn The function to execute.
 * @param ErrorClass The error class to assert.
 * @param msgIncludes The string that should be included in the error message.
 * @param msg The optional message to display if the assertion fails.
 */
export function assertThrows<E>(fn: () => void, ErrorClass: any, msgIncludes: string, msg?: string): E;

/**
 * undefined
 */
export function assertThrows(fn: () => void, errorClassOrMsg:  any | string, msgIncludesOrMsg?: string, msg?: string):any;

/**
 * Make an assertion, error will be thrown if `expr` does not have truthy value.
 * @param expr The expression to test.
 * @param msg The optional message to display if the assertion fails.
 */
export function assert(expr: unknown, undefined: undefined): asserts expr;

/**
 * Deep equality comparison used in assertions.
 * @param a The actual value
 * @param b The expected value
 */
export function equal(a: unknown, b: unknown): boolean;

/**
 * Forcefully throws a failed assertion.
 * @param msg Optional message to include in the error.
 */
export function fail(msg?: string): never;

/**
 * Use this to stub out methods that will throw when invoked.
 * @param msg Optional message to include in the error.
 */
export function unimplemented(msg?: string): never;

/**
 * Use this to assert unreachable code.
 * @param msg Optional message to include in the error.
 */
export function unreachable(msg?: string): never;

}
declare module 'jamfile:color' {

  /**
   * Enable or disable text color when styling.

  `@std/fmt/colors` automatically detects NO_COLOR environmental variable
  and disables text color. Use this API only when the automatic detection
  doesn't work.
  * @param value The boolean value to enable or disable text color
  */
  export function setColorEnabled(value: boolean): void;

  /**
   * Get whether text color change is enabled or disabled.
   */
  export function getColorEnabled(): boolean;

  /**
   * Reset the text modified.
   * @param str The text to reset
   */
  export function reset(str: string): string;

  /**
   * Make the text bold.
   * @param str The text to make bold
   */
  export function bold(str: string): string;

  /**
   * The text emits only a small amount of light.
   * @param str The text to dim
   */
  export function dim(str: string): string;

  /**
   * Make the text italic.
   * @param str The text to make italic
   */
  export function italic(str: string): string;

  /**
   * Make the text underline.
   * @param str The text to underline
   */
  export function underline(str: string): string;

  /**
   * Invert background color and text color.
   * @param str The text to invert its color
   */
  export function inverse(str: string): string;

  /**
   * Make the text hidden.
   * @param str The text to hide
   */
  export function hidden(str: string): string;

  /**
   * Put horizontal line through the center of the text.
   * @param str The text to strike through
   */
  export function strikethrough(str: string): string;

  /**
   * Set text color to black.
   * @param str The text to make black
   */
  export function black(str: string): string;

  /**
   * Set text color to red.
   * @param str The text to make red
   */
  export function red(str: string): string;

  /**
   * Set text color to green.
   * @param str The text to make green
   */
  export function green(str: string): string;

  /**
   * Set text color to yellow.
   * @param str The text to make yellow
   */
  export function yellow(str: string): string;

  /**
   * Set text color to blue.
   * @param str The text to make blue
   */
  export function blue(str: string): string;

  /**
   * Set text color to magenta.
   * @param str The text to make magenta
   */
  export function magenta(str: string): string;

  /**
   * Set text color to cyan.
   * @param str The text to make cyan
   */
  export function cyan(str: string): string;

  /**
   * Set text color to white.
   * @param str The text to make white
   */
  export function white(str: string): string;

  /**
   * Set text color to gray.
   * @param str The text to make gray
   */
  export function gray(str: string): string;

  /**
   * Set text color to bright black.
   * @param str The text to make bright black
   */
  export function brightBlack(str: string): string;

  /**
   * Set text color to bright red.
   * @param str The text to make bright red
   */
  export function brightRed(str: string): string;

  /**
   * Set text color to bright green.
   * @param str The text to make bright green
   */
  export function brightGreen(str: string): string;

  /**
   * Set text color to bright yellow.
   * @param str The text to make bright yellow
   */
  export function brightYellow(str: string): string;

  /**
   * Set text color to bright blue.
   * @param str The text to make bright blue
   */
  export function brightBlue(str: string): string;

  /**
   * Set text color to bright magenta.
   * @param str The text to make bright magenta
   */
  export function brightMagenta(str: string): string;

  /**
   * Set text color to bright cyan.
   * @param str The text to make bright cyan
   */
  export function brightCyan(str: string): string;

  /**
   * Set text color to bright white.
   * @param str The text to make bright white
   */
  export function brightWhite(str: string): string;

  /**
   * Set background color to black.
   * @param str The text to make its background black
   */
  export function bgBlack(str: string): string;

  /**
   * Set background color to red.
   * @param str The text to make its background red
   */
  export function bgRed(str: string): string;

  /**
   * Set background color to green.
   * @param str The text to make its background green
   */
  export function bgGreen(str: string): string;

  /**
   * Set background color to yellow.
   * @param str The text to make its background yellow
   */
  export function bgYellow(str: string): string;

  /**
   * Set background color to blue.
   * @param str The text to make its background blue
   */
  export function bgBlue(str: string): string;

  /**
   * Set background color to magenta.
   * @param str The text to make its background magenta
   */
  export function bgMagenta(str: string): string;

  /**
   * Set background color to cyan.
   * @param str The text to make its background cyan
   */
  export function bgCyan(str: string): string;

  /**
   * Set background color to white.
   * @param str The text to make its background white
   */
  export function bgWhite(str: string): string;

  /**
   * Set background color to bright black.
   * @param str The text to make its background bright black
   */
  export function bgBrightBlack(str: string): string;

  /**
   * Set background color to bright red.
   * @param str The text to make its background bright red
   */
  export function bgBrightRed(str: string): string;

  /**
   * Set background color to bright green.
   * @param str The text to make its background bright green
   */
  export function bgBrightGreen(str: string): string;

  /**
   * Set background color to bright yellow.
   * @param str The text to make its background bright yellow
   */
  export function bgBrightYellow(str: string): string;

  /**
   * Set background color to bright blue.
   * @param str The text to make its background bright blue
   */
  export function bgBrightBlue(str: string): string;

  /**
   * Set background color to bright magenta.
   * @param str The text to make its background bright magenta
   */
  export function bgBrightMagenta(str: string): string;

  /**
   * Set background color to bright cyan.
   * @param str The text to make its background bright cyan
   */
  export function bgBrightCyan(str: string): string;

  /**
   * Set background color to bright white.
   * @param str The text to make its background bright white
   */
  export function bgBrightWhite(str: string): string;

  /**
   * Set text color using paletted 8bit colors.
  https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
  * @param str The text color to apply paletted 8bit colors to
  * @param color The color code
  */
  export function rgb8(str: string, color: number): string;

  /**
   * Set background color using paletted 8bit colors.
  https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
  * @param str The text color to apply paletted 8bit background colors to
  * @param color code
  */
  export function bgRgb8(str: string, color: number): string;

  /** RGB 8-bits per channel. Each in range `0->255` or `0x00->0xff` */
  export interface Rgb {
    /** Red component value */
    r: number;
    /** Green component value */
    g: number;
    /** Blue component value */
    b: number;
  }

  /**
   * Set text color using 24bit rgb.
  `color` can be a number in range `0x000000` to `0xffffff` or
  an `Rgb`.
  * @param str The text color to apply 24bit rgb to
  * @param color The color code
  */
  export function rgb24(str: string, color: number | Rgb): string;

  /**
   * Set background color using 24bit rgb.
  `color` can be a number in range `0x000000` to `0xffffff` or
  an `Rgb`.
  * @param str The text color to apply 24bit rgb to
  * @param color The color code
  */
  export function bgRgb24(str: string, color: number | Rgb): string;

  /**
   * Remove ANSI escape codes from the string.
   * @param string The text to remove ANSI escape codes from
   */
  export function stripAnsiCode(string: string): string;

}

declare module "jamfile:zod" {
  export function zodToJsonSchema(zod: ZodType):JSONSchema7;

  type Primitive = string | number | symbol | bigint | boolean | null | undefined;
type Scalars = Primitive | Primitive[];

namespace util {
    type AssertEqual<T, U> = (<V>() => V extends T ? 1 : 2) extends <V>() => V extends U ? 1 : 2 ? true : false;
    export type isAny<T> = 0 extends 1 & T ? true : false;
    export const assertEqual: <A, B>(val: AssertEqual<A, B>) => AssertEqual<A, B>;
    export function assertIs<T>(_arg: T): void;
    export function assertNever(_x: never): never;
    export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;
    export type OmitKeys<T, K extends string> = Pick<T, Exclude<keyof T, K>>;
    export type MakePartial<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
    export type Exactly<T, X> = T & Record<Exclude<keyof X, keyof T>, never>;
    export const arrayToEnum: <T extends string, U extends [T, ...T[]]>(items: U) => { [k in U[number]]: k; };
    export const getValidEnumValues: (obj: any) => any[];
    export const objectValues: (obj: any) => any[];
    export const objectKeys: ObjectConstructor["keys"];
    export const find: <T>(arr: T[], checker: (arg: T) => any) => T | undefined;
    export type identity<T> = objectUtil.identity<T>;
    export type flatten<T> = objectUtil.flatten<T>;
    export type noUndefined<T> = T extends undefined ? never : T;
    export const isInteger: NumberConstructor["isInteger"];
    export function joinValues<T extends any[]>(array: T, separator?: string): string;
    export const jsonStringifyReplacer: (_: string, value: any) => any;
    export {  };
}
namespace objectUtil {
    export type MergeShapes<U, V> = {
        [k in Exclude<keyof U, keyof V>]: U[k];
    } & V;
    type optionalKeys<T extends object> = {
        [k in keyof T]: undefined extends T[k] ? k : never;
    }[keyof T];
    type requiredKeys<T extends object> = {
        [k in keyof T]: undefined extends T[k] ? never : k;
    }[keyof T];
    export type addQuestionMarks<T extends object, _O = any> = {
        [K in requiredKeys<T>]: T[K];
    } & {
        [K in optionalKeys<T>]?: T[K];
    } & {
        [k in keyof T]?: unknown;
    };
    export type identity<T> = T;
    export type flatten<T> = identity<{
        [k in keyof T]: T[k];
    }>;
    export type noNeverKeys<T> = {
        [k in keyof T]: [T[k]] extends [never] ? never : k;
    }[keyof T];
    export type noNever<T> = identity<{
        [k in noNeverKeys<T>]: k extends keyof T ? T[k] : never;
    }>;
    export const mergeShapes: <U, T>(first: U, second: T) => T & U;
    export type extendShape<A extends object, B extends object> = {
        [K in keyof A as K extends keyof B ? never : K]: A[K];
    } & {
        [K in keyof B]: B[K];
    };
    export {  };
}
const ZodParsedType: {
    string: "string";
    number: "number";
    bigint: "bigint";
    boolean: "boolean";
    symbol: "symbol";
    undefined: "undefined";
    object: "object";
    function: "function";
    map: "map";
    nan: "nan";
    integer: "integer";
    float: "float";
    date: "date";
    null: "null";
    array: "array";
    unknown: "unknown";
    promise: "promise";
    void: "void";
    never: "never";
    set: "set";
};
type ZodParsedType = keyof typeof ZodParsedType;
const getParsedType: (data: any) => ZodParsedType;

type allKeys<T> = T extends any ? keyof T : never;
type inferFlattenedErrors<T extends ZodType<any, any, any>, U = string> = typeToFlattenedError<TypeOf<T>, U>;
type typeToFlattenedError<T, U = string> = {
    formErrors: U[];
    fieldErrors: {
        [P in allKeys<T>]?: U[];
    };
};
const ZodIssueCode: {
    invalid_type: "invalid_type";
    invalid_literal: "invalid_literal";
    custom: "custom";
    invalid_union: "invalid_union";
    invalid_union_discriminator: "invalid_union_discriminator";
    invalid_enum_value: "invalid_enum_value";
    unrecognized_keys: "unrecognized_keys";
    invalid_arguments: "invalid_arguments";
    invalid_return_type: "invalid_return_type";
    invalid_date: "invalid_date";
    invalid_string: "invalid_string";
    too_small: "too_small";
    too_big: "too_big";
    invalid_intersection_types: "invalid_intersection_types";
    not_multiple_of: "not_multiple_of";
    not_finite: "not_finite";
};
type ZodIssueCode = keyof typeof ZodIssueCode;
type ZodIssueBase = {
    path: (string | number)[];
    message?: string;
};
interface ZodInvalidTypeIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_type;
    expected: ZodParsedType;
    received: ZodParsedType;
}
interface ZodInvalidLiteralIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_literal;
    expected: unknown;
    received: unknown;
}
interface ZodUnrecognizedKeysIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.unrecognized_keys;
    keys: string[];
}
interface ZodInvalidUnionIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_union;
    unionErrors: ZodError[];
}
interface ZodInvalidUnionDiscriminatorIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_union_discriminator;
    options: Primitive[];
}
interface ZodInvalidEnumValueIssue extends ZodIssueBase {
    received: string | number;
    code: typeof ZodIssueCode.invalid_enum_value;
    options: (string | number)[];
}
interface ZodInvalidArgumentsIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_arguments;
    argumentsError: ZodError;
}
interface ZodInvalidReturnTypeIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_return_type;
    returnTypeError: ZodError;
}
interface ZodInvalidDateIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_date;
}
type StringValidation = "email" | "url" | "emoji" | "uuid" | "nanoid" | "regex" | "cuid" | "cuid2" | "ulid" | "datetime" | "date" | "time" | "duration" | "ip" | "cidr" | "base64" | "jwt" | "base64url" | {
    includes: string;
    position?: number;
} | {
    startsWith: string;
} | {
    endsWith: string;
};
interface ZodInvalidStringIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_string;
    validation: StringValidation;
}
interface ZodTooSmallIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.too_small;
    minimum: number | bigint;
    inclusive: boolean;
    exact?: boolean;
    type: "array" | "string" | "number" | "set" | "date" | "bigint";
}
interface ZodTooBigIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.too_big;
    maximum: number | bigint;
    inclusive: boolean;
    exact?: boolean;
    type: "array" | "string" | "number" | "set" | "date" | "bigint";
}
interface ZodInvalidIntersectionTypesIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.invalid_intersection_types;
}
interface ZodNotMultipleOfIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.not_multiple_of;
    multipleOf: number | bigint;
}
interface ZodNotFiniteIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.not_finite;
}
interface ZodCustomIssue extends ZodIssueBase {
    code: typeof ZodIssueCode.custom;
    params?: {
        [k: string]: any;
    };
}
type DenormalizedError = {
    [k: string]: DenormalizedError | string[];
};
type ZodIssueOptionalMessage = ZodInvalidTypeIssue | ZodInvalidLiteralIssue | ZodUnrecognizedKeysIssue | ZodInvalidUnionIssue | ZodInvalidUnionDiscriminatorIssue | ZodInvalidEnumValueIssue | ZodInvalidArgumentsIssue | ZodInvalidReturnTypeIssue | ZodInvalidDateIssue | ZodInvalidStringIssue | ZodTooSmallIssue | ZodTooBigIssue | ZodInvalidIntersectionTypesIssue | ZodNotMultipleOfIssue | ZodNotFiniteIssue | ZodCustomIssue;
type ZodIssue = ZodIssueOptionalMessage & {
    fatal?: boolean;
    message: string;
};
const quotelessJson: (obj: any) => string;
type recursiveZodFormattedError<T> = T extends [any, ...any[]] ? {
    [K in keyof T]?: ZodFormattedError<T[K]>;
} : T extends any[] ? {
    [k: number]: ZodFormattedError<T[number]>;
} : T extends object ? {
    [K in keyof T]?: ZodFormattedError<T[K]>;
} : unknown;
type ZodFormattedError<T, U = string> = {
    _errors: U[];
} & recursiveZodFormattedError<NonNullable<T>>;
type inferFormattedError<T extends ZodType<any, any, any>, U = string> = ZodFormattedError<TypeOf<T>, U>;
class ZodError<T = any> extends Error {
    issues: ZodIssue[];
    get errors(): ZodIssue[];
    constructor(issues: ZodIssue[]);
    format(): ZodFormattedError<T>;
    format<U>(mapper: (issue: ZodIssue) => U): ZodFormattedError<T, U>;
    static create: (issues: ZodIssue[]) => ZodError<any>;
    static assert(value: unknown): asserts value is ZodError;
    toString(): string;
    get message(): string;
    get isEmpty(): boolean;
    addIssue: (sub: ZodIssue) => void;
    addIssues: (subs?: ZodIssue[]) => void;
    flatten(): typeToFlattenedError<T>;
    flatten<U>(mapper?: (issue: ZodIssue) => U): typeToFlattenedError<T, U>;
    get formErrors(): typeToFlattenedError<T, string>;
}
type stripPath<T extends object> = T extends any ? util.OmitKeys<T, "path"> : never;
type IssueData = stripPath<ZodIssueOptionalMessage> & {
    path?: (string | number)[];
    fatal?: boolean;
};
type ErrorMapCtx = {
    defaultError: string;
    data: any;
};
type ZodErrorMap = (issue: ZodIssueOptionalMessage, _ctx: ErrorMapCtx) => {
    message: string;
};

const errorMap: ZodErrorMap;
//# sourceMappingURL=en.d.ts.map

function setErrorMap(map: ZodErrorMap): void;
function getErrorMap(): ZodErrorMap;

const makeIssue: (params: {
    data: any;
    path: (string | number)[];
    errorMaps: ZodErrorMap[];
    issueData: IssueData;
}) => ZodIssue;
type ParseParams = {
    path: (string | number)[];
    errorMap: ZodErrorMap;
    async: boolean;
};
type ParsePathComponent = string | number;
type ParsePath = ParsePathComponent[];
const EMPTY_PATH: ParsePath;
interface ParseContext {
    readonly common: {
        readonly issues: ZodIssue[];
        readonly contextualErrorMap?: ZodErrorMap;
        readonly async: boolean;
    };
    readonly path: ParsePath;
    readonly schemaErrorMap?: ZodErrorMap;
    readonly parent: ParseContext | null;
    readonly data: any;
    readonly parsedType: ZodParsedType;
}
type ParseInput = {
    data: any;
    path: (string | number)[];
    parent: ParseContext;
};
function addIssueToContext(ctx: ParseContext, issueData: IssueData): void;
type ObjectPair = {
    key: SyncParseReturnType<any>;
    value: SyncParseReturnType<any>;
};
class ParseStatus {
    value: "aborted" | "dirty" | "valid";
    dirty(): void;
    abort(): void;
    static mergeArray(status: ParseStatus, results: SyncParseReturnType<any>[]): SyncParseReturnType;
    static mergeObjectAsync(status: ParseStatus, pairs: {
        key: ParseReturnType<any>;
        value: ParseReturnType<any>;
    }[]): Promise<SyncParseReturnType<any>>;
    static mergeObjectSync(status: ParseStatus, pairs: {
        key: SyncParseReturnType<any>;
        value: SyncParseReturnType<any>;
        alwaysSet?: boolean;
    }[]): SyncParseReturnType;
}
interface ParseResult {
    status: "aborted" | "dirty" | "valid";
    data: any;
}
type INVALID = {
    status: "aborted";
};
const INVALID: INVALID;
type DIRTY<T> = {
    status: "dirty";
    value: T;
};
const DIRTY: <T>(value: T) => DIRTY<T>;
type OK<T> = {
    status: "valid";
    value: T;
};
const OK: <T>(value: T) => OK<T>;
type SyncParseReturnType<T = any> = OK<T> | DIRTY<T> | INVALID;
type AsyncParseReturnType<T> = Promise<SyncParseReturnType<T>>;
type ParseReturnType<T> = SyncParseReturnType<T> | AsyncParseReturnType<T>;
const isAborted: (x: ParseReturnType<any>) => x is INVALID;
const isDirty: <T>(x: ParseReturnType<T>) => x is OK<T> | DIRTY<T>;
const isValid: <T>(x: ParseReturnType<T>) => x is OK<T>;
const isAsync: <T>(x: ParseReturnType<T>) => x is AsyncParseReturnType<T>;

namespace enumUtil {
    type UnionToIntersectionFn<T> = (T extends unknown ? (k: () => T) => void : never) extends (k: infer Intersection) => void ? Intersection : never;
    type GetUnionLast<T> = UnionToIntersectionFn<T> extends () => infer Last ? Last : never;
    type UnionToTuple<T, Tuple extends unknown[] = []> = [T] extends [never] ? Tuple : UnionToTuple<Exclude<T, GetUnionLast<T>>, [GetUnionLast<T>, ...Tuple]>;
    type CastToStringTuple<T> = T extends [string, ...string[]] ? T : never;
    export type UnionToTupleString<T> = CastToStringTuple<UnionToTuple<T>>;
    export {  };
}

namespace errorUtil {
    type ErrMessage = string | {
        message?: string;
    };
    const errToObj: (message?: ErrMessage) => {
        message?: string | undefined;
    };
    const toString: (message?: ErrMessage) => string | undefined;
}

namespace partialUtil {
    type DeepPartial<T extends ZodTypeAny> = T extends ZodObject<ZodRawShape> ? ZodObject<{
        [k in keyof T["shape"]]: ZodOptional<DeepPartial<T["shape"][k]>>;
    }, T["_def"]["unknownKeys"], T["_def"]["catchall"]> : T extends ZodArray<infer Type, infer Card> ? ZodArray<DeepPartial<Type>, Card> : T extends ZodOptional<infer Type> ? ZodOptional<DeepPartial<Type>> : T extends ZodNullable<infer Type> ? ZodNullable<DeepPartial<Type>> : T extends ZodTuple<infer Items> ? {
        [k in keyof Items]: Items[k] extends ZodTypeAny ? DeepPartial<Items[k]> : never;
    } extends infer PI ? PI extends ZodTupleItems ? ZodTuple<PI> : never : never : T;
}

/**
 * The Standard Schema interface.
 */
type StandardSchemaV1<Input = unknown, Output = Input> = {
    /**
     * The Standard Schema properties.
     */
    readonly "~standard": StandardSchemaV1.Props<Input, Output>;
};
namespace StandardSchemaV1 {
    /**
     * The Standard Schema properties interface.
     */
    export interface Props<Input = unknown, Output = Input> {
        /**
         * The version number of the standard.
         */
        readonly version: 1;
        /**
         * The vendor name of the schema library.
         */
        readonly vendor: string;
        /**
         * Validates unknown input values.
         */
        readonly validate: (value: unknown) => Result<Output> | Promise<Result<Output>>;
        /**
         * Inferred types associated with the schema.
         */
        readonly types?: Types<Input, Output> | undefined;
    }
    /**
     * The result interface of the validate function.
     */
    export type Result<Output> = SuccessResult<Output> | FailureResult;
    /**
     * The result interface if validation succeeds.
     */
    export interface SuccessResult<Output> {
        /**
         * The typed output value.
         */
        readonly value: Output;
        /**
         * The non-existent issues.
         */
        readonly issues?: undefined;
    }
    /**
     * The result interface if validation fails.
     */
    export interface FailureResult {
        /**
         * The issues of failed validation.
         */
        readonly issues: ReadonlyArray<Issue>;
    }
    /**
     * The issue interface of the failure output.
     */
    export interface Issue {
        /**
         * The error message of the issue.
         */
        readonly message: string;
        /**
         * The path of the issue, if any.
         */
        readonly path?: ReadonlyArray<PropertyKey | PathSegment> | undefined;
    }
    /**
     * The path segment interface of the issue.
     */
    export interface PathSegment {
        /**
         * The key representing a path segment.
         */
        readonly key: PropertyKey;
    }
    /**
     * The Standard Schema types interface.
     */
    export interface Types<Input = unknown, Output = Input> {
        /**
         * The input type of the schema.
         */
        readonly input: Input;
        /**
         * The output type of the schema.
         */
        readonly output: Output;
    }
    /**
     * Infers the input type of a Standard Schema.
     */
    export type InferInput<Schema extends StandardSchemaV1> = NonNullable<Schema["~standard"]["types"]>["input"];
    /**
     * Infers the output type of a Standard Schema.
     */
    export type InferOutput<Schema extends StandardSchemaV1> = NonNullable<Schema["~standard"]["types"]>["output"];
    export {  };
}

interface RefinementCtx {
    addIssue: (arg: IssueData) => void;
    path: (string | number)[];
}
type ZodRawShape = {
    [k: string]: ZodTypeAny;
};
type ZodTypeAny = ZodType<any, any, any>;
type TypeOf<T extends ZodType<any, any, any>> = T["_output"];
type input<T extends ZodType<any, any, any>> = T["_input"];
type output<T extends ZodType<any, any, any>> = T["_output"];

type CustomErrorParams = Partial<util.Omit<ZodCustomIssue, "code">>;
interface ZodTypeDef {
    errorMap?: ZodErrorMap;
    description?: string;
}
type RawCreateParams = {
    errorMap?: ZodErrorMap;
    invalid_type_error?: string;
    required_error?: string;
    message?: string;
    description?: string;
} | undefined;
type ProcessedCreateParams = {
    errorMap?: ZodErrorMap;
    description?: string;
};
type SafeParseSuccess<Output> = {
    success: true;
    data: Output;
    error?: never;
};
type SafeParseError<Input> = {
    success: false;
    error: ZodError<Input>;
    data?: never;
};
type SafeParseReturnType<Input, Output> = SafeParseSuccess<Output> | SafeParseError<Input>;
abstract class ZodType<Output = any, Def extends ZodTypeDef = ZodTypeDef, Input = Output> {
    readonly _type: Output;
    readonly _output: Output;
    readonly _input: Input;
    readonly _def: Def;
    get description(): string | undefined;
    "~standard": StandardSchemaV1.Props<Input, Output>;
    abstract _parse(input: ParseInput): ParseReturnType<Output>;
    _getType(input: ParseInput): string;
    _getOrReturnCtx(input: ParseInput, ctx?: ParseContext | undefined): ParseContext;
    _processInputParams(input: ParseInput): {
        status: ParseStatus;
        ctx: ParseContext;
    };
    _parseSync(input: ParseInput): SyncParseReturnType<Output>;
    _parseAsync(input: ParseInput): AsyncParseReturnType<Output>;
    parse(data: unknown, params?: Partial<ParseParams>): Output;
    safeParse(data: unknown, params?: Partial<ParseParams>): SafeParseReturnType<Input, Output>;
    "~validate"(data: unknown): StandardSchemaV1.Result<Output> | Promise<StandardSchemaV1.Result<Output>>;
    parseAsync(data: unknown, params?: Partial<ParseParams>): Promise<Output>;
    safeParseAsync(data: unknown, params?: Partial<ParseParams>): Promise<SafeParseReturnType<Input, Output>>;
    /** Alias of safeParseAsync */
    spa: (data: unknown, params?: Partial<ParseParams>) => Promise<SafeParseReturnType<Input, Output>>;
    refine<RefinedOutput extends Output>(check: (arg: Output) => arg is RefinedOutput, message?: string | CustomErrorParams | ((arg: Output) => CustomErrorParams)): ZodEffects<this, RefinedOutput, Input>;
    refine(check: (arg: Output) => unknown | Promise<unknown>, message?: string | CustomErrorParams | ((arg: Output) => CustomErrorParams)): ZodEffects<this, Output, Input>;
    refinement<RefinedOutput extends Output>(check: (arg: Output) => arg is RefinedOutput, refinementData: IssueData | ((arg: Output, ctx: RefinementCtx) => IssueData)): ZodEffects<this, RefinedOutput, Input>;
    refinement(check: (arg: Output) => boolean, refinementData: IssueData | ((arg: Output, ctx: RefinementCtx) => IssueData)): ZodEffects<this, Output, Input>;
    _refinement(refinement: RefinementEffect<Output>["refinement"]): ZodEffects<this, Output, Input>;
    superRefine<RefinedOutput extends Output>(refinement: (arg: Output, ctx: RefinementCtx) => arg is RefinedOutput): ZodEffects<this, RefinedOutput, Input>;
    superRefine(refinement: (arg: Output, ctx: RefinementCtx) => void): ZodEffects<this, Output, Input>;
    superRefine(refinement: (arg: Output, ctx: RefinementCtx) => Promise<void>): ZodEffects<this, Output, Input>;
    constructor(def: Def);
    optional(): ZodOptional<this>;
    nullable(): ZodNullable<this>;
    nullish(): ZodOptional<ZodNullable<this>>;
    array(): ZodArray<this>;
    promise(): ZodPromise<this>;
    or<T extends ZodTypeAny>(option: T): ZodUnion<[this, T]>;
    and<T extends ZodTypeAny>(incoming: T): ZodIntersection<this, T>;
    transform<NewOut>(transform: (arg: Output, ctx: RefinementCtx) => NewOut | Promise<NewOut>): ZodEffects<this, NewOut>;
    default(def: util.noUndefined<Input>): ZodDefault<this>;
    default(def: () => util.noUndefined<Input>): ZodDefault<this>;
    brand<B extends string | number | symbol>(brand?: B): ZodBranded<this, B>;
    catch(def: Output): ZodCatch<this>;
    catch(def: (ctx: {
        error: ZodError;
        input: Input;
    }) => Output): ZodCatch<this>;
    describe(description: string): this;
    pipe<T extends ZodTypeAny>(target: T): ZodPipeline<this, T>;
    readonly(): ZodReadonly<this>;
    isOptional(): boolean;
    isNullable(): boolean;
}
type IpVersion = "v4" | "v6";
type ZodStringCheck = {
    kind: "min";
    value: number;
    message?: string;
} | {
    kind: "max";
    value: number;
    message?: string;
} | {
    kind: "length";
    value: number;
    message?: string;
} | {
    kind: "email";
    message?: string;
} | {
    kind: "url";
    message?: string;
} | {
    kind: "emoji";
    message?: string;
} | {
    kind: "uuid";
    message?: string;
} | {
    kind: "nanoid";
    message?: string;
} | {
    kind: "cuid";
    message?: string;
} | {
    kind: "includes";
    value: string;
    position?: number;
    message?: string;
} | {
    kind: "cuid2";
    message?: string;
} | {
    kind: "ulid";
    message?: string;
} | {
    kind: "startsWith";
    value: string;
    message?: string;
} | {
    kind: "endsWith";
    value: string;
    message?: string;
} | {
    kind: "regex";
    regex: RegExp;
    message?: string;
} | {
    kind: "trim";
    message?: string;
} | {
    kind: "toLowerCase";
    message?: string;
} | {
    kind: "toUpperCase";
    message?: string;
} | {
    kind: "jwt";
    alg?: string;
    message?: string;
} | {
    kind: "datetime";
    offset: boolean;
    local: boolean;
    precision: number | null;
    message?: string;
} | {
    kind: "date";
    message?: string;
} | {
    kind: "time";
    precision: number | null;
    message?: string;
} | {
    kind: "duration";
    message?: string;
} | {
    kind: "ip";
    version?: IpVersion;
    message?: string;
} | {
    kind: "cidr";
    version?: IpVersion;
    message?: string;
} | {
    kind: "base64";
    message?: string;
} | {
    kind: "base64url";
    message?: string;
};
interface ZodStringDef extends ZodTypeDef {
    checks: ZodStringCheck[];
    typeName: ZodFirstPartyTypeKind.ZodString;
    coerce: boolean;
}
function datetimeRegex(args: {
    precision?: number | null;
    offset?: boolean;
    local?: boolean;
}): RegExp;
class ZodString extends ZodType<string, ZodStringDef, string> {
    _parse(input: ParseInput): ParseReturnType<string>;
    protected _regex(regex: RegExp, validation: StringValidation, message?: errorUtil.ErrMessage): ZodEffects<this, string, string>;
    _addCheck(check: ZodStringCheck): ZodString;
    email(message?: errorUtil.ErrMessage): ZodString;
    url(message?: errorUtil.ErrMessage): ZodString;
    emoji(message?: errorUtil.ErrMessage): ZodString;
    uuid(message?: errorUtil.ErrMessage): ZodString;
    nanoid(message?: errorUtil.ErrMessage): ZodString;
    cuid(message?: errorUtil.ErrMessage): ZodString;
    cuid2(message?: errorUtil.ErrMessage): ZodString;
    ulid(message?: errorUtil.ErrMessage): ZodString;
    base64(message?: errorUtil.ErrMessage): ZodString;
    base64url(message?: errorUtil.ErrMessage): ZodString;
    jwt(options?: {
        alg?: string;
        message?: string;
    }): ZodString;
    ip(options?: string | {
        version?: IpVersion;
        message?: string;
    }): ZodString;
    cidr(options?: string | {
        version?: IpVersion;
        message?: string;
    }): ZodString;
    datetime(options?: string | {
        message?: string | undefined;
        precision?: number | null;
        offset?: boolean;
        local?: boolean;
    }): ZodString;
    date(message?: string): ZodString;
    time(options?: string | {
        message?: string | undefined;
        precision?: number | null;
    }): ZodString;
    duration(message?: errorUtil.ErrMessage): ZodString;
    regex(regex: RegExp, message?: errorUtil.ErrMessage): ZodString;
    includes(value: string, options?: {
        message?: string;
        position?: number;
    }): ZodString;
    startsWith(value: string, message?: errorUtil.ErrMessage): ZodString;
    endsWith(value: string, message?: errorUtil.ErrMessage): ZodString;
    min(minLength: number, message?: errorUtil.ErrMessage): ZodString;
    max(maxLength: number, message?: errorUtil.ErrMessage): ZodString;
    length(len: number, message?: errorUtil.ErrMessage): ZodString;
    /**
     * Equivalent to `.min(1)`
     */
    nonempty(message?: errorUtil.ErrMessage): ZodString;
    trim(): ZodString;
    toLowerCase(): ZodString;
    toUpperCase(): ZodString;
    get isDatetime(): boolean;
    get isDate(): boolean;
    get isTime(): boolean;
    get isDuration(): boolean;
    get isEmail(): boolean;
    get isURL(): boolean;
    get isEmoji(): boolean;
    get isUUID(): boolean;
    get isNANOID(): boolean;
    get isCUID(): boolean;
    get isCUID2(): boolean;
    get isULID(): boolean;
    get isIP(): boolean;
    get isCIDR(): boolean;
    get isBase64(): boolean;
    get isBase64url(): boolean;
    get minLength(): number | null;
    get maxLength(): number | null;
    static create: (params?: RawCreateParams & {
        coerce?: true;
    }) => ZodString;
}
type ZodNumberCheck = {
    kind: "min";
    value: number;
    inclusive: boolean;
    message?: string;
} | {
    kind: "max";
    value: number;
    inclusive: boolean;
    message?: string;
} | {
    kind: "int";
    message?: string;
} | {
    kind: "multipleOf";
    value: number;
    message?: string;
} | {
    kind: "finite";
    message?: string;
};
interface ZodNumberDef extends ZodTypeDef {
    checks: ZodNumberCheck[];
    typeName: ZodFirstPartyTypeKind.ZodNumber;
    coerce: boolean;
}
class ZodNumber extends ZodType<number, ZodNumberDef, number> {
    _parse(input: ParseInput): ParseReturnType<number>;
    static create: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodNumber;
    gte(value: number, message?: errorUtil.ErrMessage): ZodNumber;
    min: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
    gt(value: number, message?: errorUtil.ErrMessage): ZodNumber;
    lte(value: number, message?: errorUtil.ErrMessage): ZodNumber;
    max: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
    lt(value: number, message?: errorUtil.ErrMessage): ZodNumber;
    protected setLimit(kind: "min" | "max", value: number, inclusive: boolean, message?: string): ZodNumber;
    _addCheck(check: ZodNumberCheck): ZodNumber;
    int(message?: errorUtil.ErrMessage): ZodNumber;
    positive(message?: errorUtil.ErrMessage): ZodNumber;
    negative(message?: errorUtil.ErrMessage): ZodNumber;
    nonpositive(message?: errorUtil.ErrMessage): ZodNumber;
    nonnegative(message?: errorUtil.ErrMessage): ZodNumber;
    multipleOf(value: number, message?: errorUtil.ErrMessage): ZodNumber;
    step: (value: number, message?: errorUtil.ErrMessage) => ZodNumber;
    finite(message?: errorUtil.ErrMessage): ZodNumber;
    safe(message?: errorUtil.ErrMessage): ZodNumber;
    get minValue(): number | null;
    get maxValue(): number | null;
    get isInt(): boolean;
    get isFinite(): boolean;
}
type ZodBigIntCheck = {
    kind: "min";
    value: bigint;
    inclusive: boolean;
    message?: string;
} | {
    kind: "max";
    value: bigint;
    inclusive: boolean;
    message?: string;
} | {
    kind: "multipleOf";
    value: bigint;
    message?: string;
};
interface ZodBigIntDef extends ZodTypeDef {
    checks: ZodBigIntCheck[];
    typeName: ZodFirstPartyTypeKind.ZodBigInt;
    coerce: boolean;
}
class ZodBigInt extends ZodType<bigint, ZodBigIntDef, bigint> {
    _parse(input: ParseInput): ParseReturnType<bigint>;
    _getInvalidInput(input: ParseInput): INVALID;
    static create: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodBigInt;
    gte(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
    min: (value: bigint, message?: errorUtil.ErrMessage) => ZodBigInt;
    gt(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
    lte(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
    max: (value: bigint, message?: errorUtil.ErrMessage) => ZodBigInt;
    lt(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
    protected setLimit(kind: "min" | "max", value: bigint, inclusive: boolean, message?: string): ZodBigInt;
    _addCheck(check: ZodBigIntCheck): ZodBigInt;
    positive(message?: errorUtil.ErrMessage): ZodBigInt;
    negative(message?: errorUtil.ErrMessage): ZodBigInt;
    nonpositive(message?: errorUtil.ErrMessage): ZodBigInt;
    nonnegative(message?: errorUtil.ErrMessage): ZodBigInt;
    multipleOf(value: bigint, message?: errorUtil.ErrMessage): ZodBigInt;
    get minValue(): bigint | null;
    get maxValue(): bigint | null;
}
interface ZodBooleanDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodBoolean;
    coerce: boolean;
}
class ZodBoolean extends ZodType<boolean, ZodBooleanDef, boolean> {
    _parse(input: ParseInput): ParseReturnType<boolean>;
    static create: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodBoolean;
}
type ZodDateCheck = {
    kind: "min";
    value: number;
    message?: string;
} | {
    kind: "max";
    value: number;
    message?: string;
};
interface ZodDateDef extends ZodTypeDef {
    checks: ZodDateCheck[];
    coerce: boolean;
    typeName: ZodFirstPartyTypeKind.ZodDate;
}
class ZodDate extends ZodType<Date, ZodDateDef, Date> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    _addCheck(check: ZodDateCheck): ZodDate;
    min(minDate: Date, message?: errorUtil.ErrMessage): ZodDate;
    max(maxDate: Date, message?: errorUtil.ErrMessage): ZodDate;
    get minDate(): Date | null;
    get maxDate(): Date | null;
    static create: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodDate;
}
interface ZodSymbolDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodSymbol;
}
class ZodSymbol extends ZodType<symbol, ZodSymbolDef, symbol> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodSymbol;
}
interface ZodUndefinedDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodUndefined;
}
class ZodUndefined extends ZodType<undefined, ZodUndefinedDef, undefined> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    params?: RawCreateParams;
    static create: (params?: RawCreateParams) => ZodUndefined;
}
interface ZodNullDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodNull;
}
class ZodNull extends ZodType<null, ZodNullDef, null> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodNull;
}
interface ZodAnyDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodAny;
}
class ZodAny extends ZodType<any, ZodAnyDef, any> {
    _any: true;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodAny;
}
interface ZodUnknownDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodUnknown;
}
class ZodUnknown extends ZodType<unknown, ZodUnknownDef, unknown> {
    _unknown: true;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodUnknown;
}
interface ZodNeverDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodNever;
}
class ZodNever extends ZodType<never, ZodNeverDef, never> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodNever;
}
interface ZodVoidDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodVoid;
}
class ZodVoid extends ZodType<void, ZodVoidDef, void> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: (params?: RawCreateParams) => ZodVoid;
}
interface ZodArrayDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    type: T;
    typeName: ZodFirstPartyTypeKind.ZodArray;
    exactLength: {
        value: number;
        message?: string;
    } | null;
    minLength: {
        value: number;
        message?: string;
    } | null;
    maxLength: {
        value: number;
        message?: string;
    } | null;
}
type ArrayCardinality = "many" | "atleastone";
type arrayOutputType<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> = Cardinality extends "atleastone" ? [T["_output"], ...T["_output"][]] : T["_output"][];
class ZodArray<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> extends ZodType<arrayOutputType<T, Cardinality>, ZodArrayDef<T>, Cardinality extends "atleastone" ? [T["_input"], ...T["_input"][]] : T["_input"][]> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get element(): T;
    min(minLength: number, message?: errorUtil.ErrMessage): this;
    max(maxLength: number, message?: errorUtil.ErrMessage): this;
    length(len: number, message?: errorUtil.ErrMessage): this;
    nonempty(message?: errorUtil.ErrMessage): ZodArray<T, "atleastone">;
    static create: <T_1 extends ZodTypeAny>(schema: T_1, params?: RawCreateParams) => ZodArray<T_1, "many">;
}
type ZodNonEmptyArray<T extends ZodTypeAny> = ZodArray<T, "atleastone">;
type UnknownKeysParam = "passthrough" | "strict" | "strip";
interface ZodObjectDef<T extends ZodRawShape = ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodObject;
    shape: () => T;
    catchall: Catchall;
    unknownKeys: UnknownKeys;
}
type mergeTypes<A, B> = {
    [k in keyof A | keyof B]: k extends keyof B ? B[k] : k extends keyof A ? A[k] : never;
};
type objectOutputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectUtil.flatten<objectUtil.addQuestionMarks<baseObjectOutputType<Shape>>> & CatchallOutput<Catchall> & PassthroughType<UnknownKeys>;
type baseObjectOutputType<Shape extends ZodRawShape> = {
    [k in keyof Shape]: Shape[k]["_output"];
};
type objectInputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectUtil.flatten<baseObjectInputType<Shape>> & CatchallInput<Catchall> & PassthroughType<UnknownKeys>;
type baseObjectInputType<Shape extends ZodRawShape> = objectUtil.addQuestionMarks<{
    [k in keyof Shape]: Shape[k]["_input"];
}>;
type CatchallOutput<T extends ZodType> = ZodType extends T ? unknown : {
    [k: string]: T["_output"];
};
type CatchallInput<T extends ZodType> = ZodType extends T ? unknown : {
    [k: string]: T["_input"];
};
type PassthroughType<T extends UnknownKeysParam> = T extends "passthrough" ? {
    [k: string]: unknown;
} : unknown;
type deoptional<T extends ZodTypeAny> = T extends ZodOptional<infer U> ? deoptional<U> : T extends ZodNullable<infer U> ? ZodNullable<deoptional<U>> : T;
type SomeZodObject = ZodObject<ZodRawShape, UnknownKeysParam, ZodTypeAny>;
type noUnrecognized<Obj extends object, Shape extends object> = {
    [k in keyof Obj]: k extends keyof Shape ? Obj[k] : never;
};
class ZodObject<T extends ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny, Output = objectOutputType<T, Catchall, UnknownKeys>, Input = objectInputType<T, Catchall, UnknownKeys>> extends ZodType<Output, ZodObjectDef<T, UnknownKeys, Catchall>, Input> {
    private _cached;
    _getCached(): {
        shape: T;
        keys: string[];
    };
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get shape(): T;
    strict(message?: errorUtil.ErrMessage): ZodObject<T, "strict", Catchall>;
    strip(): ZodObject<T, "strip", Catchall>;
    passthrough(): ZodObject<T, "passthrough", Catchall>;
    /**
     * @deprecated In most cases, this is no longer needed - unknown properties are now silently stripped.
     * If you want to pass through unknown properties, use `.passthrough()` instead.
     */
    nonstrict: () => ZodObject<T, "passthrough", Catchall>;
    extend<Augmentation extends ZodRawShape>(augmentation: Augmentation): ZodObject<objectUtil.extendShape<T, Augmentation>, UnknownKeys, Catchall>;
    /**
     * @deprecated Use `.extend` instead
     *  */
    augment: <Augmentation extends ZodRawShape>(augmentation: Augmentation) => ZodObject<objectUtil.extendShape<T, Augmentation>, UnknownKeys, Catchall>;
    /**
     * Prior to zod@1.0.12 there was a bug in the
     * inferred type of merged objects. Please
     * upgrade if you are experiencing issues.
     */
    merge<Incoming extends AnyZodObject, Augmentation extends Incoming["shape"]>(merging: Incoming): ZodObject<objectUtil.extendShape<T, Augmentation>, Incoming["_def"]["unknownKeys"], Incoming["_def"]["catchall"]>;
    setKey<Key extends string, Schema extends ZodTypeAny>(key: Key, schema: Schema): ZodObject<T & {
        [k in Key]: Schema;
    }, UnknownKeys, Catchall>;
    catchall<Index extends ZodTypeAny>(index: Index): ZodObject<T, UnknownKeys, Index>;
    pick<Mask extends util.Exactly<{
        [k in keyof T]?: true;
    }, Mask>>(mask: Mask): ZodObject<Pick<T, Extract<keyof T, keyof Mask>>, UnknownKeys, Catchall>;
    omit<Mask extends util.Exactly<{
        [k in keyof T]?: true;
    }, Mask>>(mask: Mask): ZodObject<Omit<T, keyof Mask>, UnknownKeys, Catchall>;
    /**
     * @deprecated
     */
    deepPartial(): partialUtil.DeepPartial<this>;
    partial(): ZodObject<{
        [k in keyof T]: ZodOptional<T[k]>;
    }, UnknownKeys, Catchall>;
    partial<Mask extends util.Exactly<{
        [k in keyof T]?: true;
    }, Mask>>(mask: Mask): ZodObject<objectUtil.noNever<{
        [k in keyof T]: k extends keyof Mask ? ZodOptional<T[k]> : T[k];
    }>, UnknownKeys, Catchall>;
    required(): ZodObject<{
        [k in keyof T]: deoptional<T[k]>;
    }, UnknownKeys, Catchall>;
    required<Mask extends util.Exactly<{
        [k in keyof T]?: true;
    }, Mask>>(mask: Mask): ZodObject<objectUtil.noNever<{
        [k in keyof T]: k extends keyof Mask ? deoptional<T[k]> : T[k];
    }>, UnknownKeys, Catchall>;
    keyof(): ZodEnum<enumUtil.UnionToTupleString<keyof T>>;
    static create: <T_1 extends ZodRawShape>(shape: T_1, params?: RawCreateParams) => ZodObject<T_1, "strip", ZodTypeAny, objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any> extends infer T_2 ? { [k in keyof T_2]: objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any>[k]; } : never, baseObjectInputType<T_1> extends infer T_3 ? { [k_1 in keyof T_3]: baseObjectInputType<T_1>[k_1]; } : never>;
    static strictCreate: <T_1 extends ZodRawShape>(shape: T_1, params?: RawCreateParams) => ZodObject<T_1, "strict", ZodTypeAny, objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any> extends infer T_2 ? { [k in keyof T_2]: objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any>[k]; } : never, baseObjectInputType<T_1> extends infer T_3 ? { [k_1 in keyof T_3]: baseObjectInputType<T_1>[k_1]; } : never>;
    static lazycreate: <T_1 extends ZodRawShape>(shape: () => T_1, params?: RawCreateParams) => ZodObject<T_1, "strip", ZodTypeAny, objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any> extends infer T_2 ? { [k in keyof T_2]: objectUtil.addQuestionMarks<baseObjectOutputType<T_1>, any>[k]; } : never, baseObjectInputType<T_1> extends infer T_3 ? { [k_1 in keyof T_3]: baseObjectInputType<T_1>[k_1]; } : never>;
}
type AnyZodObject = ZodObject<any, any, any>;
type ZodUnionOptions = Readonly<[ZodTypeAny, ...ZodTypeAny[]]>;
interface ZodUnionDef<T extends ZodUnionOptions = Readonly<[
    ZodTypeAny,
    ZodTypeAny,
    ...ZodTypeAny[]
]>> extends ZodTypeDef {
    options: T;
    typeName: ZodFirstPartyTypeKind.ZodUnion;
}
class ZodUnion<T extends ZodUnionOptions> extends ZodType<T[number]["_output"], ZodUnionDef<T>, T[number]["_input"]> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get options(): T;
    static create: <T_1 extends readonly [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]>(types: T_1, params?: RawCreateParams) => ZodUnion<T_1>;
}
type ZodDiscriminatedUnionOption<Discriminator extends string> = ZodObject<{
    [key in Discriminator]: ZodTypeAny;
} & ZodRawShape, UnknownKeysParam, ZodTypeAny>;
interface ZodDiscriminatedUnionDef<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<string>[] = ZodDiscriminatedUnionOption<string>[]> extends ZodTypeDef {
    discriminator: Discriminator;
    options: Options;
    optionsMap: Map<Primitive, ZodDiscriminatedUnionOption<any>>;
    typeName: ZodFirstPartyTypeKind.ZodDiscriminatedUnion;
}
class ZodDiscriminatedUnion<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<Discriminator>[]> extends ZodType<output<Options[number]>, ZodDiscriminatedUnionDef<Discriminator, Options>, input<Options[number]>> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get discriminator(): Discriminator;
    get options(): Options;
    get optionsMap(): Map<Primitive, ZodDiscriminatedUnionOption<any>>;
    /**
     * The constructor of the discriminated union schema. Its behaviour is very similar to that of the normal z.union() constructor.
     * However, it only allows a union of objects, all of which need to share a discriminator property. This property must
     * have a different value for each object in the union.
     * @param discriminator the name of the discriminator property
     * @param types an array of object schemas
     * @param params
     */
    static create<Discriminator extends string, Types extends readonly [
        ZodDiscriminatedUnionOption<Discriminator>,
        ...ZodDiscriminatedUnionOption<Discriminator>[]
    ]>(discriminator: Discriminator, options: Types, params?: RawCreateParams): ZodDiscriminatedUnion<Discriminator, Types>;
}
interface ZodIntersectionDef<T extends ZodTypeAny = ZodTypeAny, U extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    left: T;
    right: U;
    typeName: ZodFirstPartyTypeKind.ZodIntersection;
}
class ZodIntersection<T extends ZodTypeAny, U extends ZodTypeAny> extends ZodType<T["_output"] & U["_output"], ZodIntersectionDef<T, U>, T["_input"] & U["_input"]> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <T_1 extends ZodTypeAny, U_1 extends ZodTypeAny>(left: T_1, right: U_1, params?: RawCreateParams) => ZodIntersection<T_1, U_1>;
}
type ZodTupleItems = [ZodTypeAny, ...ZodTypeAny[]];
type AssertArray<T> = T extends any[] ? T : never;
type OutputTypeOfTuple<T extends ZodTupleItems | []> = AssertArray<{
    [k in keyof T]: T[k] extends ZodType<any, any, any> ? T[k]["_output"] : never;
}>;
type OutputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = Rest extends ZodTypeAny ? [...OutputTypeOfTuple<T>, ...Rest["_output"][]] : OutputTypeOfTuple<T>;
type InputTypeOfTuple<T extends ZodTupleItems | []> = AssertArray<{
    [k in keyof T]: T[k] extends ZodType<any, any, any> ? T[k]["_input"] : never;
}>;
type InputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = Rest extends ZodTypeAny ? [...InputTypeOfTuple<T>, ...Rest["_input"][]] : InputTypeOfTuple<T>;
interface ZodTupleDef<T extends ZodTupleItems | [] = ZodTupleItems, Rest extends ZodTypeAny | null = null> extends ZodTypeDef {
    items: T;
    rest: Rest;
    typeName: ZodFirstPartyTypeKind.ZodTuple;
}
type AnyZodTuple = ZodTuple<[
    ZodTypeAny,
    ...ZodTypeAny[]
] | [], ZodTypeAny | null>;
class ZodTuple<T extends [ZodTypeAny, ...ZodTypeAny[]] | [] = [ZodTypeAny, ...ZodTypeAny[]], Rest extends ZodTypeAny | null = null> extends ZodType<OutputTypeOfTupleWithRest<T, Rest>, ZodTupleDef<T, Rest>, InputTypeOfTupleWithRest<T, Rest>> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get items(): T;
    rest<Rest extends ZodTypeAny>(rest: Rest): ZodTuple<T, Rest>;
    static create: <T_1 extends [] | [ZodTypeAny, ...ZodTypeAny[]]>(schemas: T_1, params?: RawCreateParams) => ZodTuple<T_1, null>;
}
interface ZodRecordDef<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    valueType: Value;
    keyType: Key;
    typeName: ZodFirstPartyTypeKind.ZodRecord;
}
type KeySchema = ZodType<string | number | symbol, any, any>;
type RecordType<K extends string | number | symbol, V> = [
    string
] extends [K] ? Record<K, V> : [number] extends [K] ? Record<K, V> : [symbol] extends [K] ? Record<K, V> : [BRAND<string | number | symbol>] extends [K] ? Record<K, V> : Partial<Record<K, V>>;
class ZodRecord<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> extends ZodType<RecordType<Key["_output"], Value["_output"]>, ZodRecordDef<Key, Value>, RecordType<Key["_input"], Value["_input"]>> {
    get keySchema(): Key;
    get valueSchema(): Value;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get element(): Value;
    static create<Value extends ZodTypeAny>(valueType: Value, params?: RawCreateParams): ZodRecord<ZodString, Value>;
    static create<Keys extends KeySchema, Value extends ZodTypeAny>(keySchema: Keys, valueType: Value, params?: RawCreateParams): ZodRecord<Keys, Value>;
}
interface ZodMapDef<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    valueType: Value;
    keyType: Key;
    typeName: ZodFirstPartyTypeKind.ZodMap;
}
class ZodMap<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> extends ZodType<Map<Key["_output"], Value["_output"]>, ZodMapDef<Key, Value>, Map<Key["_input"], Value["_input"]>> {
    get keySchema(): Key;
    get valueSchema(): Value;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <Key_1 extends ZodTypeAny = ZodTypeAny, Value_1 extends ZodTypeAny = ZodTypeAny>(keyType: Key_1, valueType: Value_1, params?: RawCreateParams) => ZodMap<Key_1, Value_1>;
}
interface ZodSetDef<Value extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    valueType: Value;
    typeName: ZodFirstPartyTypeKind.ZodSet;
    minSize: {
        value: number;
        message?: string;
    } | null;
    maxSize: {
        value: number;
        message?: string;
    } | null;
}
class ZodSet<Value extends ZodTypeAny = ZodTypeAny> extends ZodType<Set<Value["_output"]>, ZodSetDef<Value>, Set<Value["_input"]>> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    min(minSize: number, message?: errorUtil.ErrMessage): this;
    max(maxSize: number, message?: errorUtil.ErrMessage): this;
    size(size: number, message?: errorUtil.ErrMessage): this;
    nonempty(message?: errorUtil.ErrMessage): ZodSet<Value>;
    static create: <Value_1 extends ZodTypeAny = ZodTypeAny>(valueType: Value_1, params?: RawCreateParams) => ZodSet<Value_1>;
}
interface ZodFunctionDef<Args extends ZodTuple<any, any> = ZodTuple<any, any>, Returns extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    args: Args;
    returns: Returns;
    typeName: ZodFirstPartyTypeKind.ZodFunction;
}
type OuterTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = Args["_input"] extends Array<any> ? (...args: Args["_input"]) => Returns["_output"] : never;
type InnerTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = Args["_output"] extends Array<any> ? (...args: Args["_output"]) => Returns["_input"] : never;
class ZodFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> extends ZodType<OuterTypeOfFunction<Args, Returns>, ZodFunctionDef<Args, Returns>, InnerTypeOfFunction<Args, Returns>> {
    _parse(input: ParseInput): ParseReturnType<any>;
    parameters(): Args;
    returnType(): Returns;
    args<Items extends Parameters<(typeof ZodTuple)["create"]>[0]>(...items: Items): ZodFunction<ZodTuple<Items, ZodUnknown>, Returns>;
    returns<NewReturnType extends ZodType<any, any, any>>(returnType: NewReturnType): ZodFunction<Args, NewReturnType>;
    implement<F extends InnerTypeOfFunction<Args, Returns>>(func: F): ReturnType<F> extends Returns["_output"] ? (...args: Args["_input"]) => ReturnType<F> : OuterTypeOfFunction<Args, Returns>;
    strictImplement(func: InnerTypeOfFunction<Args, Returns>): InnerTypeOfFunction<Args, Returns>;
    validate: <F extends InnerTypeOfFunction<Args, Returns>>(func: F) => ReturnType<F> extends Returns["_output"] ? (...args: Args["_input"]) => ReturnType<F> : OuterTypeOfFunction<Args, Returns>;
    static create(): ZodFunction<ZodTuple<[], ZodUnknown>, ZodUnknown>;
    static create<T extends AnyZodTuple = ZodTuple<[], ZodUnknown>>(args: T): ZodFunction<T, ZodUnknown>;
    static create<T extends AnyZodTuple, U extends ZodTypeAny>(args: T, returns: U): ZodFunction<T, U>;
    static create<T extends AnyZodTuple = ZodTuple<[], ZodUnknown>, U extends ZodTypeAny = ZodUnknown>(args: T, returns: U, params?: RawCreateParams): ZodFunction<T, U>;
}
interface ZodLazyDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    getter: () => T;
    typeName: ZodFirstPartyTypeKind.ZodLazy;
}
class ZodLazy<T extends ZodTypeAny> extends ZodType<output<T>, ZodLazyDef<T>, input<T>> {
    get schema(): T;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <T_1 extends ZodTypeAny>(getter: () => T_1, params?: RawCreateParams) => ZodLazy<T_1>;
}
interface ZodLiteralDef<T = any> extends ZodTypeDef {
    value: T;
    typeName: ZodFirstPartyTypeKind.ZodLiteral;
}
class ZodLiteral<T> extends ZodType<T, ZodLiteralDef<T>, T> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get value(): T;
    static create: <T_1 extends Primitive>(value: T_1, params?: RawCreateParams) => ZodLiteral<T_1>;
}
type ArrayKeys = keyof any[];
type Indices<T> = Exclude<keyof T, ArrayKeys>;
type EnumValues<T extends string = string> = readonly [T, ...T[]];
type Values<T extends EnumValues> = {
    [k in T[number]]: k;
};
interface ZodEnumDef<T extends EnumValues = EnumValues> extends ZodTypeDef {
    values: T;
    typeName: ZodFirstPartyTypeKind.ZodEnum;
}
type Writeable<T> = {
    -readonly [P in keyof T]: T[P];
};
type FilterEnum<Values, ToExclude> = Values extends [] ? [] : Values extends [infer Head, ...infer Rest] ? Head extends ToExclude ? FilterEnum<Rest, ToExclude> : [Head, ...FilterEnum<Rest, ToExclude>] : never;
type typecast<A, T> = A extends T ? A : never;
function createZodEnum<U extends string, T extends Readonly<[U, ...U[]]>>(values: T, params?: RawCreateParams): ZodEnum<Writeable<T>>;
function createZodEnum<U extends string, T extends [U, ...U[]]>(values: T, params?: RawCreateParams): ZodEnum<T>;
class ZodEnum<T extends [string, ...string[]]> extends ZodType<T[number], ZodEnumDef<T>, T[number]> {
    #private;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    get options(): T;
    get enum(): Values<T>;
    get Values(): Values<T>;
    get Enum(): Values<T>;
    extract<ToExtract extends readonly [T[number], ...T[number][]]>(values: ToExtract, newDef?: RawCreateParams): ZodEnum<Writeable<ToExtract>>;
    exclude<ToExclude extends readonly [T[number], ...T[number][]]>(values: ToExclude, newDef?: RawCreateParams): ZodEnum<typecast<Writeable<FilterEnum<T, ToExclude[number]>>, [string, ...string[]]>>;
    static create: typeof createZodEnum;
}
interface ZodNativeEnumDef<T extends EnumLike = EnumLike> extends ZodTypeDef {
    values: T;
    typeName: ZodFirstPartyTypeKind.ZodNativeEnum;
}
type EnumLike = {
    [k: string]: string | number;
    [nu: number]: string;
};
class ZodNativeEnum<T extends EnumLike> extends ZodType<T[keyof T], ZodNativeEnumDef<T>, T[keyof T]> {
    #private;
    _parse(input: ParseInput): ParseReturnType<T[keyof T]>;
    get enum(): T;
    static create: <T_1 extends EnumLike>(values: T_1, params?: RawCreateParams) => ZodNativeEnum<T_1>;
}
interface ZodPromiseDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    type: T;
    typeName: ZodFirstPartyTypeKind.ZodPromise;
}
class ZodPromise<T extends ZodTypeAny> extends ZodType<Promise<T["_output"]>, ZodPromiseDef<T>, Promise<T["_input"]>> {
    unwrap(): T;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <T_1 extends ZodTypeAny>(schema: T_1, params?: RawCreateParams) => ZodPromise<T_1>;
}
type Refinement<T> = (arg: T, ctx: RefinementCtx) => any;
type SuperRefinement<T> = (arg: T, ctx: RefinementCtx) => void | Promise<void>;
type RefinementEffect<T> = {
    type: "refinement";
    refinement: (arg: T, ctx: RefinementCtx) => any;
};
type TransformEffect<T> = {
    type: "transform";
    transform: (arg: T, ctx: RefinementCtx) => any;
};
type PreprocessEffect<T> = {
    type: "preprocess";
    transform: (arg: T, ctx: RefinementCtx) => any;
};
type Effect<T> = RefinementEffect<T> | TransformEffect<T> | PreprocessEffect<T>;
interface ZodEffectsDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    schema: T;
    typeName: ZodFirstPartyTypeKind.ZodEffects;
    effect: Effect<any>;
}
class ZodEffects<T extends ZodTypeAny, Output = output<T>, Input = input<T>> extends ZodType<Output, ZodEffectsDef<T>, Input> {
    innerType(): T;
    sourceType(): T;
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <I extends ZodTypeAny>(schema: I, effect: Effect<I["_output"]>, params?: RawCreateParams) => ZodEffects<I, I["_output"]>;
    static createWithPreprocess: <I extends ZodTypeAny>(preprocess: (arg: unknown, ctx: RefinementCtx) => unknown, schema: I, params?: RawCreateParams) => ZodEffects<I, I["_output"], unknown>;
}

interface ZodOptionalDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    innerType: T;
    typeName: ZodFirstPartyTypeKind.ZodOptional;
}
type ZodOptionalType<T extends ZodTypeAny> = ZodOptional<T>;
class ZodOptional<T extends ZodTypeAny> extends ZodType<T["_output"] | undefined, ZodOptionalDef<T>, T["_input"] | undefined> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    unwrap(): T;
    static create: <T_1 extends ZodTypeAny>(type: T_1, params?: RawCreateParams) => ZodOptional<T_1>;
}
interface ZodNullableDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    innerType: T;
    typeName: ZodFirstPartyTypeKind.ZodNullable;
}
type ZodNullableType<T extends ZodTypeAny> = ZodNullable<T>;
class ZodNullable<T extends ZodTypeAny> extends ZodType<T["_output"] | null, ZodNullableDef<T>, T["_input"] | null> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    unwrap(): T;
    static create: <T_1 extends ZodTypeAny>(type: T_1, params?: RawCreateParams) => ZodNullable<T_1>;
}
interface ZodDefaultDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    innerType: T;
    defaultValue: () => util.noUndefined<T["_input"]>;
    typeName: ZodFirstPartyTypeKind.ZodDefault;
}
class ZodDefault<T extends ZodTypeAny> extends ZodType<util.noUndefined<T["_output"]>, ZodDefaultDef<T>, T["_input"] | undefined> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    removeDefault(): T;
    static create: <T_1 extends ZodTypeAny>(type: T_1, params: {
        errorMap?: ZodErrorMap | undefined;
        invalid_type_error?: string | undefined;
        required_error?: string | undefined;
        message?: string | undefined;
        description?: string | undefined;
    } & {
        default: T_1["_input"] | (() => util.noUndefined<T_1["_input"]>);
    }) => ZodDefault<T_1>;
}
interface ZodCatchDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    innerType: T;
    catchValue: (ctx: {
        error: ZodError;
        input: unknown;
    }) => T["_input"];
    typeName: ZodFirstPartyTypeKind.ZodCatch;
}
class ZodCatch<T extends ZodTypeAny> extends ZodType<T["_output"], ZodCatchDef<T>, unknown> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    removeCatch(): T;
    static create: <T_1 extends ZodTypeAny>(type: T_1, params: {
        errorMap?: ZodErrorMap | undefined;
        invalid_type_error?: string | undefined;
        required_error?: string | undefined;
        message?: string | undefined;
        description?: string | undefined;
    } & {
        catch: T_1["_output"] | (() => T_1["_output"]);
    }) => ZodCatch<T_1>;
}
interface ZodNaNDef extends ZodTypeDef {
    typeName: ZodFirstPartyTypeKind.ZodNaN;
}
class ZodNaN extends ZodType<number, ZodNaNDef, number> {
    _parse(input: ParseInput): ParseReturnType<any>;
    static create: (params?: RawCreateParams) => ZodNaN;
}
interface ZodBrandedDef<T extends ZodTypeAny> extends ZodTypeDef {
    type: T;
    typeName: ZodFirstPartyTypeKind.ZodBranded;
}
const BRAND: unique symbol;
type BRAND<T extends string | number | symbol> = {
    [BRAND]: {
        [k in T]: true;
    };
};
class ZodBranded<T extends ZodTypeAny, B extends string | number | symbol> extends ZodType<T["_output"] & BRAND<B>, ZodBrandedDef<T>, T["_input"]> {
    _parse(input: ParseInput): ParseReturnType<any>;
    unwrap(): T;
}
interface ZodPipelineDef<A extends ZodTypeAny, B extends ZodTypeAny> extends ZodTypeDef {
    in: A;
    out: B;
    typeName: ZodFirstPartyTypeKind.ZodPipeline;
}
class ZodPipeline<A extends ZodTypeAny, B extends ZodTypeAny> extends ZodType<B["_output"], ZodPipelineDef<A, B>, A["_input"]> {
    _parse(input: ParseInput): ParseReturnType<any>;
    static create<A extends ZodTypeAny, B extends ZodTypeAny>(a: A, b: B): ZodPipeline<A, B>;
}
type BuiltIn = (((...args: any[]) => any) | (new (...args: any[]) => any)) | {
    readonly [Symbol.toStringTag]: string;
} | Date | Error | Generator | Promise<unknown> | RegExp;
type MakeReadonly<T> = T extends Map<infer K, infer V> ? ReadonlyMap<K, V> : T extends Set<infer V> ? ReadonlySet<V> : T extends [infer Head, ...infer Tail] ? readonly [Head, ...Tail] : T extends Array<infer V> ? ReadonlyArray<V> : T extends BuiltIn ? T : Readonly<T>;
interface ZodReadonlyDef<T extends ZodTypeAny = ZodTypeAny> extends ZodTypeDef {
    innerType: T;
    typeName: ZodFirstPartyTypeKind.ZodReadonly;
}
class ZodReadonly<T extends ZodTypeAny> extends ZodType<MakeReadonly<T["_output"]>, ZodReadonlyDef<T>, MakeReadonly<T["_input"]>> {
    _parse(input: ParseInput): ParseReturnType<this["_output"]>;
    static create: <T_1 extends ZodTypeAny>(type: T_1, params?: RawCreateParams) => ZodReadonly<T_1>;
    unwrap(): T;
}
type CustomParams = CustomErrorParams & {
    fatal?: boolean;
};
function custom<T>(check?: (data: any) => any, params?: string | CustomParams | ((input: any) => CustomParams),
/**
 * @deprecated
 *
 * Pass `fatal` into the params object instead:
 *
 * ```ts
 * z.string().custom((val) => val.length > 5, { fatal: false })
 * ```
 *
 */
fatal?: boolean): ZodType<T, ZodTypeDef, T>;

const late: {
    object: <T extends ZodRawShape>(shape: () => T, params?: RawCreateParams) => ZodObject<T, "strip">;
};
enum ZodFirstPartyTypeKind {
    ZodString = "ZodString",
    ZodNumber = "ZodNumber",
    ZodNaN = "ZodNaN",
    ZodBigInt = "ZodBigInt",
    ZodBoolean = "ZodBoolean",
    ZodDate = "ZodDate",
    ZodSymbol = "ZodSymbol",
    ZodUndefined = "ZodUndefined",
    ZodNull = "ZodNull",
    ZodAny = "ZodAny",
    ZodUnknown = "ZodUnknown",
    ZodNever = "ZodNever",
    ZodVoid = "ZodVoid",
    ZodArray = "ZodArray",
    ZodObject = "ZodObject",
    ZodUnion = "ZodUnion",
    ZodDiscriminatedUnion = "ZodDiscriminatedUnion",
    ZodIntersection = "ZodIntersection",
    ZodTuple = "ZodTuple",
    ZodRecord = "ZodRecord",
    ZodMap = "ZodMap",
    ZodSet = "ZodSet",
    ZodFunction = "ZodFunction",
    ZodLazy = "ZodLazy",
    ZodLiteral = "ZodLiteral",
    ZodEnum = "ZodEnum",
    ZodEffects = "ZodEffects",
    ZodNativeEnum = "ZodNativeEnum",
    ZodOptional = "ZodOptional",
    ZodNullable = "ZodNullable",
    ZodDefault = "ZodDefault",
    ZodCatch = "ZodCatch",
    ZodPromise = "ZodPromise",
    ZodBranded = "ZodBranded",
    ZodPipeline = "ZodPipeline",
    ZodReadonly = "ZodReadonly"
}
type ZodFirstPartySchemaTypes = ZodString | ZodNumber | ZodNaN | ZodBigInt | ZodBoolean | ZodDate | ZodUndefined | ZodNull | ZodAny | ZodUnknown | ZodNever | ZodVoid | ZodArray<any, any> | ZodObject<any, any, any> | ZodUnion<any> | ZodDiscriminatedUnion<any, any> | ZodIntersection<any, any> | ZodTuple<any, any> | ZodRecord<any, any> | ZodMap<any> | ZodSet<any> | ZodFunction<any, any> | ZodLazy<any> | ZodLiteral<any> | ZodEnum<any> | ZodEffects<any, any, any> | ZodNativeEnum<any> | ZodOptional<any> | ZodNullable<any> | ZodDefault<any> | ZodCatch<any> | ZodPromise<any> | ZodBranded<any, any> | ZodPipeline<any, any> | ZodReadonly<any> | ZodSymbol;
abstract class Class {
    constructor(..._: any[]);
}
const instanceOfType: <T extends typeof Class>(cls: T, params?: CustomParams) => ZodType<InstanceType<T>, ZodTypeDef, InstanceType<T>>;
const stringType: (params?: RawCreateParams & {
    coerce?: true;
}) => ZodString;
const numberType: (params?: RawCreateParams & {
    coerce?: boolean;
}) => ZodNumber;
const nanType: (params?: RawCreateParams) => ZodNaN;
const bigIntType: (params?: RawCreateParams & {
    coerce?: boolean;
}) => ZodBigInt;
const booleanType: (params?: RawCreateParams & {
    coerce?: boolean;
}) => ZodBoolean;
const dateType: (params?: RawCreateParams & {
    coerce?: boolean;
}) => ZodDate;
const symbolType: (params?: RawCreateParams) => ZodSymbol;
const undefinedType: (params?: RawCreateParams) => ZodUndefined;
const nullType: (params?: RawCreateParams) => ZodNull;
const anyType: (params?: RawCreateParams) => ZodAny;
const unknownType: (params?: RawCreateParams) => ZodUnknown;
const neverType: (params?: RawCreateParams) => ZodNever;
const voidType: (params?: RawCreateParams) => ZodVoid;
const arrayType: <T extends ZodTypeAny>(schema: T, params?: RawCreateParams) => ZodArray<T>;
const objectType: <T extends ZodRawShape>(shape: T, params?: RawCreateParams) => ZodObject<T, "strip", ZodTypeAny, objectOutputType<T, ZodTypeAny, "strip">, objectInputType<T, ZodTypeAny, "strip">>;
const strictObjectType: <T extends ZodRawShape>(shape: T, params?: RawCreateParams) => ZodObject<T, "strict">;
const unionType: <T extends readonly [ZodTypeAny, ZodTypeAny, ...ZodTypeAny[]]>(types: T, params?: RawCreateParams) => ZodUnion<T>;
const discriminatedUnionType: typeof ZodDiscriminatedUnion.create;
const intersectionType: <T extends ZodTypeAny, U extends ZodTypeAny>(left: T, right: U, params?: RawCreateParams) => ZodIntersection<T, U>;
const tupleType: <T extends [] | [ZodTypeAny, ...ZodTypeAny[]]>(schemas: T, params?: RawCreateParams) => ZodTuple<T, null>;
const recordType: typeof ZodRecord.create;
const mapType: <Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny>(keyType: Key, valueType: Value, params?: RawCreateParams) => ZodMap<Key, Value>;
const setType: <Value extends ZodTypeAny = ZodTypeAny>(valueType: Value, params?: RawCreateParams) => ZodSet<Value>;
const functionType: typeof ZodFunction.create;
const lazyType: <T extends ZodTypeAny>(getter: () => T, params?: RawCreateParams) => ZodLazy<T>;
const literalType: <T extends Primitive>(value: T, params?: RawCreateParams) => ZodLiteral<T>;
const enumType: typeof createZodEnum;
const nativeEnumType: <T extends EnumLike>(values: T, params?: RawCreateParams) => ZodNativeEnum<T>;
const promiseType: <T extends ZodTypeAny>(schema: T, params?: RawCreateParams) => ZodPromise<T>;
const effectsType: <I extends ZodTypeAny>(schema: I, effect: Effect<I["_output"]>, params?: RawCreateParams) => ZodEffects<I, I["_output"]>;
const optionalType: <T extends ZodTypeAny>(type: T, params?: RawCreateParams) => ZodOptional<T>;
const nullableType: <T extends ZodTypeAny>(type: T, params?: RawCreateParams) => ZodNullable<T>;
const preprocessType: <I extends ZodTypeAny>(preprocess: (arg: unknown, ctx: RefinementCtx) => unknown, schema: I, params?: RawCreateParams) => ZodEffects<I, I["_output"], unknown>;
const pipelineType: typeof ZodPipeline.create;
const ostring: () => ZodOptional<ZodString>;
const onumber: () => ZodOptional<ZodNumber>;
const oboolean: () => ZodOptional<ZodBoolean>;
const coerce: {
    string: (params?: RawCreateParams & {
        coerce?: true;
    }) => ZodString;
    number: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodNumber;
    boolean: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodBoolean;
    bigint: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodBigInt;
    date: (params?: RawCreateParams & {
        coerce?: boolean;
    }) => ZodDate;
};

const NEVER: never;

//# sourceMappingURL=external.d.ts.map

type z_AnyZodObject = AnyZodObject;
type z_AnyZodTuple = AnyZodTuple;
type z_ArrayCardinality = ArrayCardinality;
type z_ArrayKeys = ArrayKeys;
type z_AssertArray<T> = AssertArray<T>;
type z_AsyncParseReturnType<T> = AsyncParseReturnType<T>;
type z_BRAND<T extends string | number | symbol> = BRAND<T>;
type z_CatchallInput<T extends ZodType> = CatchallInput<T>;
type z_CatchallOutput<T extends ZodType> = CatchallOutput<T>;
type z_CustomErrorParams = CustomErrorParams;
const z_DIRTY: typeof DIRTY;
type z_DenormalizedError = DenormalizedError;
const z_EMPTY_PATH: typeof EMPTY_PATH;
type z_Effect<T> = Effect<T>;
type z_EnumLike = EnumLike;
type z_EnumValues<T extends string = string> = EnumValues<T>;
type z_ErrorMapCtx = ErrorMapCtx;
type z_FilterEnum<Values, ToExclude> = FilterEnum<Values, ToExclude>;
const z_INVALID: typeof INVALID;
type z_Indices<T> = Indices<T>;
type z_InnerTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = InnerTypeOfFunction<Args, Returns>;
type z_InputTypeOfTuple<T extends ZodTupleItems | []> = InputTypeOfTuple<T>;
type z_InputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = InputTypeOfTupleWithRest<T, Rest>;
type z_IpVersion = IpVersion;
type z_IssueData = IssueData;
type z_KeySchema = KeySchema;
const z_NEVER: typeof NEVER;
const z_OK: typeof OK;
type z_ObjectPair = ObjectPair;
type z_OuterTypeOfFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = OuterTypeOfFunction<Args, Returns>;
type z_OutputTypeOfTuple<T extends ZodTupleItems | []> = OutputTypeOfTuple<T>;
type z_OutputTypeOfTupleWithRest<T extends ZodTupleItems | [], Rest extends ZodTypeAny | null = null> = OutputTypeOfTupleWithRest<T, Rest>;
type z_ParseContext = ParseContext;
type z_ParseInput = ParseInput;
type z_ParseParams = ParseParams;
type z_ParsePath = ParsePath;
type z_ParsePathComponent = ParsePathComponent;
type z_ParseResult = ParseResult;
type z_ParseReturnType<T> = ParseReturnType<T>;
type z_ParseStatus = ParseStatus;
const z_ParseStatus: typeof ParseStatus;
type z_PassthroughType<T extends UnknownKeysParam> = PassthroughType<T>;
type z_PreprocessEffect<T> = PreprocessEffect<T>;
type z_Primitive = Primitive;
type z_ProcessedCreateParams = ProcessedCreateParams;
type z_RawCreateParams = RawCreateParams;
type z_RecordType<K extends string | number | symbol, V> = RecordType<K, V>;
type z_Refinement<T> = Refinement<T>;
type z_RefinementCtx = RefinementCtx;
type z_RefinementEffect<T> = RefinementEffect<T>;
type z_SafeParseError<Input> = SafeParseError<Input>;
type z_SafeParseReturnType<Input, Output> = SafeParseReturnType<Input, Output>;
type z_SafeParseSuccess<Output> = SafeParseSuccess<Output>;
type z_Scalars = Scalars;
type z_SomeZodObject = SomeZodObject;
type z_StringValidation = StringValidation;
type z_SuperRefinement<T> = SuperRefinement<T>;
type z_SyncParseReturnType<T = any> = SyncParseReturnType<T>;
type z_TransformEffect<T> = TransformEffect<T>;
type z_TypeOf<T extends ZodType<any, any, any>> = TypeOf<T>;
type z_UnknownKeysParam = UnknownKeysParam;
type z_Values<T extends EnumValues> = Values<T>;
type z_Writeable<T> = Writeable<T>;
type z_ZodAny = ZodAny;
const z_ZodAny: typeof ZodAny;
type z_ZodAnyDef = ZodAnyDef;
type z_ZodArray<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> = ZodArray<T, Cardinality>;
const z_ZodArray: typeof ZodArray;
type z_ZodArrayDef<T extends ZodTypeAny = ZodTypeAny> = ZodArrayDef<T>;
type z_ZodBigInt = ZodBigInt;
const z_ZodBigInt: typeof ZodBigInt;
type z_ZodBigIntCheck = ZodBigIntCheck;
type z_ZodBigIntDef = ZodBigIntDef;
type z_ZodBoolean = ZodBoolean;
const z_ZodBoolean: typeof ZodBoolean;
type z_ZodBooleanDef = ZodBooleanDef;
type z_ZodBranded<T extends ZodTypeAny, B extends string | number | symbol> = ZodBranded<T, B>;
const z_ZodBranded: typeof ZodBranded;
type z_ZodBrandedDef<T extends ZodTypeAny> = ZodBrandedDef<T>;
type z_ZodCatch<T extends ZodTypeAny> = ZodCatch<T>;
const z_ZodCatch: typeof ZodCatch;
type z_ZodCatchDef<T extends ZodTypeAny = ZodTypeAny> = ZodCatchDef<T>;
type z_ZodCustomIssue = ZodCustomIssue;
type z_ZodDate = ZodDate;
const z_ZodDate: typeof ZodDate;
type z_ZodDateCheck = ZodDateCheck;
type z_ZodDateDef = ZodDateDef;
type z_ZodDefault<T extends ZodTypeAny> = ZodDefault<T>;
const z_ZodDefault: typeof ZodDefault;
type z_ZodDefaultDef<T extends ZodTypeAny = ZodTypeAny> = ZodDefaultDef<T>;
type z_ZodDiscriminatedUnion<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<Discriminator>[]> = ZodDiscriminatedUnion<Discriminator, Options>;
const z_ZodDiscriminatedUnion: typeof ZodDiscriminatedUnion;
type z_ZodDiscriminatedUnionDef<Discriminator extends string, Options extends readonly ZodDiscriminatedUnionOption<string>[] = ZodDiscriminatedUnionOption<string>[]> = ZodDiscriminatedUnionDef<Discriminator, Options>;
type z_ZodDiscriminatedUnionOption<Discriminator extends string> = ZodDiscriminatedUnionOption<Discriminator>;
type z_ZodEffects<T extends ZodTypeAny, Output = output<T>, Input = input<T>> = ZodEffects<T, Output, Input>;
const z_ZodEffects: typeof ZodEffects;
type z_ZodEffectsDef<T extends ZodTypeAny = ZodTypeAny> = ZodEffectsDef<T>;
type z_ZodEnum<T extends [string, ...string[]]> = ZodEnum<T>;
const z_ZodEnum: typeof ZodEnum;
type z_ZodEnumDef<T extends EnumValues = EnumValues> = ZodEnumDef<T>;
type z_ZodError<T = any> = ZodError<T>;
const z_ZodError: typeof ZodError;
type z_ZodErrorMap = ZodErrorMap;
type z_ZodFirstPartySchemaTypes = ZodFirstPartySchemaTypes;
type z_ZodFirstPartyTypeKind = ZodFirstPartyTypeKind;
const z_ZodFirstPartyTypeKind: typeof ZodFirstPartyTypeKind;
type z_ZodFormattedError<T, U = string> = ZodFormattedError<T, U>;
type z_ZodFunction<Args extends ZodTuple<any, any>, Returns extends ZodTypeAny> = ZodFunction<Args, Returns>;
const z_ZodFunction: typeof ZodFunction;
type z_ZodFunctionDef<Args extends ZodTuple<any, any> = ZodTuple<any, any>, Returns extends ZodTypeAny = ZodTypeAny> = ZodFunctionDef<Args, Returns>;
type z_ZodIntersection<T extends ZodTypeAny, U extends ZodTypeAny> = ZodIntersection<T, U>;
const z_ZodIntersection: typeof ZodIntersection;
type z_ZodIntersectionDef<T extends ZodTypeAny = ZodTypeAny, U extends ZodTypeAny = ZodTypeAny> = ZodIntersectionDef<T, U>;
type z_ZodInvalidArgumentsIssue = ZodInvalidArgumentsIssue;
type z_ZodInvalidDateIssue = ZodInvalidDateIssue;
type z_ZodInvalidEnumValueIssue = ZodInvalidEnumValueIssue;
type z_ZodInvalidIntersectionTypesIssue = ZodInvalidIntersectionTypesIssue;
type z_ZodInvalidLiteralIssue = ZodInvalidLiteralIssue;
type z_ZodInvalidReturnTypeIssue = ZodInvalidReturnTypeIssue;
type z_ZodInvalidStringIssue = ZodInvalidStringIssue;
type z_ZodInvalidTypeIssue = ZodInvalidTypeIssue;
type z_ZodInvalidUnionDiscriminatorIssue = ZodInvalidUnionDiscriminatorIssue;
type z_ZodInvalidUnionIssue = ZodInvalidUnionIssue;
type z_ZodIssue = ZodIssue;
type z_ZodIssueBase = ZodIssueBase;
type z_ZodIssueCode = ZodIssueCode;
type z_ZodIssueOptionalMessage = ZodIssueOptionalMessage;
type z_ZodLazy<T extends ZodTypeAny> = ZodLazy<T>;
const z_ZodLazy: typeof ZodLazy;
type z_ZodLazyDef<T extends ZodTypeAny = ZodTypeAny> = ZodLazyDef<T>;
type z_ZodLiteral<T> = ZodLiteral<T>;
const z_ZodLiteral: typeof ZodLiteral;
type z_ZodLiteralDef<T = any> = ZodLiteralDef<T>;
type z_ZodMap<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> = ZodMap<Key, Value>;
const z_ZodMap: typeof ZodMap;
type z_ZodMapDef<Key extends ZodTypeAny = ZodTypeAny, Value extends ZodTypeAny = ZodTypeAny> = ZodMapDef<Key, Value>;
type z_ZodNaN = ZodNaN;
const z_ZodNaN: typeof ZodNaN;
type z_ZodNaNDef = ZodNaNDef;
type z_ZodNativeEnum<T extends EnumLike> = ZodNativeEnum<T>;
const z_ZodNativeEnum: typeof ZodNativeEnum;
type z_ZodNativeEnumDef<T extends EnumLike = EnumLike> = ZodNativeEnumDef<T>;
type z_ZodNever = ZodNever;
const z_ZodNever: typeof ZodNever;
type z_ZodNeverDef = ZodNeverDef;
type z_ZodNonEmptyArray<T extends ZodTypeAny> = ZodNonEmptyArray<T>;
type z_ZodNotFiniteIssue = ZodNotFiniteIssue;
type z_ZodNotMultipleOfIssue = ZodNotMultipleOfIssue;
type z_ZodNull = ZodNull;
const z_ZodNull: typeof ZodNull;
type z_ZodNullDef = ZodNullDef;
type z_ZodNullable<T extends ZodTypeAny> = ZodNullable<T>;
const z_ZodNullable: typeof ZodNullable;
type z_ZodNullableDef<T extends ZodTypeAny = ZodTypeAny> = ZodNullableDef<T>;
type z_ZodNullableType<T extends ZodTypeAny> = ZodNullableType<T>;
type z_ZodNumber = ZodNumber;
const z_ZodNumber: typeof ZodNumber;
type z_ZodNumberCheck = ZodNumberCheck;
type z_ZodNumberDef = ZodNumberDef;
type z_ZodObject<T extends ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny, Output = objectOutputType<T, Catchall, UnknownKeys>, Input = objectInputType<T, Catchall, UnknownKeys>> = ZodObject<T, UnknownKeys, Catchall, Output, Input>;
const z_ZodObject: typeof ZodObject;
type z_ZodObjectDef<T extends ZodRawShape = ZodRawShape, UnknownKeys extends UnknownKeysParam = UnknownKeysParam, Catchall extends ZodTypeAny = ZodTypeAny> = ZodObjectDef<T, UnknownKeys, Catchall>;
type z_ZodOptional<T extends ZodTypeAny> = ZodOptional<T>;
const z_ZodOptional: typeof ZodOptional;
type z_ZodOptionalDef<T extends ZodTypeAny = ZodTypeAny> = ZodOptionalDef<T>;
type z_ZodOptionalType<T extends ZodTypeAny> = ZodOptionalType<T>;
type z_ZodParsedType = ZodParsedType;
type z_ZodPipeline<A extends ZodTypeAny, B extends ZodTypeAny> = ZodPipeline<A, B>;
const z_ZodPipeline: typeof ZodPipeline;
type z_ZodPipelineDef<A extends ZodTypeAny, B extends ZodTypeAny> = ZodPipelineDef<A, B>;
type z_ZodPromise<T extends ZodTypeAny> = ZodPromise<T>;
const z_ZodPromise: typeof ZodPromise;
type z_ZodPromiseDef<T extends ZodTypeAny = ZodTypeAny> = ZodPromiseDef<T>;
type z_ZodRawShape = ZodRawShape;
type z_ZodReadonly<T extends ZodTypeAny> = ZodReadonly<T>;
const z_ZodReadonly: typeof ZodReadonly;
type z_ZodReadonlyDef<T extends ZodTypeAny = ZodTypeAny> = ZodReadonlyDef<T>;
type z_ZodRecord<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> = ZodRecord<Key, Value>;
const z_ZodRecord: typeof ZodRecord;
type z_ZodRecordDef<Key extends KeySchema = ZodString, Value extends ZodTypeAny = ZodTypeAny> = ZodRecordDef<Key, Value>;
type z_ZodSet<Value extends ZodTypeAny = ZodTypeAny> = ZodSet<Value>;
const z_ZodSet: typeof ZodSet;
type z_ZodSetDef<Value extends ZodTypeAny = ZodTypeAny> = ZodSetDef<Value>;
type z_ZodString = ZodString;
const z_ZodString: typeof ZodString;
type z_ZodStringCheck = ZodStringCheck;
type z_ZodStringDef = ZodStringDef;
type z_ZodSymbol = ZodSymbol;
const z_ZodSymbol: typeof ZodSymbol;
type z_ZodSymbolDef = ZodSymbolDef;
type z_ZodTooBigIssue = ZodTooBigIssue;
type z_ZodTooSmallIssue = ZodTooSmallIssue;
type z_ZodTuple<T extends [ZodTypeAny, ...ZodTypeAny[]] | [] = [ZodTypeAny, ...ZodTypeAny[]], Rest extends ZodTypeAny | null = null> = ZodTuple<T, Rest>;
const z_ZodTuple: typeof ZodTuple;
type z_ZodTupleDef<T extends ZodTupleItems | [] = ZodTupleItems, Rest extends ZodTypeAny | null = null> = ZodTupleDef<T, Rest>;
type z_ZodTupleItems = ZodTupleItems;
type z_ZodType<Output = any, Def extends ZodTypeDef = ZodTypeDef, Input = Output> = ZodType<Output, Def, Input>;
const z_ZodType: typeof ZodType;
type z_ZodTypeAny = ZodTypeAny;
type z_ZodTypeDef = ZodTypeDef;
type z_ZodUndefined = ZodUndefined;
const z_ZodUndefined: typeof ZodUndefined;
type z_ZodUndefinedDef = ZodUndefinedDef;
type z_ZodUnion<T extends ZodUnionOptions> = ZodUnion<T>;
const z_ZodUnion: typeof ZodUnion;
type z_ZodUnionDef<T extends ZodUnionOptions = Readonly<[
    ZodTypeAny,
    ZodTypeAny,
    ...ZodTypeAny[]
]>> = ZodUnionDef<T>;
type z_ZodUnionOptions = ZodUnionOptions;
type z_ZodUnknown = ZodUnknown;
const z_ZodUnknown: typeof ZodUnknown;
type z_ZodUnknownDef = ZodUnknownDef;
type z_ZodUnrecognizedKeysIssue = ZodUnrecognizedKeysIssue;
type z_ZodVoid = ZodVoid;
const z_ZodVoid: typeof ZodVoid;
type z_ZodVoidDef = ZodVoidDef;
const z_addIssueToContext: typeof addIssueToContext;
type z_arrayOutputType<T extends ZodTypeAny, Cardinality extends ArrayCardinality = "many"> = arrayOutputType<T, Cardinality>;
type z_baseObjectInputType<Shape extends ZodRawShape> = baseObjectInputType<Shape>;
type z_baseObjectOutputType<Shape extends ZodRawShape> = baseObjectOutputType<Shape>;
const z_coerce: typeof coerce;
const z_custom: typeof custom;
const z_datetimeRegex: typeof datetimeRegex;
type z_deoptional<T extends ZodTypeAny> = deoptional<T>;
const z_getErrorMap: typeof getErrorMap;
const z_getParsedType: typeof getParsedType;
type z_inferFlattenedErrors<T extends ZodType<any, any, any>, U = string> = inferFlattenedErrors<T, U>;
type z_inferFormattedError<T extends ZodType<any, any, any>, U = string> = inferFormattedError<T, U>;
type z_input<T extends ZodType<any, any, any>> = input<T>;
const z_isAborted: typeof isAborted;
const z_isAsync: typeof isAsync;
const z_isDirty: typeof isDirty;
const z_isValid: typeof isValid;
const z_late: typeof late;
const z_makeIssue: typeof makeIssue;
type z_mergeTypes<A, B> = mergeTypes<A, B>;
type z_noUnrecognized<Obj extends object, Shape extends object> = noUnrecognized<Obj, Shape>;
type z_objectInputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectInputType<Shape, Catchall, UnknownKeys>;
type z_objectOutputType<Shape extends ZodRawShape, Catchall extends ZodTypeAny, UnknownKeys extends UnknownKeysParam = UnknownKeysParam> = objectOutputType<Shape, Catchall, UnknownKeys>;
const z_objectUtil: typeof objectUtil;
const z_oboolean: typeof oboolean;
const z_onumber: typeof onumber;
const z_ostring: typeof ostring;
type z_output<T extends ZodType<any, any, any>> = output<T>;
const z_quotelessJson: typeof quotelessJson;
const z_setErrorMap: typeof setErrorMap;
type z_typeToFlattenedError<T, U = string> = typeToFlattenedError<T, U>;
type z_typecast<A, T> = typecast<A, T>;
const z_util: typeof util;
namespace z {
  export { type z_AnyZodObject as AnyZodObject, type z_AnyZodTuple as AnyZodTuple, type z_ArrayCardinality as ArrayCardinality, type z_ArrayKeys as ArrayKeys, type z_AssertArray as AssertArray, type z_AsyncParseReturnType as AsyncParseReturnType, type z_BRAND as BRAND, type z_CatchallInput as CatchallInput, type z_CatchallOutput as CatchallOutput, type z_CustomErrorParams as CustomErrorParams, z_DIRTY as DIRTY, type z_DenormalizedError as DenormalizedError, z_EMPTY_PATH as EMPTY_PATH, type z_Effect as Effect, type z_EnumLike as EnumLike, type z_EnumValues as EnumValues, type z_ErrorMapCtx as ErrorMapCtx, type z_FilterEnum as FilterEnum, z_INVALID as INVALID, type z_Indices as Indices, type z_InnerTypeOfFunction as InnerTypeOfFunction, type z_InputTypeOfTuple as InputTypeOfTuple, type z_InputTypeOfTupleWithRest as InputTypeOfTupleWithRest, type z_IpVersion as IpVersion, type z_IssueData as IssueData, type z_KeySchema as KeySchema, z_NEVER as NEVER, z_OK as OK, type z_ObjectPair as ObjectPair, type z_OuterTypeOfFunction as OuterTypeOfFunction, type z_OutputTypeOfTuple as OutputTypeOfTuple, type z_OutputTypeOfTupleWithRest as OutputTypeOfTupleWithRest, type z_ParseContext as ParseContext, type z_ParseInput as ParseInput, type z_ParseParams as ParseParams, type z_ParsePath as ParsePath, type z_ParsePathComponent as ParsePathComponent, type z_ParseResult as ParseResult, type z_ParseReturnType as ParseReturnType, z_ParseStatus as ParseStatus, type z_PassthroughType as PassthroughType, type z_PreprocessEffect as PreprocessEffect, type z_Primitive as Primitive, type z_ProcessedCreateParams as ProcessedCreateParams, type z_RawCreateParams as RawCreateParams, type z_RecordType as RecordType, type z_Refinement as Refinement, type z_RefinementCtx as RefinementCtx, type z_RefinementEffect as RefinementEffect, type z_SafeParseError as SafeParseError, type z_SafeParseReturnType as SafeParseReturnType, type z_SafeParseSuccess as SafeParseSuccess, type z_Scalars as Scalars, ZodType as Schema, type z_SomeZodObject as SomeZodObject, type z_StringValidation as StringValidation, type z_SuperRefinement as SuperRefinement, type z_SyncParseReturnType as SyncParseReturnType, type z_TransformEffect as TransformEffect, type z_TypeOf as TypeOf, type z_UnknownKeysParam as UnknownKeysParam, type z_Values as Values, type z_Writeable as Writeable, z_ZodAny as ZodAny, type z_ZodAnyDef as ZodAnyDef, z_ZodArray as ZodArray, type z_ZodArrayDef as ZodArrayDef, z_ZodBigInt as ZodBigInt, type z_ZodBigIntCheck as ZodBigIntCheck, type z_ZodBigIntDef as ZodBigIntDef, z_ZodBoolean as ZodBoolean, type z_ZodBooleanDef as ZodBooleanDef, z_ZodBranded as ZodBranded, type z_ZodBrandedDef as ZodBrandedDef, z_ZodCatch as ZodCatch, type z_ZodCatchDef as ZodCatchDef, type z_ZodCustomIssue as ZodCustomIssue, z_ZodDate as ZodDate, type z_ZodDateCheck as ZodDateCheck, type z_ZodDateDef as ZodDateDef, z_ZodDefault as ZodDefault, type z_ZodDefaultDef as ZodDefaultDef, z_ZodDiscriminatedUnion as ZodDiscriminatedUnion, type z_ZodDiscriminatedUnionDef as ZodDiscriminatedUnionDef, type z_ZodDiscriminatedUnionOption as ZodDiscriminatedUnionOption, z_ZodEffects as ZodEffects, type z_ZodEffectsDef as ZodEffectsDef, z_ZodEnum as ZodEnum, type z_ZodEnumDef as ZodEnumDef, z_ZodError as ZodError, type z_ZodErrorMap as ZodErrorMap, type z_ZodFirstPartySchemaTypes as ZodFirstPartySchemaTypes, z_ZodFirstPartyTypeKind as ZodFirstPartyTypeKind, type z_ZodFormattedError as ZodFormattedError, z_ZodFunction as ZodFunction, type z_ZodFunctionDef as ZodFunctionDef, z_ZodIntersection as ZodIntersection, type z_ZodIntersectionDef as ZodIntersectionDef, type z_ZodInvalidArgumentsIssue as ZodInvalidArgumentsIssue, type z_ZodInvalidDateIssue as ZodInvalidDateIssue, type z_ZodInvalidEnumValueIssue as ZodInvalidEnumValueIssue, type z_ZodInvalidIntersectionTypesIssue as ZodInvalidIntersectionTypesIssue, type z_ZodInvalidLiteralIssue as ZodInvalidLiteralIssue, type z_ZodInvalidReturnTypeIssue as ZodInvalidReturnTypeIssue, type z_ZodInvalidStringIssue as ZodInvalidStringIssue, type z_ZodInvalidTypeIssue as ZodInvalidTypeIssue, type z_ZodInvalidUnionDiscriminatorIssue as ZodInvalidUnionDiscriminatorIssue, type z_ZodInvalidUnionIssue as ZodInvalidUnionIssue, type z_ZodIssue as ZodIssue, type z_ZodIssueBase as ZodIssueBase, type z_ZodIssueCode as ZodIssueCode, type z_ZodIssueOptionalMessage as ZodIssueOptionalMessage, z_ZodLazy as ZodLazy, type z_ZodLazyDef as ZodLazyDef, z_ZodLiteral as ZodLiteral, type z_ZodLiteralDef as ZodLiteralDef, z_ZodMap as ZodMap, type z_ZodMapDef as ZodMapDef, z_ZodNaN as ZodNaN, type z_ZodNaNDef as ZodNaNDef, z_ZodNativeEnum as ZodNativeEnum, type z_ZodNativeEnumDef as ZodNativeEnumDef, z_ZodNever as ZodNever, type z_ZodNeverDef as ZodNeverDef, type z_ZodNonEmptyArray as ZodNonEmptyArray, type z_ZodNotFiniteIssue as ZodNotFiniteIssue, type z_ZodNotMultipleOfIssue as ZodNotMultipleOfIssue, z_ZodNull as ZodNull, type z_ZodNullDef as ZodNullDef, z_ZodNullable as ZodNullable, type z_ZodNullableDef as ZodNullableDef, type z_ZodNullableType as ZodNullableType, z_ZodNumber as ZodNumber, type z_ZodNumberCheck as ZodNumberCheck, type z_ZodNumberDef as ZodNumberDef, z_ZodObject as ZodObject, type z_ZodObjectDef as ZodObjectDef, z_ZodOptional as ZodOptional, type z_ZodOptionalDef as ZodOptionalDef, type z_ZodOptionalType as ZodOptionalType, type z_ZodParsedType as ZodParsedType, z_ZodPipeline as ZodPipeline, type z_ZodPipelineDef as ZodPipelineDef, z_ZodPromise as ZodPromise, type z_ZodPromiseDef as ZodPromiseDef, type z_ZodRawShape as ZodRawShape, z_ZodReadonly as ZodReadonly, type z_ZodReadonlyDef as ZodReadonlyDef, z_ZodRecord as ZodRecord, type z_ZodRecordDef as ZodRecordDef, ZodType as ZodSchema, z_ZodSet as ZodSet, type z_ZodSetDef as ZodSetDef, z_ZodString as ZodString, type z_ZodStringCheck as ZodStringCheck, type z_ZodStringDef as ZodStringDef, z_ZodSymbol as ZodSymbol, type z_ZodSymbolDef as ZodSymbolDef, type z_ZodTooBigIssue as ZodTooBigIssue, type z_ZodTooSmallIssue as ZodTooSmallIssue, ZodEffects as ZodTransformer, z_ZodTuple as ZodTuple, type z_ZodTupleDef as ZodTupleDef, type z_ZodTupleItems as ZodTupleItems, z_ZodType as ZodType, type z_ZodTypeAny as ZodTypeAny, type z_ZodTypeDef as ZodTypeDef, z_ZodUndefined as ZodUndefined, type z_ZodUndefinedDef as ZodUndefinedDef, z_ZodUnion as ZodUnion, type z_ZodUnionDef as ZodUnionDef, type z_ZodUnionOptions as ZodUnionOptions, z_ZodUnknown as ZodUnknown, type z_ZodUnknownDef as ZodUnknownDef, type z_ZodUnrecognizedKeysIssue as ZodUnrecognizedKeysIssue, z_ZodVoid as ZodVoid, type z_ZodVoidDef as ZodVoidDef, z_addIssueToContext as addIssueToContext, anyType as any, arrayType as array, type z_arrayOutputType as arrayOutputType, type z_baseObjectInputType as baseObjectInputType, type z_baseObjectOutputType as baseObjectOutputType, bigIntType as bigint, booleanType as boolean, z_coerce as coerce, z_custom as custom, dateType as date, z_datetimeRegex as datetimeRegex, errorMap as defaultErrorMap, type z_deoptional as deoptional, discriminatedUnionType as discriminatedUnion, effectsType as effect, enumType as enum, functionType as function, z_getErrorMap as getErrorMap, z_getParsedType as getParsedType, type TypeOf as infer, type z_inferFlattenedErrors as inferFlattenedErrors, type z_inferFormattedError as inferFormattedError, type z_input as input, instanceOfType as instanceof, intersectionType as intersection, z_isAborted as isAborted, z_isAsync as isAsync, z_isDirty as isDirty, z_isValid as isValid, z_late as late, lazyType as lazy, literalType as literal, z_makeIssue as makeIssue, mapType as map, type z_mergeTypes as mergeTypes, nanType as nan, nativeEnumType as nativeEnum, neverType as never, type z_noUnrecognized as noUnrecognized, nullType as null, nullableType as nullable, numberType as number, objectType as object, type z_objectInputType as objectInputType, type z_objectOutputType as objectOutputType, z_objectUtil as objectUtil, z_oboolean as oboolean, z_onumber as onumber, optionalType as optional, z_ostring as ostring, type z_output as output, pipelineType as pipeline, preprocessType as preprocess, promiseType as promise, z_quotelessJson as quotelessJson, recordType as record, setType as set, z_setErrorMap as setErrorMap, strictObjectType as strictObject, stringType as string, symbolType as symbol, effectsType as transformer, tupleType as tuple, type z_typeToFlattenedError as typeToFlattenedError, type z_typecast as typecast, undefinedType as undefined, unionType as union, unknownType as unknown, z_util as util, voidType as void };
}

//# sourceMappingURL=index.d.ts.map

export { type AnyZodObject, type AnyZodTuple, type ArrayCardinality, type ArrayKeys, type AssertArray, type AsyncParseReturnType, BRAND, type CatchallInput, type CatchallOutput, type CustomErrorParams, DIRTY, type DenormalizedError, EMPTY_PATH, type Effect, type EnumLike, type EnumValues, type ErrorMapCtx, type FilterEnum, INVALID, type Indices, type InnerTypeOfFunction, type InputTypeOfTuple, type InputTypeOfTupleWithRest, type IpVersion, type IssueData, type KeySchema, NEVER, OK, type ObjectPair, type OuterTypeOfFunction, type OutputTypeOfTuple, type OutputTypeOfTupleWithRest, type ParseContext, type ParseInput, type ParseParams, type ParsePath, type ParsePathComponent, type ParseResult, type ParseReturnType, ParseStatus, type PassthroughType, type PreprocessEffect, type Primitive, type ProcessedCreateParams, type RawCreateParams, type RecordType, type Refinement, type RefinementCtx, type RefinementEffect, type SafeParseError, type SafeParseReturnType, type SafeParseSuccess, type Scalars, ZodType as Schema, type SomeZodObject, type StringValidation, type SuperRefinement, type SyncParseReturnType, type TransformEffect, type TypeOf, type UnknownKeysParam, type Values, type Writeable, ZodAny, type ZodAnyDef, ZodArray, type ZodArrayDef, ZodBigInt, type ZodBigIntCheck, type ZodBigIntDef, ZodBoolean, type ZodBooleanDef, ZodBranded, type ZodBrandedDef, ZodCatch, type ZodCatchDef, type ZodCustomIssue, ZodDate, type ZodDateCheck, type ZodDateDef, ZodDefault, type ZodDefaultDef, ZodDiscriminatedUnion, type ZodDiscriminatedUnionDef, type ZodDiscriminatedUnionOption, ZodEffects, type ZodEffectsDef, ZodEnum, type ZodEnumDef, ZodError, type ZodErrorMap, type ZodFirstPartySchemaTypes, ZodFirstPartyTypeKind, type ZodFormattedError, ZodFunction, type ZodFunctionDef, ZodIntersection, type ZodIntersectionDef, type ZodInvalidArgumentsIssue, type ZodInvalidDateIssue, type ZodInvalidEnumValueIssue, type ZodInvalidIntersectionTypesIssue, type ZodInvalidLiteralIssue, type ZodInvalidReturnTypeIssue, type ZodInvalidStringIssue, type ZodInvalidTypeIssue, type ZodInvalidUnionDiscriminatorIssue, type ZodInvalidUnionIssue, type ZodIssue, type ZodIssueBase, ZodIssueCode, type ZodIssueOptionalMessage, ZodLazy, type ZodLazyDef, ZodLiteral, type ZodLiteralDef, ZodMap, type ZodMapDef, ZodNaN, type ZodNaNDef, ZodNativeEnum, type ZodNativeEnumDef, ZodNever, type ZodNeverDef, type ZodNonEmptyArray, type ZodNotFiniteIssue, type ZodNotMultipleOfIssue, ZodNull, type ZodNullDef, ZodNullable, type ZodNullableDef, type ZodNullableType, ZodNumber, type ZodNumberCheck, type ZodNumberDef, ZodObject, type ZodObjectDef, ZodOptional, type ZodOptionalDef, type ZodOptionalType, ZodParsedType, ZodPipeline, type ZodPipelineDef, ZodPromise, type ZodPromiseDef, type ZodRawShape, ZodReadonly, type ZodReadonlyDef, ZodRecord, type ZodRecordDef, ZodType as ZodSchema, ZodSet, type ZodSetDef, ZodString, type ZodStringCheck, type ZodStringDef, ZodSymbol, type ZodSymbolDef, type ZodTooBigIssue, type ZodTooSmallIssue, ZodEffects as ZodTransformer, ZodTuple, type ZodTupleDef, type ZodTupleItems, ZodType, type ZodTypeAny, type ZodTypeDef, ZodUndefined, type ZodUndefinedDef, ZodUnion, type ZodUnionDef, type ZodUnionOptions, ZodUnknown, type ZodUnknownDef, type ZodUnrecognizedKeysIssue, ZodVoid, type ZodVoidDef, addIssueToContext, anyType as any, arrayType as array, type arrayOutputType, type baseObjectInputType, type baseObjectOutputType, bigIntType as bigint, booleanType as boolean, coerce, custom, dateType as date, datetimeRegex, z as default, errorMap as defaultErrorMap, type deoptional, discriminatedUnionType as discriminatedUnion, effectsType as effect, enumType as enum, functionType as function, getErrorMap, getParsedType, type TypeOf as infer, type inferFlattenedErrors, type inferFormattedError, type input, instanceOfType as instanceof, intersectionType as intersection, isAborted, isAsync, isDirty, isValid, late, lazyType as lazy, literalType as literal, makeIssue, mapType as map, type mergeTypes, nanType as nan, nativeEnumType as nativeEnum, neverType as never, type noUnrecognized, nullType as null, nullableType as nullable, numberType as number, objectType as object, type objectInputType, type objectOutputType, objectUtil, oboolean, onumber, optionalType as optional, ostring, type output, pipelineType as pipeline, preprocessType as preprocess, promiseType as promise, quotelessJson, recordType as record, setType as set, setErrorMap, strictObjectType as strictObject, stringType as string, symbolType as symbol, effectsType as transformer, tupleType as tuple, type typeToFlattenedError, type typecast, undefinedType as undefined, unionType as union, unknownType as unknown, util, voidType as void, z };
}



// ==================================================================================================
// JSON Schema Draft 07
// ==================================================================================================
// https://tools.ietf.org/html/draft-handrews-json-schema-validation-01
// --------------------------------------------------------------------------------------------------

/**
 * Primitive type
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.1.1
 */
type JSONSchema7TypeName =
| "string" //
| "number"
| "integer"
| "boolean"
| "object"
| "array"
| "null";

/**
* Primitive type
* @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.1.1
*/
type JSONSchema7Type =
| string //
| number
| boolean
| JSONSchema7Object
| JSONSchema7Array
| null;

// Workaround for infinite type recursion
interface JSONSchema7Object {
[key: string]: JSONSchema7Type;
}

// Workaround for infinite type recursion
// https://github.com/Microsoft/TypeScript/issues/3496#issuecomment-128553540
interface JSONSchema7Array extends Array<JSONSchema7Type> {}

/**
* Meta schema
*
* Recommended values:
* - 'http://json-schema.org/schema#'
* - 'http://json-schema.org/hyper-schema#'
* - 'http://json-schema.org/draft-07/schema#'
* - 'http://json-schema.org/draft-07/hyper-schema#'
*
* @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-5
*/
type JSONSchema7Version = string;

/**
* JSON Schema v7
* @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01
*/
type JSONSchema7Definition = JSONSchema7 | boolean;
interface JSONSchema7 {
$id?: string | undefined;
$ref?: string | undefined;
$schema?: JSONSchema7Version | undefined;
$comment?: string | undefined;

/**
 * @see https://datatracker.ietf.org/doc/html/draft-bhutton-json-schema-00#section-8.2.4
 * @see https://datatracker.ietf.org/doc/html/draft-bhutton-json-schema-validation-00#appendix-A
 */
$defs?: {
    [key: string]: JSONSchema7Definition;
} | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.1
 */
type?: JSONSchema7TypeName | JSONSchema7TypeName[] | undefined;
enum?: JSONSchema7Type[] | undefined;
const?: JSONSchema7Type | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.2
 */
multipleOf?: number | undefined;
maximum?: number | undefined;
exclusiveMaximum?: number | undefined;
minimum?: number | undefined;
exclusiveMinimum?: number | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.3
 */
maxLength?: number | undefined;
minLength?: number | undefined;
pattern?: string | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.4
 */
items?: JSONSchema7Definition | JSONSchema7Definition[] | undefined;
additionalItems?: JSONSchema7Definition | undefined;
maxItems?: number | undefined;
minItems?: number | undefined;
uniqueItems?: boolean | undefined;
contains?: JSONSchema7Definition | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.5
 */
maxProperties?: number | undefined;
minProperties?: number | undefined;
required?: string[] | undefined;
properties?: {
    [key: string]: JSONSchema7Definition;
} | undefined;
patternProperties?: {
    [key: string]: JSONSchema7Definition;
} | undefined;
additionalProperties?: JSONSchema7Definition | undefined;
dependencies?: {
    [key: string]: JSONSchema7Definition | string[];
} | undefined;
propertyNames?: JSONSchema7Definition | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.6
 */
if?: JSONSchema7Definition | undefined;
then?: JSONSchema7Definition | undefined;
else?: JSONSchema7Definition | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-6.7
 */
allOf?: JSONSchema7Definition[] | undefined;
anyOf?: JSONSchema7Definition[] | undefined;
oneOf?: JSONSchema7Definition[] | undefined;
not?: JSONSchema7Definition | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-7
 */
format?: string | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-8
 */
contentMediaType?: string | undefined;
contentEncoding?: string | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-9
 */
definitions?: {
    [key: string]: JSONSchema7Definition;
} | undefined;

/**
 * @see https://tools.ietf.org/html/draft-handrews-json-schema-validation-01#section-10
 */
title?: string | undefined;
description?: string | undefined;
default?: JSONSchema7Type | undefined;
readOnly?: boolean | undefined;
writeOnly?: boolean | undefined;
examples?: JSONSchema7Type | undefined;
}
