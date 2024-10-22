/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=4 sts=4 sw=4 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│                                                                              │
│ Bestline ── Library for interactive pseudoteletypewriter command             │
│             sessions using ANSI Standard X3.64 control sequences             │
│                                                                              │
│ OVERVIEW                                                                     │
│                                                                              │
│   Bestline is a fork of linenoise (a popular readline alternative)           │
│   that fixes its bugs and adds the missing features while reducing           │
│   binary footprint (surprisingly) by removing bloated dependencies           │
│   which means you can finally have a permissively-licensed command           │
│   prompt w/ a 30kb footprint that's nearly as good as gnu readline           │
│                                                                              │
│ EXAMPLE                                                                      │
│                                                                              │
│   main() {                                                                   │
│       char *line;                                                            │
│       while ((line = bestlineWithHistory("IN> ", "foo"))) {                  │
│           fputs("OUT> ", stdout);                                            │
│           fputs(line, stdout);                                               │
│           fputs("\n", stdout);                                               │
│           free(line);                                                        │
│       }                                                                      │
│   }                                                                          │
│                                                                              │
│ CHANGES                                                                      │
│                                                                              │
│   - Remove bell                                                              │
│   - Add kill ring                                                            │
│   - Fix flickering                                                           │
│   - Add UTF-8 editing                                                        │
│   - Add CTRL-R search                                                        │
│   - Support unlimited lines                                                  │
│   - Add parentheses awareness                                                │
│   - React to terminal resizing                                               │
│   - Don't generate .data section                                             │
│   - Support terminal flow control                                            │
│   - Make history loading 10x faster                                          │
│   - Make multiline mode the only mode                                        │
│   - Accommodate O_NONBLOCK file descriptors                                  │
│   - Restore raw mode on process foregrounding                                │
│   - Make source code compatible with C++ compilers                           │
│   - Fix corruption issues by using generalized parsing                       │
│   - Implement nearly all GNU readline editing shortcuts                      │
│   - Remove heavyweight dependencies like printf/sprintf                      │
│   - Remove ISIG→^C→EAGAIN hack and use ephemeral handlers                    │
│   - Support running on Windows in MinTTY or CMD.EXE on Win10+                │
│   - Support diacratics, русский, Ελληνικά, 漢字, 仮名, 한글                  │
│                                                                              │
│ SHORTCUTS                                                                    │
│                                                                              │
│   CTRL-E         END                                                         │
│   CTRL-A         START                                                       │
│   CTRL-B         BACK                                                        │
│   CTRL-F         FORWARD                                                     │
│   CTRL-L         CLEAR                                                       │
│   CTRL-H         BACKSPACE                                                   │
│   CTRL-D         DELETE                                                      │
│   CTRL-Y         YANK                                                        │
│   CTRL-D         EOF (IF EMPTY)                                              │
│   CTRL-N         NEXT HISTORY                                                │
│   CTRL-P         PREVIOUS HISTORY                                            │
│   CTRL-R         SEARCH HISTORY                                              │
│   CTRL-G         CANCEL SEARCH                                               │
│   CTRL-J         INSERT NEWLINE                                              │
│   ALT-<          BEGINNING OF HISTORY                                        │
│   ALT->          END OF HISTORY                                              │
│   ALT-F          FORWARD WORD                                                │
│   ALT-B          BACKWARD WORD                                               │
│   CTRL-ALT-F     FORWARD EXPR                                                │
│   CTRL-ALT-B     BACKWARD EXPR                                               │
│   ALT-RIGHT      FORWARD EXPR                                                │
│   ALT-LEFT       BACKWARD EXPR                                               │
│   ALT-SHIFT-B    BARF EXPR                                                   │
│   ALT-SHIFT-S    SLURP EXPR                                                  │
│   ALT-SHIFT-R    RAISE EXPR                                                  │
│   CTRL-K         KILL LINE FORWARDS                                          │
│   CTRL-U         KILL LINE BACKWARDS                                         │
│   ALT-H          KILL WORD BACKWARDS                                         │
│   CTRL-W         KILL WORD BACKWARDS                                         │
│   CTRL-ALT-H     KILL WORD BACKWARDS                                         │
│   ALT-D          KILL WORD FORWARDS                                          │
│   ALT-Y          ROTATE KILL RING AND YANK AGAIN                             │
│   ALT-\          SQUEEZE ADJACENT WHITESPACE                                 │
│   CTRL-T         TRANSPOSE                                                   │
│   ALT-T          TRANSPOSE WORD                                              │
│   ALT-U          UPPERCASE WORD                                              │
│   ALT-L          LOWERCASE WORD                                              │
│   ALT-C          CAPITALIZE WORD                                             │
│   CTRL-C CTRL-C  INTERRUPT PROCESS                                           │
│   CTRL-Z         SUSPEND PROCESS                                             │
│   CTRL-\         QUIT PROCESS                                                │
│   CTRL-S         PAUSE OUTPUT                                                │
│   CTRL-Q         UNPAUSE OUTPUT (IF PAUSED)                                  │
│   CTRL-Q         ESCAPED INSERT                                              │
│   CTRL-SPACE     SET MARK                                                    │
│   CTRL-X CTRL-X  GOTO MARK                                                   │
│   PROTIP         REMAP CAPS LOCK TO CTRL                                     │
│                                                                              │
╞══════════════════════════════════════════════════════════════════════════════╡
│                                                                              │
│ Copyright 2018-2021 Justine Tunney <jtunney@gmail.com>                       │
│ Copyright 2010-2016 Salvatore Sanfilippo <antirez@gmail.com>                 │
│ Copyright 2010-2013 Pieter Noordhuis <pcnoordhuis@gmail.com>                 │
│                                                                              │
│ All rights reserved.                                                         │
│                                                                              │
│ Redistribution and use in source and binary forms, with or without           │
│ modification, are permitted provided that the following conditions are       │
│ met:                                                                         │
│                                                                              │
│  *  Redistributions of source code must retain the above copyright           │
│     notice, this list of conditions and the following disclaimer.            │
│                                                                              │
│  *  Redistributions in binary form must reproduce the above copyright        │
│     notice, this list of conditions and the following disclaimer in the      │
│     documentation and/or other materials provided with the distribution.     │
│                                                                              │
│ THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS          │
│ "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT            │
│ LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR        │
│ A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT         │
│ HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,       │
│ SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT             │
│ LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,        │
│ DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY        │
│ THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT          │
│ (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE        │
│ OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.         │
│                                                                              │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "bestline.h"

#define _POSIX_C_SOURCE 1 /* so GCC builds in ANSI mode */
#define _XOPEN_SOURCE 700 /* so GCC builds in ANSI mode */
#define _DARWIN_C_SOURCE 1 /* so SIGWINCH / IUTF8 on XNU */
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>
#ifndef SIGWINCH
#define SIGWINCH 28 /* GNU/Systemd + XNU + FreeBSD + NetBSD + OpenBSD */
#endif
#ifndef IUTF8
#define IUTF8 0
#endif

__asm__(".ident\t\"\\n\\n\
Bestline (BSD-2)\\n\
Copyright 2018-2020 Justine Tunney <jtunney@gmail.com>\\n\
Copyright 2010-2016 Salvatore Sanfilippo <antirez@gmail.com>\\n\
Copyright 2010-2013 Pieter Noordhuis <pcnoordhuis@gmail.com>\"");

#ifndef BESTLINE_MAX_RING
#define BESTLINE_MAX_RING 8
#endif

#ifndef BESTLINE_MAX_HISTORY
#define BESTLINE_MAX_HISTORY 1024
#endif

#define BESTLINE_HISTORY_PREV +1
#define BESTLINE_HISTORY_NEXT -1

#define Ctrl(C) ((C) ^ 0100)
#define Min(X, Y) ((Y) > (X) ? (X) : (Y))
#define Max(X, Y) ((Y) < (X) ? (X) : (Y))
#define Case(X, Y) \
    case X: \
        Y; \
        break
#define Read16le(X) ((255 & (X)[0]) << 000 | (255 & (X)[1]) << 010)
#define Read32le(X) \
    ((unsigned)(255 & (X)[0]) << 000 | (unsigned)(255 & (X)[1]) << 010 | \
     (unsigned)(255 & (X)[2]) << 020 | (unsigned)(255 & (X)[3]) << 030)

struct abuf {
    char *b;
    unsigned len;
    unsigned cap;
};

struct rune {
    unsigned c;
    unsigned n;
};

struct bestlineRing {
    unsigned i;
    char *p[BESTLINE_MAX_RING];
};

/* The bestlineState structure represents the state during line editing.
 * We pass this state to functions implementing specific editing
 * functionalities. */
struct bestlineState {
    int ifd; /* terminal stdin file descriptor */
    int ofd; /* terminal stdout file descriptor */
    struct winsize ws; /* rows and columns in terminal */
    char *buf; /* edited line buffer */
    const char *prompt; /* prompt to display */
    int hindex; /* history index */
    int rows; /* rows being used */
    int oldpos; /* previous refresh cursor position */
    unsigned buflen; /* edited line buffer size */
    unsigned pos; /* current buffer index */
    unsigned len; /* current edited line length */
    unsigned mark; /* saved cursor position */
    unsigned yi, yj; /* boundaries of last yank */
    char seq[2][16]; /* keystroke history for yanking code */
    char final; /* set to true on last update */
    char dirty; /* if an update was squashed */
    struct abuf full; /* used for multiline mode */
};

static const char *const kUnsupported[] = {"dumb", "cons25", "emacs"};

static int gotint;
static int gotcont;
static int gotwinch;
static signed char rawmode;
static char maskmode;
static char emacsmode;
static char llamamode;
static char balancemode;
static char ispaused;
static char iscapital;
static unsigned historylen;
static struct bestlineRing ring;
static struct sigaction orig_cont;
static struct sigaction orig_winch;
static struct termios orig_termios;
static char *history[BESTLINE_MAX_HISTORY];
static bestlineXlatCallback *xlatCallback;
static bestlineHintsCallback *hintsCallback;
static bestlineFreeHintsCallback *freeHintsCallback;
static bestlineCompletionCallback *completionCallback;

static void bestlineAtExit(void);
static void bestlineRefreshLine(struct bestlineState *);

static void bestlineOnInt(int sig) {
    gotint = sig;
}

static void bestlineOnCont(int sig) {
    gotcont = sig;
}

static void bestlineOnWinch(int sig) {
    gotwinch = sig;
}

static char IsControl(unsigned c) {
    return c <= 0x1F || (0x7F <= c && c <= 0x9F);
}

static int GetMonospaceCharacterWidth(unsigned c) {
    return !IsControl(c) +
           (c >= 0x1100 && (c <= 0x115f || c == 0x2329 || c == 0x232a ||
                            (c >= 0x2e80 && c <= 0xa4cf && c != 0x303f) ||
                            (c >= 0xac00 && c <= 0xd7a3) || (c >= 0xf900 && c <= 0xfaff) ||
                            (c >= 0xfe10 && c <= 0xfe19) || (c >= 0xfe30 && c <= 0xfe6f) ||
                            (c >= 0xff00 && c <= 0xff60) || (c >= 0xffe0 && c <= 0xffe6) ||
                            (c >= 0x20000 && c <= 0x2fffd) || (c >= 0x30000 && c <= 0x3fffd)));
}

/**
 * Returns nonzero if 𝑐 isn't alphanumeric.
 *
 * Line reading interfaces generally define this operation as UNICODE
 * characters that aren't in the letter category (Lu, Ll, Lt, Lm, Lo)
 * and aren't in the number categorie (Nd, Nl, No). We also add a few
 * other things like blocks and emoji (So).
 */
char bestlineIsSeparator(unsigned c) {
    int m, l, r, n;
    if (c < 0200) {
        return !(('0' <= c && c <= '9') || ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z'));
    }
    if (c <= 0xffff) {
        static const unsigned short kGlyphs[][2] = {
            {0x00aa, 0x00aa}, /*     1x English */
            {0x00b2, 0x00b3}, /*     2x English Arabic */
            {0x00b5, 0x00b5}, /*     1x Greek */
            {0x00b9, 0x00ba}, /*     2x English Arabic */
            {0x00bc, 0x00be}, /*     3x Vulgar English Arabic */
            {0x00c0, 0x00d6}, /*    23x Watin */
            {0x00d8, 0x00f6}, /*    31x Watin */
            {0x0100, 0x02c1}, /*   450x Watin-AB,IPA,Spacemod */
            {0x02c6, 0x02d1}, /*    12x Spacemod */
            {0x02e0, 0x02e4}, /*     5x Spacemod */
            {0x02ec, 0x02ec}, /*     1x Spacemod */
            {0x02ee, 0x02ee}, /*     1x Spacemod */
            {0x0370, 0x0374}, /*     5x Greek */
            {0x0376, 0x0377}, /*     2x Greek */
            {0x037a, 0x037d}, /*     4x Greek */
            {0x037f, 0x037f}, /*     1x Greek */
            {0x0386, 0x0386}, /*     1x Greek */
            {0x0388, 0x038a}, /*     3x Greek */
            {0x038c, 0x038c}, /*     1x Greek */
            {0x038e, 0x03a1}, /*    20x Greek */
            {0x03a3, 0x03f5}, /*    83x Greek */
            {0x03f7, 0x0481}, /*   139x Greek */
            {0x048a, 0x052f}, /*   166x Cyrillic */
            {0x0531, 0x0556}, /*    38x Armenian */
            {0x0560, 0x0588}, /*    41x Armenian */
            {0x05d0, 0x05ea}, /*    27x Hebrew */
            {0x0620, 0x064a}, /*    43x Arabic */
            {0x0660, 0x0669}, /*    10x Arabic */
            {0x0671, 0x06d3}, /*    99x Arabic */
            {0x06ee, 0x06fc}, /*    15x Arabic */
            {0x0712, 0x072f}, /*    30x Syriac */
            {0x074d, 0x07a5}, /*    89x Syriac,Arabic2,Thaana */
            {0x07c0, 0x07ea}, /*    43x NKo */
            {0x0800, 0x0815}, /*    22x Samaritan */
            {0x0840, 0x0858}, /*    25x Mandaic */
            {0x0904, 0x0939}, /*    54x Devanagari */
            {0x0993, 0x09a8}, /*    22x Bengali */
            {0x09e6, 0x09f1}, /*    12x Bengali */
            {0x0a13, 0x0a28}, /*    22x Gurmukhi */
            {0x0a66, 0x0a6f}, /*    10x Gurmukhi */
            {0x0a93, 0x0aa8}, /*    22x Gujarati */
            {0x0b13, 0x0b28}, /*    22x Oriya */
            {0x0c92, 0x0ca8}, /*    23x Kannada */
            {0x0caa, 0x0cb3}, /*    10x Kannada */
            {0x0ce6, 0x0cef}, /*    10x Kannada */
            {0x0d12, 0x0d3a}, /*    41x Malayalam */
            {0x0d85, 0x0d96}, /*    18x Sinhala */
            {0x0d9a, 0x0db1}, /*    24x Sinhala */
            {0x0de6, 0x0def}, /*    10x Sinhala */
            {0x0e01, 0x0e30}, /*    48x Thai */
            {0x0e8c, 0x0ea3}, /*    24x Lao */
            {0x0f20, 0x0f33}, /*    20x Tibetan */
            {0x0f49, 0x0f6c}, /*    36x Tibetan */
            {0x109e, 0x10c5}, /*    40x Myanmar,Georgian */
            {0x10d0, 0x10fa}, /*    43x Georgian */
            {0x10fc, 0x1248}, /*   333x Georgian,Hangul,Ethiopic */
            {0x13a0, 0x13f5}, /*    86x Cherokee */
            {0x1401, 0x166d}, /*   621x Aboriginal */
            {0x16a0, 0x16ea}, /*    75x Runic */
            {0x1700, 0x170c}, /*    13x Tagalog */
            {0x1780, 0x17b3}, /*    52x Khmer */
            {0x1820, 0x1878}, /*    89x Mongolian */
            {0x1a00, 0x1a16}, /*    23x Buginese */
            {0x1a20, 0x1a54}, /*    53x Tai Tham */
            {0x1a80, 0x1a89}, /*    10x Tai Tham */
            {0x1a90, 0x1a99}, /*    10x Tai Tham */
            {0x1b05, 0x1b33}, /*    47x Balinese */
            {0x1b50, 0x1b59}, /*    10x Balinese */
            {0x1b83, 0x1ba0}, /*    30x Sundanese */
            {0x1bae, 0x1be5}, /*    56x Sundanese */
            {0x1c90, 0x1cba}, /*    43x Georgian2 */
            {0x1cbd, 0x1cbf}, /*     3x Georgian2 */
            {0x1e00, 0x1f15}, /*   278x Watin-C,Greek2 */
            {0x2070, 0x2071}, /*     2x Supersub */
            {0x2074, 0x2079}, /*     6x Supersub */
            {0x207f, 0x2089}, /*    11x Supersub */
            {0x2090, 0x209c}, /*    13x Supersub */
            {0x2100, 0x2117}, /*    24x Letterlike */
            {0x2119, 0x213f}, /*    39x Letterlike */
            {0x2145, 0x214a}, /*     6x Letterlike */
            {0x214c, 0x218b}, /*    64x Letterlike,Numbery */
            {0x21af, 0x21cd}, /*    31x Arrows */
            {0x21d5, 0x21f3}, /*    31x Arrows */
            {0x230c, 0x231f}, /*    20x Technical */
            {0x232b, 0x237b}, /*    81x Technical */
            {0x237d, 0x239a}, /*    30x Technical */
            {0x23b4, 0x23db}, /*    40x Technical */
            {0x23e2, 0x2426}, /*    69x Technical,ControlPictures */
            {0x2460, 0x25b6}, /*   343x Enclosed,Boxes,Blocks,Shapes */
            {0x25c2, 0x25f7}, /*    54x Shapes */
            {0x2600, 0x266e}, /*   111x Symbols */
            {0x2670, 0x2767}, /*   248x Symbols,Dingbats */
            {0x2776, 0x27bf}, /*    74x Dingbats */
            {0x2800, 0x28ff}, /*   256x Braille */
            {0x2c00, 0x2c2e}, /*    47x Glagolitic */
            {0x2c30, 0x2c5e}, /*    47x Glagolitic */
            {0x2c60, 0x2ce4}, /*   133x Watin-D */
            {0x2d00, 0x2d25}, /*    38x Georgian2 */
            {0x2d30, 0x2d67}, /*    56x Tifinagh */
            {0x2d80, 0x2d96}, /*    23x Ethiopic2 */
            {0x2e2f, 0x2e2f}, /*     1x Punctuation2 */
            {0x3005, 0x3007}, /*     3x CJK Symbols & Punctuation */
            {0x3021, 0x3029}, /*     9x CJK Symbols & Punctuation */
            {0x3031, 0x3035}, /*     5x CJK Symbols & Punctuation */
            {0x3038, 0x303c}, /*     5x CJK Symbols & Punctuation */
            {0x3041, 0x3096}, /*    86x Hiragana */
            {0x30a1, 0x30fa}, /*    90x Katakana */
            {0x3105, 0x312f}, /*    43x Bopomofo */
            {0x3131, 0x318e}, /*    94x Hangul Compatibility Jamo */
            {0x31a0, 0x31ba}, /*    27x Bopomofo Extended */
            {0x31f0, 0x31ff}, /*    16x Katakana Phonetic Extensions */
            {0x3220, 0x3229}, /*    10x Enclosed CJK Letters & Months */
            {0x3248, 0x324f}, /*     8x Enclosed CJK Letters & Months */
            {0x3251, 0x325f}, /*    15x Enclosed CJK Letters & Months */
            {0x3280, 0x3289}, /*    10x Enclosed CJK Letters & Months */
            {0x32b1, 0x32bf}, /*    15x Enclosed CJK Letters & Months */
            {0x3400, 0x4db5}, /*  6582x CJK Unified Ideographs Extension A */
            {0x4dc0, 0x9fef}, /* 21040x Yijing Hexagram, CJK Unified Ideographs */
            {0xa000, 0xa48c}, /*  1165x Yi Syllables */
            {0xa4d0, 0xa4fd}, /*    46x Lisu */
            {0xa500, 0xa60c}, /*   269x Vai */
            {0xa610, 0xa62b}, /*    28x Vai */
            {0xa6a0, 0xa6ef}, /*    80x Bamum */
            {0xa80c, 0xa822}, /*    23x Syloti Nagri */
            {0xa840, 0xa873}, /*    52x Phags-pa */
            {0xa882, 0xa8b3}, /*    50x Saurashtra */
            {0xa8d0, 0xa8d9}, /*    10x Saurashtra */
            {0xa900, 0xa925}, /*    38x Kayah Li */
            {0xa930, 0xa946}, /*    23x Rejang */
            {0xa960, 0xa97c}, /*    29x Hangul Jamo Extended-A */
            {0xa984, 0xa9b2}, /*    47x Javanese */
            {0xa9cf, 0xa9d9}, /*    11x Javanese */
            {0xaa00, 0xaa28}, /*    41x Cham */
            {0xaa50, 0xaa59}, /*    10x Cham */
            {0xabf0, 0xabf9}, /*    10x Meetei Mayek */
            {0xac00, 0xd7a3}, /* 11172x Hangul Syllables */
            {0xf900, 0xfa6d}, /*   366x CJK Compatibility Ideographs */
            {0xfa70, 0xfad9}, /*   106x CJK Compatibility Ideographs */
            {0xfb1f, 0xfb28}, /*    10x Alphabetic Presentation Forms */
            {0xfb2a, 0xfb36}, /*    13x Alphabetic Presentation Forms */
            {0xfb46, 0xfbb1}, /*   108x Alphabetic Presentation Forms */
            {0xfbd3, 0xfd3d}, /*   363x Arabic Presentation Forms-A */
            {0xfe76, 0xfefc}, /*   135x Arabic Presentation Forms-B */
            {0xff10, 0xff19}, /*    10x Dubs */
            {0xff21, 0xff3a}, /*    26x Dubs */
            {0xff41, 0xff5a}, /*    26x Dubs */
            {0xff66, 0xffbe}, /*    89x Dubs */
            {0xffc2, 0xffc7}, /*     6x Dubs */
            {0xffca, 0xffcf}, /*     6x Dubs */
            {0xffd2, 0xffd7}, /*     6x Dubs */
            {0xffda, 0xffdc}, /*     3x Dubs */
        };
        l = 0;
        r = n = sizeof(kGlyphs) / sizeof(kGlyphs[0]);
        while (l < r) {
            m = (l + r) >> 1;
            if (kGlyphs[m][1] < c) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return !(l < n && kGlyphs[l][0] <= c && c <= kGlyphs[l][1]);
    } else {
        static const unsigned kAstralGlyphs[][2] = {
            {0x10107, 0x10133}, /*    45x Aegean */
            {0x10140, 0x10178}, /*    57x Ancient Greek Numbers */
            {0x1018a, 0x1018b}, /*     2x Ancient Greek Numbers */
            {0x10280, 0x1029c}, /*    29x Lycian */
            {0x102a0, 0x102d0}, /*    49x Carian */
            {0x102e1, 0x102fb}, /*    27x Coptic Epact Numbers */
            {0x10300, 0x10323}, /*    36x Old Italic */
            {0x1032d, 0x1034a}, /*    30x Old Italic, Gothic */
            {0x10350, 0x10375}, /*    38x Old Permic */
            {0x10380, 0x1039d}, /*    30x Ugaritic */
            {0x103a0, 0x103c3}, /*    36x Old Persian */
            {0x103c8, 0x103cf}, /*     8x Old Persian */
            {0x103d1, 0x103d5}, /*     5x Old Persian */
            {0x10400, 0x1049d}, /*    158x Deseret, Shavian, Osmanya */
            {0x104b0, 0x104d3}, /*    36x Osage */
            {0x104d8, 0x104fb}, /*    36x Osage */
            {0x10500, 0x10527}, /*    40x Elbasan */
            {0x10530, 0x10563}, /*    52x Caucasian Albanian */
            {0x10600, 0x10736}, /*   311x Linear A */
            {0x10800, 0x10805}, /*     6x Cypriot Syllabary */
            {0x1080a, 0x10835}, /*    44x Cypriot Syllabary */
            {0x10837, 0x10838}, /*     2x Cypriot Syllabary */
            {0x1083f, 0x1089e}, /*    86x Cypriot,ImperialAramaic,Palmyrene,Nabataean */
            {0x108e0, 0x108f2}, /*    19x Hatran */
            {0x108f4, 0x108f5}, /*     2x Hatran */
            {0x108fb, 0x1091b}, /*    33x Hatran */
            {0x10920, 0x10939}, /*    26x Lydian */
            {0x10980, 0x109b7}, /*    56x Meroitic Hieroglyphs */
            {0x109bc, 0x109cf}, /*    20x Meroitic Cursive */
            {0x109d2, 0x10a00}, /*    47x Meroitic Cursive */
            {0x10a10, 0x10a13}, /*     4x Kharoshthi */
            {0x10a15, 0x10a17}, /*     3x Kharoshthi */
            {0x10a19, 0x10a35}, /*    29x Kharoshthi */
            {0x10a40, 0x10a48}, /*     9x Kharoshthi */
            {0x10a60, 0x10a7e}, /*    31x Old South Arabian */
            {0x10a80, 0x10a9f}, /*    32x Old North Arabian */
            {0x10ac0, 0x10ac7}, /*     8x Manichaean */
            {0x10ac9, 0x10ae4}, /*    28x Manichaean */
            {0x10aeb, 0x10aef}, /*     5x Manichaean */
            {0x10b00, 0x10b35}, /*    54x Avestan */
            {0x10b40, 0x10b55}, /*    22x Inscriptional Parthian */
            {0x10b58, 0x10b72}, /*    27x Inscriptional Parthian and Pahlavi */
            {0x10b78, 0x10b91}, /*    26x Inscriptional Pahlavi, Psalter Pahlavi */
            {0x10c00, 0x10c48}, /*    73x Old Turkic */
            {0x10c80, 0x10cb2}, /*    51x Old Hungarian */
            {0x10cc0, 0x10cf2}, /*    51x Old Hungarian */
            {0x10cfa, 0x10d23}, /*    42x Old Hungarian, Hanifi Rohingya */
            {0x10d30, 0x10d39}, /*    10x Hanifi Rohingya */
            {0x10e60, 0x10e7e}, /*    31x Rumi Numeral Symbols */
            {0x10f00, 0x10f27}, /*    40x Old Sogdian */
            {0x10f30, 0x10f45}, /*    22x Sogdian */
            {0x10f51, 0x10f54}, /*     4x Sogdian */
            {0x10fe0, 0x10ff6}, /*    23x Elymaic */
            {0x11003, 0x11037}, /*    53x Brahmi */
            {0x11052, 0x1106f}, /*    30x Brahmi */
            {0x11083, 0x110af}, /*    45x Kaithi */
            {0x110d0, 0x110e8}, /*    25x Sora Sompeng */
            {0x110f0, 0x110f9}, /*    10x Sora Sompeng */
            {0x11103, 0x11126}, /*    36x Chakma */
            {0x11136, 0x1113f}, /*    10x Chakma */
            {0x11144, 0x11144}, /*     1x Chakma */
            {0x11150, 0x11172}, /*    35x Mahajani */
            {0x11176, 0x11176}, /*     1x Mahajani */
            {0x11183, 0x111b2}, /*    48x Sharada */
            {0x111c1, 0x111c4}, /*     4x Sharada */
            {0x111d0, 0x111da}, /*    11x Sharada */
            {0x111dc, 0x111dc}, /*     1x Sharada */
            {0x111e1, 0x111f4}, /*    20x Sinhala Archaic Numbers */
            {0x11200, 0x11211}, /*    18x Khojki */
            {0x11213, 0x1122b}, /*    25x Khojki */
            {0x11280, 0x11286}, /*     7x Multani */
            {0x11288, 0x11288}, /*     1x Multani */
            {0x1128a, 0x1128d}, /*     4x Multani */
            {0x1128f, 0x1129d}, /*    15x Multani */
            {0x1129f, 0x112a8}, /*    10x Multani */
            {0x112b0, 0x112de}, /*    47x Khudawadi */
            {0x112f0, 0x112f9}, /*    10x Khudawadi */
            {0x11305, 0x1130c}, /*     8x Grantha */
            {0x1130f, 0x11310}, /*     2x Grantha */
            {0x11313, 0x11328}, /*    22x Grantha */
            {0x1132a, 0x11330}, /*     7x Grantha */
            {0x11332, 0x11333}, /*     2x Grantha */
            {0x11335, 0x11339}, /*     5x Grantha */
            {0x1133d, 0x1133d}, /*     1x Grantha */
            {0x11350, 0x11350}, /*     1x Grantha */
            {0x1135d, 0x11361}, /*     5x Grantha */
            {0x11400, 0x11434}, /*    53x Newa */
            {0x11447, 0x1144a}, /*     4x Newa */
            {0x11450, 0x11459}, /*    10x Newa */
            {0x1145f, 0x1145f}, /*     1x Newa */
            {0x11480, 0x114af}, /*    48x Tirhuta */
            {0x114c4, 0x114c5}, /*     2x Tirhuta */
            {0x114c7, 0x114c7}, /*     1x Tirhuta */
            {0x114d0, 0x114d9}, /*    10x Tirhuta */
            {0x11580, 0x115ae}, /*    47x Siddham */
            {0x115d8, 0x115db}, /*     4x Siddham */
            {0x11600, 0x1162f}, /*    48x Modi */
            {0x11644, 0x11644}, /*     1x Modi */
            {0x11650, 0x11659}, /*    10x Modi */
            {0x11680, 0x116aa}, /*    43x Takri */
            {0x116b8, 0x116b8}, /*     1x Takri */
            {0x116c0, 0x116c9}, /*    10x Takri */
            {0x11700, 0x1171a}, /*    27x Ahom */
            {0x11730, 0x1173b}, /*    12x Ahom */
            {0x11800, 0x1182b}, /*    44x Dogra */
            {0x118a0, 0x118f2}, /*    83x Warang Citi */
            {0x118ff, 0x118ff}, /*     1x Warang Citi */
            {0x119a0, 0x119a7}, /*     8x Nandinagari */
            {0x119aa, 0x119d0}, /*    39x Nandinagari */
            {0x119e1, 0x119e1}, /*     1x Nandinagari */
            {0x119e3, 0x119e3}, /*     1x Nandinagari */
            {0x11a00, 0x11a00}, /*     1x Zanabazar Square */
            {0x11a0b, 0x11a32}, /*    40x Zanabazar Square */
            {0x11a3a, 0x11a3a}, /*     1x Zanabazar Square */
            {0x11a50, 0x11a50}, /*     1x Soyombo */
            {0x11a5c, 0x11a89}, /*    46x Soyombo */
            {0x11a9d, 0x11a9d}, /*     1x Soyombo */
            {0x11ac0, 0x11af8}, /*    57x Pau Cin Hau */
            {0x11c00, 0x11c08}, /*     9x Bhaiksuki */
            {0x11c0a, 0x11c2e}, /*    37x Bhaiksuki */
            {0x11c40, 0x11c40}, /*     1x Bhaiksuki */
            {0x11c50, 0x11c6c}, /*    29x Bhaiksuki */
            {0x11c72, 0x11c8f}, /*    30x Marchen */
            {0x11d00, 0x11d06}, /*     7x Masaram Gondi */
            {0x11d08, 0x11d09}, /*     2x Masaram Gondi */
            {0x11d0b, 0x11d30}, /*    38x Masaram Gondi */
            {0x11d46, 0x11d46}, /*     1x Masaram Gondi */
            {0x11d50, 0x11d59}, /*    10x Masaram Gondi */
            {0x11d60, 0x11d65}, /*     6x Gunjala Gondi */
            {0x11d67, 0x11d68}, /*     2x Gunjala Gondi */
            {0x11d6a, 0x11d89}, /*    32x Gunjala Gondi */
            {0x11d98, 0x11d98}, /*     1x Gunjala Gondi */
            {0x11da0, 0x11da9}, /*    10x Gunjala Gondi */
            {0x11ee0, 0x11ef2}, /*    19x Makasar */
            {0x11fc0, 0x11fd4}, /*    21x Tamil Supplement */
            {0x12000, 0x12399}, /*   922x Cuneiform */
            {0x12400, 0x1246e}, /*   111x Cuneiform Numbers & Punctuation */
            {0x12480, 0x12543}, /*   196x Early Dynastic Cuneiform */
            {0x13000, 0x1342e}, /*  1071x Egyptian Hieroglyphs */
            {0x14400, 0x14646}, /*   583x Anatolian Hieroglyphs */
            {0x16800, 0x16a38}, /*   569x Bamum Supplement */
            {0x16a40, 0x16a5e}, /*    31x Mro */
            {0x16a60, 0x16a69}, /*    10x Mro */
            {0x16ad0, 0x16aed}, /*    30x Bassa Vah */
            {0x16b00, 0x16b2f}, /*    48x Pahawh Hmong */
            {0x16b40, 0x16b43}, /*     4x Pahawh Hmong */
            {0x16b50, 0x16b59}, /*    10x Pahawh Hmong */
            {0x16b5b, 0x16b61}, /*     7x Pahawh Hmong */
            {0x16b63, 0x16b77}, /*    21x Pahawh Hmong */
            {0x16b7d, 0x16b8f}, /*    19x Pahawh Hmong */
            {0x16e40, 0x16e96}, /*    87x Medefaidrin */
            {0x16f00, 0x16f4a}, /*    75x Miao */
            {0x16f50, 0x16f50}, /*     1x Miao */
            {0x16f93, 0x16f9f}, /*    13x Miao */
            {0x16fe0, 0x16fe1}, /*     2x Ideographic Symbols & Punctuation */
            {0x16fe3, 0x16fe3}, /*     1x Ideographic Symbols & Punctuation */
            {0x17000, 0x187f7}, /*  6136x Tangut */
            {0x18800, 0x18af2}, /*   755x Tangut Components */
            {0x1b000, 0x1b11e}, /*   287x Kana Supplement */
            {0x1b150, 0x1b152}, /*     3x Small Kana Extension */
            {0x1b164, 0x1b167}, /*     4x Small Kana Extension */
            {0x1b170, 0x1b2fb}, /*   396x Nushu */
            {0x1bc00, 0x1bc6a}, /*   107x Duployan */
            {0x1bc70, 0x1bc7c}, /*    13x Duployan */
            {0x1bc80, 0x1bc88}, /*     9x Duployan */
            {0x1bc90, 0x1bc99}, /*    10x Duployan */
            {0x1d2e0, 0x1d2f3}, /*    20x Mayan Numerals */
            {0x1d360, 0x1d378}, /*    25x Counting Rod Numerals */
            {0x1d400, 0x1d454}, /*    85x 𝐀..𝑔 Math */
            {0x1d456, 0x1d49c}, /*    71x 𝑖..𝒜 Math */
            {0x1d49e, 0x1d49f}, /*     2x 𝒞..𝒟 Math */
            {0x1d4a2, 0x1d4a2}, /*     1x 𝒢..𝒢 Math */
            {0x1d4a5, 0x1d4a6}, /*     2x 𝒥..𝒦 Math */
            {0x1d4a9, 0x1d4ac}, /*     4x 𝒩..𝒬 Math */
            {0x1d4ae, 0x1d4b9}, /*    12x 𝒮..𝒹 Math */
            {0x1d4bb, 0x1d4bb}, /*     1x 𝒻..𝒻 Math */
            {0x1d4bd, 0x1d4c3}, /*     7x 𝒽..𝓃 Math */
            {0x1d4c5, 0x1d505}, /*    65x 𝓅..𝔅 Math */
            {0x1d507, 0x1d50a}, /*     4x 𝔇..𝔊 Math */
            {0x1d50d, 0x1d514}, /*     8x 𝔍..𝔔 Math */
            {0x1d516, 0x1d51c}, /*     7x 𝔖..𝔜 Math */
            {0x1d51e, 0x1d539}, /*    28x 𝔞..𝔹 Math */
            {0x1d53b, 0x1d53e}, /*     4x 𝔻..𝔾 Math */
            {0x1d540, 0x1d544}, /*     5x 𝕀..𝕄 Math */
            {0x1d546, 0x1d546}, /*     1x 𝕆..𝕆 Math */
            {0x1d54a, 0x1d550}, /*     7x 𝕊..𝕐 Math */
            {0x1d552, 0x1d6a5}, /*   340x 𝕒..𝚥 Math */
            {0x1d6a8, 0x1d6c0}, /*    25x 𝚨..𝛀 Math */
            {0x1d6c2, 0x1d6da}, /*    25x 𝛂..𝛚 Math */
            {0x1d6dc, 0x1d6fa}, /*    31x 𝛜..𝛺 Math */
            {0x1d6fc, 0x1d714}, /*    25x 𝛼..𝜔 Math */
            {0x1d716, 0x1d734}, /*    31x 𝜖..𝜴 Math */
            {0x1d736, 0x1d74e}, /*    25x 𝜶..𝝎 Math */
            {0x1d750, 0x1d76e}, /*    31x 𝝐..𝝮 Math */
            {0x1d770, 0x1d788}, /*    25x 𝝰..𝞈 Math */
            {0x1d78a, 0x1d7a8}, /*    31x 𝞊..𝞨 Math */
            {0x1d7aa, 0x1d7c2}, /*    25x 𝞪..𝟂 Math */
            {0x1d7c4, 0x1d7cb}, /*     8x 𝟄..𝟋 Math */
            {0x1d7ce, 0x1d9ff}, /*   562x Math, Sutton SignWriting */
            {0x1f100, 0x1f10c}, /*    13x Enclosed Alphanumeric Supplement */
            {0x20000, 0x2a6d6}, /* 42711x CJK Unified Ideographs Extension B */
            {0x2a700, 0x2b734}, /*  4149x CJK Unified Ideographs Extension C */
            {0x2b740, 0x2b81d}, /*   222x CJK Unified Ideographs Extension D */
            {0x2b820, 0x2cea1}, /*  5762x CJK Unified Ideographs Extension E */
            {0x2ceb0, 0x2ebe0}, /*  7473x CJK Unified Ideographs Extension F */
            {0x2f800, 0x2fa1d}, /*   542x CJK Compatibility Ideographs Supplement */
        };
        l = 0;
        r = n = sizeof(kAstralGlyphs) / sizeof(kAstralGlyphs[0]);
        while (l < r) {
            m = (l + r) >> 1;
            if (kAstralGlyphs[m][1] < c) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return !(l < n && kAstralGlyphs[l][0] <= c && c <= kAstralGlyphs[l][1]);
    }
}

unsigned bestlineLowercase(unsigned c) {
    int m, l, r, n;
    if (c < 0200) {
        if ('A' <= c && c <= 'Z') {
            return c + 32;
        } else {
            return c;
        }
    } else if (c <= 0xffff) {
        if ((0x0100 <= c && c <= 0x0176) || /* 60x Ā..ā → ā..ŵ Watin-A */
            (0x01de <= c && c <= 0x01ee) || /*  9x Ǟ..Ǯ → ǟ..ǯ Watin-B */
            (0x01f8 <= c && c <= 0x021e) || /* 20x Ǹ..Ȟ → ǹ..ȟ Watin-B */
            (0x0222 <= c && c <= 0x0232) || /*  9x Ȣ..Ȳ → ȣ..ȳ Watin-B */
            (0x1e00 <= c && c <= 0x1eff)) { /*256x Ḁ..Ỿ → ḁ..ỿ Watin-C */
            if (c == 0x0130)
                return c - 199;
            if (c == 0x1e9e)
                return c;
            return c + (~c & 1);
        } else if (0x01cf <= c && c <= 0x01db) {
            return c + (c & 1); /* 7x Ǐ..Ǜ → ǐ..ǜ Watin-B */
        } else if (0x13a0 <= c && c <= 0x13ef) {
            return c + 38864; /* 80x Ꭰ ..Ꮿ  → ꭰ ..ꮿ  Cherokee */
        } else {
            static const struct {
                unsigned short a;
                unsigned short b;
                short d;
            } kLower[] = {
                {0x00c0, 0x00d6, +32}, /* 23x À ..Ö  → à ..ö  Watin */
                {0x00d8, 0x00de, +32}, /*  7x Ø ..Þ  → ø ..þ  Watin */
                {0x0178, 0x0178, -121}, /*  1x Ÿ ..Ÿ  → ÿ ..ÿ  Watin-A */
                {0x0179, 0x0179, +1}, /*  1x Ź ..Ź  → ź ..ź  Watin-A */
                {0x017b, 0x017b, +1}, /*  1x Ż ..Ż  → ż ..ż  Watin-A */
                {0x017d, 0x017d, +1}, /*  1x Ž ..Ž  → ž ..ž  Watin-A */
                {0x0181, 0x0181, +210}, /*  1x Ɓ ..Ɓ  → ɓ ..ɓ  Watin-B */
                {0x0182, 0x0182, +1}, /*  1x Ƃ ..Ƃ  → ƃ ..ƃ  Watin-B */
                {0x0184, 0x0184, +1}, /*  1x Ƅ ..Ƅ  → ƅ ..ƅ  Watin-B */
                {0x0186, 0x0186, +206}, /*  1x Ɔ ..Ɔ  → ɔ ..ɔ  Watin-B */
                {0x0187, 0x0187, +1}, /*  1x Ƈ ..Ƈ  → ƈ ..ƈ  Watin-B */
                {0x0189, 0x018a, +205}, /*  2x Ɖ ..Ɗ  → ɖ ..ɗ  Watin-B */
                {0x018b, 0x018b, +1}, /*  1x Ƌ ..Ƌ  → ƌ ..ƌ  Watin-B */
                {0x018e, 0x018e, +79}, /*  1x Ǝ ..Ǝ  → ǝ ..ǝ  Watin-B */
                {0x018f, 0x018f, +202}, /*  1x Ə ..Ə  → ə ..ə  Watin-B */
                {0x0190, 0x0190, +203}, /*  1x Ɛ ..Ɛ  → ɛ ..ɛ  Watin-B */
                {0x0191, 0x0191, +1}, /*  1x Ƒ ..Ƒ  → ƒ ..ƒ  Watin-B */
                {0x0193, 0x0193, +205}, /*  1x Ɠ ..Ɠ  → ɠ ..ɠ  Watin-B */
                {0x0194, 0x0194, +207}, /*  1x Ɣ ..Ɣ  → ɣ ..ɣ  Watin-B */
                {0x0196, 0x0196, +211}, /*  1x Ɩ ..Ɩ  → ɩ ..ɩ  Watin-B */
                {0x0197, 0x0197, +209}, /*  1x Ɨ ..Ɨ  → ɨ ..ɨ  Watin-B */
                {0x0198, 0x0198, +1}, /*  1x Ƙ ..Ƙ  → ƙ ..ƙ  Watin-B */
                {0x019c, 0x019c, +211}, /*  1x Ɯ ..Ɯ  → ɯ ..ɯ  Watin-B */
                {0x019d, 0x019d, +213}, /*  1x Ɲ ..Ɲ  → ɲ ..ɲ  Watin-B */
                {0x019f, 0x019f, +214}, /*  1x Ɵ ..Ɵ  → ɵ ..ɵ  Watin-B */
                {0x01a0, 0x01a0, +1}, /*  1x Ơ ..Ơ  → ơ ..ơ  Watin-B */
                {0x01a2, 0x01a2, +1}, /*  1x Ƣ ..Ƣ  → ƣ ..ƣ  Watin-B */
                {0x01a4, 0x01a4, +1}, /*  1x Ƥ ..Ƥ  → ƥ ..ƥ  Watin-B */
                {0x01a6, 0x01a6, +218}, /*  1x Ʀ ..Ʀ  → ʀ ..ʀ  Watin-B */
                {0x01a7, 0x01a7, +1}, /*  1x Ƨ ..Ƨ  → ƨ ..ƨ  Watin-B */
                {0x01a9, 0x01a9, +218}, /*  1x Ʃ ..Ʃ  → ʃ ..ʃ  Watin-B */
                {0x01ac, 0x01ac, +1}, /*  1x Ƭ ..Ƭ  → ƭ ..ƭ  Watin-B */
                {0x01ae, 0x01ae, +218}, /*  1x Ʈ ..Ʈ  → ʈ ..ʈ  Watin-B */
                {0x01af, 0x01af, +1}, /*  1x Ư ..Ư  → ư ..ư  Watin-B */
                {0x01b1, 0x01b2, +217}, /*  2x Ʊ ..Ʋ  → ʊ ..ʋ  Watin-B */
                {0x01b3, 0x01b3, +1}, /*  1x Ƴ ..Ƴ  → ƴ ..ƴ  Watin-B */
                {0x01b5, 0x01b5, +1}, /*  1x Ƶ ..Ƶ  → ƶ ..ƶ  Watin-B */
                {0x01b7, 0x01b7, +219}, /*  1x Ʒ ..Ʒ  → ʒ ..ʒ  Watin-B */
                {0x01b8, 0x01b8, +1}, /*  1x Ƹ ..Ƹ  → ƹ ..ƹ  Watin-B */
                {0x01bc, 0x01bc, +1}, /*  1x Ƽ ..Ƽ  → ƽ ..ƽ  Watin-B */
                {0x01c4, 0x01c4, +2}, /*  1x Ǆ ..Ǆ  → ǆ ..ǆ  Watin-B */
                {0x01c5, 0x01c5, +1}, /*  1x ǅ ..ǅ  → ǆ ..ǆ  Watin-B */
                {0x01c7, 0x01c7, +2}, /*  1x Ǉ ..Ǉ  → ǉ ..ǉ  Watin-B */
                {0x01c8, 0x01c8, +1}, /*  1x ǈ ..ǈ  → ǉ ..ǉ  Watin-B */
                {0x01ca, 0x01ca, +2}, /*  1x Ǌ ..Ǌ  → ǌ ..ǌ  Watin-B */
                {0x01cb, 0x01cb, +1}, /*  1x ǋ ..ǋ  → ǌ ..ǌ  Watin-B */
                {0x01cd, 0x01cd, +1}, /*  1x Ǎ ..Ǎ  → ǎ ..ǎ  Watin-B */
                {0x01f1, 0x01f1, +2}, /*  1x Ǳ ..Ǳ  → ǳ ..ǳ  Watin-B */
                {0x01f2, 0x01f2, +1}, /*  1x ǲ ..ǲ  → ǳ ..ǳ  Watin-B */
                {0x01f4, 0x01f4, +1}, /*  1x Ǵ ..Ǵ  → ǵ ..ǵ  Watin-B */
                {0x01f6, 0x01f6, -97}, /*  1x Ƕ ..Ƕ  → ƕ ..ƕ  Watin-B */
                {0x01f7, 0x01f7, -56}, /*  1x Ƿ ..Ƿ  → ƿ ..ƿ  Watin-B */
                {0x0220, 0x0220, -130}, /*  1x Ƞ ..Ƞ  → ƞ ..ƞ  Watin-B */
                {0x023b, 0x023b, +1}, /*  1x Ȼ ..Ȼ  → ȼ ..ȼ  Watin-B */
                {0x023d, 0x023d, -163}, /*  1x Ƚ ..Ƚ  → ƚ ..ƚ  Watin-B */
                {0x0241, 0x0241, +1}, /*  1x Ɂ ..Ɂ  → ɂ ..ɂ  Watin-B */
                {0x0243, 0x0243, -195}, /*  1x Ƀ ..Ƀ  → ƀ ..ƀ  Watin-B */
                {0x0244, 0x0244, +69}, /*  1x Ʉ ..Ʉ  → ʉ ..ʉ  Watin-B */
                {0x0245, 0x0245, +71}, /*  1x Ʌ ..Ʌ  → ʌ ..ʌ  Watin-B */
                {0x0246, 0x0246, +1}, /*  1x Ɇ ..Ɇ  → ɇ ..ɇ  Watin-B */
                {0x0248, 0x0248, +1}, /*  1x Ɉ ..Ɉ  → ɉ ..ɉ  Watin-B */
                {0x024a, 0x024a, +1}, /*  1x Ɋ ..Ɋ  → ɋ ..ɋ  Watin-B */
                {0x024c, 0x024c, +1}, /*  1x Ɍ ..Ɍ  → ɍ ..ɍ  Watin-B */
                {0x024e, 0x024e, +1}, /*  1x Ɏ ..Ɏ  → ɏ ..ɏ  Watin-B */
                {0x0386, 0x0386, +38}, /*  1x Ά ..Ά  → ά ..ά  Greek */
                {0x0388, 0x038a, +37}, /*  3x Έ ..Ί  → έ ..ί  Greek */
                {0x038c, 0x038c, +64}, /*  1x Ό ..Ό  → ό ..ό  Greek */
                {0x038e, 0x038f, +63}, /*  2x Ύ ..Ώ  → ύ ..ώ  Greek */
                {0x0391, 0x03a1, +32}, /* 17x Α ..Ρ  → α ..ρ  Greek */
                {0x03a3, 0x03ab, +32}, /*  9x Σ ..Ϋ  → σ ..ϋ  Greek */
                {0x03dc, 0x03dc, +1}, /*  1x Ϝ ..Ϝ  → ϝ ..ϝ  Greek */
                {0x03f4, 0x03f4, -60}, /*  1x ϴ ..ϴ  → θ ..θ  Greek */
                {0x0400, 0x040f, +80}, /* 16x Ѐ ..Џ  → ѐ ..џ  Cyrillic */
                {0x0410, 0x042f, +32}, /* 32x А ..Я  → а ..я  Cyrillic */
                {0x0460, 0x0460, +1}, /*  1x Ѡ ..Ѡ  → ѡ ..ѡ  Cyrillic */
                {0x0462, 0x0462, +1}, /*  1x Ѣ ..Ѣ  → ѣ ..ѣ  Cyrillic */
                {0x0464, 0x0464, +1}, /*  1x Ѥ ..Ѥ  → ѥ ..ѥ  Cyrillic */
                {0x0472, 0x0472, +1}, /*  1x Ѳ ..Ѳ  → ѳ ..ѳ  Cyrillic */
                {0x0490, 0x0490, +1}, /*  1x Ґ ..Ґ  → ґ ..ґ  Cyrillic */
                {0x0498, 0x0498, +1}, /*  1x Ҙ ..Ҙ  → ҙ ..ҙ  Cyrillic */
                {0x049a, 0x049a, +1}, /*  1x Қ ..Қ  → қ ..қ  Cyrillic */
                {0x0531, 0x0556, +48}, /* 38x Ա ..Ֆ  → ա ..ֆ  Armenian */
                {0x10a0, 0x10c5, +7264}, /* 38x Ⴀ ..Ⴥ  → ⴀ ..ⴥ  Georgian */
                {0x10c7, 0x10c7, +7264}, /*  1x Ⴧ ..Ⴧ  → ⴧ ..ⴧ  Georgian */
                {0x10cd, 0x10cd, +7264}, /*  1x Ⴭ ..Ⴭ  → ⴭ ..ⴭ  Georgian */
                {0x13f0, 0x13f5, +8}, /*  6x Ᏸ ..Ᏽ  → ᏸ ..ᏽ  Cherokee */
                {0x1c90, 0x1cba, -3008}, /* 43x Ა ..Ჺ  → ა ..ჺ  Georgian2 */
                {0x1cbd, 0x1cbf, -3008}, /*  3x Ჽ ..Ჿ  → ჽ ..ჿ  Georgian2 */
                {0x1f08, 0x1f0f, -8}, /*  8x Ἀ ..Ἇ  → ἀ ..ἇ  Greek2 */
                {0x1f18, 0x1f1d, -8}, /*  6x Ἐ ..Ἕ  → ἐ ..ἕ  Greek2 */
                {0x1f28, 0x1f2f, -8}, /*  8x Ἠ ..Ἧ  → ἠ ..ἧ  Greek2 */
                {0x1f38, 0x1f3f, -8}, /*  8x Ἰ ..Ἷ  → ἰ ..ἷ  Greek2 */
                {0x1f48, 0x1f4d, -8}, /*  6x Ὀ ..Ὅ  → ὀ ..ὅ  Greek2 */
                {0x1f59, 0x1f59, -8}, /*  1x Ὑ ..Ὑ  → ὑ ..ὑ  Greek2 */
                {0x1f5b, 0x1f5b, -8}, /*  1x Ὓ ..Ὓ  → ὓ ..ὓ  Greek2 */
                {0x1f5d, 0x1f5d, -8}, /*  1x Ὕ ..Ὕ  → ὕ ..ὕ  Greek2 */
                {0x1f5f, 0x1f5f, -8}, /*  1x Ὗ ..Ὗ  → ὗ ..ὗ  Greek2 */
                {0x1f68, 0x1f6f, -8}, /*  8x Ὠ ..Ὧ  → ὠ ..ὧ  Greek2 */
                {0x1f88, 0x1f8f, -8}, /*  8x ᾈ ..ᾏ  → ᾀ ..ᾇ  Greek2 */
                {0x1f98, 0x1f9f, -8}, /*  8x ᾘ ..ᾟ  → ᾐ ..ᾗ  Greek2 */
                {0x1fa8, 0x1faf, -8}, /*  8x ᾨ ..ᾯ  → ᾠ ..ᾧ  Greek2 */
                {0x1fb8, 0x1fb9, -8}, /*  2x Ᾰ ..Ᾱ  → ᾰ ..ᾱ  Greek2 */
                {0x1fba, 0x1fbb, -74}, /*  2x Ὰ ..Ά  → ὰ ..ά  Greek2 */
                {0x1fbc, 0x1fbc, -9}, /*  1x ᾼ ..ᾼ  → ᾳ ..ᾳ  Greek2 */
                {0x1fc8, 0x1fcb, -86}, /*  4x Ὲ ..Ή  → ὲ ..ή  Greek2 */
                {0x1fcc, 0x1fcc, -9}, /*  1x ῌ ..ῌ  → ῃ ..ῃ  Greek2 */
                {0x1fd8, 0x1fd9, -8}, /*  2x Ῐ ..Ῑ  → ῐ ..ῑ  Greek2 */
                {0x1fda, 0x1fdb, -100}, /*  2x Ὶ ..Ί  → ὶ ..ί  Greek2 */
                {0x1fe8, 0x1fe9, -8}, /*  2x Ῠ ..Ῡ  → ῠ ..ῡ  Greek2 */
                {0x1fea, 0x1feb, -112}, /*  2x Ὺ ..Ύ  → ὺ ..ύ  Greek2 */
                {0x1fec, 0x1fec, -7}, /*  1x Ῥ ..Ῥ  → ῥ ..ῥ  Greek2 */
                {0x1ff8, 0x1ff9, -128}, /*  2x Ὸ ..Ό  → ὸ ..ό  Greek2 */
                {0x1ffa, 0x1ffb, -126}, /*  2x Ὼ ..Ώ  → ὼ ..ώ  Greek2 */
                {0x1ffc, 0x1ffc, -9}, /*  1x ῼ ..ῼ  → ῳ ..ῳ  Greek2 */
                {0x2126, 0x2126, -7517}, /*  1x Ω ..Ω  → ω ..ω  Letterlike */
                {0x212a, 0x212a, -8383}, /*  1x K ..K  → k ..k  Letterlike */
                {0x212b, 0x212b, -8262}, /*  1x Å ..Å  → å ..å  Letterlike */
                {0x2132, 0x2132, +28}, /*  1x Ⅎ ..Ⅎ  → ⅎ ..ⅎ  Letterlike */
                {0x2160, 0x216f, +16}, /* 16x Ⅰ ..Ⅿ  → ⅰ ..ⅿ  Numbery */
                {0x2183, 0x2183, +1}, /*  1x Ↄ ..Ↄ  → ↄ ..ↄ  Numbery */
                {0x24b6, 0x24cf, +26}, /* 26x Ⓐ ..Ⓩ  → ⓐ ..ⓩ  Enclosed */
                {0x2c00, 0x2c2e, +48}, /* 47x Ⰰ ..Ⱞ  → ⰰ ..ⱞ  Glagolitic */
                {0xff21, 0xff3a, +32}, /* 26x Ａ..Ｚ → ａ..ｚ Dubs */
            };
            l = 0;
            r = n = sizeof(kLower) / sizeof(kLower[0]);
            while (l < r) {
                m = (l + r) >> 1;
                if (kLower[m].b < c) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            if (l < n && kLower[l].a <= c && c <= kLower[l].b) {
                return c + kLower[l].d;
            } else {
                return c;
            }
        }
    } else {
        static struct {
            unsigned a;
            unsigned b;
            short d;
        } kAstralLower[] = {
            {0x10400, 0x10427, +40}, /* 40x 𐐀 ..𐐧  → 𐐨 ..𐑏  Deseret */
            {0x104b0, 0x104d3, +40}, /* 36x 𐒰 ..𐓓  → 𐓘 ..𐓻  Osage */
            {0x1d400, 0x1d419, +26}, /* 26x 𝐀 ..𝐙  → 𝐚 ..𝐳  Math */
            {0x1d43c, 0x1d44d, +26}, /* 18x 𝐼 ..𝑍  → 𝑖 ..𝑧  Math */
            {0x1d468, 0x1d481, +26}, /* 26x 𝑨 ..𝒁  → 𝒂 ..𝒛  Math */
            {0x1d4ae, 0x1d4b5, +26}, /*  8x 𝒮 ..𝒵  → 𝓈 ..𝓏  Math */
            {0x1d4d0, 0x1d4e9, +26}, /* 26x 𝓐 ..𝓩  → 𝓪 ..𝔃  Math */
            {0x1d50d, 0x1d514, +26}, /*  8x 𝔍 ..𝔔  → 𝔧 ..𝔮  Math */
            {0x1d56c, 0x1d585, +26}, /* 26x 𝕬 ..𝖅  → 𝖆 ..𝖟  Math */
            {0x1d5a0, 0x1d5b9, +26}, /* 26x 𝖠 ..𝖹  → 𝖺 ..𝗓  Math */
            {0x1d5d4, 0x1d5ed, +26}, /* 26x 𝗔 ..𝗭  → 𝗮 ..𝘇  Math */
            {0x1d608, 0x1d621, +26}, /* 26x 𝘈 ..𝘡  → 𝘢 ..𝘻  Math */
            {0x1d63c, 0x1d655, -442}, /* 26x 𝘼 ..𝙕  → 𝒂 ..𝒛  Math */
            {0x1d670, 0x1d689, +26}, /* 26x 𝙰 ..𝚉  → 𝚊 ..𝚣  Math */
            {0x1d6a8, 0x1d6b8, +26}, /* 17x 𝚨 ..𝚸  → 𝛂 ..𝛒  Math */
            {0x1d6e2, 0x1d6f2, +26}, /* 17x 𝛢 ..𝛲  → 𝛼 ..𝜌  Math */
            {0x1d71c, 0x1d72c, +26}, /* 17x 𝜜 ..𝜬  → 𝜶 ..𝝆  Math */
            {0x1d756, 0x1d766, +26}, /* 17x 𝝖 ..𝝦  → 𝝰 ..𝞀  Math */
            {0x1d790, 0x1d7a0, -90}, /* 17x 𝞐 ..𝞠  → 𝜶 ..𝝆  Math */
        };
        l = 0;
        r = n = sizeof(kAstralLower) / sizeof(kAstralLower[0]);
        while (l < r) {
            m = (l + r) >> 1;
            if (kAstralLower[m].b < c) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        if (l < n && kAstralLower[l].a <= c && c <= kAstralLower[l].b) {
            return c + kAstralLower[l].d;
        } else {
            return c;
        }
    }
}

unsigned bestlineUppercase(unsigned c) {
    int m, l, r, n;
    if (c < 0200) {
        if ('a' <= c && c <= 'z') {
            return c - 32;
        } else {
            return c;
        }
    } else if (c <= 0xffff) {
        if ((0x0101 <= c && c <= 0x0177) || /* 60x ā..ŵ → Ā..ā Watin-A */
            (0x01df <= c && c <= 0x01ef) || /*  9x ǟ..ǯ → Ǟ..Ǯ Watin-B */
            (0x01f8 <= c && c <= 0x021e) || /* 20x ǹ..ȟ → Ǹ..Ȟ Watin-B */
            (0x0222 <= c && c <= 0x0232) || /*  9x ȣ..ȳ → Ȣ..Ȳ Watin-B */
            (0x1e01 <= c && c <= 0x1eff)) { /*256x ḁ..ỿ → Ḁ..Ỿ Watin-C */
            if (c == 0x0131)
                return c + 232;
            if (c == 0x1e9e)
                return c;
            return c - (c & 1);
        } else if (0x01d0 <= c && c <= 0x01dc) {
            return c - (~c & 1); /* 7x ǐ..ǜ → Ǐ..Ǜ Watin-B */
        } else if (0xab70 <= c && c <= 0xabbf) {
            return c - 38864; /* 80x ꭰ ..ꮿ  → Ꭰ ..Ꮿ  Cherokee Supplement */
        } else {
            static const struct {
                unsigned short a;
                unsigned short b;
                short d;
            } kUpper[] = {
                {0x00b5, 0x00b5, +743}, /*  1x µ ..µ  → Μ ..Μ  Watin */
                {0x00e0, 0x00f6, -32}, /* 23x à ..ö  → À ..Ö  Watin */
                {0x00f8, 0x00fe, -32}, /*  7x ø ..þ  → Ø ..Þ  Watin */
                {0x00ff, 0x00ff, +121}, /*  1x ÿ ..ÿ  → Ÿ ..Ÿ  Watin */
                {0x017a, 0x017a, -1}, /*  1x ź ..ź  → Ź ..Ź  Watin-A */
                {0x017c, 0x017c, -1}, /*  1x ż ..ż  → Ż ..Ż  Watin-A */
                {0x017e, 0x017e, -1}, /*  1x ž ..ž  → Ž ..Ž  Watin-A */
                {0x017f, 0x017f, -300}, /*  1x ſ ..ſ  → S ..S  Watin-A */
                {0x0180, 0x0180, +195}, /*  1x ƀ ..ƀ  → Ƀ ..Ƀ  Watin-B */
                {0x0183, 0x0183, -1}, /*  1x ƃ ..ƃ  → Ƃ ..Ƃ  Watin-B */
                {0x0185, 0x0185, -1}, /*  1x ƅ ..ƅ  → Ƅ ..Ƅ  Watin-B */
                {0x0188, 0x0188, -1}, /*  1x ƈ ..ƈ  → Ƈ ..Ƈ  Watin-B */
                {0x018c, 0x018c, -1}, /*  1x ƌ ..ƌ  → Ƌ ..Ƌ  Watin-B */
                {0x0192, 0x0192, -1}, /*  1x ƒ ..ƒ  → Ƒ ..Ƒ  Watin-B */
                {0x0195, 0x0195, +97}, /*  1x ƕ ..ƕ  → Ƕ ..Ƕ  Watin-B */
                {0x0199, 0x0199, -1}, /*  1x ƙ ..ƙ  → Ƙ ..Ƙ  Watin-B */
                {0x019a, 0x019a, +163}, /*  1x ƚ ..ƚ  → Ƚ ..Ƚ  Watin-B */
                {0x019e, 0x019e, +130}, /*  1x ƞ ..ƞ  → Ƞ ..Ƞ  Watin-B */
                {0x01a1, 0x01a1, -1}, /*  1x ơ ..ơ  → Ơ ..Ơ  Watin-B */
                {0x01a3, 0x01a3, -1}, /*  1x ƣ ..ƣ  → Ƣ ..Ƣ  Watin-B */
                {0x01a5, 0x01a5, -1}, /*  1x ƥ ..ƥ  → Ƥ ..Ƥ  Watin-B */
                {0x01a8, 0x01a8, -1}, /*  1x ƨ ..ƨ  → Ƨ ..Ƨ  Watin-B */
                {0x01ad, 0x01ad, -1}, /*  1x ƭ ..ƭ  → Ƭ ..Ƭ  Watin-B */
                {0x01b0, 0x01b0, -1}, /*  1x ư ..ư  → Ư ..Ư  Watin-B */
                {0x01b4, 0x01b4, -1}, /*  1x ƴ ..ƴ  → Ƴ ..Ƴ  Watin-B */
                {0x01b6, 0x01b6, -1}, /*  1x ƶ ..ƶ  → Ƶ ..Ƶ  Watin-B */
                {0x01b9, 0x01b9, -1}, /*  1x ƹ ..ƹ  → Ƹ ..Ƹ  Watin-B */
                {0x01bd, 0x01bd, -1}, /*  1x ƽ ..ƽ  → Ƽ ..Ƽ  Watin-B */
                {0x01bf, 0x01bf, +56}, /*  1x ƿ ..ƿ  → Ƿ ..Ƿ  Watin-B */
                {0x01c5, 0x01c5, -1}, /*  1x ǅ ..ǅ  → Ǆ ..Ǆ  Watin-B */
                {0x01c6, 0x01c6, -2}, /*  1x ǆ ..ǆ  → Ǆ ..Ǆ  Watin-B */
                {0x01c8, 0x01c8, -1}, /*  1x ǈ ..ǈ  → Ǉ ..Ǉ  Watin-B */
                {0x01c9, 0x01c9, -2}, /*  1x ǉ ..ǉ  → Ǉ ..Ǉ  Watin-B */
                {0x01cb, 0x01cb, -1}, /*  1x ǋ ..ǋ  → Ǌ ..Ǌ  Watin-B */
                {0x01cc, 0x01cc, -2}, /*  1x ǌ ..ǌ  → Ǌ ..Ǌ  Watin-B */
                {0x01ce, 0x01ce, -1}, /*  1x ǎ ..ǎ  → Ǎ ..Ǎ  Watin-B */
                {0x01dd, 0x01dd, -79}, /*  1x ǝ ..ǝ  → Ǝ ..Ǝ  Watin-B */
                {0x01f2, 0x01f2, -1}, /*  1x ǲ ..ǲ  → Ǳ ..Ǳ  Watin-B */
                {0x01f3, 0x01f3, -2}, /*  1x ǳ ..ǳ  → Ǳ ..Ǳ  Watin-B */
                {0x01f5, 0x01f5, -1}, /*  1x ǵ ..ǵ  → Ǵ ..Ǵ  Watin-B */
                {0x023c, 0x023c, -1}, /*  1x ȼ ..ȼ  → Ȼ ..Ȼ  Watin-B */
                {0x023f, 0x0240, +10815}, /*  2x ȿ ..ɀ  → Ȿ ..Ɀ  Watin-B */
                {0x0242, 0x0242, -1}, /*  1x ɂ ..ɂ  → Ɂ ..Ɂ  Watin-B */
                {0x0247, 0x0247, -1}, /*  1x ɇ ..ɇ  → Ɇ ..Ɇ  Watin-B */
                {0x0249, 0x0249, -1}, /*  1x ɉ ..ɉ  → Ɉ ..Ɉ  Watin-B */
                {0x024b, 0x024b, -1}, /*  1x ɋ ..ɋ  → Ɋ ..Ɋ  Watin-B */
                {0x024d, 0x024d, -1}, /*  1x ɍ ..ɍ  → Ɍ ..Ɍ  Watin-B */
                {0x024f, 0x024f, -1}, /*  1x ɏ ..ɏ  → Ɏ ..Ɏ  Watin-B */
                {0x037b, 0x037d, +130}, /*  3x ͻ ..ͽ  → Ͻ ..Ͽ  Greek */
                {0x03ac, 0x03ac, -38}, /*  1x ά ..ά  → Ά ..Ά  Greek */
                {0x03ad, 0x03af, -37}, /*  3x έ ..ί  → Έ ..Ί  Greek */
                {0x03b1, 0x03c1, -32}, /* 17x α ..ρ  → Α ..Ρ  Greek */
                {0x03c2, 0x03c2, -31}, /*  1x ς ..ς  → Σ ..Σ  Greek */
                {0x03c3, 0x03cb, -32}, /*  9x σ ..ϋ  → Σ ..Ϋ  Greek */
                {0x03cc, 0x03cc, -64}, /*  1x ό ..ό  → Ό ..Ό  Greek */
                {0x03cd, 0x03ce, -63}, /*  2x ύ ..ώ  → Ύ ..Ώ  Greek */
                {0x03d0, 0x03d0, -62}, /*  1x ϐ ..ϐ  → Β ..Β  Greek */
                {0x03d1, 0x03d1, -57}, /*  1x ϑ ..ϑ  → Θ ..Θ  Greek */
                {0x03d5, 0x03d5, -47}, /*  1x ϕ ..ϕ  → Φ ..Φ  Greek */
                {0x03d6, 0x03d6, -54}, /*  1x ϖ ..ϖ  → Π ..Π  Greek */
                {0x03dd, 0x03dd, -1}, /*  1x ϝ ..ϝ  → Ϝ ..Ϝ  Greek */
                {0x03f0, 0x03f0, -86}, /*  1x ϰ ..ϰ  → Κ ..Κ  Greek */
                {0x03f1, 0x03f1, -80}, /*  1x ϱ ..ϱ  → Ρ ..Ρ  Greek */
                {0x03f5, 0x03f5, -96}, /*  1x ϵ ..ϵ  → Ε ..Ε  Greek */
                {0x0430, 0x044f, -32}, /* 32x а ..я  → А ..Я  Cyrillic */
                {0x0450, 0x045f, -80}, /* 16x ѐ ..џ  → Ѐ ..Џ  Cyrillic */
                {0x0461, 0x0461, -1}, /*  1x ѡ ..ѡ  → Ѡ ..Ѡ  Cyrillic */
                {0x0463, 0x0463, -1}, /*  1x ѣ ..ѣ  → Ѣ ..Ѣ  Cyrillic */
                {0x0465, 0x0465, -1}, /*  1x ѥ ..ѥ  → Ѥ ..Ѥ  Cyrillic */
                {0x0473, 0x0473, -1}, /*  1x ѳ ..ѳ  → Ѳ ..Ѳ  Cyrillic */
                {0x0491, 0x0491, -1}, /*  1x ґ ..ґ  → Ґ ..Ґ  Cyrillic */
                {0x0499, 0x0499, -1}, /*  1x ҙ ..ҙ  → Ҙ ..Ҙ  Cyrillic */
                {0x049b, 0x049b, -1}, /*  1x қ ..қ  → Қ ..Қ  Cyrillic */
                {0x0561, 0x0586, -48}, /* 38x ա ..ֆ  → Ա ..Ֆ  Armenian */
                {0x10d0, 0x10fa, +3008}, /* 43x ა ..ჺ  → Ა ..Ჺ  Georgian */
                {0x10fd, 0x10ff, +3008}, /*  3x ჽ ..ჿ  → Ჽ ..Ჿ  Georgian */
                {0x13f8, 0x13fd, -8}, /*  6x ᏸ ..ᏽ  → Ᏸ ..Ᏽ  Cherokee */
                {0x214e, 0x214e, -28}, /*  1x ⅎ ..ⅎ  → Ⅎ ..Ⅎ  Letterlike */
                {0x2170, 0x217f, -16}, /* 16x ⅰ ..ⅿ  → Ⅰ ..Ⅿ  Numbery */
                {0x2184, 0x2184, -1}, /*  1x ↄ ..ↄ  → Ↄ ..Ↄ  Numbery */
                {0x24d0, 0x24e9, -26}, /* 26x ⓐ ..ⓩ  → Ⓐ ..Ⓩ  Enclosed */
                {0x2c30, 0x2c5e, -48}, /* 47x ⰰ ..ⱞ  → Ⰰ ..Ⱞ  Glagolitic */
                {0x2d00, 0x2d25, -7264}, /* 38x ⴀ ..ⴥ  → Ⴀ ..Ⴥ  Georgian2 */
                {0x2d27, 0x2d27, -7264}, /*  1x ⴧ ..ⴧ  → Ⴧ ..Ⴧ  Georgian2 */
                {0x2d2d, 0x2d2d, -7264}, /*  1x ⴭ ..ⴭ  → Ⴭ ..Ⴭ  Georgian2 */
                {0xff41, 0xff5a, -32}, /* 26x ａ..ｚ → Ａ..Ｚ Dubs */
            };
            l = 0;
            r = n = sizeof(kUpper) / sizeof(kUpper[0]);
            while (l < r) {
                m = (l + r) >> 1;
                if (kUpper[m].b < c) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            if (l < n && kUpper[l].a <= c && c <= kUpper[l].b) {
                return c + kUpper[l].d;
            } else {
                return c;
            }
        }
    } else {
        static const struct {
            unsigned a;
            unsigned b;
            short d;
        } kAstralUpper[] = {
            {0x10428, 0x1044f, -40}, /* 40x 𐐨..𐑏 → 𐐀..𐐧 Deseret */
            {0x104d8, 0x104fb, -40}, /* 36x 𐓘..𐓻 → 𐒰..𐓓 Osage */
            {0x1d41a, 0x1d433, -26}, /* 26x 𝐚..𝐳 → 𝐀..𝐙 Math */
            {0x1d456, 0x1d467, -26}, /* 18x 𝑖..𝑧 → 𝐼..𝑍 Math */
            {0x1d482, 0x1d49b, -26}, /* 26x 𝒂..𝒛 → 𝑨..𝒁 Math */
            {0x1d4c8, 0x1d4cf, -26}, /*  8x 𝓈..𝓏 → 𝒮..𝒵 Math */
            {0x1d4ea, 0x1d503, -26}, /* 26x 𝓪..𝔃 → 𝓐..𝓩 Math */
            {0x1d527, 0x1d52e, -26}, /*  8x 𝔧..𝔮 → 𝔍..𝔔 Math */
            {0x1d586, 0x1d59f, -26}, /* 26x 𝖆..𝖟 → 𝕬..𝖅 Math */
            {0x1d5ba, 0x1d5d3, -26}, /* 26x 𝖺..𝗓 → 𝖠..𝖹 Math */
            {0x1d5ee, 0x1d607, -26}, /* 26x 𝗮..𝘇 → 𝗔..𝗭 Math */
            {0x1d622, 0x1d63b, -26}, /* 26x 𝘢..𝘻 → 𝘈..𝘡 Math */
            {0x1d68a, 0x1d6a3, +442}, /* 26x 𝒂..𝒛 → 𝘼..𝙕 Math */
            {0x1d6c2, 0x1d6d2, -26}, /* 26x 𝚊..𝚣 → 𝙰..𝚉 Math */
            {0x1d6fc, 0x1d70c, -26}, /* 17x 𝛂..𝛒 → 𝚨..𝚸 Math */
            {0x1d736, 0x1d746, -26}, /* 17x 𝛼..𝜌 → 𝛢..𝛲 Math */
            {0x1d770, 0x1d780, -26}, /* 17x 𝜶..𝝆 → 𝜜..𝜬 Math */
            {0x1d770, 0x1d756, -26}, /* 17x 𝝰..𝞀 → 𝝖..𝝦 Math */
            {0x1d736, 0x1d790, -90}, /* 17x 𝜶..𝝆 → 𝞐..𝞠 Math */
        };
        l = 0;
        r = n = sizeof(kAstralUpper) / sizeof(kAstralUpper[0]);
        while (l < r) {
            m = (l + r) >> 1;
            if (kAstralUpper[m].b < c) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        if (l < n && kAstralUpper[l].a <= c && c <= kAstralUpper[l].b) {
            return c + kAstralUpper[l].d;
        } else {
            return c;
        }
    }
}

char bestlineNotSeparator(unsigned c) {
    return !bestlineIsSeparator(c);
}

static unsigned GetMirror(const unsigned short A[][2], size_t n, unsigned c) {
    int l, m, r;
    l = 0;
    r = n - 1;
    while (l <= r) {
        m = (l + r) >> 1;
        if (A[m][0] < c) {
            l = m + 1;
        } else if (A[m][0] > c) {
            r = m - 1;
        } else {
            return A[m][1];
        }
    }
    return 0;
}

unsigned bestlineMirrorLeft(unsigned c) {
    static const unsigned short kMirrorRight[][2] = {
        {L')', L'('},   {L']', L'['},   {L'}', L'{'},   {L'⁆', L'⁅'},   {L'⁾', L'⁽'},
        {L'₎', L'₍'},   {L'⌉', L'⌈'},   {L'⌋', L'⌊'},   {L'〉', L'〈'}, {L'❩', L'❨'},
        {L'❫', L'❪'},   {L'❭', L'❬'},   {L'❯', L'❮'},   {L'❱', L'❰'},   {L'❳', L'❲'},
        {L'❵', L'❴'},   {L'⟆', L'⟅'},   {L'⟧', L'⟦'},   {L'⟩', L'⟨'},   {L'⟫', L'⟪'},
        {L'⟭', L'⟬'},   {L'⟯', L'⟮'},   {L'⦄', L'⦃'},   {L'⦆', L'⦅'},   {L'⦈', L'⦇'},
        {L'⦊', L'⦉'},   {L'⦌', L'⦋'},   {L'⦎', L'⦏'},   {L'⦐', L'⦍'},   {L'⦒', L'⦑'},
        {L'⦔', L'⦓'},   {L'⦘', L'⦗'},   {L'⧙', L'⧘'},   {L'⧛', L'⧚'},   {L'⧽', L'⧼'},
        {L'﹚', L'﹙'}, {L'﹜', L'﹛'}, {L'﹞', L'﹝'}, {L'）', L'（'}, {L'］', L'［'},
        {L'｝', L'｛'}, {L'｣', L'｢'},
    };
    return GetMirror(kMirrorRight, sizeof(kMirrorRight) / sizeof(kMirrorRight[0]), c);
}

unsigned bestlineMirrorRight(unsigned c) {
    static const unsigned short kMirrorLeft[][2] = {
        {L'(', L')'},   {L'[', L']'},   {L'{', L'}'},   {L'⁅', L'⁆'},   {L'⁽', L'⁾'},
        {L'₍', L'₎'},   {L'⌈', L'⌉'},   {L'⌊', L'⌋'},   {L'〈', L'〉'}, {L'❨', L'❩'},
        {L'❪', L'❫'},   {L'❬', L'❭'},   {L'❮', L'❯'},   {L'❰', L'❱'},   {L'❲', L'❳'},
        {L'❴', L'❵'},   {L'⟅', L'⟆'},   {L'⟦', L'⟧'},   {L'⟨', L'⟩'},   {L'⟪', L'⟫'},
        {L'⟬', L'⟭'},   {L'⟮', L'⟯'},   {L'⦃', L'⦄'},   {L'⦅', L'⦆'},   {L'⦇', L'⦈'},
        {L'⦉', L'⦊'},   {L'⦋', L'⦌'},   {L'⦍', L'⦐'},   {L'⦏', L'⦎'},   {L'⦑', L'⦒'},
        {L'⦓', L'⦔'},   {L'⦗', L'⦘'},   {L'⧘', L'⧙'},   {L'⧚', L'⧛'},   {L'⧼', L'⧽'},
        {L'﹙', L'﹚'}, {L'﹛', L'﹜'}, {L'﹝', L'﹞'}, {L'（', L'）'}, {L'［', L'］'},
        {L'｛', L'｝'}, {L'｢', L'｣'},
    };
    return GetMirror(kMirrorLeft, sizeof(kMirrorLeft) / sizeof(kMirrorLeft[0]), c);
}

static char StartsWith(const char *s, const char *prefix) {
    for (;;) {
        if (!*prefix)
            return 1;
        if (!*s)
            return 0;
        if (*s++ != *prefix++)
            return 0;
    }
}

static char EndsWith(const char *s, const char *suffix) {
    size_t n, m;
    n = strlen(s);
    m = strlen(suffix);
    if (m > n)
        return 0;
    return !memcmp(s + n - m, suffix, m);
}

char bestlineIsXeparator(unsigned c) {
    return (bestlineIsSeparator(c) && !bestlineMirrorLeft(c) && !bestlineMirrorRight(c));
}

static unsigned Capitalize(unsigned c) {
    if (!iscapital) {
        c = bestlineUppercase(c);
        iscapital = 1;
    }
    return c;
}

static inline int Bsr(unsigned long long x) {
#if defined(__GNUC__) && !defined(__STRICT_ANSI__)
    int b;
    b = __builtin_clzll(x);
    b ^= sizeof(unsigned long long) * CHAR_BIT - 1;
    return b;
#else
    static const char kDebruijn[64] = {
        0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61, 54, 58, 35, 52, 50, 42,
        21, 44, 38, 32, 29, 23, 17, 11, 4,  62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43,
        31, 22, 10, 45, 25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63,
    };
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return kDebruijn[(x * 0x03f79d71b4cb0a89) >> 58];
#endif
}

static struct rune DecodeUtf8(int c) {
    struct rune r;
    if (c < 252) {
        r.n = Bsr(255 & ~c);
        r.c = c & (((1 << r.n) - 1) | 3);
        r.n = 6 - r.n;
    } else {
        r.c = c & 3;
        r.n = 5;
    }
    return r;
}

static unsigned long long EncodeUtf8(unsigned c) {
    static const unsigned short kTpEnc[32 - 7] = {
        1 | 0300 << 8, 1 | 0300 << 8, 1 | 0300 << 8, 1 | 0300 << 8, 2 | 0340 << 8,
        2 | 0340 << 8, 2 | 0340 << 8, 2 | 0340 << 8, 2 | 0340 << 8, 3 | 0360 << 8,
        3 | 0360 << 8, 3 | 0360 << 8, 3 | 0360 << 8, 3 | 0360 << 8, 4 | 0370 << 8,
        4 | 0370 << 8, 4 | 0370 << 8, 4 | 0370 << 8, 4 | 0370 << 8, 5 | 0374 << 8,
        5 | 0374 << 8, 5 | 0374 << 8, 5 | 0374 << 8, 5 | 0374 << 8, 5 | 0374 << 8,
    };
    int e, n;
    unsigned long long w;
    if (c < 0200)
        return c;
    e = kTpEnc[Bsr(c) - 7];
    n = e & 0xff;
    w = 0;
    do {
        w |= 0200 | (c & 077);
        w <<= 8;
        c >>= 6;
    } while (--n);
    return c | w | e >> 8;
}

static struct rune GetUtf8(const char *p, size_t n) {
    struct rune r;
    if ((r.n = r.c = 0) < n && (r.c = p[r.n++] & 255) >= 0300) {
        r.c = DecodeUtf8(r.c).c;
        while (r.n < n && (p[r.n] & 0300) == 0200) {
            r.c = r.c << 6 | (p[r.n++] & 077);
        }
    }
    return r;
}

static char *FormatUnsigned(char *p, unsigned x) {
    char t;
    size_t i, a, b;
    i = 0;
    do {
        p[i++] = x % 10 + '0';
        x = x / 10;
    } while (x > 0);
    p[i] = '\0';
    if (i) {
        for (a = 0, b = i - 1; a < b; ++a, --b) {
            t = p[a];
            p[a] = p[b];
            p[b] = t;
        }
    }
    return p + i;
}

static void abInit(struct abuf *a) {
    a->len = 0;
    a->cap = 16;
    a->b = (char *)malloc(a->cap);
    a->b[0] = 0;
}

static char abGrow(struct abuf *a, int need) {
    int cap;
    char *b;
    cap = a->cap;
    do
        cap += cap / 2;
    while (cap < need);
    if (!(b = (char *)realloc(a->b, cap * sizeof(*a->b))))
        return 0;
    a->cap = cap;
    a->b = b;
    return 1;
}

static void abAppendw(struct abuf *a, unsigned long long w) {
    char *p;
    if (a->len + 8 > a->cap && !abGrow(a, a->len + 8))
        return;
    p = a->b + a->len;
    p[0] = (0x00000000000000FF & w) >> 000;
    p[1] = (0x000000000000FF00 & w) >> 010;
    p[2] = (0x0000000000FF0000 & w) >> 020;
    p[3] = (0x00000000FF000000 & w) >> 030;
    p[4] = (0x000000FF00000000 & w) >> 040;
    p[5] = (0x0000FF0000000000 & w) >> 050;
    p[6] = (0x00FF000000000000 & w) >> 060;
    p[7] = (0xFF00000000000000 & w) >> 070;
    a->len += w ? (Bsr(w) >> 3) + 1 : 1;
}

static void abAppend(struct abuf *a, const char *s, int len) {
    if (a->len + len + 1 > a->cap && !abGrow(a, a->len + len + 1))
        return;
    memcpy(a->b + a->len, s, len);
    a->b[a->len + len] = 0;
    a->len += len;
}

static void abAppends(struct abuf *a, const char *s) {
    abAppend(a, s, strlen(s));
}

static void abAppendu(struct abuf *a, unsigned u) {
    char b[11];
    abAppend(a, b, FormatUnsigned(b, u) - b);
}

static void abFree(struct abuf *a) {
    free(a->b);
    a->b = 0;
}

static size_t GetFdSize(int fd) {
    struct stat st;
    st.st_size = 0;
    fstat(fd, &st);
    return st.st_size;
}

static char IsCharDev(int fd) {
    struct stat st;
    st.st_mode = 0;
    fstat(fd, &st);
    return (st.st_mode & S_IFMT) == S_IFCHR;
}

static int MyRead(int fd, void *c, int);
static int MyWrite(int fd, const void *c, int);
static int MyPoll(int fd, int events, int to);

static int (*_MyRead)(int fd, void *c, int n) = MyRead;
static int (*_MyWrite)(int fd, const void *c, int n) = MyWrite;
static int (*_MyPoll)(int fd, int events, int to) = MyPoll;

static int WaitUntilReady(int fd, int events) {
    return _MyPoll(fd, events, -1);
}

static char HasPendingInput(int fd) {
    return _MyPoll(fd, POLLIN, 0) == 1;
}

static char *GetLineBlock(FILE *f) {
    ssize_t rc;
    char *p = 0;
    size_t n, c = 0;
    if ((rc = getdelim(&p, &c, '\n', f)) != EOF) {
        for (n = rc; n; --n) {
            if (p[n - 1] == '\r' || p[n - 1] == '\n') {
                p[n - 1] = 0;
            } else {
                break;
            }
        }
        return p;
    } else {
        free(p);
        return 0;
    }
}

long bestlineReadCharacter(int fd, char *p, unsigned long n) {
    int e;
    size_t i;
    ssize_t rc;
    struct rune r;
    unsigned char c;
    enum { kAscii, kUtf8, kEsc, kCsi1, kCsi2, kSs, kNf, kStr, kStr2, kDone } t;
    i = 0;
    r.c = 0;
    r.n = 0;
    e = errno;
    t = kAscii;
    if (n)
        p[0] = 0;
    do {
        for (;;) {
            if (gotint) {
                errno = EINTR;
                return -1;
            }
            if (n) {
                rc = _MyRead(fd, &c, 1);
            } else {
                rc = _MyRead(fd, 0, 0);
            }
            if (rc == -1 && errno == EINTR) {
                if (!i) {
                    return -1;
                }
            } else if (rc == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (WaitUntilReady(fd, POLLIN) == -1) {
                    if (rc == -1 && errno == EINTR) {
                        if (!i) {
                            return -1;
                        }
                    } else {
                        return -1;
                    }
                }
            } else if (rc == -1) {
                return -1;
            } else if (!rc) {
                if (!i) {
                    errno = e;
                    return 0;
                } else {
                    errno = EILSEQ;
                    return -1;
                }
            } else {
                break;
            }
        }
        if (i + 1 < n) {
            p[i] = c;
            p[i + 1] = 0;
        } else if (i < n) {
            p[i] = 0;
        }
        ++i;
        switch (t) {
        Whoopsie:
            if (n)
                p[0] = c;
            t = kAscii;
            i = 1;
            /* fallthrough */
        case kAscii:
            if (c < 0200) {
                if (c == 033) {
                    t = kEsc;
                } else {
                    t = kDone;
                }
            } else if (c >= 0300) {
                t = kUtf8;
                r = DecodeUtf8(c);
            } else {
                /* ignore overlong sequences */
            }
            break;
        case kUtf8:
            if ((c & 0300) == 0200) {
                r.c <<= 6;
                r.c |= c & 077;
                if (!--r.n) {
                    switch (r.c) {
                    case 033:
                        t = kEsc; /* parsed but not canonicalized */
                        break;
                    case 0x9b:
                        t = kCsi1; /* unusual but legal */
                        break;
                    case 0x8e: /* SS2 (Single Shift Two) */
                    case 0x8f: /* SS3 (Single Shift Three) */
                        t = kSs;
                        break;
                    case 0x90: /* DCS (Device Control String) */
                    case 0x98: /* SOS (Start of String) */
                    case 0x9d: /* OSC (Operating System Command) */
                    case 0x9e: /* PM  (Privacy Message) */
                    case 0x9f: /* APC (Application Program Command) */
                        t = kStr;
                        break;
                    default:
                        t = kDone;
                        break;
                    }
                }
            } else {
                goto Whoopsie; /* ignore underlong sequences if not eof */
            }
            break;
        case kEsc:
            if (0x20 <= c && c <= 0x2f) { /* Nf */
                /*
                 * Almost no one uses ANSI Nf sequences
                 * They overlaps with alt+graphic keystrokes
                 * We care more about being able to type alt-/
                 */
                if (c == ' ' || c == '#') {
                    t = kNf;
                } else {
                    t = kDone;
                }
            } else if (0x30 <= c && c <= 0x3f) { /* Fp */
                t = kDone;
            } else if (0x20 <= c && c <= 0x5F) { /* Fe */
                switch (c) {
                case '[':
                    t = kCsi1;
                    break;
                case 'N': /* SS2 (Single Shift Two) */
                case 'O': /* SS3 (Single Shift Three) */
                    t = kSs;
                    break;
                case 'P': /* DCS (Device Control String) */
                case 'X': /* SOS (Start of String) */
                case ']': /* OSC (Operating System Command) */
                case '^': /* PM  (Privacy Message) */
                case '_': /* APC (Application Program Command) */
                    t = kStr;
                    break;
                default:
                    t = kDone;
                    break;
                }
            } else if (0x60 <= c && c <= 0x7e) { /* Fs */
                t = kDone;
            } else if (c == 033) {
                if (i < 3) {
                    /* alt chording */
                } else {
                    t = kDone; /* esc mashing */
                    i = 1;
                }
            } else {
                t = kDone;
            }
            break;
        case kSs:
            t = kDone;
            break;
        case kNf:
            if (0x30 <= c && c <= 0x7e) {
                t = kDone;
            } else if (!(0x20 <= c && c <= 0x2f)) {
                goto Whoopsie;
            }
            break;
        case kCsi1:
            if (0x20 <= c && c <= 0x2f) {
                t = kCsi2;
            } else if (c == '[' && ((i == 3) || (i == 4 && p[1] == 033))) {
                /* linux function keys */
            } else if (0x40 <= c && c <= 0x7e) {
                t = kDone;
            } else if (!(0x30 <= c && c <= 0x3f)) {
                goto Whoopsie;
            }
            break;
        case kCsi2:
            if (0x40 <= c && c <= 0x7e) {
                t = kDone;
            } else if (!(0x20 <= c && c <= 0x2f)) {
                goto Whoopsie;
            }
            break;
        case kStr:
            switch (c) {
            case '\a':
                t = kDone;
                break;
            case 0033: /* ESC */
            case 0302: /* C1 (UTF-8) */
                t = kStr2;
                break;
            default:
                break;
            }
            break;
        case kStr2:
            switch (c) {
            case '\a':
            case '\\': /* ST (ASCII) */
            case 0234: /* ST (UTF-8) */
                t = kDone;
                break;
            default:
                t = kStr;
                break;
            }
            break;
        default:
            assert(0);
        }
    } while (t != kDone);
    errno = e;
    return i;
}

static char *GetLineChar(int fin, int fout) {
    size_t got;
    ssize_t rc;
    char seq[16];
    struct abuf a;
    struct sigaction sa[3];
    abInit(&a);
    gotint = 0;
    sigemptyset(&sa->sa_mask);
    sa->sa_flags = 0;
    sa->sa_handler = bestlineOnInt;
    sigaction(SIGINT, sa, sa + 1);
    sigaction(SIGQUIT, sa, sa + 2);
    for (;;) {
        if (gotint) {
            rc = -1;
            break;
        }
        if ((rc = bestlineReadCharacter(fin, seq, sizeof(seq))) == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (WaitUntilReady(fin, POLLIN) > 0) {
                    continue;
                }
            }
            if (errno == EINTR) {
                continue;
            } else {
                break;
            }
        }
        if (!(got = rc)) {
            if (a.len) {
                break;
            } else {
                rc = -1;
                break;
            }
        }
        if (seq[0] == '\r') {
            if (HasPendingInput(fin)) {
                if ((rc = bestlineReadCharacter(fin, seq + 1, sizeof(seq) - 1)) > 0) {
                    if (seq[0] == '\n') {
                        break;
                    }
                } else {
                    rc = -1;
                    break;
                }
            } else {
                _MyWrite(fout, "\n", 1);
                break;
            }
        } else if (seq[0] == Ctrl('D')) {
            break;
        } else if (seq[0] == '\n') {
            break;
        } else if (seq[0] == '\b') {
            while (a.len && (a.b[a.len - 1] & 0300) == 0200)
                --a.len;
            if (a.len)
                --a.len;
        }
        if (!IsControl(seq[0])) {
            abAppend(&a, seq, got);
        }
    }
    sigaction(SIGQUIT, sa + 2, 0);
    sigaction(SIGINT, sa + 1, 0);
    if (gotint) {
        abFree(&a);
        raise(gotint);
        errno = EINTR;
        rc = -1;
    }
    if (rc != -1) {
        return a.b;
    } else {
        abFree(&a);
        return 0;
    }
}

static char *GetLine(FILE *in, FILE *out) {
    if (!IsCharDev(fileno(in))) {
        return GetLineBlock(in);
    } else {
        return GetLineChar(fileno(in), fileno(out));
    }
}

static char *Copy(char *d, const char *s, size_t n) {
    memcpy(d, s, n);
    return d + n;
}

static int CompareStrings(const char *a, const char *b) {
    size_t i;
    int x, y, c;
    for (i = 0;; ++i) {
        x = bestlineLowercase(a[i] & 255);
        y = bestlineLowercase(b[i] & 255);
        if ((c = x - y) || !x) {
            return c;
        }
    }
}

static const char *FindSubstringReverse(const char *p, size_t n, const char *q, size_t m) {
    size_t i;
    if (m <= n) {
        n -= m;
        do {
            for (i = 0; i < m; ++i) {
                if (p[n + i] != q[i]) {
                    break;
                }
            }
            if (i == m) {
                return p + n;
            }
        } while (n--);
    }
    return 0;
}

static int ParseUnsigned(const char *s, void *e) {
    int c, x;
    for (x = 0; (c = *s++);) {
        if ('0' <= c && c <= '9') {
            x = Min(c - '0' + x * 10, 32767);
        } else {
            break;
        }
    }
    if (e)
        *(const char **)e = s;
    return x;
}

/**
 * Returns UNICODE CJK Monospace Width of string.
 *
 * Control codes and ANSI sequences have a width of zero. We only parse
 * a limited subset of ANSI here since we don't store ANSI codes in the
 * linenoiseState::buf, but we do encourage CSI color codes in prompts.
 */
static size_t GetMonospaceWidth(const char *p, size_t n, char *out_haswides) {
    int c, d;
    size_t i, w;
    struct rune r;
    char haswides;
    enum { kAscii, kUtf8, kEsc, kCsi1, kCsi2 } t;
    for (haswides = r.c = r.n = w = i = 0, t = kAscii; i < n; ++i) {
        c = p[i] & 255;
        switch (t) {
        Whoopsie:
            t = kAscii;
            /* fallthrough */
        case kAscii:
            if (c < 0200) {
                if (c == 033) {
                    t = kEsc;
                } else {
                    ++w;
                }
            } else if (c >= 0300) {
                t = kUtf8;
                r = DecodeUtf8(c);
            }
            break;
        case kUtf8:
            if ((c & 0300) == 0200) {
                r.c <<= 6;
                r.c |= c & 077;
                if (!--r.n) {
                    d = GetMonospaceCharacterWidth(r.c);
                    d = Max(0, d);
                    w += d;
                    haswides |= d > 1;
                    t = kAscii;
                    break;
                }
            } else {
                goto Whoopsie;
            }
            break;
        case kEsc:
            if (c == '[') {
                t = kCsi1;
            } else {
                t = kAscii;
            }
            break;
        case kCsi1:
            if (0x20 <= c && c <= 0x2f) {
                t = kCsi2;
            } else if (0x40 <= c && c <= 0x7e) {
                t = kAscii;
            } else if (!(0x30 <= c && c <= 0x3f)) {
                goto Whoopsie;
            }
            break;
        case kCsi2:
            if (0x40 <= c && c <= 0x7e) {
                t = kAscii;
            } else if (!(0x20 <= c && c <= 0x2f)) {
                goto Whoopsie;
            }
            break;
        default:
            assert(0);
        }
    }
    if (out_haswides) {
        *out_haswides = haswides;
    }
    return w;
}

static int bestlineIsUnsupportedTerm(void) {
    size_t i;
    char *term;
    static char once, res;
    if (!once) {
        if ((term = getenv("TERM"))) {
            for (i = 0; i < sizeof(kUnsupported) / sizeof(*kUnsupported); i++) {
                if (!CompareStrings(term, kUnsupported[i])) {
                    res = 1;
                    break;
                }
            }
        }
        once = 1;
    }
    return res;
}

static int enableRawMode(int fd) {
    struct termios raw;
    struct sigaction sa;
    if (tcgetattr(fd, &orig_termios) != -1) {
        raw = orig_termios;
        raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_iflag |= IUTF8;
        raw.c_cflag |= CS8;
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(fd, TCSANOW, &raw) != -1) {
            sa.sa_flags = 0;
            sa.sa_handler = bestlineOnCont;
            sigemptyset(&sa.sa_mask);
            sigaction(SIGCONT, &sa, &orig_cont);
            sa.sa_handler = bestlineOnWinch;
            sigaction(SIGWINCH, &sa, &orig_winch);
            rawmode = fd;
            gotwinch = 0;
            gotcont = 0;
            return 0;
        }
    }
    errno = ENOTTY;
    return -1;
}

static void bestlineUnpause(int fd) {
    if (ispaused) {
        tcflow(fd, TCOON);
        ispaused = 0;
    }
}

void bestlineDisableRawMode(void) {
    if (rawmode != -1) {
        bestlineUnpause(rawmode);
        sigaction(SIGCONT, &orig_cont, 0);
        sigaction(SIGWINCH, &orig_winch, 0);
        tcsetattr(rawmode, TCSANOW, &orig_termios);
        rawmode = -1;
    }
}

static int bestlineWrite(int fd, const void *p, size_t n) {
    ssize_t rc;
    size_t wrote;
    do {
        for (;;) {
            if (gotint) {
                errno = EINTR;
                return -1;
            }
            if (ispaused) {
                return 0;
            }
            rc = _MyWrite(fd, p, n);
            if (rc == -1 && errno == EINTR) {
                continue;
            } else if (rc == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (WaitUntilReady(fd, POLLOUT) == -1) {
                    if (errno == EINTR) {
                        continue;
                    } else {
                        return -1;
                    }
                }
            } else {
                break;
            }
        }
        if (rc != -1) {
            wrote = rc;
            n -= wrote;
            p = (char *)p + wrote;
        } else {
            return -1;
        }
    } while (n);
    return 0;
}

static int bestlineWriteStr(int fd, const char *p) {
    return bestlineWrite(fd, p, strlen(p));
}

static ssize_t bestlineRead(int fd, char *buf, size_t size, struct bestlineState *l) {
    size_t got;
    ssize_t rc;
    int refreshme;
    do {
        refreshme = 0;
        if (gotint) {
            errno = EINTR;
            return -1;
        }
        if (gotcont && rawmode != -1) {
            enableRawMode(rawmode);
            if (l)
                refreshme = 1;
        }
        if (gotwinch && l) {
            refreshme = 1;
        }
        if (refreshme)
            bestlineRefreshLine(l);
        rc = bestlineReadCharacter(fd, buf, size);
    } while (rc == -1 && errno == EINTR);
    if (rc != -1) {
        got = rc;
        if (got > 0 && l) {
            memcpy(l->seq[1], l->seq[0], sizeof(l->seq[0]));
            memset(l->seq[0], 0, sizeof(l->seq[0]));
            memcpy(l->seq[0], buf, Min(Min(size, got), sizeof(l->seq[0]) - 1));
        }
    }
    return rc;
}

/**
 * Returns number of columns in current terminal.
 *
 * 1. Checks COLUMNS environment variable (set by Emacs)
 * 2. Tries asking termios (works for pseudoteletypewriters)
 * 3. Falls back to inband signalling (works w/ pipe or serial)
 * 4. Otherwise we conservatively assume 80 columns
 *
 * @param ws should be initialized by caller to zero before first call
 * @param ifd is input file descriptor
 * @param ofd is output file descriptor
 * @return window size
 */
static struct winsize GetTerminalSize(struct winsize ws, int ifd, int ofd) {
    int x;
    ssize_t n;
    char *p, *s, b[16];
    ioctl(ofd, TIOCGWINSZ, &ws);
    if ((!ws.ws_row && (s = getenv("ROWS")) && (x = ParseUnsigned(s, 0)))) {
        ws.ws_row = x;
    }
    if ((!ws.ws_col && (s = getenv("COLUMNS")) && (x = ParseUnsigned(s, 0)))) {
        ws.ws_col = x;
    }
    if (((!ws.ws_col || !ws.ws_row) && bestlineRead(ifd, 0, 0, 0) != -1 &&
         bestlineWriteStr(ofd, "\0337" /* save position */
                               "\033[9979;9979H" /* move cursor to bottom right corner */
                               "\033[6n" /* report position */
                               "\0338") != -1 && /* restore position */
         (n = bestlineRead(ifd, b, sizeof(b), 0)) != -1 &&
         n && b[0] == 033 && b[1] == '[' && b[n - 1] == 'R')) {
        p = b + 2;
        if ((x = ParseUnsigned(p, &p)))
            ws.ws_row = x;
        if (*p++ == ';' && (x = ParseUnsigned(p, 0)))
            ws.ws_col = x;
    }
    if (!ws.ws_col)
        ws.ws_col = 80;
    if (!ws.ws_row)
        ws.ws_row = 24;
    return ws;
}

/* Clear the screen. Used to handle ctrl+l */
void bestlineClearScreen(int fd) {
    bestlineWriteStr(fd, "\033[H" /* move cursor to top left corner */
                         "\033[2J"); /* erase display */
}

static void bestlineBeep(void) {
    /* THE TERMINAL BELL IS DEAD - HISTORY HAS KILLED IT */
}

static char bestlineGrow(struct bestlineState *ls, size_t n) {
    char *p;
    size_t m;
    m = ls->buflen;
    if (m >= n)
        return 1;
    do
        m += m >> 1;
    while (m < n);
    if (!(p = (char *)realloc(ls->buf, m * sizeof(*ls->buf))))
        return 0;
    ls->buf = p;
    ls->buflen = m;
    return 1;
}

/* This is an helper function for bestlineEdit() and is called when the
 * user types the <tab> key in order to complete the string currently in the
 * input.
 *
 * The state of the editing is encapsulated into the pointed bestlineState
 * structure as described in the structure definition. */
static ssize_t bestlineCompleteLine(struct bestlineState *ls, char *seq, int size) {
    ssize_t nread;
    size_t i, n, stop;
    bestlineCompletions lc;
    struct bestlineState original, saved;
    nread = 0;
    memset(&lc, 0, sizeof(lc));
    completionCallback(ls->buf, ls->pos, &lc);
    if (!lc.len) {
        bestlineBeep();
    } else {
        i = 0;
        stop = 0;
        original = *ls;
        while (!stop) {
            /* Show completion or original buffer */
            if (i < lc.len) {
                saved = *ls;
                ls->len = strlen(lc.cvec[i]);
                ls->pos = original.pos + ls->len - original.len;
                ls->buf = lc.cvec[i];
                bestlineRefreshLine(ls);
                ls->len = saved.len;
                ls->pos = saved.pos;
                ls->buf = saved.buf;
            } else {
                bestlineRefreshLine(ls);
            }
            if ((nread = bestlineRead(ls->ifd, seq, size, ls)) <= 0) {
                bestlineFreeCompletions(&lc);
                return -1;
            }
            switch (seq[0]) {
            case '\t':
                i = (i + 1) % (lc.len + 1);
                if (i == lc.len) {
                    bestlineBeep();
                }
                break;
            default:
                if (i < lc.len) {
                    n = strlen(lc.cvec[i]);
                    if (bestlineGrow(ls, n + 1)) {
                        memcpy(ls->buf, lc.cvec[i], n + 1);
                        ls->len = n;
                        ls->pos = original.pos + n - original.len;
                    }
                }
                stop = 1;
                break;
            }
        }
    }
    bestlineFreeCompletions(&lc);
    return nread;
}

static void bestlineEditHistoryGoto(struct bestlineState *l, unsigned i) {
    size_t n;
    if (historylen <= 1)
        return;
    if (i > historylen - 1)
        return;
    i = Max(Min(i, historylen - 1), 0);
    free(history[historylen - 1 - l->hindex]);
    history[historylen - 1 - l->hindex] = strdup(l->buf);
    l->hindex = i;
    n = strlen(history[historylen - 1 - l->hindex]);
    bestlineGrow(l, n + 1);
    n = Min(n, l->buflen - 1);
    memcpy(l->buf, history[historylen - 1 - l->hindex], n);
    l->buf[n] = 0;
    l->len = l->pos = n;
    bestlineRefreshLine(l);
}

static void bestlineEditHistoryMove(struct bestlineState *l, int dx) {
    bestlineEditHistoryGoto(l, l->hindex + dx);
}

static char *bestlineMakeSearchPrompt(struct abuf *ab, int fail, const char *s, int n) {
    ab->len = 0;
    abAppendw(ab, '(');
    if (fail)
        abAppends(ab, "failed ");
    abAppends(ab, "reverse-i-search `\033[4m");
    abAppend(ab, s, n);
    abAppends(ab, "\033[24m");
    abAppends(ab, s + n);
    abAppendw(ab, Read32le("') "));
    return ab->b;
}

static int bestlineSearch(struct bestlineState *l, char *seq, int size) {
    char *p;
    char isstale;
    struct abuf ab;
    struct abuf prompt;
    unsigned i, j, k, matlen;
    const char *oldprompt, *q;
    int rc, fail, added, oldpos, oldindex;
    if (historylen <= 1)
        return 0;
    abInit(&ab);
    abInit(&prompt);
    oldpos = l->pos;
    oldprompt = l->prompt;
    oldindex = l->hindex;
    for (fail = matlen = 0;;) {
        l->prompt = bestlineMakeSearchPrompt(&prompt, fail, ab.b, matlen);
        bestlineRefreshLine(l);
        fail = 1;
        added = 0;
        j = l->pos;
        i = l->hindex;
        rc = bestlineRead(l->ifd, seq, size, l);
        if (rc > 0) {
            if (seq[0] == Ctrl('?') || seq[0] == Ctrl('H')) {
                if (ab.len) {
                    --ab.len;
                    matlen = Min(matlen, ab.len);
                }
            } else if (seq[0] == Ctrl('R')) {
                if (j) {
                    --j;
                } else if (i + 1 < historylen) {
                    ++i;
                    j = strlen(history[historylen - 1 - i]);
                }
            } else if (seq[0] == Ctrl('G')) {
                bestlineEditHistoryGoto(l, oldindex);
                l->pos = oldpos;
                rc = 0;
                break;
            } else if (IsControl(seq[0])) { /* only sees canonical c0 */
                break;
            } else {
                abAppend(&ab, seq, rc);
                added = rc;
            }
        } else {
            break;
        }
        isstale = 0;
        while (i < historylen) {
            p = history[historylen - 1 - i];
            k = strlen(p);
            if (!isstale) {
                j = Min(k, j + ab.len);
            } else {
                isstale = 0;
                j = k;
            }
            if ((q = FindSubstringReverse(p, j, ab.b, ab.len))) {
                bestlineEditHistoryGoto(l, i);
                l->pos = q - p;
                fail = 0;
                if (added) {
                    matlen += added;
                    added = 0;
                }
                break;
            } else {
                isstale = 1;
                ++i;
            }
        }
    }
    l->prompt = oldprompt;
    bestlineRefreshLine(l);
    abFree(&prompt);
    abFree(&ab);
    bestlineRefreshLine(l);
    return rc;
}

static void bestlineRingFree(void) {
    size_t i;
    for (i = 0; i < BESTLINE_MAX_RING; ++i) {
        if (ring.p[i]) {
            free(ring.p[i]);
            ring.p[i] = 0;
        }
    }
}

static void bestlineRingPush(const char *p, size_t n) {
    char *q;
    if (!n)
        return;
    if (!(q = (char *)malloc(n + 1)))
        return;
    ring.i = (ring.i + 1) % BESTLINE_MAX_RING;
    free(ring.p[ring.i]);
    ring.p[ring.i] = (char *)memcpy(q, p, n);
    ring.p[ring.i][n] = 0;
}

static void bestlineRingRotate(void) {
    size_t i;
    for (i = 0; i < BESTLINE_MAX_RING; ++i) {
        ring.i = (ring.i - 1) % BESTLINE_MAX_RING;
        if (ring.p[ring.i])
            break;
    }
}

static char *bestlineRefreshHints(struct bestlineState *l) {
    char *hint;
    struct abuf ab;
    const char *ansi1 = "\033[90m", *ansi2 = "\033[39m";
    if (!hintsCallback)
        return 0;
    if (!(hint = hintsCallback(l->buf, &ansi1, &ansi2)))
        return 0;
    abInit(&ab);
    if (ansi1)
        abAppends(&ab, ansi1);
    abAppends(&ab, hint);
    if (ansi2)
        abAppends(&ab, ansi2);
    if (freeHintsCallback)
        freeHintsCallback(hint);
    return ab.b;
}

static size_t Backward(struct bestlineState *l, size_t pos) {
    if (pos) {
        do
            --pos;
        while (pos && (l->buf[pos] & 0300) == 0200);
    }
    return pos;
}

static int bestlineEditMirrorLeft(struct bestlineState *l, int res[2]) {
    unsigned c, pos, left, right, depth, index;
    if ((pos = Backward(l, l->pos))) {
        right = GetUtf8(l->buf + pos, l->len - pos).c;
        if ((left = bestlineMirrorLeft(right))) {
            depth = 0;
            index = pos;
            do {
                pos = Backward(l, pos);
                c = GetUtf8(l->buf + pos, l->len - pos).c;
                if (c == right) {
                    ++depth;
                } else if (c == left) {
                    if (depth) {
                        --depth;
                    } else {
                        res[0] = pos;
                        res[1] = index;
                        return 0;
                    }
                }
            } while (pos);
        }
    }
    return -1;
}

static int bestlineEditMirrorRight(struct bestlineState *l, int res[2]) {
    struct rune rune;
    unsigned pos, left, right, depth, index;
    pos = l->pos;
    rune = GetUtf8(l->buf + pos, l->len - pos);
    left = rune.c;
    if ((right = bestlineMirrorRight(left))) {
        depth = 0;
        index = pos;
        do {
            pos += rune.n;
            rune = GetUtf8(l->buf + pos, l->len - pos);
            if (rune.c == left) {
                ++depth;
            } else if (rune.c == right) {
                if (depth) {
                    --depth;
                } else {
                    res[0] = index;
                    res[1] = pos;
                    return 0;
                }
            }
        } while (pos + rune.n < l->len);
    }
    return -1;
}

static int bestlineEditMirror(struct bestlineState *l, int res[2]) {
    int rc;
    rc = bestlineEditMirrorLeft(l, res);
    if (rc == -1)
        rc = bestlineEditMirrorRight(l, res);
    return rc;
}

static void bestlineRefreshLineImpl(struct bestlineState *l, int force) {
    char *hint;
    char flipit;
    char hasflip;
    char haswides;
    struct abuf ab;
    const char *buf;
    struct rune rune;
    struct winsize oldsize;
    int fd, plen, rows, len, pos;
    unsigned x, xn, yn, width, pwidth;
    int i, t, cx, cy, tn, resized, flip[2];

    /*
     * synchonize the i/o state
     */
    if (ispaused) {
        if (force) {
            bestlineUnpause(l->ofd);
        } else {
            return;
        }
    }
    if (!force && HasPendingInput(l->ifd)) {
        l->dirty = 1;
        return;
    }
    oldsize = l->ws;
    if ((resized = gotwinch) && rawmode != -1) {
        gotwinch = 0;
        l->ws = GetTerminalSize(l->ws, l->ifd, l->ofd);
    }
    hasflip = !l->final && !bestlineEditMirror(l, flip);

StartOver:
    fd = l->ofd;
    buf = l->buf;
    pos = l->pos;
    len = l->len;
    xn = l->ws.ws_col;
    yn = l->ws.ws_row;
    plen = strlen(l->prompt);
    pwidth = GetMonospaceWidth(l->prompt, plen, 0);
    width = GetMonospaceWidth(buf, len, &haswides);

    /*
     * handle the case where the line is larger than the whole display
     * gnu readline actually isn't able to deal with this situation!!!
     * we kludge xn to address the edge case of wide chars on the edge
     */
    for (tn = xn - haswides * 2;;) {
        if (pwidth + width + 1 < tn * yn)
            break; /* we're fine */
        if (!len || width < 2)
            break; /* we can't do anything */
        if (pwidth + 2 > tn * yn)
            break; /* we can't do anything */
        if (pos > len / 2) {
            /* hide content on the left if we're editing on the right */
            rune = GetUtf8(buf, len);
            buf += rune.n;
            len -= rune.n;
            pos -= rune.n;
        } else {
            /* hide content on the right if we're editing on left */
            t = len;
            while (len && (buf[len - 1] & 0300) == 0200)
                --len;
            if (len)
                --len;
            rune = GetUtf8(buf + len, t - len);
        }
        if ((t = GetMonospaceCharacterWidth(rune.c)) > 0) {
            width -= t;
        }
    }
    pos = Max(0, Min(pos, len));

    /*
     * now generate the terminal codes to update the line
     *
     * since we support unlimited lines it's important that we don't
     * clear the screen before we draw the screen. doing that causes
     * flickering. the key with terminals is to overwrite cells, and
     * then use \e[K and \e[J to clear everything else.
     *
     * we make the assumption that prompts and hints may contain ansi
     * sequences, but the buffer does not.
     *
     * we need to handle the edge case where a wide character like 度
     * might be at the edge of the window, when there's one cell left.
     * so we can't use division based on string width to compute the
     * coordinates and have to track it as we go.
     */
    cy = -1;
    cx = -1;
    rows = 1;
    abInit(&ab);
    abAppendw(&ab, '\r'); /* start of line */
    if (l->rows - l->oldpos - 1 > 0) {
        abAppends(&ab, "\033[");
        abAppendu(&ab, l->rows - l->oldpos - 1);
        abAppendw(&ab, 'A'); /* cursor up clamped */
    }
    abAppends(&ab, l->prompt);
    x = pwidth;
    for (i = 0; i < len; i += rune.n) {
        rune = GetUtf8(buf + i, len - i);
        if (x && x + rune.n > xn) {
            if (cy >= 0)
                ++cy;
            if (x < xn) {
                abAppends(&ab, "\033[K"); /* clear line forward */
            }
            abAppends(&ab, "\r" /* start of line */
                           "\n"); /* cursor down unclamped */
            ++rows;
            x = 0;
        }
        if (i == pos) {
            cy = 0;
            cx = x;
        }
        if (maskmode) {
            abAppendw(&ab, '*');
        } else {
            flipit = hasflip && (i == flip[0] || i == flip[1]);
            if (flipit)
                abAppends(&ab, "\033[1m");
            abAppendw(&ab, EncodeUtf8(rune.c));
            if (flipit)
                abAppends(&ab, "\033[22m");
        }
        t = GetMonospaceCharacterWidth(rune.c);
        t = Max(0, t);
        x += t;
    }
    if (!l->final && (hint = bestlineRefreshHints(l))) {
        if (GetMonospaceWidth(hint, strlen(hint), 0) < xn - x) {
            if (cx < 0) {
                cx = x;
            }
            abAppends(&ab, hint);
        }
        free(hint);
    }
    abAppendw(&ab, Read32le("\033[J")); /* erase display forwards */

    /*
     * if we are at the very end of the screen with our prompt, we need
     * to emit a newline and move the prompt to the first column.
     */
    if (pos && pos == len && x >= xn) {
        abAppendw(&ab, Read32le("\n\r\0"));
        ++rows;
    }

    /*
     * move cursor to right position
     */
    if (cy > 0) {
        abAppends(&ab, "\033[");
        abAppendu(&ab, cy);
        abAppendw(&ab, 'A'); /* cursor up */
    }
    if (cx > 0) {
        abAppendw(&ab, Read32le("\r\033["));
        abAppendu(&ab, cx);
        abAppendw(&ab, 'C'); /* cursor right */
    } else if (!cx) {
        abAppendw(&ab, '\r'); /* start */
    }

    /*
     * now get ready to progress state
     * we use a mostly correct kludge when the tty resizes
     */
    l->rows = rows;
    if (resized && oldsize.ws_col > l->ws.ws_col) {
        resized = 0;
        abFree(&ab);
        goto StartOver;
    }
    l->dirty = 0;
    l->oldpos = Max(0, cy);

    /*
     * send codes to terminal
     */
    bestlineWrite(fd, ab.b, ab.len);
    abFree(&ab);
}

static void bestlineRefreshLine(struct bestlineState *l) {
    bestlineRefreshLineImpl(l, 0);
}

static void bestlineRefreshLineForce(struct bestlineState *l) {
    bestlineRefreshLineImpl(l, 1);
}

static void bestlineEditInsert(struct bestlineState *l, const char *p, size_t n) {
    if (!bestlineGrow(l, l->len + n + 1))
        return;
    memmove(l->buf + l->pos + n, l->buf + l->pos, l->len - l->pos);
    memcpy(l->buf + l->pos, p, n);
    l->pos += n;
    l->len += n;
    l->buf[l->len] = 0;
    bestlineRefreshLine(l);
}

static void bestlineEditHome(struct bestlineState *l) {
    l->pos = 0;
    bestlineRefreshLine(l);
}

static void bestlineEditEnd(struct bestlineState *l) {
    l->pos = l->len;
    bestlineRefreshLine(l);
}

static void bestlineEditUp(struct bestlineState *l) {
    bestlineEditHistoryMove(l, BESTLINE_HISTORY_PREV);
}

static void bestlineEditDown(struct bestlineState *l) {
    bestlineEditHistoryMove(l, BESTLINE_HISTORY_NEXT);
}

static void bestlineEditBof(struct bestlineState *l) {
    bestlineEditHistoryGoto(l, historylen - 1);
}

static void bestlineEditEof(struct bestlineState *l) {
    bestlineEditHistoryGoto(l, 0);
}

static void bestlineEditRefresh(struct bestlineState *l) {
    bestlineClearScreen(l->ofd);
    bestlineRefreshLine(l);
}

static size_t Forward(struct bestlineState *l, size_t pos) {
    return pos + GetUtf8(l->buf + pos, l->len - pos).n;
}

static size_t Backwards(struct bestlineState *l, size_t pos, char pred(unsigned)) {
    size_t i;
    struct rune r;
    while (pos) {
        i = Backward(l, pos);
        r = GetUtf8(l->buf + i, l->len - i);
        if (pred(r.c)) {
            pos = i;
        } else {
            break;
        }
    }
    return pos;
}

static size_t Forwards(struct bestlineState *l, size_t pos, char pred(unsigned)) {
    struct rune r;
    while (pos < l->len) {
        r = GetUtf8(l->buf + pos, l->len - pos);
        if (pred(r.c)) {
            pos += r.n;
        } else {
            break;
        }
    }
    return pos;
}

static size_t ForwardWord(struct bestlineState *l, size_t pos) {
    pos = Forwards(l, pos, bestlineIsSeparator);
    pos = Forwards(l, pos, bestlineNotSeparator);
    return pos;
}

static size_t BackwardWord(struct bestlineState *l, size_t pos) {
    pos = Backwards(l, pos, bestlineIsSeparator);
    pos = Backwards(l, pos, bestlineNotSeparator);
    return pos;
}

static size_t EscapeWord(struct bestlineState *l, size_t i) {
    size_t j;
    struct rune r;
    for (; i && i < l->len; i += r.n) {
        if (i < l->len) {
            r = GetUtf8(l->buf + i, l->len - i);
            if (bestlineIsSeparator(r.c))
                break;
        }
        if ((j = i)) {
            do
                --j;
            while (j && (l->buf[j] & 0300) == 0200);
            r = GetUtf8(l->buf + j, l->len - j);
            if (bestlineIsSeparator(r.c))
                break;
        }
    }
    return i;
}

static void bestlineEditLeft(struct bestlineState *l) {
    l->pos = Backward(l, l->pos);
    bestlineRefreshLine(l);
}

static void bestlineEditRight(struct bestlineState *l) {
    if (l->pos == l->len)
        return;
    do
        l->pos++;
    while (l->pos < l->len && (l->buf[l->pos] & 0300) == 0200);
    bestlineRefreshLine(l);
}

static void bestlineEditLeftWord(struct bestlineState *l) {
    l->pos = BackwardWord(l, l->pos);
    bestlineRefreshLine(l);
}

static void bestlineEditRightWord(struct bestlineState *l) {
    l->pos = ForwardWord(l, l->pos);
    bestlineRefreshLine(l);
}

static void bestlineEditLeftExpr(struct bestlineState *l) {
    int mark[2];
    l->pos = Backwards(l, l->pos, bestlineIsXeparator);
    if (!bestlineEditMirrorLeft(l, mark)) {
        l->pos = mark[0];
    } else {
        l->pos = Backwards(l, l->pos, bestlineNotSeparator);
    }
    bestlineRefreshLine(l);
}

static void bestlineEditRightExpr(struct bestlineState *l) {
    int mark[2];
    l->pos = Forwards(l, l->pos, bestlineIsXeparator);
    if (!bestlineEditMirrorRight(l, mark)) {
        l->pos = Forward(l, mark[1]);
    } else {
        l->pos = Forwards(l, l->pos, bestlineNotSeparator);
    }
    bestlineRefreshLine(l);
}

static void bestlineEditDelete(struct bestlineState *l) {
    size_t i;
    if (l->pos == l->len)
        return;
    i = Forward(l, l->pos);
    memmove(l->buf + l->pos, l->buf + i, l->len - i + 1);
    l->len -= i - l->pos;
    bestlineRefreshLine(l);
}

static void bestlineEditRubout(struct bestlineState *l) {
    size_t i;
    if (!l->pos)
        return;
    i = Backward(l, l->pos);
    memmove(l->buf + i, l->buf + l->pos, l->len - l->pos + 1);
    l->len -= l->pos - i;
    l->pos = i;
    bestlineRefreshLine(l);
}

static void bestlineEditDeleteWord(struct bestlineState *l) {
    size_t i;
    if (l->pos == l->len)
        return;
    i = ForwardWord(l, l->pos);
    bestlineRingPush(l->buf + l->pos, i - l->pos);
    memmove(l->buf + l->pos, l->buf + i, l->len - i + 1);
    l->len -= i - l->pos;
    bestlineRefreshLine(l);
}

static void bestlineEditRuboutWord(struct bestlineState *l) {
    size_t i;
    if (!l->pos)
        return;
    i = BackwardWord(l, l->pos);
    bestlineRingPush(l->buf + i, l->pos - i);
    memmove(l->buf + i, l->buf + l->pos, l->len - l->pos + 1);
    l->len -= l->pos - i;
    l->pos = i;
    bestlineRefreshLine(l);
}

static void bestlineEditXlatWord(struct bestlineState *l, unsigned xlat(unsigned)) {
    unsigned c;
    size_t i, j;
    struct rune r;
    struct abuf ab;
    abInit(&ab);
    i = Forwards(l, l->pos, bestlineIsSeparator);
    for (j = i; j < l->len; j += r.n) {
        r = GetUtf8(l->buf + j, l->len - j);
        if (bestlineIsSeparator(r.c))
            break;
        if ((c = xlat(r.c)) != r.c) {
            abAppendw(&ab, EncodeUtf8(c));
        } else { /* avoid canonicalization */
            abAppend(&ab, l->buf + j, r.n);
        }
    }
    if (ab.len && bestlineGrow(l, i + ab.len + l->len - j + 1)) {
        l->pos = i + ab.len;
        abAppend(&ab, l->buf + j, l->len - j);
        l->len = i + ab.len;
        memcpy(l->buf + i, ab.b, ab.len + 1);
        bestlineRefreshLine(l);
    }
    abFree(&ab);
}

static void bestlineEditLowercaseWord(struct bestlineState *l) {
    bestlineEditXlatWord(l, bestlineLowercase);
}

static void bestlineEditUppercaseWord(struct bestlineState *l) {
    bestlineEditXlatWord(l, bestlineUppercase);
}

static void bestlineEditCapitalizeWord(struct bestlineState *l) {
    iscapital = 0;
    bestlineEditXlatWord(l, Capitalize);
}

static void bestlineEditKillLeft(struct bestlineState *l) {
    size_t diff, old_pos;
    bestlineRingPush(l->buf, l->pos);
    old_pos = l->pos;
    l->pos = 0;
    diff = old_pos - l->pos;
    memmove(l->buf + l->pos, l->buf + old_pos, l->len - old_pos + 1);
    l->len -= diff;
    bestlineRefreshLine(l);
}

static void bestlineEditKillRight(struct bestlineState *l) {
    bestlineRingPush(l->buf + l->pos, l->len - l->pos);
    l->buf[l->pos] = '\0';
    l->len = l->pos;
    bestlineRefreshLine(l);
}

static void bestlineEditYank(struct bestlineState *l) {
    char *p;
    size_t n;
    if (!ring.p[ring.i])
        return;
    n = strlen(ring.p[ring.i]);
    if (!bestlineGrow(l, l->len + n + 1))
        return;
    if (!(p = (char *)malloc(l->len - l->pos + 1)))
        return;
    memcpy(p, l->buf + l->pos, l->len - l->pos + 1);
    memcpy(l->buf + l->pos, ring.p[ring.i], n);
    memcpy(l->buf + l->pos + n, p, l->len - l->pos + 1);
    free(p);
    l->yi = l->pos;
    l->yj = l->pos + n;
    l->pos += n;
    l->len += n;
    bestlineRefreshLine(l);
}

static void bestlineEditRotate(struct bestlineState *l) {
    if ((l->seq[1][0] == Ctrl('Y') || (l->seq[1][0] == 033 && l->seq[1][1] == 'y'))) {
        if (l->yi < l->len && l->yj <= l->len) {
            memmove(l->buf + l->yi, l->buf + l->yj, l->len - l->yj + 1);
            l->len -= l->yj - l->yi;
            l->pos -= l->yj - l->yi;
        }
        bestlineRingRotate();
        bestlineEditYank(l);
    }
}

static void bestlineEditTranspose(struct bestlineState *l) {
    char *q, *p;
    size_t a, b, c;
    b = l->pos;
    if (b == l->len)
        --b;
    a = Backward(l, b);
    c = Forward(l, b);
    if (!(a < b && b < c))
        return;
    p = q = (char *)malloc(c - a);
    p = Copy(p, l->buf + b, c - b);
    p = Copy(p, l->buf + a, b - a);
    assert((size_t)(p - q) == c - a);
    memcpy(l->buf + a, q, p - q);
    l->pos = c;
    free(q);
    bestlineRefreshLine(l);
}

static void bestlineEditTransposeWords(struct bestlineState *l) {
    char *q, *p;
    size_t i, pi, xi, xj, yi, yj;
    i = l->pos;
    if (i == l->len) {
        i = Backwards(l, i, bestlineIsSeparator);
        i = Backwards(l, i, bestlineNotSeparator);
    }
    pi = EscapeWord(l, i);
    xj = Backwards(l, pi, bestlineIsSeparator);
    xi = Backwards(l, xj, bestlineNotSeparator);
    yi = Forwards(l, pi, bestlineIsSeparator);
    yj = Forwards(l, yi, bestlineNotSeparator);
    if (!(xi < xj && xj < yi && yi < yj))
        return;
    p = q = (char *)malloc(yj - xi);
    p = Copy(p, l->buf + yi, yj - yi);
    p = Copy(p, l->buf + xj, yi - xj);
    p = Copy(p, l->buf + xi, xj - xi);
    assert((size_t)(p - q) == yj - xi);
    memcpy(l->buf + xi, q, p - q);
    l->pos = yj;
    free(q);
    bestlineRefreshLine(l);
}

static void bestlineEditSqueeze(struct bestlineState *l) {
    size_t i, j;
    i = Backwards(l, l->pos, bestlineIsSeparator);
    j = Forwards(l, l->pos, bestlineIsSeparator);
    if (!(i < j))
        return;
    memmove(l->buf + i, l->buf + j, l->len - j + 1);
    l->len -= j - i;
    l->pos = i;
    bestlineRefreshLine(l);
}

static void bestlineEditMark(struct bestlineState *l) {
    l->mark = l->pos;
}

static void bestlineEditGoto(struct bestlineState *l) {
    if (l->mark > l->len)
        return;
    l->pos = Min(l->mark, l->len);
    bestlineRefreshLine(l);
}

static size_t bestlineEscape(char *d, const char *s, size_t n) {
    char *p;
    size_t i;
    unsigned c, w, l;
    for (p = d, l = i = 0; i < n; ++i) {
        switch ((c = s[i] & 255)) {
            Case('\a', w = Read16le("\\a"));
            Case('\b', w = Read16le("\\b"));
            Case('\t', w = Read16le("\\t"));
            Case('\n', w = Read16le("\\n"));
            Case('\v', w = Read16le("\\v"));
            Case('\f', w = Read16le("\\f"));
            Case('\r', w = Read16le("\\r"));
            Case('"', w = Read16le("\\\""));
            Case('\'', w = Read16le("\\\'"));
            Case('\\', w = Read16le("\\\\"));
        default:
            if (c <= 0x1F || c == 0x7F || (c == '?' && l == '?')) {
                w = Read16le("\\x");
                w |= "0123456789abcdef"[(c & 0xF0) >> 4] << 020;
                w |= "0123456789abcdef"[(c & 0x0F) >> 0] << 030;
            } else {
                w = c;
            }
            break;
        }
        p[0] = (w & 0x000000ff) >> 000;
        p[1] = (w & 0x0000ff00) >> 010;
        p[2] = (w & 0x00ff0000) >> 020;
        p[3] = (w & 0xff000000) >> 030;
        p += (Bsr(w) >> 3) + 1;
        l = w;
    }
    return p - d;
}

static void bestlineEditInsertEscape(struct bestlineState *l) {
    size_t m;
    ssize_t n;
    char seq[16];
    char esc[sizeof(seq) * 4];
    if ((n = bestlineRead(l->ifd, seq, sizeof(seq), l)) > 0) {
        m = bestlineEscape(esc, seq, n);
        bestlineEditInsert(l, esc, m);
    }
}

static void bestlineEditInterrupt(void) {
    gotint = SIGINT;
}

static void bestlineEditQuit(void) {
    gotint = SIGQUIT;
}

static void bestlineEditSuspend(void) {
    raise(SIGSTOP);
}

static void bestlineEditPause(struct bestlineState *l) {
    tcflow(l->ofd, TCOOFF);
    ispaused = 1;
}

static void bestlineEditCtrlq(struct bestlineState *l) {
    if (ispaused) {
        bestlineUnpause(l->ofd);
        bestlineRefreshLineForce(l);
    } else {
        bestlineEditInsertEscape(l);
    }
}

/**
 * Moves last item inside current s-expression to outside, e.g.
 *
 *     (a| b c)
 *     (a| b) c
 *
 * The cursor position changes only if a paren is moved before it:
 *
 *     (a b    c   |)
 *     (a b)    c   |
 *
 * To accommodate non-LISP languages we connect unspaced outer symbols:
 *
 *     f(a,| b, g())
 *     f(a,| b), g()
 *
 * Our standard keybinding is ALT-SHIFT-B.
 */
static void bestlineEditBarf(struct bestlineState *l) {
    struct rune r;
    unsigned long w;
    size_t i, pos, depth = 0;
    unsigned lhs, rhs, end, *stack = 0;
    /* go as far right within current s-expr as possible */
    for (pos = l->pos;; pos += r.n) {
        if (pos == l->len)
            goto Finish;
        r = GetUtf8(l->buf + pos, l->len - pos);
        if (depth) {
            if (r.c == stack[depth - 1]) {
                --depth;
            }
        } else {
            if ((rhs = bestlineMirrorRight(r.c))) {
                stack = (unsigned *)realloc(stack, ++depth * sizeof(*stack));
                stack[depth - 1] = rhs;
            } else if (bestlineMirrorLeft(r.c)) {
                end = pos;
                break;
            }
        }
    }
    /* go back one item */
    pos = Backwards(l, pos, bestlineIsXeparator);
    for (;; pos = i) {
        if (!pos)
            goto Finish;
        i = Backward(l, pos);
        r = GetUtf8(l->buf + i, l->len - i);
        if (depth) {
            if (r.c == stack[depth - 1]) {
                --depth;
            }
        } else {
            if ((lhs = bestlineMirrorLeft(r.c))) {
                stack = (unsigned *)realloc(stack, ++depth * sizeof(*stack));
                stack[depth - 1] = lhs;
            } else if (bestlineIsSeparator(r.c)) {
                break;
            }
        }
    }
    pos = Backwards(l, pos, bestlineIsXeparator);
    /* now move the text */
    r = GetUtf8(l->buf + end, l->len - end);
    memmove(l->buf + pos + r.n, l->buf + pos, end - pos);
    w = EncodeUtf8(r.c);
    for (i = 0; i < r.n; ++i) {
        l->buf[pos + i] = w;
        w >>= 8;
    }
    if (l->pos > pos) {
        l->pos += r.n;
    }
    bestlineRefreshLine(l);
Finish:
    free(stack);
}

/**
 * Moves first item outside current s-expression to inside, e.g.
 *
 *     (a| b) c d
 *     (a| b c) d
 *
 * To accommodate non-LISP languages we connect unspaced outer symbols:
 *
 *     f(a,| b), g()
 *     f(a,| b, g())
 *
 * Our standard keybinding is ALT-SHIFT-S.
 */
static void bestlineEditSlurp(struct bestlineState *l) {
    char rp[6];
    struct rune r;
    size_t pos, depth = 0;
    unsigned rhs, point = 0, start = 0, *stack = 0;
    /* go to outside edge of current s-expr */
    for (pos = l->pos; pos < l->len; pos += r.n) {
        r = GetUtf8(l->buf + pos, l->len - pos);
        if (depth) {
            if (r.c == stack[depth - 1]) {
                --depth;
            }
        } else {
            if ((rhs = bestlineMirrorRight(r.c))) {
                stack = (unsigned *)realloc(stack, ++depth * sizeof(*stack));
                stack[depth - 1] = rhs;
            } else if (bestlineMirrorLeft(r.c)) {
                point = pos;
                pos += r.n;
                start = pos;
                break;
            }
        }
    }
    /* go forward one item */
    pos = Forwards(l, pos, bestlineIsXeparator);
    for (; pos < l->len; pos += r.n) {
        r = GetUtf8(l->buf + pos, l->len - pos);
        if (depth) {
            if (r.c == stack[depth - 1]) {
                --depth;
            }
        } else {
            if ((rhs = bestlineMirrorRight(r.c))) {
                stack = (unsigned *)realloc(stack, ++depth * sizeof(*stack));
                stack[depth - 1] = rhs;
            } else if (bestlineIsSeparator(r.c)) {
                break;
            }
        }
    }
    /* now move the text */
    memcpy(rp, l->buf + point, start - point);
    memmove(l->buf + point, l->buf + start, pos - start);
    memcpy(l->buf + pos - (start - point), rp, start - point);
    bestlineRefreshLine(l);
    free(stack);
}

static void bestlineEditRaise(struct bestlineState *l) {
    (void)l;
}

static char IsBalanced(struct abuf *buf) {
    unsigned i, d;
    for (d = i = 0; i < buf->len; ++i) {
        if (buf->b[i] == '(')
            ++d;
        else if (d > 0 && buf->b[i] == ')')
            --d;
    }
    return d == 0;
}

/**
 * Runs bestline engine.
 *
 * This function is the core of the line editing capability of bestline.
 * It expects 'fd' to be already in "raw mode" so that every key pressed
 * will be returned ASAP to read().
 *
 * The resulting string is put into 'buf' when the user type enter, or
 * when ctrl+d is typed.
 *
 * Returns chomped character count in buf >=0 or -1 on eof / error
 */
static ssize_t bestlineEdit(int stdin_fd, int stdout_fd, const char *prompt, const char *init,
                            char **obuf) {
    ssize_t rc;
    char seq[16];
    const char *promptnotnull, *promptlastnl;
    size_t nread;
    int pastemode;
    struct rune rune;
    unsigned long long w;
    struct bestlineState l;
    pastemode = 0;
    memset(&l, 0, sizeof(l));
    if (!(l.buf = (char *)malloc((l.buflen = 32))))
        return -1;
    l.buf[0] = 0;
    l.ifd = stdin_fd;
    l.ofd = stdout_fd;
    promptnotnull = prompt ? prompt : "";
    promptlastnl = strrchr(promptnotnull, '\n');
    l.prompt = promptlastnl ? promptlastnl + 1 : promptnotnull;
    l.ws = GetTerminalSize(l.ws, l.ifd, l.ofd);
    abInit(&l.full);
    bestlineHistoryAdd("");
    bestlineWriteStr(l.ofd, promptnotnull);
    init = init ? init : "";
    bestlineEditInsert(&l, init, strlen(init));
    while (1) {
        if (l.dirty)
            bestlineRefreshLineForce(&l);
        rc = bestlineRead(l.ifd, seq, sizeof(seq), &l);
        if (rc > 0) {
            if (seq[0] == Ctrl('R')) {
                rc = bestlineSearch(&l, seq, sizeof(seq));
                if (!rc)
                    continue;
            } else if (seq[0] == '\t' && completionCallback) {
                rc = bestlineCompleteLine(&l, seq, sizeof(seq));
                if (!rc)
                    continue;
            }
        }
        if (rc > 0) {
            nread = rc;
        } else if (!rc && l.len) {
            nread = 1;
            seq[0] = '\r';
            seq[1] = 0;
        } else {
            if (historylen) {
                free(history[--historylen]);
                history[historylen] = 0;
            }
            free(l.buf);
            abFree(&l.full);
            return -1;
        }
        switch (seq[0]) {
            Case(Ctrl('P'), bestlineEditUp(&l));
            Case(Ctrl('E'), bestlineEditEnd(&l));
            Case(Ctrl('N'), bestlineEditDown(&l));
            Case(Ctrl('A'), bestlineEditHome(&l));
            Case(Ctrl('B'), bestlineEditLeft(&l));
            Case(Ctrl('@'), bestlineEditMark(&l));
            Case(Ctrl('Y'), bestlineEditYank(&l));
            Case(Ctrl('Q'), bestlineEditCtrlq(&l));
            Case(Ctrl('F'), bestlineEditRight(&l));
            Case(Ctrl('\\'), bestlineEditQuit());
            Case(Ctrl('S'), bestlineEditPause(&l));
            Case(Ctrl('?'), bestlineEditRubout(&l));
            Case(Ctrl('H'), bestlineEditRubout(&l));
            Case(Ctrl('L'), bestlineEditRefresh(&l));
            Case(Ctrl('Z'), bestlineEditSuspend());
            Case(Ctrl('U'), bestlineEditKillLeft(&l));
            Case(Ctrl('T'), bestlineEditTranspose(&l));
            Case(Ctrl('K'), bestlineEditKillRight(&l));
            Case(Ctrl('W'), bestlineEditRuboutWord(&l));
        case Ctrl('C'):
            if (emacsmode) {
                if (bestlineRead(l.ifd, seq, sizeof(seq), &l) != 1)
                    break;
                switch (seq[0]) {
                    Case(Ctrl('C'), bestlineEditInterrupt());
                    Case(Ctrl('B'), bestlineEditBarf(&l));
                    Case(Ctrl('S'), bestlineEditSlurp(&l));
                    Case(Ctrl('R'), bestlineEditRaise(&l));
                default:
                    break;
                }
            } else {
                bestlineEditInterrupt();
            }
            break;
        case Ctrl('X'):
            if (l.seq[1][0] == Ctrl('X')) {
                bestlineEditGoto(&l);
            }
            break;
        case Ctrl('D'):
            if (l.len) {
                bestlineEditDelete(&l);
            } else {
                if (historylen) {
                    free(history[--historylen]);
                    history[historylen] = 0;
                }
                free(l.buf);
                abFree(&l.full);
                return -1;
            }
            break;
        case '\n':
            l.final = 1;
            bestlineEditEnd(&l);
            bestlineRefreshLineForce(&l);
            l.final = 0;
            abAppend(&l.full, l.buf, l.len);
            l.prompt = "... ";
            abAppends(&l.full, "\n");
            l.len = 0;
            l.pos = 0;
            bestlineWriteStr(stdout_fd, "\r\n");
            bestlineRefreshLineForce(&l);
            break;
        case '\r': {
            char is_finished = 1;
            char needs_strip = 0;
            if (historylen) {
                free(history[--historylen]);
                history[historylen] = 0;
            }
            l.final = 1;
            bestlineEditEnd(&l);
            bestlineRefreshLineForce(&l);
            l.final = 0;
            abAppend(&l.full, l.buf, l.len);
            if (pastemode)
                is_finished = 0;
            if (balancemode)
                if (!IsBalanced(&l.full))
                    is_finished = 0;
            if (llamamode)
                if (StartsWith(l.full.b, "\"\"\""))
                    needs_strip = is_finished = l.full.len > 6 && EndsWith(l.full.b, "\"\"\"");
            if (is_finished) {
                if (needs_strip) {
                    int len = l.full.len - 6;
                    *obuf = strndup(l.full.b + 3, len);
                    abFree(&l.full);
                    free(l.buf);
                    return len;
                } else {
                    *obuf = l.full.b;
                    free(l.buf);
                    return l.full.len;
                }
            } else {
                l.prompt = "... ";
                abAppends(&l.full, "\n");
                l.len = 0;
                l.pos = 0;
                bestlineWriteStr(stdout_fd, "\r\n");
                bestlineRefreshLineForce(&l);
            }
            break;
        }
        case 033:
            if (nread < 2)
                break;
            switch (seq[1]) {
                Case('<', bestlineEditBof(&l));
                Case('>', bestlineEditEof(&l));
                Case('B', bestlineEditBarf(&l));
                Case('S', bestlineEditSlurp(&l));
                Case('R', bestlineEditRaise(&l));
                Case('y', bestlineEditRotate(&l));
                Case('\\', bestlineEditSqueeze(&l));
                Case('b', bestlineEditLeftWord(&l));
                Case('f', bestlineEditRightWord(&l));
                Case('h', bestlineEditRuboutWord(&l));
                Case('d', bestlineEditDeleteWord(&l));
                Case('l', bestlineEditLowercaseWord(&l));
                Case('u', bestlineEditUppercaseWord(&l));
                Case('c', bestlineEditCapitalizeWord(&l));
                Case('t', bestlineEditTransposeWords(&l));
                Case(Ctrl('B'), bestlineEditLeftExpr(&l));
                Case(Ctrl('F'), bestlineEditRightExpr(&l));
                Case(Ctrl('H'), bestlineEditRuboutWord(&l));
            case '[':
                if (nread == 6 && !memcmp(seq, "\033[200~", 6)) {
                    pastemode = 1;
                    break;
                }
                if (nread == 6 && !memcmp(seq, "\033[201~", 6)) {
                    pastemode = 0;
                    break;
                }
                if (nread < 3)
                    break;
                if (seq[2] >= '0' && seq[2] <= '9') {
                    if (nread < 4)
                        break;
                    if (seq[3] == '~') {
                        switch (seq[2]) {
                            Case('1', bestlineEditHome(&l)); /* \e[1~ */
                            Case('3', bestlineEditDelete(&l)); /* \e[3~ */
                            Case('4', bestlineEditEnd(&l)); /* \e[4~ */
                        default:
                            break;
                        }
                    }
                } else {
                    switch (seq[2]) {
                        Case('A', bestlineEditUp(&l));
                        Case('B', bestlineEditDown(&l));
                        Case('C', bestlineEditRight(&l));
                        Case('D', bestlineEditLeft(&l));
                        Case('H', bestlineEditHome(&l));
                        Case('F', bestlineEditEnd(&l));
                    default:
                        break;
                    }
                }
                break;
            case 'O':
                if (nread < 3)
                    break;
                switch (seq[2]) {
                    Case('A', bestlineEditUp(&l));
                    Case('B', bestlineEditDown(&l));
                    Case('C', bestlineEditRight(&l));
                    Case('D', bestlineEditLeft(&l));
                    Case('H', bestlineEditHome(&l));
                    Case('F', bestlineEditEnd(&l));
                default:
                    break;
                }
                break;
            case 033:
                if (nread < 3)
                    break;
                switch (seq[2]) {
                case '[':
                    if (nread < 4)
                        break;
                    switch (seq[3]) {
                        Case('C', bestlineEditRightExpr(&l)); /* \e\e[C alt-right */
                        Case('D', bestlineEditLeftExpr(&l)); /* \e\e[D alt-left */
                    default:
                        break;
                    }
                    break;
                case 'O':
                    if (nread < 4)
                        break;
                    switch (seq[3]) {
                        Case('C', bestlineEditRightExpr(&l)); /* \e\eOC alt-right */
                        Case('D', bestlineEditLeftExpr(&l)); /* \e\eOD alt-left */
                    default:
                        break;
                    }
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
            break;
        default:
            if (!IsControl(seq[0])) { /* only sees canonical c0 */
                if (xlatCallback) {
                    rune = GetUtf8(seq, nread);
                    w = EncodeUtf8(xlatCallback(rune.c));
                    nread = 0;
                    do {
                        seq[nread++] = w;
                    } while ((w >>= 8));
                }
                bestlineEditInsert(&l, seq, nread);
            }
            break;
        }
    }
}

void bestlineFree(void *ptr) {
    free(ptr);
}

void bestlineHistoryFree(void) {
    size_t i;
    for (i = 0; i < BESTLINE_MAX_HISTORY; i++) {
        if (history[i]) {
            free(history[i]);
            history[i] = 0;
        }
    }
    historylen = 0;
}

static void bestlineAtExit(void) {
    bestlineDisableRawMode();
    bestlineHistoryFree();
    bestlineRingFree();
}

int bestlineHistoryAdd(const char *line) {
    char *linecopy;
    if (!BESTLINE_MAX_HISTORY)
        return 0;
    if (historylen && !strcmp(history[historylen - 1], line))
        return 0;
    if (!(linecopy = strdup(line)))
        return 0;
    if (historylen == BESTLINE_MAX_HISTORY) {
        free(history[0]);
        memmove(history, history + 1, sizeof(char *) * (BESTLINE_MAX_HISTORY - 1));
        historylen--;
    }
    history[historylen++] = linecopy;
    return 1;
}

/**
 * Saves line editing history to file.
 *
 * @return 0 on success, or -1 w/ errno
 */
int bestlineHistorySave(const char *filename) {
    FILE *fp;
    unsigned j;
    mode_t old_umask;
    old_umask = umask(S_IXUSR | S_IRWXG | S_IRWXO);
    fp = fopen(filename, "w");
    umask(old_umask);
    if (!fp)
        return -1;
    chmod(filename, S_IRUSR | S_IWUSR);
    for (j = 0; j < historylen; j++) {
        fputs(history[j], fp);
        fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

/**
 * Loads history from the specified file.
 *
 * If the file doesn't exist, zero is returned and this will do nothing.
 * If the file does exists and the operation succeeded zero is returned
 * otherwise on error -1 is returned.
 *
 * @return 0 on success, or -1 w/ errno
 */
int bestlineHistoryLoad(const char *filename) {
    char **h;
    int rc, fd, err;
    size_t i, j, k, n, t;
    char *m, *e, *p, *q, *f, *s;
    err = errno, rc = 0;
    if (!BESTLINE_MAX_HISTORY)
        return 0;
    if (!(h = (char **)calloc(2 * BESTLINE_MAX_HISTORY, sizeof(char *))))
        return -1;
    if ((fd = open(filename, O_RDONLY)) != -1) {
        if ((n = GetFdSize(fd))) {
            if ((m = (char *)mmap(0, n, PROT_READ, MAP_SHARED, fd, 0)) != MAP_FAILED) {
                for (i = 0, e = (p = m) + n; p < e; p = f + 1) {
                    if (!(q = (char *)memchr(p, '\n', e - p)))
                        q = e;
                    for (f = q; q > p; --q) {
                        if (q[-1] != '\n' && q[-1] != '\r')
                            break;
                    }
                    if (q > p) {
                        h[i * 2 + 0] = p;
                        h[i * 2 + 1] = q;
                        i = (i + 1) % BESTLINE_MAX_HISTORY;
                    }
                }
                bestlineHistoryFree();
                for (j = 0; j < BESTLINE_MAX_HISTORY; ++j) {
                    if (h[(k = (i + j) % BESTLINE_MAX_HISTORY) * 2]) {
                        if ((s = (char *)malloc((t = h[k * 2 + 1] - h[k * 2]) + 1))) {
                            memcpy(s, h[k * 2], t), s[t] = 0;
                            history[historylen++] = s;
                        }
                    }
                }
                munmap(m, n);
            } else {
                rc = -1;
            }
        }
        close(fd);
    } else if (errno == ENOENT) {
        errno = err;
    } else {
        rc = -1;
    }
    free(h);
    return rc;
}

/**
 * Like bestlineRaw, but with the additional parameter init used as the buffer
 * initial value.
 */
char *bestlineRawInit(const char *prompt, const char *init, int infd, int outfd) {
    char *buf;
    ssize_t rc;
    static char once;
    struct sigaction sa[3];
    if (!once)
        atexit(bestlineAtExit), once = 1;
    if (enableRawMode(infd) == -1)
        return 0;
    buf = 0;
    gotint = 0;
    sigemptyset(&sa->sa_mask);
    sa->sa_flags = 0;
    sa->sa_handler = bestlineOnInt;
    sigaction(SIGINT, sa, sa + 1);
    sigaction(SIGQUIT, sa, sa + 2);
    bestlineWriteStr(outfd, "\033[?2004h"); // enable bracketed paste mode
    rc = bestlineEdit(infd, outfd, prompt, init, &buf);
    bestlineWriteStr(outfd, "\033[?2004l"); // disable bracketed paste mode
    bestlineDisableRawMode();
    sigaction(SIGQUIT, sa + 2, 0);
    sigaction(SIGINT, sa + 1, 0);
    if (gotint) {
        free(buf);
        buf = 0;
        raise(gotint);
        errno = EINTR;
        rc = -1;
    }
    bestlineWriteStr(outfd, "\r\n");
    if (rc != -1) {
        return buf;
    } else {
        free(buf);
        return 0;
    }
}

/**
 * Reads line interactively.
 *
 * This function can be used instead of bestline() in cases where we
 * know for certain we're dealing with a terminal, which means we can
 * avoid linking any stdio code.
 *
 * @return chomped allocated string of read line or null on eof/error
 */
char *bestlineRaw(const char *prompt, int infd, int outfd) {
    return bestlineRawInit(prompt, "", infd, outfd);
}

/**
 * Like bestline, but with the additional parameter init used as the buffer
 * initial value. The init parameter is only used if the terminal has basic
 * capabilites.
 */
char *bestlineInit(const char *prompt, const char *init) {
    if (prompt && *prompt && (strchr(prompt, '\t') || strchr(prompt + 1, '\r'))) {
        errno = EINVAL;
        return 0;
    }
    if ((!isatty(fileno(stdin)) || !isatty(fileno(stdout)))) {
        if (prompt && *prompt && (IsCharDev(fileno(stdin)) && IsCharDev(fileno(stdout)))) {
            fputs(prompt, stdout);
            fflush(stdout);
        }
        return GetLine(stdin, stdout);
    } else if (bestlineIsUnsupportedTerm()) {
        if (prompt && *prompt) {
            fputs(prompt, stdout);
            fflush(stdout);
        }
        return GetLine(stdin, stdout);
    } else {
        fflush(stdout);
        return bestlineRawInit(prompt, init, fileno(stdin), fileno(stdout));
    }
}

/**
 * Reads line intelligently.
 *
 * The high level function that is the main API of the bestline library.
 * This function checks if the terminal has basic capabilities, just checking
 * for a blacklist of inarticulate terminals, and later either calls the line
 * editing function or uses dummy fgets() so that you will be able to type
 * something even in the most desperate of the conditions.
 *
 * @param prompt is printed before asking for input if we have a term
 *     and this may be set to empty or null to disable and prompt may
 *     contain ansi escape sequences, color, utf8, etc.
 * @return chomped allocated string of read line or null on eof/error
 */
char *bestline(const char *prompt) {
    return bestlineInit(prompt, "");
}

/**
 * Reads line intelligently w/ history, e.g.
 *
 *     // see ~/.foo_history
 *     main() {
 *         char *line;
 *         while ((line = bestlineWithHistory("IN> ", "foo"))) {
 *             printf("OUT> %s\n", line);
 *             free(line);
 *         }
 *     }
 *
 * @param prompt is printed before asking for input if we have a term
 *     and this may be set to empty or null to disable and prompt may
 *     contain ansi escape sequences, color, utf8, etc.
 * @param prog is name of your app, used to generate history filename
 *     however if it contains a slash / dot then we'll assume prog is
 *     the history filename which as determined by the caller
 * @return chomped allocated string of read line or null on eof/error
 */
char *bestlineWithHistory(const char *prompt, const char *prog) {
    char *line;
    struct abuf path;
    const char *a, *b;
    abInit(&path);
    if (prog) {
        if (strchr(prog, '/') || strchr(prog, '.')) {
            abAppends(&path, prog);
        } else {
            b = "";
            if (!(a = getenv("HOME"))) {
                if (!(a = getenv("HOMEDRIVE")) || !(b = getenv("HOMEPATH"))) {
                    a = "";
                }
            }
            if (*a) {
                abAppends(&path, a);
                abAppends(&path, b);
                abAppendw(&path, '/');
            }
            abAppendw(&path, '.');
            abAppends(&path, prog);
            abAppends(&path, "_history");
        }
    }
    if (path.len) {
        bestlineHistoryLoad(path.b);
    }
    line = bestline(prompt);
    if (path.len && line && *line) {
        /* history here is inefficient but helpful when the user has multiple
         * repls open at the same time, so history propagates between them */
        bestlineHistoryLoad(path.b);
        bestlineHistoryAdd(line);
        bestlineHistorySave(path.b);
    }
    abFree(&path);
    return line;
}

/**
 * Registers tab completion callback.
 */
void bestlineSetCompletionCallback(bestlineCompletionCallback *fn) {
    completionCallback = fn;
}

/**
 * Registers hints callback.
 *
 * Register a hits function to be called to show hits to the user at the
 * right of the prompt.
 */
void bestlineSetHintsCallback(bestlineHintsCallback *fn) {
    hintsCallback = fn;
}

/**
 * Sets free hints callback.
 *
 * This registers a function to free the hints returned by the hints
 * callback registered with bestlineSetHintsCallback().
 */
void bestlineSetFreeHintsCallback(bestlineFreeHintsCallback *fn) {
    freeHintsCallback = fn;
}

/**
 * Sets character translation callback.
 */
void bestlineSetXlatCallback(bestlineXlatCallback *fn) {
    xlatCallback = fn;
}

/**
 * Adds completion.
 *
 * This function is used by the callback function registered by the user
 * in order to add completion options given the input string when the
 * user typed <tab>. See the example.c source code for a very easy to
 * understand example.
 */
void bestlineAddCompletion(bestlineCompletions *lc, const char *str) {
    size_t len;
    char *copy, **cvec;
    if ((copy = (char *)malloc((len = strlen(str)) + 1))) {
        memcpy(copy, str, len + 1);
        if ((cvec = (char **)realloc(lc->cvec, (lc->len + 1) * sizeof(*lc->cvec)))) {
            lc->cvec = cvec;
            lc->cvec[lc->len++] = copy;
        } else {
            free(copy);
        }
    }
}

/**
 * Frees list of completion option populated by bestlineAddCompletion().
 */
void bestlineFreeCompletions(bestlineCompletions *lc) {
    size_t i;
    for (i = 0; i < lc->len; i++)
        free(lc->cvec[i]);
    if (lc->cvec)
        free(lc->cvec);
}

/**
 * Enables "mask mode".
 *
 * When it is enabled, instead of the input that the user is typing, the
 * terminal will just display a corresponding number of asterisks, like
 * "****". This is useful for passwords and other secrets that should
 * not be displayed.
 *
 * @see bestlineMaskModeDisable()
 */
void bestlineMaskModeEnable(void) {
    maskmode = 1;
}

/**
 * Disables "mask mode".
 */
void bestlineMaskModeDisable(void) {
    maskmode = 0;
}

/**
 * Enables or disables "balance mode".
 *
 * When it is enabled, bestline() will block until parentheses are
 * balanced. This is useful for code but not for free text.
 */
void bestlineBalanceMode(char mode) {
    balancemode = mode;
}

/**
 * Enables or disables "ollama mode".
 *
 * This enables you to type multiline input by putting triple quotes at
 * the beginning and end. For example:
 *
 *     >>> """
 *     ... second line
 *     ... third line
 *     ... """
 *
 * Would yield the string `"\nsecond line\nthird line\n"`.
 *
 * @param mode is 1 to enable, or 0 to disable
 */
void bestlineLlamaMode(char mode) {
    llamamode = mode;
}

/**
 * Enables Emacs mode.
 *
 * This mode remaps CTRL-C so you can use additional shortcuts, like C-c
 * C-s for slurp. By default, CTRL-C raises SIGINT for exiting programs.
 */
void bestlineEmacsMode(char mode) {
    emacsmode = mode;
}

/**
 * Allows implementation of user functions for read, write, and poll
 * with the intention of polling for background I/O.
 */

static int MyRead(int fd, void *c, int n) {
    return read(fd, c, n);
}

static int MyWrite(int fd, const void *c, int n) {
    return write(fd, c, n);
}

static int MyPoll(int fd, int events, int to) {
    struct pollfd p[1];
    p[0].fd = fd;
    p[0].events = events;
    return poll(p, 1, to);
}

void bestlineUserIO(int (*userReadFn)(int, void *, int), int (*userWriteFn)(int, const void *, int),
                    int (*userPollFn)(int, int, int)) {
    if (userReadFn)
        _MyRead = userReadFn;
    else
        _MyRead = MyRead;
    if (userWriteFn)
        _MyWrite = userWriteFn;
    else
        _MyWrite = MyWrite;
    if (userPollFn)
        _MyPoll = userPollFn;
    else
        _MyPoll = MyPoll;
}
