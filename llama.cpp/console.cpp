// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;tab-width:8;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi

#include "console.h"

#include <vector>
#include <iostream>
#include <climits>
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <termios.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

namespace console {

//
// Console state
//

static bool      advanced_display = false;
static bool      simple_io        = true;
static bool      should_close_tty = false;
static display_t current_display  = reset;
static FILE*     out              = stdout;
static FILE*     tty              = nullptr;
static termios   initial_state;

//
// Init and cleanup
//

void init(bool use_simple_io, bool use_advanced_display) {
    should_close_tty = false;
    simple_io = use_simple_io;
    advanced_display = use_advanced_display;
    if (!simple_io) {
        tty = fopen("/dev/tty", "w+e");
        if (tty) {
            should_close_tty = true;
        } else if (IsLinux() || IsOpenbsd()) {
            // this could happen because pledge() blocked us
            tty = fdopen(0, "w+e");
        }
        if (tty != nullptr) {
            if (!tcgetattr(fileno(tty), &initial_state)) {
                out = tty;
                struct termios new_termios = initial_state;
                new_termios.c_lflag &= ~(ICANON | ECHO);
                new_termios.c_cc[VMIN] = 1;
                new_termios.c_cc[VTIME] = 0;
                tcsetattr(fileno(tty), TCSANOW, &new_termios);
            } else {
                simple_io = true;
                fclose(tty);
                tty = 0;
            }
        } else {
            simple_io = true;
        }
    }
    setlocale(LC_ALL, "");
}

void cleanup() {
    // Reset console display
    set_display(reset);
    // Restore settings
    if (!simple_io) {
        if (tty != nullptr) {
            fflush(tty);
            tcsetattr(fileno(tty), TCSANOW, &initial_state);
            if (should_close_tty) {
                fclose(tty);
            }
            tty = nullptr;
            out = stdout;
        }
    }
}

//
// Display and IO
//

// Keep track of current display and only emit ANSI code if it changes
void set_display(display_t display) {
    if (advanced_display && current_display != display) {
        fflush(stdout);
        switch(display) {
            case reset:
                fprintf(out, ANSI_COLOR_RESET);
                break;
            case prompt:
                fprintf(out, ANSI_COLOR_YELLOW);
                break;
            case user_input:
                fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                break;
            case error:
                fprintf(out, ANSI_BOLD ANSI_COLOR_RED);
        }
        current_display = display;
        fflush(out);
    }
}

static char32_t getchar32() {
    wchar_t wc = getwchar();
    if (static_cast<wint_t>(wc) == WEOF) {
        return WEOF;
    }
    return static_cast<char32_t>(wc);
}

static void pop_cursor() {
    putc('\b', out);
}

static int estimateWidth(char32_t codepoint) {
    return wcwidth(codepoint);
}

static int put_codepoint(const char* utf8_codepoint, size_t length, int expectedWidth) {
    // We can trust expectedWidth if we've got one
    if (expectedWidth >= 0 || tty == nullptr) {
        fwrite(utf8_codepoint, length, 1, out);
        return expectedWidth;
    }

    fputs("\033[6n", tty); // Query cursor position
    int x1;
    int y1;
    int x2;
    int y2;
    int results = 0;
    results = fscanf(tty, "\033[%d;%dR", &y1, &x1);

    fwrite(utf8_codepoint, length, 1, tty);

    fputs("\033[6n", tty); // Query cursor position
    results += fscanf(tty, "\033[%d;%dR", &y2, &x2);

    if (results != 4) {
        return expectedWidth;
    }

    int width = x2 - x1;
    if (width < 0) {
        // Calculate the width considering text wrapping
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        width += w.ws_col;
    }
    return width;
}

static void replace_last(char ch) {
    fprintf(out, "\b%c", ch);
}

static void append_utf8(char32_t ch, std::string & out) {
    if (ch <= 0x7F) {
        out.push_back(static_cast<unsigned char>(ch));
    } else if (ch <= 0x7FF) {
        out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0xFFFF) {
        out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else if (ch <= 0x10FFFF) {
        out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
        out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
    } else {
        // Invalid Unicode code point
    }
}

// Helper function to remove the last UTF-8 character from a string
static void pop_back_utf8_char(std::string & line) {
    if (line.empty()) {
        return;
    }

    size_t pos = line.length() - 1;

    // Find the start of the last UTF-8 character (checking up to 4 bytes back)
    for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
        if ((line[pos] & 0xC0) != 0x80) {
            break; // Found the start of the character
        }
    }
    line.erase(pos);
}

static bool readline_advanced(std::string & line, bool multiline_input) {
    if (out != stdout) {
        fflush(stdout);
    }

    line.clear();
    std::vector<int> widths;
    bool is_special_char = false;
    bool end_of_stream = false;

    char32_t input_char;
    while (true) {
        fflush(out); // Ensure all output is displayed before waiting for input
        input_char = getchar32();

        if (input_char == '\r' || input_char == '\n') {
            break;
        }

        if (input_char == (char32_t) WEOF || input_char == 0x04 /* Ctrl+D*/) {
            end_of_stream = true;
            break;
        }

        if (is_special_char) {
            set_display(user_input);
            replace_last(line.back());
            is_special_char = false;
        }

        if (input_char == '\033') { // Escape sequence
            char32_t code = getchar32();
            if (code == '[' || code == 0x1B) {
                // Discard the rest of the escape sequence
                while ((code = getchar32()) != (char32_t) WEOF) {
                    if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
                        break;
                    }
                }
            }
        } else if (input_char == 0x08 || input_char == 0x7F) { // Backspace
            if (!widths.empty()) {
                int count;
                do {
                    count = widths.back();
                    widths.pop_back();
                    // Move cursor back, print space, and move cursor back again
                    for (int i = 0; i < count; i++) {
                        replace_last(' ');
                        pop_cursor();
                    }
                    pop_back_utf8_char(line);
                } while (count == 0 && !widths.empty());
            }
        } else {
            int offset = line.length();
            append_utf8(input_char, line);
            int width = put_codepoint(line.c_str() + offset, line.length() - offset, estimateWidth(input_char));
            if (width < 0) {
                width = 0;
            }
            widths.push_back(width);
        }

        if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
            set_display(prompt);
            replace_last(line.back());
            is_special_char = true;
        }
    }

    bool has_more = multiline_input;
    if (is_special_char) {
        replace_last(' ');
        pop_cursor();

        char last = line.back();
        line.pop_back();
        if (last == '\\') {
            line += '\n';
            fputc('\n', out);
            has_more = !has_more;
        } else {
            // llama will just eat the single space, it won't act as a space
            if (line.length() == 1 && line.back() == ' ') {
                line.clear();
                pop_cursor();
            }
            has_more = false;
        }
    } else {
        if (end_of_stream) {
            has_more = false;
        } else {
            line += '\n';
            fputc('\n', out);
        }
    }

    fflush(out);
    return has_more;
}

static bool readline_simple(std::string & line, bool multiline_input) {
    if (!std::getline(std::cin, line)) {
        // Input stream is bad or EOF received
        line.clear();
        return false;
    }
    if (!line.empty()) {
        char last = line.back();
        if (last == '/') { // Always return control on '/' symbol
            line.pop_back();
            return false;
        }
        if (last == '\\') { // '\\' changes the default action
            line.pop_back();
            multiline_input = !multiline_input;
        }
    }
    line += '\n';

    // By default, continue input if multiline_input is set
    return multiline_input;
}

bool readline(std::string & line, bool multiline_input) {
    set_display(user_input);
    if (simple_io) {
        return readline_simple(line, multiline_input);
    }
    return readline_advanced(line, multiline_input);
}

}  // namespace console
