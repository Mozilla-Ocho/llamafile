// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
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

#include "chatbot.h"

#include <assert.h>
#include <cosmo.h>
#include <ctype.h>
#include <glob.h>
#include <math.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "llama.cpp/common.h"
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/llava/clip.h"
#include "llama.cpp/llava/llava.h"
#include "llama.cpp/server/server.h"
#include "llamafile/bestline.h"
#include "llamafile/compute.h"
#include "llamafile/datauri.h"
#include "llamafile/highlight.h"
#include "llamafile/image.h"
#include "llamafile/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/string.h"
#include "llamafile/xterm.h"

#define IMAGE_PLACEHOLDER_TOKEN -31337

enum Role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
};

struct ServerArgs {
    int argc;
    char **argv;
};

static bool g_manual_mode;
static bool g_has_ephemeral;
static bool g_said_something;
static char g_last_printed_char;
static int g_system_prompt_tokens;
static gpt_params g_params;
static clip_ctx *g_clip;
static llama_model *g_model;
static llama_context *g_ctx;
static std::vector<llama_pos> g_undo;
static std::vector<llama_pos> g_stack;
static std::vector<llama_token> g_history;
static enum Role g_role = ROLE_USER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static std::string g_listen_url;
static volatile sig_atomic_t g_got_sigint;

static void on_sigint(int sig) {
    g_got_sigint = 1;
}

static int tokens_used(void) {
    return g_history.size();
}

static bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
}

static void fix_stack(std::vector<llama_pos> *stack) {
    while (!stack->empty() && stack->back() > tokens_used())
        stack->pop_back();
}

static void fix_stacks(void) {
    fix_stack(&g_undo);
    fix_stack(&g_stack);
    g_system_prompt_tokens = MIN(g_system_prompt_tokens, tokens_used());
}

static std::vector<llama_pos> adjust_stack(llama_pos erase_begin, llama_pos erase_end,
                                           const std::vector<llama_pos> &stack) {
    std::vector<llama_pos> builder;
    for (llama_pos pos : stack) {
        if (erase_begin <= pos && pos < erase_end)
            continue;
        if (pos >= erase_end)
            pos -= erase_end - erase_begin;
        builder.push_back(pos);
    }
    return builder;
}

static void adjust_stacks(llama_pos erase_begin, llama_pos erase_end) {
    g_undo = adjust_stack(erase_begin, erase_end, g_undo);
    g_stack = adjust_stack(erase_begin, erase_end, g_stack);
}

static void record_undo(void) {
    if (g_undo.empty() || g_undo.back() != tokens_used())
        g_undo.push_back(tokens_used());
}

static bool is_directory(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode);
}

static const char *get_role_name(enum Role role) {
    switch (role) {
    case ROLE_USER:
        return "user";
    case ROLE_ASSISTANT:
        return "assistant";
    case ROLE_SYSTEM:
        return "system";
    default:
        __builtin_unreachable();
    }
}

static const char *get_role_color(enum Role role) {
    switch (role) {
    case ROLE_USER:
        return GREEN;
    case ROLE_ASSISTANT:
        return MAGENTA;
    case ROLE_SYSTEM:
        return YELLOW;
    default:
        __builtin_unreachable();
    }
}

static enum Role get_next_role(enum Role role) {
    switch (role) {
    case ROLE_USER:
        return ROLE_ASSISTANT;
    case ROLE_ASSISTANT:
        return ROLE_USER;
    case ROLE_SYSTEM:
        return ROLE_USER;
    default:
        __builtin_unreachable();
    }
}

static enum Role cycle_role(enum Role role) {
    switch (role) {
    case ROLE_USER:
        return ROLE_ASSISTANT;
    case ROLE_ASSISTANT:
        return ROLE_SYSTEM;
    case ROLE_SYSTEM:
        return ROLE_USER;
    default:
        __builtin_unreachable();
    }
}

static bool has_binary(const std::string_view s) {
    return s.find('\0') != std::string_view::npos;
}

static std::string describe_compute(void) {
    if (llama_n_gpu_layers(g_model) > 0 && llamafile_has_gpu()) {
        if (llamafile_has_metal()) {
            return "Apple Metal GPU";
        } else {
            std::vector<std::string> vec;
            int count = ggml_backend_cuda_get_device_count();
            for (int i = 0; i < count; i++) {
                char buf[128];
                ggml_backend_cuda_get_device_description(i, buf, sizeof(buf));
                vec.emplace_back(buf);
            }
            return lf::join(vec, ", ");
        }
    } else {
        return llamafile_describe_cpu();
    }
}

static std::string token_to_piece(const struct llama_context *ctx, llama_token token,
                                  bool special) {
    if (token == IMAGE_PLACEHOLDER_TOKEN)
        return "⁑";
    return llama_token_to_piece(ctx, token, special);
}

static std::string describe_token(llama_token token) {
    if (token == llama_token_bos(g_model))
        return "§";
    if (token == llama_token_eos(g_model))
        return "∎";
    if (token == llama_token_cls(g_model))
        return "⌘";
    if (token == llama_token_sep(g_model))
        return "⋯";
    if (token == llama_token_pad(g_model))
        return "␣";
    if (token == llama_token_nl(g_model))
        return "↵";
    if (llama_token_is_eog(g_model, token))
        return "⌟";
    if (llama_token_is_control(g_model, token))
        return "∷";
    std::string s = token_to_piece(g_ctx, token, DONT_RENDER_SPECIAL_TOKENS);
    if (s.empty())
        return "↯";
    return s;
}

static std::string describe_erasure(llama_pos begin, llama_pos end) {
    unassert(begin <= end);
    unassert(end <= tokens_used());
    std::string description;
    llama_pos pos = begin;
    while (pos < end && description.size() < 63)
        description += describe_token(g_history[pos++]);
    if (!description.empty() && pos < end)
        description += " ...";
    description = lf::collapse(description);
    if (pos == end && description.empty())
        description = "<absolute end>";
    return description;
}

static std::string describe_position(llama_pos pos) {
    unassert(pos <= tokens_used());
    std::string description;
    while (pos > 0 && description.size() < 63)
        description = describe_token(g_history[--pos]) + description;
    if (!description.empty() && pos > 0)
        description = std::string("... ") + description;
    description = lf::collapse(description);
    if (!pos && description.empty())
        description = "<absolute beginning>";
    return description;
}

static void print(const std::string_view &s) {
    for (char c : s) {
        g_last_printed_char = c;
        fputc(c, stdout);
        if (c == '\n')
            g_has_ephemeral = false;
    }
}

static void ensure_newline() {
    if (g_last_printed_char != '\n')
        print("\n");
}

static void print_ephemeral(const std::string_view &description) {
    fprintf(stderr, " " BRIGHT_BLACK "%.*s" UNFOREGROUND "\r", (int)description.size(),
            description.data());
    g_has_ephemeral = true;
}

static void clear_ephemeral(void) {
    if (g_has_ephemeral) {
        fprintf(stderr, CLEAR_FORWARD);
        g_has_ephemeral = false;
    }
}

static void err(const char *fmt, ...) {
    va_list ap;
    clear_ephemeral();
    ensure_newline();
    va_start(ap, fmt);
    fputs(BRIGHT_RED, stderr);
    vfprintf(stderr, fmt, ap);
    fputs(RESET "\n", stderr);
    va_end(ap);
}

static bool out_of_context(int extra) {
    err("error: ran out of context window at %d tokens\n"
        "consider passing `-c %d` at startup for the maximum\n"
        "you can free up more space using /forget or /clear",
        tokens_used() + extra, llama_n_ctx_train(g_model));
    return false;
}

static bool eval_tokens(std::vector<llama_token> tokens) {
    int N = (int)tokens.size();
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            print_ephemeral(lf::format("loading prompt %d%%...", (int)((double)i / N * 100)));
        int n_eval = (int)tokens.size() - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval, tokens_used(), 0)))
            return out_of_context(n_eval);
        g_history.insert(g_history.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
    }
    clear_ephemeral();
    // this function is what computes /stats. we need to call it now
    // since llama_decode() kicks the can down the road to functions
    // like llama_sampling_sample(). that is bad because the chatbot
    // returns control to the repl rather than sampling when loading
    // system and image prompts.
    llama_synchronize(g_ctx);
    return true;
}

static bool eval_image_embed(const struct llava_image_embed *image_embed) {
    int N = image_embed->n_image_pos;
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    int n_embd = llama_n_embd(llama_get_model(g_ctx));
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            print_ephemeral(lf::format("loading image %d%%...", (int)((double)i / N * 100)));
        int n_eval = N - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        llama_batch batch = {
            .n_tokens = n_eval,
            .embd = image_embed->embed + i * n_embd,
            .all_pos_0 = tokens_used(),
            .all_pos_1 = 1,
            .all_seq_id = 0,
        };
        if (llama_decode(g_ctx, batch))
            return out_of_context(n_eval);
        for (int i = 0; i < n_eval; ++i)
            g_history.push_back(IMAGE_PLACEHOLDER_TOKEN);
    }
    clear_ephemeral();
    return true;
}

static bool eval_image(const std::string_view binary) {
    unassert(g_clip);
    llava_image_embed *image_embed;
    print_ephemeral("analyzing image...");
    image_embed = llava_image_embed_make_with_bytes(
        g_clip, FLAG_threads_batch, (const unsigned char *)binary.data(), binary.size());
    clear_ephemeral();
    if (!image_embed) {
        err("failed to load image");
        return false;
    }
    bool ok = eval_image_embed(image_embed);
    llava_image_embed_free(image_embed);
    return ok;
}

static bool eval_token(int id) {
    return eval_tokens({id});
}

static bool eval_plain_text(const std::string &str, bool add_special, bool parse_special) {
    return eval_tokens(llama_tokenize(g_model, str, add_special, parse_special));
}

static bool eval_string(std::string_view s, bool add_special, bool parse_special) {
    size_t i = 0;
    for (;;) {
        size_t pos = s.find("data:", i);
        if (pos == std::string_view::npos)
            return eval_plain_text(std::string(s), add_special, parse_special);
        DataUri uri;
        size_t end = uri.parse(s.substr(pos + 5));
        if (end == std::string_view::npos) {
            i = pos + 5;
            continue;
        }
        std::string image = uri.decode();
        if (!lf::is_image(image)) {
            i = pos + 5;
            continue;
        }
        if (!eval_plain_text(std::string(s.substr(0, pos)), add_special, parse_special))
            return false;
        if (!eval_image(image))
            return false;
        s = s.substr(pos + 5 + end);
        i = 0;
    }
}

static void rewind(int pos) {
    unassert(pos <= tokens_used());
    llama_kv_cache_seq_rm(g_ctx, 0, pos, -1);
    g_history.resize(pos);
}

static void on_manual(const std::vector<std::string> &args) {
    if (args.size() == 1) {
        g_manual_mode = !g_manual_mode;
    } else if (args.size() == 2 && (args[1] == "on" || args[1] == "off")) {
        g_manual_mode = args[1] == "on";
    } else {
        err("error: bad /manual command\n"
            "usage: /manual [on|off]");
        return;
    }
    fprintf(stderr, FAINT "manual mode %s" RESET "\n", g_manual_mode ? "enabled" : "disabled");
    if (!g_manual_mode)
        g_role = ROLE_USER;
}

static void on_context(const std::vector<std::string> &args) {
    int configured_context = llama_n_ctx(g_ctx);
    int max_context = llama_n_ctx_train(g_model);
    printf("%d out of %d context tokens used (%d tokens remaining)\n", tokens_used(),
           configured_context, configured_context - tokens_used());
    if (configured_context < max_context)
        printf("use the `-c %d` flag at startup for maximum context\n", max_context);
}

static void on_clear(const std::vector<std::string> &args) {
    rewind(g_system_prompt_tokens);
    g_stack.clear();
    fix_stacks();
}

static void print_stack(void) {
    for (size_t i = g_stack.size(); i--;)
        printf("%12d " FAINT "(%s)" RESET "\n", g_stack[i], describe_position(g_stack[i]).c_str());
}

static void on_push(const std::vector<std::string> &args) {
    g_stack.push_back(tokens_used());
    print_stack();
}

static void on_pop(const std::vector<std::string> &args) {
    if (g_stack.empty()) {
        err("error: context length stack is empty");
        return;
    }
    printf(BOLD "%12d" RESET " restored " FAINT "(%s)" RESET "\n", g_stack.back(),
           describe_position(g_stack.back()).c_str());
    rewind(g_stack.back());
    g_stack.pop_back();
    fix_stacks();
    print_stack();
}

static void on_undo(const std::vector<std::string> &args) {
    while (!g_undo.empty() && g_undo.back() == tokens_used())
        g_undo.pop_back();
    if (g_undo.empty()) {
        err("error: no further undo actions possible");
        return;
    }
    printf(FAINT "restoring conversation to: %s" RESET "\n",
           describe_position(g_undo.back()).c_str());
    rewind(g_undo.back());
    g_undo.pop_back();
    fix_stacks();
}

static void on_forget(const std::vector<std::string> &args) {
    if (g_undo.size() < 2) {
        err("error: nothing left to forget");
        return;
    }
    int erase_count;
    llama_pos erase_begin = g_undo[1];
    llama_pos erase_end = g_undo.size() > 2 ? g_undo[2] : tokens_used();
    if (!(erase_count = erase_end - erase_begin)) {
        err("error: nothing left to forget");
        return;
    }
    printf(FAINT "forgetting: %s" RESET "\n", describe_erasure(erase_begin, erase_end).c_str());
    llama_kv_cache_seq_rm(g_ctx, 0, erase_begin, erase_end);
    llama_kv_cache_seq_add(g_ctx, 0, erase_end, -1, -erase_count);
    g_history.erase(g_history.begin() + erase_begin, //
                    g_history.begin() + erase_end);
    adjust_stacks(erase_begin, erase_end);
    fix_stacks();
}

static void on_stack(const std::vector<std::string> &args) {
    if (g_stack.empty()) {
        printf(FAINT "stack is currently empty (try using /push)" RESET "\n");
        return;
    }
    print_stack();
}

static void on_stats(const std::vector<std::string> &args) {
    FLAG_log_disable = false;
    llama_print_timings(g_ctx);
    FLAG_log_disable = true;
}

static void on_dump(const std::vector<std::string> &args) {
    int fd = 1;
    if (args.size() >= 2) {
        if ((fd = creat(args[1].c_str(), 0644)) == -1) {
            perror(args[1].c_str());
            return;
        }
    }
    std::string s;
    for (auto id : g_history)
        s += token_to_piece(g_ctx, id, RENDER_SPECIAL_TOKENS);
    if (!s.empty() && s[s.size() - 1] != '\n')
        s += '\n';
    write(fd, s.data(), s.size());
    if (args.size() >= 2)
        close(fd);
}

static void on_upload(const std::vector<std::string> &args) {
    if (args.size() < 2) {
        err("error: missing file path" RESET "\n"
            "usage: /upload PATH");
        return;
    }
    if (args.size() > 2) {
        err("error: too many arguments" RESET "\n"
            "usage: /upload PATH");
        return;
    }
    const char *path = args[1].c_str();
    struct stat st;
    if (stat(path, &st) || !S_ISREG(st.st_mode)) {
        err("%s: file does not exist", path);
        return;
    }
    int tokens_used_before = tokens_used();
    std::string content;
    if (!lf::slurp(&content, path)) {
        err("%s: failed to slurp file", path);
        return;
    }
    std::string markdown;
    markdown += "- **Filename**: `";
    markdown += path;
    markdown += "`\n- **Last modified**: ";
    markdown += lf::iso8601(st.st_mtim);
    markdown += "\n\n";
    if (lf::is_image(content)) {
        if (!g_clip) {
            err("%s: need --mmproj model to process images", path);
            return;
        }
        lf::print_image(1, content, 80);
        lf::convert_image_to_uri(&markdown, content);
    } else {
        if (has_binary(content)) {
            err("%s: binary file type not supported", path);
            return;
        }
        markdown += "``````";
        markdown += lf::extname(path);
        markdown += '\n';
        markdown += content;
        if (markdown.back() != '\n')
            markdown += '\n';
        markdown += "``````";
    }
    std::vector<llama_chat_msg> chat = {{"system", std::move(markdown)}};
    if (!eval_string(
            llama_chat_apply_template(g_model, g_params.chat_template, chat, DONT_ADD_ASSISTANT),
            DONT_ADD_SPECIAL, PARSE_SPECIAL)) {
        rewind(tokens_used_before);
        return;
    }
    llama_synchronize(g_ctx);
}

static const char *on_hint_impl(const char *line) {
    if (!*line && g_manual_mode)
        return get_role_name(g_role);
    if (!*line && !g_manual_mode && !g_said_something)
        return "say something (or type /help for help)";
    static const char *const kHints[] = {
        "/clear", //
        "/context", //
        "/dump", //
        "/exit", //
        "/forget", //
        "/help", //
        "/manual", //
        "/pop", //
        "/push", //
        "/stack", //
        "/stats", //
        "/undo", //
        "/upload", //
    };
    int z = strlen(line);
    int n = sizeof(kHints) / sizeof(kHints[0]);
    int l = 0;
    int r = n - 1;
    int i = -1;
    while (l <= r) {
        int m = (l & r) + ((l ^ r) >> 1); // floor((a+b)/2)
        int c = strncmp(line, kHints[m], z);
        if (!c) {
            i = m;
            r = m - 1;
        } else if (c < 0) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }
    if (i == -1 || (i + 1 < n && !strncmp(line, kHints[i + 1], z)))
        return "";
    return kHints[i] + z;
}

static char *on_hint(const char *line, const char **ansi1, const char **ansi2) {
    *ansi1 = FAINT;
    *ansi2 = UNBOLD;
    return strdup(on_hint_impl(line));
}

static void on_completion(const char *line, int pos, bestlineCompletions *comp) {
    if (startswith(line, "/upload ")) {
        std::string pattern(line + strlen("/upload "));
        pattern += '*';
        glob_t gl;
        if (!glob(pattern.c_str(), GLOB_TILDE, 0, &gl)) {
            for (size_t i = 0; i < gl.gl_pathc; ++i) {
                std::string completion = "/upload ";
                completion += gl.gl_pathv[i];
                if (is_directory(gl.gl_pathv[i]))
                    completion += '/';
                bestlineAddCompletion(comp, completion.c_str());
            }
            globfree(&gl);
        }
    } else {
        static const char *const kCompletions[] = {
            "/clear", // usage: /clear
            "/context", // usage: /context
            "/dump", // usage: /dump [FILE]
            "/exit", // usage: /exit
            "/forget", // usage: /forget
            "/help", // usage: /help [COMMAND]
            "/manual", // usage: /manual [on|off]
            "/pop", // usage: /pop
            "/push", // usage: /push
            "/stack", // usage: /stack
            "/stats", // usage: /stats
            "/undo", // usage: /undo
            "/upload", // usage: /upload FILE
        };
        for (int i = 0; i < sizeof(kCompletions) / sizeof(*kCompletions); ++i)
            if (startswith(kCompletions[i], line))
                bestlineAddCompletion(comp, kCompletions[i]);
    }
}

// handle irc style commands like: `/arg0 arg1 arg2`
static bool handle_command(const char *command) {
    if (!strcmp(command, "/?")) {
        const std::vector<std::string> args = {"?"};
        chatbot_help(args);
        return true;
    }
    if (!(command[0] == '/' && std::isalpha(command[1])))
        return false;
    std::vector<std::string> args;
    std::istringstream iss(command + 1);
    std::string arg;
    while (iss >> arg)
        args.push_back(arg);
    if (args[0] == "exit" || args[0] == "bye") {
        exit(0);
    } else if (args[0] == "help") {
        chatbot_help(args);
    } else if (args[0] == "stats") {
        on_stats(args);
    } else if (args[0] == "context") {
        on_context(args);
    } else if (args[0] == "manual") {
        on_manual(args);
    } else if (args[0] == "clear") {
        on_clear(args);
    } else if (args[0] == "dump") {
        on_dump(args);
    } else if (args[0] == "push") {
        on_push(args);
    } else if (args[0] == "pop") {
        on_pop(args);
    } else if (args[0] == "undo") {
        on_undo(args);
    } else if (args[0] == "forget") {
        on_forget(args);
    } else if (args[0] == "stack") {
        on_stack(args);
    } else if (args[0] == "upload") {
        on_upload(args);
    } else {
        err("%s: unrecognized command", args[0].c_str());
    }
    return true;
}

static void on_server_listening(const char *host, int port) {
    pthread_mutex_lock(&g_lock);
    g_listen_url = lf::format("http://%s:%d/", host, port);
    pthread_cond_signal(&g_cond);
    pthread_mutex_unlock(&g_lock);
}

static void *server_thread(void *arg) {
    ServerArgs *sargs = (ServerArgs *)arg;
    server_log_json = false;
    g_server_background_mode = true;
    g_server_force_llama_model = g_model;
    g_server_on_listening = on_server_listening;
    exit(server_cli(sargs->argc, sargs->argv));
}

int chatbot_main(int argc, char **argv) {

    // print logo
    chatbot_logo(argv);

    // disable llamafile gpu initialization log messages
    if (!llamafile_has(argv, "--verbose"))
        FLAG_log_disable = true;

    // override defaults for some flags
    g_params.n_batch = 256; // for better progress indication
    g_params.sparams.temp = 0; // don't believe in randomness by default
    g_params.prompt = "A chat between a curious human and an artificial intelligence assistant. "
                      "The assistant gives helpful, detailed, and polite answers to the "
                      "human's questions.";

    // parse flags (sadly initializes gpu support as side-effect)
    print_ephemeral("loading backend...");
    llama_backend_init();
    if (!gpt_params_parse(argc, argv, g_params)) { // also loads gpu module
        fprintf(stderr, "error: failed to parse flags\n");
        exit(1);
    }
    clear_ephemeral();

    // setup logging
    FLAG_log_disable = false;
    if (!g_params.verbosity)
        log_disable();

    print_ephemeral("loading model...");
    llama_model_params model_params = llama_model_params_from_gpt_params(g_params);
    g_model = llama_load_model_from_file(g_params.model.c_str(), model_params);
    if (g_model == NULL) {
        clear_ephemeral();
        fprintf(stderr, "%s: failed to load model\n", g_params.model.c_str());
        exit(2);
    }
    if (g_params.n_ctx <= 0 || g_params.n_ctx > llama_n_ctx_train(g_model))
        g_params.n_ctx = llama_n_ctx_train(g_model);
    if (g_params.n_ctx < g_params.n_batch)
        g_params.n_batch = g_params.n_ctx;
    clear_ephemeral();

    bool want_server = !llamafile_has(argv, "--chat");
    if (want_server) {
        print_ephemeral("launching server...");
        pthread_t thread;
        pthread_attr_t attr;
        ServerArgs sargs = {argc, argv};
        pthread_mutex_lock(&g_lock);
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_create(&thread, &attr, server_thread, &sargs);
        pthread_attr_destroy(&attr);
        pthread_cond_wait(&g_cond, &g_lock);
        pthread_mutex_unlock(&g_lock);
        clear_ephemeral();
    }

    if (!FLAG_nologo) {
        printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n" //
               BOLD "model" UNBOLD ":    %s\n",
               lf::basename(g_params.model).c_str());
        printf(BOLD "compute" UNBOLD ":  %s\n", describe_compute().c_str());
        if (want_server)
            printf(BOLD "server" UNBOLD ":   %s\n", g_listen_url.c_str());
        printf("\n");
    }

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = llama_context_params_from_gpt_params(g_params);
    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    clear_ephemeral();
    if (!g_ctx) {
        fprintf(stderr, "error: failed to initialize context\n");
        exit(3);
    }

    if (llama_model_has_encoder(g_model))
        fprintf(stderr, "warning: this model has an encoder\n");

    if (FLAG_mmproj) {
        print_ephemeral("initializing vision model...");
        g_clip = clip_model_load(FLAG_mmproj, g_params.verbosity);
        clear_ephemeral();
        if (!g_clip) {
            fprintf(stderr, "%s: failed to initialize clip image model\n", FLAG_mmproj);
            exit(4);
        }
    }

    // setup conversation
    if (llama_should_add_bos_token(g_model))
        if (!eval_token(llama_token_bos(g_model)))
            exit(6);
    record_undo();

    // setup system prompt
    std::vector<llama_chat_msg> chat = {{"system", g_params.prompt}};
    std::string msg =
        llama_chat_apply_template(g_model, g_params.chat_template, chat, DONT_ADD_ASSISTANT);
    if (!eval_string(msg, DONT_ADD_SPECIAL, PARSE_SPECIAL))
        exit(6);
    llama_synchronize(g_ctx);
    g_system_prompt_tokens = tokens_used();
    clear_ephemeral();
    if (g_params.display_prompt)
        printf("%s\n", g_params.special ? msg.c_str() : g_params.prompt.c_str());

    // perform important setup
    HighlightMarkdown highlighter;
    ColorBleeder bleeder(&highlighter);
    struct llama_sampling_context *sampler = llama_sampling_init(g_params.sparams);
    signal(SIGINT, on_sigint);

    // run chatbot
    for (;;) {
        record_undo();
        bestlineLlamaMode(true);
        bestlineSetHintsCallback(on_hint);
        bestlineSetFreeHintsCallback(free);
        bestlineSetCompletionCallback(on_completion);
        write(1, get_role_color(g_role), strlen(get_role_color(g_role)));
        char *line = bestlineWithHistory(">>> ", "llamafile");
        write(1, UNFOREGROUND, strlen(UNFOREGROUND));
        if (!line) {
            if (g_got_sigint)
                ensure_newline();
            break;
        }
        if (is_empty(line)) {
            if (g_manual_mode) {
                g_role = cycle_role(g_role);
                write(1, "\033[F", 3);
            }
            free(line);
            continue;
        }
        g_said_something = true;
        if (handle_command(line)) {
            free(line);
            continue;
        }
        bool add_assi = !g_manual_mode;
        int tokens_used_before = tokens_used();
        std::vector<llama_chat_msg> chat = {{get_role_name(g_role), line}};
        std::string msg =
            llama_chat_apply_template(g_model, g_params.chat_template, chat, add_assi);
        if (!eval_string(msg, DONT_ADD_SPECIAL, PARSE_SPECIAL)) {
            rewind(tokens_used_before);
            continue;
        }
        if (g_manual_mode) {
            g_role = get_next_role(g_role);
            free(line);
            continue;
        }
        for (;;) {
            if (g_got_sigint) {
                eval_token(llamafile_token_eot(g_model));
                break;
            }
            llama_token id = llama_sampling_sample(sampler, g_ctx, NULL);
            llama_sampling_accept(sampler, g_ctx, id, APPLY_GRAMMAR);
            if (!eval_token(id))
                break;
            if (llama_token_is_eog(g_model, id))
                break;
            std::string s;
            bleeder.feed(&s, token_to_piece(g_ctx, id, g_params.special));
            print(s);
            fflush(stdout);
        }
        g_got_sigint = 0;
        free(line);
        std::string s;
        bleeder.flush(&s);
        print(s);
        ensure_newline();
    }

    if (g_clip) {
        print_ephemeral("freeing vision model...");
        clip_free(g_clip);
        clear_ephemeral();
    }

    print_ephemeral("freeing context...");
    llama_sampling_free(sampler);
    llama_free(g_ctx);
    clear_ephemeral();

    print_ephemeral("freeing model...");
    llama_free_model(g_model);
    clear_ephemeral();

    print_ephemeral("freeing backend...");
    llama_backend_free();
    clear_ephemeral();

    return 0;
}
