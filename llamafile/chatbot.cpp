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

#include <assert.h>
#include <cosmo.h>
#include <ctype.h>
#include <math.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#include "llama.cpp/common.h"
#include "llama.cpp/ggml-cuda.h"
#include "llama.cpp/llama.h"
#include "llama.cpp/server/server.h"
#include "llamafile/bestline.h"
#include "llamafile/compute.h"
#include "llamafile/highlight.h"
#include "llamafile/llamafile.h"

#define RESET "\e[0m"
#define BOLD "\e[1m"
#define FAINT "\e[2m"
#define UNBOLD "\e[22m"
#define RED "\e[31m"
#define GREEN "\e[32m"
#define MAGENTA "\e[35m"
#define YELLOW "\e[33m"
#define CYAN "\e[36m"
#define UNFOREGROUND "\e[39m"
#define BRIGHT_BLACK "\e[90m"
#define BRIGHT_RED "\e[91m"
#define BRIGHT_GREEN "\e[92m"
#define CLEAR_FORWARD "\e[K"

enum Role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
};

struct ServerArgs {
    int argc;
    char **argv;
};

static int g_past;
static int g_clear;
static bool g_once;
static bool g_manual;
static llama_model *g_model;
static llama_context *g_ctx;
static std::vector<int> g_stack;
static std::vector<llama_token> g_history;
static enum Role g_role = ROLE_USER;
static pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static std::string g_listen_url;
static volatile sig_atomic_t g_got_sigint;

static void on_sigint(int sig) {
    g_got_sigint = 1;
}

static bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
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

static std::string basename(const std::string_view path) {
    size_t i, e;
    if ((e = path.size())) {
        while (e > 1 && path[e - 1] == '/')
            --e;
        i = e - 1;
        while (i && path[i - 1] != '/')
            --i;
        return std::string(path.substr(i, e - i));
    } else {
        return ".";
    }
}

__attribute__((format(printf, 1, 2))) static std::string format(const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = 512;
    std::string res(size, '\0');
    int need = vsnprintf(res.data(), size, fmt, ap);
    res.resize(need + 1, '\0');
    if (need + 1 > size)
        vsnprintf(res.data(), need + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return res;
}

static std::string join(const std::vector<std::string> &vec, const std::string_view &delim) {
    std::string result;
    for (size_t i = 0; i < vec.size(); i++) {
        result += vec[i];
        if (i < vec.size() - 1)
            result += delim;
    }
    return result;
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
            return join(vec, ", ");
        }
    } else {
        return llamafile_describe_cpu();
    }
}

static void on_help(const std::vector<std::string> &args) {
    if (args.size() == 1) {
        fprintf(stderr, "\
" BOLD "available commands" RESET "\n\
  /context                 print context window usage\n\
  /dump [FILE]             print or save context window to file\n\
  /exit                    end program\n\
  /help [COMMAND]          show help\n\
  /manual [on|off]         toggle manual role mode\n\
  /stats                   print performance metrics\n\
");
    } else if (args[1] == "context") {
        fprintf(stderr, "\
usage: /context" RESET "\n\
prints information about context window usage. this helps you know how\n\
soon you're going to run out of tokens for the current conversation.\n\
");
    } else if (args[1] == "dump") {
        fprintf(stderr, "\
" BOLD "usage: /dump [FILE]" RESET "\n\
dumps raw tokens for current conversation history. special tokens are\n\
printed in the a model specific chat syntax. this is useful for seeing\n\
specifically what data is being evaluated by the model. by default it\n\
will be printed to the terminal. if a FILE argument is specified, then\n\
the raw conversation history will be written to that filename.\n\
");
    } else if (args[1] == "exit") {
        fprintf(stderr, "\
" BOLD "usage: /exit" RESET "\n\
this command will cause the process to exit. it is essentially the same\n\
as typing ctrl-d which signals an eof condition. it also does the same\n\
thing as typing ctrl-c when the >>> user input prompt is displayed.\n\
");
    } else if (args[1] == "manual") {
        fprintf(stderr, "\
" BOLD "usage: /manual [on|off]" RESET "\n\
puts the chatbot in manual mode. this is useful if you want to inject\n\
a response as the model rather than the user. it's also possible to add\n\
additional system prompts to the conversation history. when the manual\n\
mode is activated, a hint is displayed next to the '>>>' indicating\n\
the current role, which can be 'user', 'assistant', or 'system'. if\n\
enter is pressed on an empty line, then llamafile will cycle between\n\
all three roles. when /manual is specified without an argument, it will\n\
toggle manual mode. otherwise an 'on' or 'off' argument is supplied.\n\
");
    } else if (args[1] == "help") {
        fprintf(stderr, "\
" BOLD "usage: /help [COMMAND]" RESET "\n\
shows help on how to issue commands to your llamafile. if no argument is\n\
specified, then a synopsis of all available commands will be printed. if\n\
a specific command name is given (e.g. /help dump) then documentation on\n\
the usage of that specific command will be printed.\n\
");
    } else if (args[1] == "stats") {
        fprintf(stderr, "\
" BOLD "usage: /stats" RESET "\n\
prints performance statistics for current session. this includes prompt\n\
evaluation time in tokens per second, which indicates prefill speed, or\n\
how quickly llamafile is able to read text. the 'eval time' statistic\n\
gives you prediction or token generation speed, in tokens per second,\n\
which tells you how quickly llamafile is able to write text.\n\
");
    } else if (args[1] == "clear") {
        fprintf(stderr, "\
usage: /clear" RESET "\n\
start conversation over from the beginning. this command adjusts the\n\
context window to what it was after the initial system prompt. this\n\
command also erases the /push stack.\n\
");
    } else if (args[1] == "push") {
        fprintf(stderr, "\
usage: /push" RESET "\n\
save current size of context window to stack. this command may be used\n\
with /pop to backtrack a conversation.\n\
");
    } else if (args[1] == "pop") {
        fprintf(stderr, "\
usage: /pop" RESET "\n\
restores size of context window from stack. this command may be used\n\
with /push to backtrack a conversation.\n\
");
    } else {
        fprintf(stderr, BRIGHT_RED "%s: unknown command" RESET "\n", args[1].c_str());
    }
}

static void on_manual(const std::vector<std::string> &args) {
    if (args.size() == 1) {
        g_manual = !g_manual;
    } else if (args.size() == 2 && (args[1] == "on" || args[1] == "off")) {
        g_manual = args[1] == "on";
    } else {
        fprintf(stderr, BRIGHT_RED "error: bad /manual command" RESET "\n"
                                   "usage: /manual [on|off]\n");
        return;
    }
    fprintf(stderr, FAINT "manual mode %s" RESET "\n", g_manual ? "enabled" : "disabled");
    if (!g_manual)
        g_role = ROLE_USER;
}

static void on_context(const std::vector<std::string> &args) {
    int configured_context = llama_n_ctx(g_ctx);
    int max_context = llama_n_ctx_train(g_model);
    printf("%d out of %d context tokens used (%d tokens remaining)\n", g_past, configured_context,
           configured_context - g_past);
    if (configured_context < max_context)
        printf("use the `-c %d` flag at startup for maximum context\n", max_context);
}

static void print_stack(void) {
    for (size_t i = g_stack.size(); i--;)
        printf("%8d\n", g_stack[i]);
}

static void on_clear(const std::vector<std::string> &args) { // [experimental]
    g_past = g_clear;
    g_stack.clear();
    g_history.resize(g_past);
}

static void on_push(const std::vector<std::string> &args) { // [experimental]
    g_stack.push_back(g_past);
    print_stack();
}

static void on_pop(const std::vector<std::string> &args) { // [experimental]
    if (g_stack.empty()) {
        fprintf(stderr, BRIGHT_RED "error: context length stack is empty" RESET "\n");
        return;
    }
    printf(BOLD "%12d" RESET " restored\n", g_stack[g_stack.size() - 1]);
    g_past = g_stack.back();
    g_stack.pop_back();
    g_history.resize(g_past);
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
        s += llama_token_to_piece(g_ctx, id, true);
    if (!s.empty() && s[s.size() - 1] != '\n')
        s += '\n';
    write(fd, s.data(), s.size());
    if (args.size() >= 2)
        close(fd);
}

static char *on_hint(const char *line, const char **ansi1, const char **ansi2) {
    *ansi1 = FAINT;
    *ansi2 = UNBOLD;
    if (!*line && g_manual)
        return strdup(get_role_name(g_role));
    else if (!*line && !g_manual && !g_once)
        return strdup("say something (or type /help for help)");
    return strdup("");
}

static void on_completion(const char *line, int pos, bestlineCompletions *comp) {
    static const char *const kCompletions[] = {
        "/context", // usage: /context
        "/dump", // usage: /dump [FILE]
        "/exit", // usage: /exit
        "/help", // usage: /help [COMMAND]
        "/manual", // usage: /manual [on|off]
        "/stats", // usage: /stats
    };
    for (int i = 0; i < sizeof(kCompletions) / sizeof(*kCompletions); ++i)
        if (startswith(kCompletions[i], line))
            bestlineAddCompletion(comp, kCompletions[i]);
}

// handle irc style commands like: `/arg0 arg1 arg2`
static bool handle_command(const char *command) {
    if (!(command[0] == '/' && std::isalpha(command[1])))
        return false;
    std::vector<std::string> args;
    std::istringstream iss(command + 1);
    std::string arg;
    while (iss >> arg)
        args.push_back(arg);
    if (args[0] == "exit" || args[0] == "bye") {
        exit(0);
    } else if (args[0] == "help" || args[0] == "?") {
        on_help(args);
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
    } else {
        fprintf(stderr, BRIGHT_RED "%s: unrecognized command" RESET "\n", args[0].c_str());
    }
    return true;
}

static void print_logo(const char16_t *s) {
    for (int i = 0; s[i]; ++i) {
        switch (s[i]) {
        case u'█':
            printf(GREEN "█" UNFOREGROUND);
            break;
        case u'╚':
        case u'═':
        case u'╝':
        case u'╗':
        case u'║':
        case u'╔':
            printf(FAINT "%C" UNBOLD, s[i]);
            break;
        default:
            printf("%C", s[i]);
            break;
        }
    }
}

static void print_ephemeral(const std::string_view &description) {
    fprintf(stderr, " " BRIGHT_BLACK "%.*s" UNFOREGROUND "\r", (int)description.size(),
            description.data());
}

static void clear_ephemeral(void) {
    fprintf(stderr, CLEAR_FORWARD);
}

static void die_out_of_context(void) {
    fprintf(stderr,
            "\n" BRIGHT_RED
            "error: ran out of context window at %d tokens; you can use the maximum "
            "context window size by passing the flag `-c %d` to llamafile." UNFOREGROUND "\n",
            g_past, llama_n_ctx_train(g_model));
    exit(1);
}

static void eval_tokens(std::vector<llama_token> tokens, int n_batch) {
    int N = (int)tokens.size();
    if (g_past + N > llama_n_ctx(g_ctx)) {
        g_past += N;
        die_out_of_context();
    }
    for (int i = 0; i < N; i += n_batch) {
        if (N > n_batch)
            print_ephemeral(format("loading prompt %d%%...", (int)((double)i / N * 100)));
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
            n_eval = n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval, g_past, 0)))
            die_out_of_context();
        g_history.insert(g_history.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
        g_past += n_eval;
    }
}

static void eval_id(int id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    eval_tokens(tokens, 1);
}

static void eval_string(const std::string &str, int n_batch, bool add_special, bool parse_special) {
    eval_tokens(llama_tokenize(g_ctx, str, add_special, parse_special), n_batch);
}

static void on_server_listening(const char *host, int port) {
    pthread_mutex_lock(&g_lock);
    g_listen_url = format("http://%s:%d/", host, port);
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
    log_disable();

    if (llamafile_has(argv, "--ascii")) {
        printf("\
 _ _                        __ _ _\n\
| | | __ _ _ __ ___   __ _ / _(_) | ___\n\
| | |/ _` | '_ ` _ \\ / _` | |_| | |/ _ \\\n\
| | | (_| | | | | | | (_| |  _| | |  __/\n\
|_|_|\\__,_|_| |_| |_|\\__,_|_| |_|_|\\___|\n");
    } else {
        print_logo(u"\n\
██╗     ██╗      █████╗ ███╗   ███╗ █████╗ ███████╗██╗██╗     ███████╗\n\
██║     ██║     ██╔══██╗████╗ ████║██╔══██╗██╔════╝██║██║     ██╔════╝\n\
██║     ██║     ███████║██╔████╔██║███████║█████╗  ██║██║     █████╗\n\
██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══╝  ██║██║     ██╔══╝\n\
███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║     ██║███████╗███████╗\n\
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝\n");
    }

    print_ephemeral("loading backend...");
    llama_backend_init();
    gpt_params params;
    params.n_batch = 512; // for better progress indication
    params.sparams.temp = 0; // don't believe in randomness by default
    if (!gpt_params_parse(argc, argv, params)) { // also loads gpu module
        fprintf(stderr, "error: failed to parse flags\n");
        exit(1);
    }
    clear_ephemeral();

    print_ephemeral("loading model...");
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = llamafile_gpu_layers(35);
    g_model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (g_model == NULL) {
        clear_ephemeral();
        fprintf(stderr, "%s: failed to load model\n", params.model.c_str());
        exit(2);
    }
    if (params.n_ctx <= 0 || params.n_ctx > llama_n_ctx_train(g_model))
        params.n_ctx = llama_n_ctx_train(g_model);
    if (params.n_ctx < params.n_batch)
        params.n_batch = params.n_ctx;
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

    printf(BOLD "software" UNBOLD ": llamafile " LLAMAFILE_VERSION_STRING "\n" //
           BOLD "model" UNBOLD ":    %s\n",
           basename(params.model).c_str());
    printf(BOLD "compute" UNBOLD ":  %s\n", describe_compute().c_str());
    if (want_server)
        printf(BOLD "server" UNBOLD ":   %s\n", g_listen_url.c_str());
    printf("\n");

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (g_ctx == NULL) {
        clear_ephemeral();
        fprintf(stderr, "error: failed to initialize context\n");
        exit(3);
    }
    clear_ephemeral();

    if (params.prompt.empty())
        params.prompt = "A chat between a curious human and an artificial intelligence assistant. "
                        "The assistant gives helpful, detailed, and polite answers to the "
                        "human's questions.";

    bool add_bos = llama_should_add_bos_token(llama_get_model(g_ctx));
    std::vector<llama_chat_msg> chat = {{"system", params.prompt}};
    std::string msg = llama_chat_apply_template(g_model, params.chat_template, chat, false);
    eval_string(msg, params.n_batch, add_bos, true);
    g_clear = g_past;
    clear_ephemeral();
    printf("%s\n", params.special ? msg.c_str() : params.prompt.c_str());

    // perform important setup
    HighlightMarkdown highlighter;
    ColorBleeder bleeder(&highlighter);
    struct llama_sampling_context *sampler = llama_sampling_init(params.sparams);
    signal(SIGINT, on_sigint);

    // run chatbot
    for (;;) {
        bestlineLlamaMode(true);
        bestlineSetHintsCallback(on_hint);
        bestlineSetFreeHintsCallback(free);
        bestlineSetCompletionCallback(on_completion);
        write(1, get_role_color(g_role), strlen(get_role_color(g_role)));
        char *line = bestlineWithHistory(">>> ", "llamafile");
        write(1, UNFOREGROUND, strlen(UNFOREGROUND));
        if (!line) {
            if (g_got_sigint)
                printf("\n");
            break;
        }
        if (is_empty(line)) {
            if (g_manual) {
                g_role = cycle_role(g_role);
                write(1, "\033[F", 3);
            }
            free(line);
            continue;
        }
        if (handle_command(line)) {
            free(line);
            continue;
        }
        g_once = true;
        bool add_assi = !g_manual;
        std::vector<llama_chat_msg> chat = {{get_role_name(g_role), line}};
        std::string msg = llama_chat_apply_template(g_model, params.chat_template, chat, add_assi);
        eval_string(msg, params.n_batch, false, true);
        if (g_manual) {
            g_role = get_next_role(g_role);
            free(line);
            continue;
        }
        while (!g_got_sigint) {
            llama_token id = llama_sampling_sample(sampler, g_ctx, NULL);
            llama_sampling_accept(sampler, g_ctx, id, true);
            if (llama_token_is_eog(g_model, id))
                break;
            std::string s;
            bleeder.feed(&s, llama_token_to_piece(g_ctx, id, params.special));
            printf("%s", s.c_str());
            fflush(stdout);
            eval_id(id);
        }
        g_got_sigint = 0;
        free(line);
        std::string s;
        bleeder.flush(&s);
        printf("%s\n", s.c_str());
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
