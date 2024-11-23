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

#include <cassert>
#include <vector>

#include "llama.cpp/llama.h"
#include "llamafile/color.h"
#include "llamafile/llama.h"
#include "llamafile/macros.h"
#include "llamafile/string.h"

namespace lf {
namespace chatbot {

bool g_manual_mode;
enum Role g_role = ROLE_USER;
int g_system_prompt_tokens;
std::vector<llama_pos> g_stack;
std::vector<llama_pos> g_undo;
std::vector<llama_token> g_history;

const char *get_role_name(enum Role role) {
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

const char *get_role_color(enum Role role) {
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

enum Role get_next_role(enum Role role) {
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

enum Role cycle_role(enum Role role) {
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

int tokens_used(void) {
    return g_history.size();
}

std::string describe_token(llama_token token) {
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

std::string describe_erasure(llama_pos begin, llama_pos end) {
    unassert(begin <= end);
    unassert(end <= tokens_used());
    std::string description;
    llama_pos pos = begin;
    while (pos < end && description.size() < 63)
        description += describe_token(g_history[pos++]);
    if (!description.empty() && pos < end)
        description += " ...";
    description = collapse(description);
    if (pos == end && description.empty())
        description = "<absolute end>";
    return description;
}

std::string describe_position(llama_pos pos) {
    unassert(pos <= tokens_used());
    std::string description;
    while (pos > 0 && description.size() < 63)
        description = describe_token(g_history[--pos]) + description;
    if (!description.empty() && pos > 0)
        description = std::string("... ") + description;
    description = collapse(description);
    if (!pos && description.empty())
        description = "<absolute beginning>";
    return description;
}

static void fix_stack(std::vector<llama_pos> *stack) {
    while (!stack->empty() && stack->back() > tokens_used())
        stack->pop_back();
}

void fix_stacks(void) {
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

void adjust_stacks(llama_pos erase_begin, llama_pos erase_end) {
    g_undo = adjust_stack(erase_begin, erase_end, g_undo);
    g_stack = adjust_stack(erase_begin, erase_end, g_stack);
}

void record_undo(void) {
    if (g_undo.empty() || g_undo.back() != tokens_used())
        g_undo.push_back(tokens_used());
}

void on_undo(const std::vector<std::string> &args) {
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

void on_forget(const std::vector<std::string> &args) {
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

void rewind(int pos) {
    unassert(pos <= tokens_used());
    llama_kv_cache_seq_rm(g_ctx, 0, pos, -1);
    g_history.resize(pos);
}

void on_manual(const std::vector<std::string> &args) {
    if (is_base_model()) {
        err("error: /manual mode not supported on base models");
        return;
    }
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

void on_context(const std::vector<std::string> &args) {
    int configured_context = llama_n_ctx(g_ctx);
    int max_context = llama_n_ctx_train(g_model);
    printf("%d out of %d context tokens used (%d tokens remaining)\n", tokens_used(),
           configured_context, configured_context - tokens_used());
    if (configured_context < max_context)
        printf("use the `-c %d` flag at startup for maximum context\n", max_context);
}

void on_clear(const std::vector<std::string> &args) {
    rewind(g_system_prompt_tokens);
    g_stack.clear();
    fix_stacks();
}

void print_stack(void) {
    for (size_t i = g_stack.size(); i--;)
        printf("%12d " FAINT "(%s)" RESET "\n", g_stack[i], describe_position(g_stack[i]).c_str());
}

void on_push(const std::vector<std::string> &args) {
    g_stack.push_back(tokens_used());
    print_stack();
}

void on_pop(const std::vector<std::string> &args) {
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

void on_stack(const std::vector<std::string> &args) {
    if (g_stack.empty()) {
        printf(FAINT "stack is currently empty (try using /push)" RESET "\n");
        return;
    }
    print_stack();
}

void on_dump(const std::vector<std::string> &args) {
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

} // namespace chatbot
} // namespace lf
