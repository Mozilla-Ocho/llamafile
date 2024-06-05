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

#include "client.h"

#include <ctl/optional.h>
#include <ctl/string.h>
#include <errno.h>
#include <libc/fmt/itoa.h>
#include <libc/str/slice.h>
#include <limits.h>
#include <string.h>
#include <sys/auxv.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include "json.h"
#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/version.h"
#include "log.h"
#include "time.h"
#include "utils.h"

#define STANDARD_RESPONSE_HEADERS \
    "Server: llamafile/" LLAMAFILE_VERSION_STRING "\r\n" \
    "Referrer-Policy: origin\r\n" \
    "Cache-Control: private; max-age=0\r\n"

#define HasHeader(H) (!!msg.headers[H].a)
#define HeaderData(H) (ibuf.p + msg.headers[H].a)
#define HeaderLength(H) (msg.headers[H].b - msg.headers[H].a)
#define HeaderEqual(H, S) \
    SlicesEqual(S, strlen(S), HeaderData(H), HeaderLength(H))
#define HeaderEqualCase(H, S) \
    SlicesEqualCase(S, strlen(S), HeaderData(H), HeaderLength(H))

using namespace ctl;

Client::Client() : ibuf(8 * 1024 * 1024), obuf(64 * 1024 * 1024)
{
    InitHttpMessage(&msg, 0);
    url.params.p = nullptr;
}

int
Client::close()
{
    int rc = 0;
    clear();
    DestroyHttpMessage(&msg);
    if (fd != -1) {
        if (FLAG_verbose >= 2)
            LOG("close");
        rc = ::close(fd);
        fd = -1;
    }
    return rc;
}

void
Client::clear()
{
    free(url_memory);
    url_memory = nullptr;
    free(params_memory);
    params_memory = nullptr;
    free(url.params.p);
    url.params.p = nullptr;
    close_connection = false;
    payload = "";
    unread = 0;
}

void
Client::run()
{
    ibuf.n = 0;
    for (;;) {

        // read headers
        clear();
        if (!read_request())
            break;

        // process message
        if (!transport())
            break;

        // synchronize message stream
        if (close_connection)
            break;
        if (!read_payload())
            break;

        // move pipelined bytes back to beginning
        if (ibuf.n == ibuf.i) {
            ibuf.n = 0;
        } else {
            memmove(ibuf.p, ibuf.p + ibuf.i, ibuf.n - ibuf.i);
            ibuf.n -= ibuf.i;
        }
    }
}

bool
Client::read_request()
{
    int inmsglen;
    ResetHttpMessage(&msg, kHttpRequest);
    for (;;) {
        inmsglen = ParseHttpMessage(&msg, ibuf.p, ibuf.n, ibuf.c);
        if (inmsglen > 0) {
            message_started = timespec_real();
            ibuf.i = inmsglen;
            return true;
        }
        if (inmsglen == -1) {
            LOG("bad message %m");
            return false;
        }
        if (ibuf.n)
            LOG("fragmented message with %zu bytes", ibuf.n);
        ssize_t got;
        got = read(fd, ibuf.p + ibuf.n, ibuf.c - ibuf.n);
        if (!got && ibuf.n)
            LOG("unexpected eof after %zu bytes", ibuf.n);
        if (got == -1 && (ibuf.n || (errno != EAGAIN && errno != ECONNRESET)))
            LOG("read failed %m");
        if (got <= 0)
            return false;
        ibuf.n += got;
    }
}

bool
Client::transport()
{
    if (msg.version > 11) {
        close_connection = true;
        return send_error(505);
    }

    if (msg.method == kHttpConnect) {
        close_connection = true;
        return send_error(501);
    }

    if (!has_at_most_this_element(kHttpExpect, "100-continue")) {
        close_connection = true;
        return send_error(417);
    }

    if (HasHeader(kHttpTransferEncoding))
        if (!HeaderEqualCase(kHttpTransferEncoding, "identity")) {
            close_connection = true;
            return send_error(501, "Transfer-Encoding Not Implemented");
        }

    if (HasHeader(kHttpContentLength)) {
        long cl;
        cl = ParseContentLength(HeaderData(kHttpContentLength),
                                HeaderLength(kHttpContentLength));
        if (cl == -1) {
            close_connection = true;
            return send_error(400, "Bad Content-Length");
        }
        if (cl > ibuf.c - ibuf.i) {
            close_connection = true;
            return send_error(413);
        }
        unread = cl;
    } else if (msg.method == kHttpPost || msg.method == kHttpPut) {
        close_connection = true;
        return send_error(411);
    }

    if (FLAG_verbose >= 1)
        LOG("get %#.*s", msg.uri.b - msg.uri.a, ibuf.p + msg.uri.a);

    if (msg.version >= 11)
        if (HeaderEqualCase(kHttpExpect, "100-continue"))
            if (!send("HTTP/1.1 100 Continue\r\n\r\n"))
                return false;

    url_memory = ParseUrl(
      ibuf.p + msg.uri.a, msg.uri.b - msg.uri.a, &url, kUrlPlus | kUrlLatin1);
    if (!url_memory)
        __builtin_trap();

    return dispatch();
}

bool
Client::send_error(int code, const char* reason)
{
    if (!reason)
        reason = GetHttpReason(code);
    LOG("error %d %s", code, reason);
    char* p = start_response(obuf.p, code, reason);
    return send_response(obuf.p, p, string(reason) + "\r\n");
}

char*
Client::start_response(char* p, int code, const char* reason)
{
    *p++ = 'H';
    *p++ = 'T';
    *p++ = 'T';
    *p++ = 'P';
    *p++ = '/';
    *p++ = '1';
    *p++ = '.';
    *p++ = '0' + (msg.version & 1);
    *p++ = ' ';
    *p++ = '0' + code / 100;
    *p++ = '0' + code / 10 % 10;
    *p++ = '0' + code % 10;
    *p++ = ' ';
    if (!reason)
        reason = GetHttpReason(code);
    p = stpcpy(p, reason);
    *p++ = '\r';
    *p++ = '\n';
    p = stpcpy(p, STANDARD_RESPONSE_HEADERS);
    return p;
}

bool
Client::send_response(char* p0, char* p, string_view content)
{
    // append date header
    tm tm;
    p = stpcpy(p, "Date: ");
    gmtime_lockless(message_started.tv_sec, &tm);
    p = FormatHttpDateTime(p, &tm);
    *p++ = '\r';
    *p++ = '\n';

    // inform client of close() intent
    if (msg.version < 11)
        close_connection = true;
    if (HeaderEqualCase(kHttpConnection, "close"))
        close_connection = true;
    if (close_connection)
        p = stpcpy(p, "Connection: close\r\n");

    // append content length
    p = stpcpy(p, "Content-Length: ");
    p = FormatInt64(p, content.size());
    *p++ = '\r';
    *p++ = '\n';

    // finish message
    *p++ = '\r';
    *p++ = '\n';

    return send2(string_view(p0, p - p0), content);
}

bool
Client::send(const string_view s)
{
    ssize_t sent;
    if ((sent = write(fd, s.data(), s.size())) != s.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            LOG("write failed %m");
        return false;
    }
    return true;
}

bool
Client::send2(const string_view s1, const string_view s2)
{
    iovec iov[2];
    ssize_t sent;
    iov[0].iov_base = (void*)s1.data();
    iov[0].iov_len = s1.size();
    iov[1].iov_base = (void*)s2.data();
    iov[1].iov_len = s2.size();
    if ((sent = writev(fd, iov, 2)) != s1.size() + s2.size()) {
        if (sent == -1 && errno != EAGAIN && errno != ECONNRESET)
            LOG("writev failed %m");
        return false;
    }
    return true;
}

bool
Client::has_at_most_this_element(int h, const string_view s)
{
    if (!HasHeader(h))
        return true;
    if (!SlicesEqualCase(s.data(), s.size(), HeaderData(h), HeaderLength(h)))
        return false;
    struct HttpHeader* x;
    for (unsigned i = 0; i < msg.xheaders.n; ++i) {
        x = msg.xheaders.p + i;
        if (GetHttpHeader(ibuf.p + x->k.a, x->k.b - x->k.a) == h &&
            !SlicesEqualCase(
              ibuf.p + x->v.a, x->v.b - x->v.a, s.data(), s.size())) {
            return false;
        }
    }
    return true;
}

bool
Client::read_payload()
{
    while (ibuf.n - ibuf.i < unread) {
        ssize_t got;
        if ((got = read(fd, ibuf.p + ibuf.n, ibuf.c - ibuf.n)) <= 0) {
            if (!got)
                LOG("unexpected eof");
            if (got == -1)
                LOG("read failed %m");
            return false;
        }
        ibuf.n += got;
    }
    payload = string_view(ibuf.p + ibuf.i, unread);
    ibuf.i += unread;
    unread = 0;
    if (msg.method == kHttpPost && //
        HasHeader(kHttpContentType) &&
        IsMimeType(HeaderData(kHttpContentType),
                   HeaderLength(kHttpContentType),
                   "application/x-www-form-urlencoded")) {
        params_memory =
          ParseParams(payload.data(), payload.size(), &url.params);
    }

    return true;
}

bool
Client::dispatch()
{
    if (path() == "/tokenize")
        return tokenize();
    return send_error(404);
}

string_view
Client::path()
{
    if (!url.path.n)
        return "/";
    return { url.path.p, url.path.n };
}

optional<string_view>
Client::param(string_view key)
{
    for (size_t i = 0; i < url.params.n; ++i)
        if (key.size() == url.params.p[i].key.n)
            if (!memcmp(key.data(), url.params.p[i].key.p, key.size()))
                return optional(
                  string_view(url.params.p[i].val.p, url.params.p[i].val.n));
    return {};
}

static string_view
or_empty(optional<string_view> x)
{
    if (x.has_value())
        return x.value();
    return {};
}

bool
Client::tokenize()
{
    if (msg.method != kHttpGet && msg.method != kHttpPost)
        return send_error(405);

    if (!read_payload())
        return false;

    // get prompt
    //
    //   1. Allow GET "/tokenize?prompt=foo"
    //   2. Allow POST "prompt=foo" (application/x-www-form-urlencoded)
    //   3. Allow POST "foo" (text/plain)
    //
    string_view input;
    optional<string_view> prompt = param("prompt");
    if (prompt.has_value()) {
        input = prompt.value();
    } else if (HasHeader(kHttpContentType)) {
        if (IsMimeType(HeaderData(kHttpContentType),
                       HeaderLength(kHttpContentType),
                       "text/plain")) {
            input = payload;
        } else {
            return send_error(501, "Content Type Not Implemented");
        }
    } else {
        input = payload;
    }

    // get optional parameters
    bool add_special = atob(or_empty(param("add_special")), true);
    bool parse_special = atob(or_empty(param("parse_special")), false);

    // setup statistics
    rusage rustart = {};
    getrusage(RUSAGE_THREAD, &rustart);
    timespec started = timespec_real();

    // turn text into tokens
    extern llama_model* g_model;
    int maxcount = input.size() + 16;
    llama_token* toks = new llama_token[maxcount];
    int count = llama_tokenize(g_model,
                               input.data(),
                               input.size(),
                               toks,
                               maxcount,
                               add_special,
                               parse_special);
    if (count < 0) {
        delete[] toks;
        __builtin_trap();
    }

    // serialize tokens to json
    char* p = obuf.p;
    p = stpcpy(p, "{\r\n");
    p = stpcpy(p, "  \"add_special\": ");
    p = encode_bool(p, add_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"parse_special\": ");
    p = encode_bool(p, parse_special);
    p = stpcpy(p, ",\n");
    p = stpcpy(p, "  \"tokens\": [");
    for (int i = 0; i < count; ++i) {
        if (i)
            *p++ = ',';
        p = stpcpy(p, "\r\n    ");
        char s[32];
        int n = llama_token_to_piece(g_model, toks[i], s, sizeof(s), true);
        if (n < 0) {
            delete[] toks;
            __builtin_trap();
        }
        p = encode_json(p, string_view(s, n));
    }
    p = stpcpy(p, "\r\n  ]\r\n");
    p = stpcpy(p, "}\r\n");
    string_view content(obuf.p, p - obuf.p);
    delete[] toks;

    // collect statistics
    rusage ruend = {};
    getrusage(RUSAGE_THREAD, &ruend);
    timeval user = timeval_sub(ruend.ru_utime, rustart.ru_utime);
    timeval system = timeval_sub(ruend.ru_stime, rustart.ru_stime);
    timespec ended = timespec_real();
    timespec wall = timespec_sub(ended, started);
    long wall_us = timespec_tomicros(wall);
    long user_us = timeval_tomicros(user);
    long system_us = timeval_tomicros(system);

    // send response
    char* headers = p;
    p = start_response(p, 200);
    p = stpcpy(p, "Content-Type: application/json\r\n");
    p = stpcpy(p, "X-Wall-Micros: ");
    p = FormatInt64(p, wall_us);
    p = stpcpy(p, "\r\nX-User-Micros: ");
    p = FormatInt64(p, user_us);
    p = stpcpy(p, "\r\nX-System-Micros: ");
    p = FormatInt64(p, system_us);
    p = stpcpy(p, "\r\n");
    return send_response(headers, p, content);
}
