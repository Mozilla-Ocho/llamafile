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

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#include "llama.cpp/llama.h"
#include "llamafile/llamafile.h"
#include "llamafile/version.h"

#include "log.h"
#include "time.h"

#define STANDARD_RESPONSE_HEADERS \
    "Server: llamafile/" LLAMAFILE_VERSION_STRING "\r\n" \
    "Referrer-Policy: origin\r\n" \
    "Cache-Control: private; max-age=0\r\n"

using namespace ctl;

Client::Client()
  : cleanups(nullptr), ibuf(FLAG_http_ibuf_size), obuf(FLAG_http_obuf_size)
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
Client::cleanup()
{
    Cleanup* clean;
    while ((clean = cleanups)) {
        cleanups = clean->next;
        clean->func(clean->arg);
        delete clean;
    }
}

void
Client::clear()
{
    cleanup();
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
Client::defer_cleanup(void (*func)(void*), void* arg)
{
    Cleanup* clean = new Cleanup;
    clean->next = cleanups;
    clean->func = func;
    clean->arg = arg;
    cleanups = clean;
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
    cleanup();
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
    cleanup();

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
    if (path() == "/embedding")
        return embedding();
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
