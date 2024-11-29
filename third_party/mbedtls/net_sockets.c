/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright The Mbed TLS Contributors                                          │
│                                                                              │
│ Licensed under the Apache License, Version 2.0 (the "License");              │
│ you may not use this file except in compliance with the License.             │
│ You may obtain a copy of the License at                                      │
│                                                                              │
│     http://www.apache.org/licenses/LICENSE-2.0                               │
│                                                                              │
│ Unless required by applicable law or agreed to in writing, software          │
│ distributed under the License is distributed on an "AS IS" BASIS,            │
│ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     │
│ See the License for the specific language governing permissions and          │
│ limitations under the License.                                               │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <stdbool.h>
#include "third_party/mbedtls/net_sockets.h"
#include <libc/calls/calls.h>
#include <libc/calls/struct/sigaction.h>
#include <libc/calls/weirdtypes.h>
#include <libc/errno.h>
#include <libc/sock/select.h>
#include <libc/sock/struct/sockaddr6.h>
#include <libc/sysv/consts/af.h>
#include <libc/sysv/consts/f.h>
#include <libc/sysv/consts/ipproto.h>
#include <libc/sysv/consts/msg.h>
#include <libc/sysv/consts/o.h>
#include <libc/sysv/consts/sig.h>
#include <libc/sysv/consts/so.h>
#include <libc/sysv/consts/sock.h>
#include <libc/sysv/consts/sol.h>
#include "third_party/mbedtls/error.h"
#include <third_party/musl/netdb.h>
#include <libc/sock/sock.h>
#include "third_party/mbedtls/ssl.h"

#define IS_EINTR(ret) ((ret) == EINTR)

static int net_prepare(void) {
  signal(SIGPIPE, SIG_IGN);
  return 0;
}

/**
 * \brief          Initialize a context
 *                 Just makes the context ready to be used or freed safely.
 *
 * \param ctx      Context to initialize
 */
void mbedtls_net_init(mbedtls_net_context *ctx) {
  ctx->fd = -1;
}

/**
 * \brief          Initiate a connection with host:port in the given protocol
 *
 * \param ctx      Socket to use
 * \param host     Host to connect to
 * \param port     Port to connect to
 * \param proto    Protocol: MBEDTLS_NET_PROTO_TCP or MBEDTLS_NET_PROTO_UDP
 *
 * \return         0 if successful, or one of:
 *                      MBEDTLS_ERR_NET_SOCKET_FAILED,
 *                      MBEDTLS_ERR_NET_UNKNOWN_HOST,
 *                      MBEDTLS_ERR_NET_CONNECT_FAILED
 *
 * \note           Sets the socket in connected mode even with UDP.
 */
int mbedtls_net_connect(mbedtls_net_context *ctx, const char *host,
                        const char *port, int proto) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  struct addrinfo hints, *addr_list, *cur;
  if ((ret = net_prepare()) != 0) return ret;
  /* Do name resolution with both IPv6 and IPv4 */
  mbedtls_platform_zeroize(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = proto == MBEDTLS_NET_PROTO_UDP ? SOCK_DGRAM : SOCK_STREAM;
  hints.ai_protocol =
      proto == MBEDTLS_NET_PROTO_UDP ? IPPROTO_UDP : IPPROTO_TCP;
  if (getaddrinfo(host, port, &hints, &addr_list) != 0)
    return MBEDTLS_ERR_NET_UNKNOWN_HOST;
  /* Try the sockaddrs until a connection succeeds */
  ret = MBEDTLS_ERR_NET_UNKNOWN_HOST;
  for (cur = addr_list; cur != NULL; cur = cur->ai_next) {
    ctx->fd = (int)socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (ctx->fd < 0) {
      ret = MBEDTLS_ERR_NET_SOCKET_FAILED;
      continue;
    }
    if (connect(ctx->fd, cur->ai_addr, cur->ai_addrlen) == 0) {
      ret = 0;
      break;
    }
    close(ctx->fd);
    if (errno == ECANCELED) {
      ret = MBEDTLS_ERR_SSL_CANCELED;
    } else {
      ret = MBEDTLS_ERR_NET_CONNECT_FAILED;
    }
  }
  freeaddrinfo(addr_list);
  return ret;
}

/**
 * \brief          Create a receiving socket on bind_ip:port in the chosen
 *                 protocol. If bind_ip == NULL, all interfaces are bound.
 *
 * \param ctx      Socket to use
 * \param bind_ip  IP to bind to, can be NULL
 * \param port     Port number to use
 * \param proto    Protocol: MBEDTLS_NET_PROTO_TCP or MBEDTLS_NET_PROTO_UDP
 *
 * \return         0 if successful, or one of:
 *                      MBEDTLS_ERR_NET_SOCKET_FAILED,
 *                      MBEDTLS_ERR_NET_UNKNOWN_HOST,
 *                      MBEDTLS_ERR_NET_BIND_FAILED,
 *                      MBEDTLS_ERR_NET_LISTEN_FAILED
 *
 * \note           Regardless of the protocol, opens the sockets and binds it.
 *                 In addition, make the socket listening if protocol is TCP.
 */
int mbedtls_net_bind(mbedtls_net_context *ctx, const char *bind_ip,
                     const char *port, int proto) {
  int n, ret;
  struct addrinfo hints, *addr_list, *cur;
  if ((ret = net_prepare()) != 0) return ret;
  /* Bind to IPv6 and/or IPv4, but only in the desired protocol */
  mbedtls_platform_zeroize(&hints, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = proto == MBEDTLS_NET_PROTO_UDP ? SOCK_DGRAM : SOCK_STREAM;
  hints.ai_protocol =
      proto == MBEDTLS_NET_PROTO_UDP ? IPPROTO_UDP : IPPROTO_TCP;
  if (bind_ip == NULL) hints.ai_flags = AI_PASSIVE;
  if (getaddrinfo(bind_ip, port, &hints, &addr_list) != 0)
    return MBEDTLS_ERR_NET_UNKNOWN_HOST;
  /* Try the sockaddrs until a binding succeeds */
  ret = MBEDTLS_ERR_NET_UNKNOWN_HOST;
  for (cur = addr_list; cur != NULL; cur = cur->ai_next) {
    ctx->fd = (int)socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
    if (ctx->fd < 0) {
      ret = MBEDTLS_ERR_NET_SOCKET_FAILED;
      continue;
    }
    n = 1;
    if (setsockopt(ctx->fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&n,
                   sizeof(n)) != 0) {
      close(ctx->fd);
      ret = MBEDTLS_ERR_NET_SOCKET_FAILED;
      continue;
    }
    if (bind(ctx->fd, cur->ai_addr, cur->ai_addrlen) != 0) {
      close(ctx->fd);
      ret = MBEDTLS_ERR_NET_BIND_FAILED;
      continue;
    }
    /* Listen only makes sense for TCP */
    if (proto == MBEDTLS_NET_PROTO_TCP) {
      if (listen(ctx->fd, MBEDTLS_NET_LISTEN_BACKLOG) != 0) {
        close(ctx->fd);
        ret = MBEDTLS_ERR_NET_LISTEN_FAILED;
        continue;
      }
    }
    /* Bind was successful */
    ret = 0;
    break;
  }
  freeaddrinfo(addr_list);
  return ret;
}

/*
 * Check if the requested operation would be blocking on a non-blocking socket
 * and thus 'failed' with a negative return value.
 *
 * Note: on a blocking socket this function always returns 0!
 */
static int net_would_block(const mbedtls_net_context *ctx) {
  int err = errno;
  /*
   * Never return 'WOULD BLOCK' on a blocking socket
   */
  if ((fcntl(ctx->fd, F_GETFL) & O_NONBLOCK) != O_NONBLOCK) {
    errno = err;
    return 0;
  }
  errno = err;
  if (err == EAGAIN || err == EWOULDBLOCK) return 1;
  return 0;
}

/**
 * \brief           Accept a connection from a remote client
 *
 * \param bind_ctx  Relevant socket
 * \param client_ctx Will contain the connected client socket
 * \param client_ip Will contain the client IP address, can be NULL
 * \param buf_size  Size of the client_ip buffer
 * \param ip_len    Will receive the size of the client IP written,
 *                  can be NULL if client_ip is null
 *
 * \return          0 if successful, or
 *                  MBEDTLS_ERR_NET_SOCKET_FAILED,
 *                  MBEDTLS_ERR_NET_BIND_FAILED,
 *                  MBEDTLS_ERR_NET_ACCEPT_FAILED, or
 *                  MBEDTLS_ERR_NET_BUFFER_TOO_SMALL if buf_size is too small,
 *                  MBEDTLS_ERR_SSL_WANT_READ if bind_fd was set to
 *                  non-blocking and accept() would block.
 */
int mbedtls_net_accept(mbedtls_net_context *bind_ctx,
                       mbedtls_net_context *client_ctx, void *client_ip,
                       size_t buf_size, size_t *ip_len) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  int type;
  struct sockaddr_storage client_addr;
#if defined(__socklen_t_defined) || defined(_SOCKLEN_T) ||          \
    defined(_SOCKLEN_T_DECLARED) || defined(__DEFINED_socklen_t) || \
    defined(socklen_t) ||                                           \
    (defined(_POSIX_VERSION) && _POSIX_VERSION >= 200112L)
  socklen_t n = (socklen_t)sizeof(client_addr);
  socklen_t type_len = (socklen_t)sizeof(type);
#else
  int n = (int)sizeof(client_addr);
  int type_len = (int)sizeof(type);
#endif
  /* Is this a TCP or UDP socket? */
  if (getsockopt(bind_ctx->fd, SOL_SOCKET, SO_TYPE, (void *)&type, &type_len) !=
          0 ||
      (type != SOCK_STREAM && type != SOCK_DGRAM)) {
    return MBEDTLS_ERR_NET_ACCEPT_FAILED;
  }
  if (type == SOCK_STREAM) {
    /* TCP: actual accept() */
    ret = client_ctx->fd =
        (int)accept(bind_ctx->fd, (struct sockaddr *)&client_addr, &n);
  } else {
    /* UDP: wait for a message, but keep it in the queue */
    char buf[1] = {0};
    ret = (int)recvfrom(bind_ctx->fd, buf, sizeof(buf), MSG_PEEK,
                        (struct sockaddr *)&client_addr, &n);
#if defined(_WIN32)
    if (ret == SOCKET_ERROR && WSAGetLastError() == WSAEMSGSIZE) {
      /* We know buf is too small, thanks, just peeking here */
      ret = 0;
    }
#endif
  }
  if (ret < 0) {
    if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
    if (net_would_block(bind_ctx) != 0) return MBEDTLS_ERR_SSL_WANT_READ;
    return MBEDTLS_ERR_NET_ACCEPT_FAILED;
  }
  /* UDP: hijack the listening socket to communicate with the client,
   * then bind a new socket to accept new connections */
  if (type != SOCK_STREAM) {
    struct sockaddr_storage local_addr;
    int one = 1;
    if (connect(bind_ctx->fd, (struct sockaddr *)&client_addr, n) != 0) {
      if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
      return MBEDTLS_ERR_NET_ACCEPT_FAILED;
    }
    client_ctx->fd = bind_ctx->fd;
    bind_ctx->fd = -1; /* In case we exit early */
    n = sizeof(struct sockaddr_storage);
    if (getsockname(client_ctx->fd, (struct sockaddr *)&local_addr, &n) != 0 ||
        (bind_ctx->fd =
             (int)socket(local_addr.ss_family, SOCK_DGRAM, IPPROTO_UDP)) < 0 ||
        setsockopt(bind_ctx->fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one,
                   sizeof(one)) != 0) {
      return MBEDTLS_ERR_NET_SOCKET_FAILED;
    }
    if (bind(bind_ctx->fd, (struct sockaddr *)&local_addr, n) != 0) {
      return MBEDTLS_ERR_NET_BIND_FAILED;
    }
  }
  if (client_ip != NULL) {
    if (client_addr.ss_family == AF_INET) {
      struct sockaddr_in *addr4 = (struct sockaddr_in *)&client_addr;
      *ip_len = sizeof(addr4->sin_addr.s_addr);
      if (buf_size < *ip_len) return MBEDTLS_ERR_NET_BUFFER_TOO_SMALL;
      memcpy(client_ip, &addr4->sin_addr.s_addr, *ip_len);
    } else {
      struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)&client_addr;
      *ip_len = sizeof(addr6->sin6_addr.s6_addr);
      if (buf_size < *ip_len) return MBEDTLS_ERR_NET_BUFFER_TOO_SMALL;
      memcpy(client_ip, &addr6->sin6_addr.s6_addr, *ip_len);
    }
  }
  return 0;
}

/**
 * \brief          Set the socket blocking
 *
 * \param ctx      Socket to set
 *
 * \return         0 if successful, or a non-zero error code
 */
int mbedtls_net_set_block(mbedtls_net_context *ctx) {
  return fcntl(ctx->fd, F_SETFL, fcntl(ctx->fd, F_GETFL) & ~O_NONBLOCK);
}

/**
 * \brief          Set the socket non-blocking
 *
 * \param ctx      Socket to set
 *
 * \return         0 if successful, or a non-zero error code
 */
int mbedtls_net_set_nonblock(mbedtls_net_context *ctx) {
  return fcntl(ctx->fd, F_SETFL, fcntl(ctx->fd, F_GETFL) | O_NONBLOCK);
}

/**
 * \brief          Check and wait for the context to be ready for read/write
 *
 * \note           The current implementation of this function uses
 *                 select() and returns an error if the file descriptor
 *                 is \c FD_SETSIZE or greater.
 *
 * \param ctx      Socket to check
 * \param rw       Bitflag composed of MBEDTLS_NET_POLL_READ and
 *                 MBEDTLS_NET_POLL_WRITE specifying the events
 *                 to wait for:
 *                 - If MBEDTLS_NET_POLL_READ is set, the function
 *                   will return as soon as the net context is available
 *                   for reading.
 *                 - If MBEDTLS_NET_POLL_WRITE is set, the function
 *                   will return as soon as the net context is available
 *                   for writing.
 * \param timeout  Maximal amount of time to wait before returning,
 *                 in milliseconds. If \c timeout is zero, the
 *                 function returns immediately. If \c timeout is
 *                 -1u, the function blocks potentially indefinitely.
 *
 * \return         Bitmask composed of MBEDTLS_NET_POLL_READ/WRITE
 *                 on success or timeout, or a negative return code otherwise.
 */
int mbedtls_net_poll(mbedtls_net_context *ctx, uint32_t rw, uint32_t timeout) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  struct timeval tv;
  fd_set read_fds;
  fd_set write_fds;
  int fd = ctx->fd;
  if (fd < 0) return MBEDTLS_ERR_NET_INVALID_CONTEXT;
  /* A limitation of select() is that it only works with file descriptors
   * that are strictly less than FD_SETSIZE. This is a limitation of the
   * fd_set type. Error out early, because attempting to call FD_SET on a
   * large file descriptor is a buffer overflow on typical platforms. */
  if (fd >= FD_SETSIZE) return MBEDTLS_ERR_NET_POLL_FAILED;
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
  /* Ensure that memory sanitizers consider read_fds and write_fds as
   * initialized even on platforms such as<Glibc/x86_64 where FD_ZER>
   * is implemented in assembly. */
  mbedtls_platform_zeroize(&read_fds, sizeof(read_fds));
  mbedtls_platform_zeroize(&write_fds, sizeof(write_fds));
#endif
#endif
  FD_ZERO(&read_fds);
  if (rw & MBEDTLS_NET_POLL_READ) {
    rw &= ~MBEDTLS_NET_POLL_READ;
    FD_SET(fd, &read_fds);
  }
  FD_ZERO(&write_fds);
  if (rw & MBEDTLS_NET_POLL_WRITE) {
    rw &= ~MBEDTLS_NET_POLL_WRITE;
    FD_SET(fd, &write_fds);
  }
  if (rw != 0) return MBEDTLS_ERR_NET_BAD_INPUT_DATA;
  tv.tv_sec = timeout / 1000;
  tv.tv_usec = (timeout % 1000) * 1000;
  do {
    ret = select(fd + 1, &read_fds, &write_fds, NULL,
                 timeout == (uint32_t)-1 ? NULL : &tv);
  } while (IS_EINTR(ret));
  if (ret < 0) {
    if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
    return MBEDTLS_ERR_NET_POLL_FAILED;
  }
  ret = 0;
  if (FD_ISSET(fd, &read_fds)) ret |= MBEDTLS_NET_POLL_READ;
  if (FD_ISSET(fd, &write_fds)) ret |= MBEDTLS_NET_POLL_WRITE;
  return ret;
}

/**
 * \brief          Portable usleep helper
 *
 * \param usec     Amount of microseconds to sleep
 *
 * \note           Real amount of time slept will not be less than
 *                 select()'s timeout granularity (typically, 10ms).
 */
void mbedtls_net_usleep(unsigned long usec) {
  usleep(usec);
}

/**
 * \brief          Read at most 'len' characters. If no error occurs,
 *                 the actual amount read is returned.
 *
 * \param ctx      Socket
 * \param buf      The buffer to write to
 * \param len      Maximum length of the buffer
 *
 * \return         the number of bytes received,
 *                 or a non-zero error code; with a non-blocking socket,
 *                 MBEDTLS_ERR_SSL_WANT_READ indicates read() would block.
 */
int mbedtls_net_recv(void *ctx, unsigned char *buf, size_t len) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  int fd = ((mbedtls_net_context *)ctx)->fd;
  if (fd < 0) return MBEDTLS_ERR_NET_INVALID_CONTEXT;
  ret = (int)read(fd, buf, len);
  if (ret < 0) {
    if (net_would_block(ctx) != 0) return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == EPIPE || errno == ECONNRESET)
      return MBEDTLS_ERR_NET_CONN_RESET;
    if (errno == EINTR) return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
    return MBEDTLS_ERR_NET_RECV_FAILED;
  }
  return ret;
}

/**
 * \brief          Read at most 'len' characters, blocking for at most
 *                 'timeout' seconds. If no error occurs, the actual amount
 *                 read is returned.
 *
 * \note           The current implementation of this function uses
 *                 select() and returns an error if the file descriptor
 *                 is \c FD_SETSIZE or greater.
 *
 * \param ctx      Socket
 * \param buf      The buffer to write to
 * \param len      Maximum length of the buffer
 * \param timeout  Maximum number of milliseconds to wait for data
 *                 0 means no timeout (wait forever)
 *
 * \return         The number of bytes received if successful.
 *                 MBEDTLS_ERR_SSL_TIMEOUT if the operation timed out.
 *                 MBEDTLS_ERR_SSL_WANT_READ if interrupted by a signal.
 *                 Another negative error code (MBEDTLS_ERR_NET_xxx)
 *                 for other failures.
 *
 * \note           This function will block (until data becomes available or
 *                 timeout is reached) even if the socket is set to
 *                 non-blocking. Handling timeouts with non-blocking reads
 *                 requires a different strategy.
 */
int mbedtls_net_recv_timeout(void *ctx, unsigned char *buf, size_t len,
                             uint32_t timeout) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  struct timeval tv;
  fd_set read_fds;
  int fd = ((mbedtls_net_context *)ctx)->fd;
  if (fd < 0) return MBEDTLS_ERR_NET_INVALID_CONTEXT;
  /* A limitation of select() is that it only works with file descriptors
   * that are strictly less than FD_SETSIZE. This is a limitation of the
   * fd_set type. Error out early, because attempting to call FD_SET on a
   * large file descriptor is a buffer overflow on typical platforms. */
  if (fd >= FD_SETSIZE) return (MBEDTLS_ERR_NET_POLL_FAILED);
  FD_ZERO(&read_fds);
  FD_SET(fd, &read_fds);
  tv.tv_sec = timeout / 1000;
  tv.tv_usec = (timeout % 1000) * 1000;
  ret = select(fd + 1, &read_fds, NULL, NULL, timeout == 0 ? NULL : &tv);
  /* Zero fds ready means we timed out */
  if (ret == 0) return MBEDTLS_ERR_SSL_TIMEOUT;
  if (ret < 0) {
    if (errno == EINTR) return MBEDTLS_ERR_SSL_WANT_READ;
    if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
    return MBEDTLS_ERR_NET_RECV_FAILED;
  }
  /* This call will not block */
  return mbedtls_net_recv(ctx, buf, len);
}

/**
 * \brief          Write at most 'len' characters. If no error occurs,
 *                 the actual amount read is returned.
 *
 * \param ctx      Socket
 * \param buf      The buffer to read from
 * \param len      The length of the buffer
 *
 * \return         the number of bytes sent,
 *                 or a non-zero error code; with a non-blocking socket,
 *                 MBEDTLS_ERR_SSL_WANT_WRITE indicates write() would block.
 */
int mbedtls_net_send(void *ctx, const unsigned char *buf, size_t len) {
  int ret = MBEDTLS_ERR_THIS_CORRUPTION;
  int fd = ((mbedtls_net_context *)ctx)->fd;
  if (fd < 0) return MBEDTLS_ERR_NET_INVALID_CONTEXT;
  ret = (int)write(fd, buf, len);
  if (ret < 0) {
    if (net_would_block(ctx) != 0) return MBEDTLS_ERR_SSL_WANT_WRITE;
    if (errno == EPIPE || errno == ECONNRESET)
      return MBEDTLS_ERR_NET_CONN_RESET;
    if (errno == EINTR) return MBEDTLS_ERR_SSL_WANT_WRITE;
    if (errno == ECANCELED) return MBEDTLS_ERR_SSL_CANCELED;
    return MBEDTLS_ERR_NET_SEND_FAILED;
  }
  return ret;
}

/**
 * \brief          Closes down the connection and free associated data
 *
 * \param ctx      The context to close
 */
void mbedtls_net_close(mbedtls_net_context *ctx) {
  if (ctx->fd == -1) return;
  close(ctx->fd);
  ctx->fd = -1;
}

/**
 * \brief          Gracefully shutdown the connection and free associated data
 *
 * \param ctx      The context to free
 */
void mbedtls_net_free(mbedtls_net_context *ctx) {
  if (ctx->fd == -1) return;
  shutdown(ctx->fd, 2);
  close(ctx->fd);
  ctx->fd = -1;
}
