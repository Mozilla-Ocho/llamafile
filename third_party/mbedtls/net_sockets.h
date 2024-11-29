#ifndef COSMOPOLITAN_THIRD_PARTY_MBEDTLS_NET_SOCKETS_H_
#define COSMOPOLITAN_THIRD_PARTY_MBEDTLS_NET_SOCKETS_H_
COSMOPOLITAN_C_START_

#define MBEDTLS_ERR_NET_SOCKET_FAILED                     -0x0042  /*< Failed to open a socket. */
#define MBEDTLS_ERR_NET_CONNECT_FAILED                    -0x0044  /*< The connection to the given server / port failed. */
#define MBEDTLS_ERR_NET_BIND_FAILED                       -0x0046  /*< Binding of the socket failed. */
#define MBEDTLS_ERR_NET_LISTEN_FAILED                     -0x0048  /*< Could not listen on the socket. */
#define MBEDTLS_ERR_NET_ACCEPT_FAILED                     -0x004A  /*< Could not accept the incoming connection. */
#define MBEDTLS_ERR_NET_RECV_FAILED                       -0x004C  /*< Reading information from the socket failed. */
#define MBEDTLS_ERR_NET_SEND_FAILED                       -0x004E  /*< Sending information through the socket failed. */
#define MBEDTLS_ERR_NET_CONN_RESET                        -0x0050  /*< Connection was reset by peer. */
#define MBEDTLS_ERR_NET_UNKNOWN_HOST                      -0x0052  /*< Failed to get an IP address for the given hostname. */
#define MBEDTLS_ERR_NET_BUFFER_TOO_SMALL                  -0x0043  /*< Buffer is too small to hold the data. */
#define MBEDTLS_ERR_NET_INVALID_CONTEXT                   -0x0045  /*< The context is invalid, eg because it was free()ed. */
#define MBEDTLS_ERR_NET_POLL_FAILED                       -0x0047  /*< Polling the net context failed. */
#define MBEDTLS_ERR_NET_BAD_INPUT_DATA                    -0x0049  /*< Input invalid. */

#define MBEDTLS_NET_LISTEN_BACKLOG         10                      /*< The backlog that listen() should use. */

#define MBEDTLS_NET_PROTO_TCP 0                                    /*< The TCP transport protocol */
#define MBEDTLS_NET_PROTO_UDP 1                                    /*< The UDP transport protocol */

#define MBEDTLS_NET_POLL_READ  1                                   /*< Used in \c mbedtls_net_poll to check for pending data  */
#define MBEDTLS_NET_POLL_WRITE 2                                   /*< Used in \c mbedtls_net_poll to check if write possible */

/**
 * Wrapper type for sockets.
 *
 * Currently backed by just a file descriptor, but might be more in the future
 * (eg two file descriptors for combined IPv4 + IPv6 support, or additional
 * structures for hand-made UDP demultiplexing).
 */
typedef struct mbedtls_net_context
{
    int fd;             /*< The underlying file descriptor                 */
}
mbedtls_net_context;

int mbedtls_net_accept( mbedtls_net_context *, mbedtls_net_context *, void *, size_t, size_t * );
int mbedtls_net_bind( mbedtls_net_context *, const char *, const char *, int );
int mbedtls_net_connect( mbedtls_net_context *, const char *, const char *, int );
int mbedtls_net_poll( mbedtls_net_context *, uint32_t, uint32_t );
int mbedtls_net_recv( void *, unsigned char *, size_t );
int mbedtls_net_recv_timeout( void *, unsigned char *, size_t, uint32_t );
int mbedtls_net_send( void *, const unsigned char *, size_t );
int mbedtls_net_set_block( mbedtls_net_context * );
int mbedtls_net_set_nonblock( mbedtls_net_context * );
void mbedtls_net_close( mbedtls_net_context * );
void mbedtls_net_free( mbedtls_net_context * );
void mbedtls_net_init( mbedtls_net_context * );
void mbedtls_net_usleep( unsigned long );

COSMOPOLITAN_C_END_
#endif /* COSMOPOLITAN_THIRD_PARTY_MBEDTLS_NET_SOCKETS_H_ */
