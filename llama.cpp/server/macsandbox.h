#ifndef INCLUDE_MACSANDBOX_H
#define INCLUDE_MACSANDBOX_H

#include <string>

//
// Turn on sandboxing for this process.
//
// Returns non-zero on failure and populates `error_str` with
// an error message when one is available.
//
int mac_sandbox_init(std::string& error_str);

#endif // INCLUDE_MACSANDBOX_H
