#include <string>
#include <stdio.h>
#include <assert.h>
#include <dlfcn.h>
#include "macsandbox.h"

// Platform sandboxing dylib
#define MACOS_SANDBOXING_DYLIB "/usr/lib/system/libsystem_sandbox.dylib"

// Paths used for self-tests
#define MACOS_SANDBOXING_TEST_WRITE_PATH "/private/tmp/foo"
#define MACOS_SANDBOXING_TEST_READ_PATH "/private/etc/ntp.conf" // a benign file
#define MACOS_SANDBOXING_TEST_EXEC_PATH "/bin/date"

// dlsym'd functions
typedef int (*sandbox_init_with_parameters_t)(const char*, uint64_t,
                                              const char* const[], char**);
typedef int (*sandbox_check_t)(pid_t pid, const char* operation, int type, ...);

//
// A deny-by-default policy. Denies connecting to services, exec, and
// file I/O. Applies to operations/connections not already established. This is
// a restrictive policy that depends on the application already opening files
// and establishing connections to services it needs. As new features are added,
// the policy may need to be changed to allow more functionality (if for example
// a new file must be opened after the sandbox is enabled.) Logging of sandbox
// exceptions is enabled via Console.app.
//
static const char policy[] = R"SANDBOX_LITERAL(
  (version 1)
  (deny default)
  (if (defined? 'process-info*)
    (deny process-info*))
  (if (defined? 'nvram*)
    (deny nvram*))
  (if (defined? 'iokit-get-properties)
    (deny iokit-get-properties))
)SANDBOX_LITERAL";

void get_dlerror(std::string& out)
{
    char* errorp = cosmo_dlerror();
    if (errorp) {
        out = errorp;
    }
}

//
// Load the sandboxing platform dylib and turn on sandboxing for this process
// using the above `policy`.
//
// Returns non-zero on failure and populates `error_str` with an error message
// when one is available.
//
int mac_sandbox_init(std::string& error_str)
{
    // dlopen the platform dylib for sandboxing
    void* sblib = cosmo_dlopen(MACOS_SANDBOXING_DYLIB, RTLD_LOCAL|RTLD_NOW);
    if (!sblib) {
        get_dlerror(error_str);
        return 1;
    }

    // Get the function that applies a sandbox policy to the current process
    int (*sandbox_init_with_parametersp)(const char*, uint64_t,
        const char* const[], char**);
    sandbox_init_with_parametersp = (sandbox_init_with_parameters_t)
        cosmo_dlsym(sblib, "sandbox_init_with_parameters");    
    if (!sandbox_init_with_parametersp) {
        get_dlerror(error_str);
        return 1;
    }

    // Get the function that checks a policy is applied for a given PID
    int (*sandbox_checkp)(pid_t pid, const char* operation, int type, ...);
    sandbox_checkp = (sandbox_check_t)cosmo_dlsym(sblib, "sandbox_check");    
    if (!sandbox_checkp) {
        get_dlerror(error_str);
        return 1;
    }

    // Apply the policy
    char* errorbufp = NULL;
    int rv = sandbox_init_with_parametersp(policy, 0, NULL, &errorbufp);
    if (rv != 0) {
        if (errorbufp) {
            error_str = errorbufp;
        }
        return rv;
    }

    // Test that a policy is applied
    assert(sandbox_checkp(getpid(), NULL, 0) == 1);

    // Basic tests to ensure the policy works    
    assert(fopen(MACOS_SANDBOXING_TEST_WRITE_PATH, "w") == NULL);
    assert(fopen(MACOS_SANDBOXING_TEST_READ_PATH, "r") == NULL);
    assert(system(MACOS_SANDBOXING_TEST_EXEC_PATH) != 0);
    
    return 0;
}
