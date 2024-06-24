#include <random>
#include <unistd.h>

// TODO(jart): delete this with next cosmocc update

namespace std {

random_device::random_device(const string& __token)
{
    if (__token != "/dev/urandom")
        __throw_system_error(ENOENT, ("random device not supported " + __token).c_str());
}

random_device::~random_device()
{
}

unsigned
random_device::operator()()
{
    unsigned r;
    size_t n = sizeof(r);
    int err = getentropy(&r, n);
    if (err)
        __throw_system_error(errno, "random_device getentropy failed");
    return r;
}

double
random_device::entropy() const noexcept
{
    return 0;
}

} // namespace std
