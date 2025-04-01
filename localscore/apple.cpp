#include <cosmo.h>
#include <dlfcn.h>

#include "apple.h"
#include "llama.cpp/common.h"

static void *imp(void *lib, const char *sym) {
    void *fun = cosmo_dlsym(lib, sym);
    if (!fun)
        tinylog(__func__, ": error: failed to import symbol: ", sym, "\n", NULL);
    return fun;
}

static struct IOReport {
    CFDictionaryRef (*IOReportCopyChannelsInGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
    IOReportSubscriptionRef (*IOReportCreateSubscription)(CVoidRef, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef);
    CFDictionaryRef (*IOReportCreateSamples)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
    CFDictionaryRef (*IOReportCreateSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
    CFStringRef (*IOReportChannelGetChannelName)(CFDictionaryRef);
    int64_t (*IOReportSimpleGetIntegerValue)(CFDictionaryRef, int32_t);
    CFStringRef (*IOReportChannelGetUnitLabel)(CFDictionaryRef);
} io_report;

static struct CoreFoundation {
    CFMutableDictionaryRef (*CFDictionaryCreateMutableCopy)(CVoidRef, long, CFDictionaryRef);
    long (*CFDictionaryGetCount)(CFDictionaryRef);
    void (*CFShow)(CFTypeRef);
    CVoidRef (*CFDictionaryGetValue)(CFDictionaryRef, CVoidRef);
    CFStringRef (*CFStringCreateWithCString)(CVoidRef, const char*, int);
    void (*CFRelease)(CFTypeRef); 
    int (*CFArrayGetCount)(CFArrayRef);
    CFTypeRef (*CFArrayGetValueAtIndex)(CFArrayRef, int);
    CFArrayRef (*CFArrayCreateCopy)(CVoidRef, CFArrayRef);
    bool (*CFStringGetCString)(CFStringRef, char *, int, int);
} core_foundation;

bool init_apple_mon() {
    void *lib = cosmo_dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY);
    if (!lib) {
        tinylog(__func__, ": error: failed to open IOKit framework\n", NULL);
        return false;
    }

    bool ok = true;

    ok &= !!(io_report.IOReportCopyChannelsInGroup = (CFDictionaryRef (*)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t))imp(lib, "IOReportCopyChannelsInGroup"));
    ok &= !!(io_report.IOReportCreateSubscription = (IOReportSubscriptionRef (*)(CVoidRef, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t, CFTypeRef))imp(lib, "IOReportCreateSubscription"));
    ok &= !!(io_report.IOReportCreateSamples = (CFDictionaryRef (*)(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef))imp(lib, "IOReportCreateSamples"));
    ok &= !!(io_report.IOReportCreateSamplesDelta = (CFDictionaryRef (*)(CFDictionaryRef, CFDictionaryRef, CFTypeRef))imp(lib, "IOReportCreateSamplesDelta"));
    ok &= !!(io_report.IOReportChannelGetChannelName = (CFStringRef (*)(CFDictionaryRef))imp(lib, "IOReportChannelGetChannelName"));
    ok &= !!(io_report.IOReportSimpleGetIntegerValue = (int64_t (*)(CFDictionaryRef, int32_t))imp(lib, "IOReportSimpleGetIntegerValue"));
    ok &= !!(io_report.IOReportChannelGetUnitLabel = (CFStringRef (*)(CFDictionaryRef))imp(lib, "IOReportChannelGetUnitLabel"));

    if (!ok) {
        tinylog(__func__, ": error: not all IOReport symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    ok &= !!(core_foundation.CFDictionaryCreateMutableCopy = (CFMutableDictionaryRef (*)(void*, long, CFDictionaryRef))imp(lib, "CFDictionaryCreateMutableCopy"));
    ok &= !!(core_foundation.CFDictionaryGetCount = (long (*)(CFDictionaryRef))imp(lib, "CFDictionaryGetCount"));
    ok &= !!(core_foundation.CFShow = (void (*)(CFTypeRef))imp(lib, "CFShow"));
    ok &= !!(core_foundation.CFDictionaryGetValue = ( void* (*)(CFDictionaryRef,  void*))imp(lib, "CFDictionaryGetValue"));
    ok &= !!(core_foundation.CFStringCreateWithCString = (CFStringRef (*)(CVoidRef, const char*, int))imp(lib, "CFStringCreateWithCString"));
    ok &= !!(core_foundation.CFRelease = (void (*)(CFTypeRef))imp(lib, "CFRelease"));
    ok &= !!(core_foundation.CFArrayGetCount = (int (*)(CFArrayRef))imp(lib, "CFArrayGetCount"));
    ok &= !!(core_foundation.CFArrayGetValueAtIndex = (CFTypeRef (*)(CFArrayRef, int))imp(lib, "CFArrayGetValueAtIndex"));
    ok &= !!(core_foundation.CFArrayCreateCopy = (CFArrayRef (*)(CVoidRef, CFArrayRef))imp(lib, "CFArrayCreateCopy"));
    ok &= !!(core_foundation.CFStringGetCString = (bool (*)(CFStringRef, char *, int, int))imp(lib, "CFStringGetCString"));

    if (!ok) {
        tinylog(__func__, ": error: not all CoreFoundation symbols could be imported\n", NULL);
        cosmo_dlclose(lib);
        return false;
    }

    return true;
}

static bool get_cstring_from_cfstring(CFStringRef cfString, char* buffer, size_t bufferSize) {
    return core_foundation.CFStringGetCString(cfString, buffer, bufferSize, 0x08000100);
}

static char* get_unit_label(CFDictionaryRef item) {
    static char unit[64];
    CFStringRef u = io_report.IOReportChannelGetUnitLabel(item);
    if (u) {
        if (!get_cstring_from_cfstring(u, unit, sizeof(unit))) {
            strcpy(unit, "Unknown");
        }
        core_foundation.CFRelease(u);
    } else {
        strcpy(unit, "N/A");
    }
    return unit;
}

static double get_item_energy_millijoules(CFDictionaryRef item, const char* name) {
    char* unit = get_unit_label(item);
    double energy = (double)io_report.IOReportSimpleGetIntegerValue(item, 0);
    double energy_millijoules = 0;

    if (strcmp(name, "CPU Energy") == 0 || strcmp(name, "GPU Energy") == 0 || strstr(name, "ANE") != NULL) {
        if (strcmp(unit, "mJ") == 0) {
            energy_millijoules = energy;
        } else if (strcmp(unit, "uJ") == 0) {
            energy_millijoules = energy / 1e3;
        } else if (strcmp(unit, "nJ") == 0) {
            energy_millijoules = energy / 1e6;
        } else {
            printf("Unknown unit: %s for channel: %s\n", unit, name);
            return 0;
        }
    }

    return energy_millijoules;
}


void am_release(void* obj) {
    core_foundation.CFRelease(obj);
}

void am_print_object(CFTypeRef obj) {
    core_foundation.CFShow(obj);
}

CFMutableDictionaryRef am_get_power_channels() {
    CFStringRef energy_str = core_foundation.CFStringCreateWithCString(NULL, "Energy Model", 0x08000100);
    CFDictionaryRef channels = io_report.IOReportCopyChannelsInGroup(energy_str, NULL, 0, 0, 0);
    core_foundation.CFRelease(energy_str);

    CFMutableDictionaryRef channels_mut = core_foundation.CFDictionaryCreateMutableCopy(NULL, core_foundation.CFDictionaryGetCount(channels), channels);
    core_foundation.CFRelease(channels);

    return channels_mut;
}

IOReportSubscriptionRef am_get_subscription(CFMutableDictionaryRef channels_mut) {
    CFMutableDictionaryRef subscription;
    IOReportSubscriptionRef s = io_report.IOReportCreateSubscription(NULL, channels_mut, &subscription, 0, NULL);
    return s;
}

// TODO need some way of freeing the CFDictionaryRef?
CFDictionaryRef am_sample_power(IOReportSubscriptionRef sub, CFMutableDictionaryRef channels) {
    return io_report.IOReportCreateSamples(sub, channels, NULL);
}

double am_sample_to_millijoules(CFDictionaryRef sample) {
    CFStringRef key = core_foundation.CFStringCreateWithCString(NULL, "IOReportChannels", 0x08000100);
    CFArrayRef report = core_foundation.CFDictionaryGetValue(sample, key);
    core_foundation.CFRelease(key);

    CFIndex count = core_foundation.CFArrayGetCount(report);
    double total_energy_millijoules = 0;

    for (CFIndex i = 0; i < count; i++) {
        CFDictionaryRef item = core_foundation.CFArrayGetValueAtIndex(report, i);
        CFStringRef n = io_report.IOReportChannelGetChannelName(item);
        char name[64] = {0};
        
        if (!core_foundation.CFStringGetCString(n, name, sizeof(name), 0x08000100)) {
            printf("Failed to get channel name\n");
            core_foundation.CFRelease(n);
            continue;
        }

        total_energy_millijoules += get_item_energy_millijoules(item, name);
        core_foundation.CFRelease(n);
    }

    return total_energy_millijoules;
}