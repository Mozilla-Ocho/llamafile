typedef void* CFStringRef;
typedef void* CFDictionaryRef;
typedef void* CFMutableDictionaryRef;
typedef void* CFTypeRef;
typedef void* CFArrayRef;
typedef void* IOReportSubscriptionRef;
typedef void* CVoidRef;
typedef int CFIndex;

bool init_apple_mon();

void am_release(void* obj);

void am_print_object(CFTypeRef obj);

CFMutableDictionaryRef am_get_power_channels();

IOReportSubscriptionRef am_get_subscription(CFMutableDictionaryRef power_channel);

CFDictionaryRef am_sample_power(IOReportSubscriptionRef sub, CFMutableDictionaryRef channels);

double am_sample_to_millijoules(CFDictionaryRef sample);