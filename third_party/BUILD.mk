include third_party/double-conversion/BUILD.mk
include third_party/mbedtls/BUILD.mk
include third_party/quickjs/BUILD.mk
include third_party/sqlite/BUILD.mk
include third_party/stb/BUILD.mk

.PHONY: o/$(MODE)/third_party
o/$(MODE)/third_party: 						\
		o/$(MODE)/third_party/double-conversion		\
		o/$(MODE)/third_party/mbedtls			\
		o/$(MODE)/third_party/sqlite			\
		o/$(MODE)/third_party/stb			\
