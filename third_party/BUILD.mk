include third_party/double-conversion/BUILD.mk
include third_party/stb/BUILD.mk

.PHONY: o/$(MODE)/third_party
o/$(MODE)/third_party: 						\
		o/$(MODE)/third_party/double-conversion		\
		o/$(MODE)/third_party/stb			\
