#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

LINK.o = $(CXX) $(CCFLAGS) $(LDFLAGS)
COMPILE.c = $(CC) $(CCFLAGS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.cc = $(CXX) $(CCFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

o/$(MODE)/%.a:
	$(AR) $(ARFLAGS) $@ $^

o/$(MODE)/%.o: %.c
	@mkdir -p $(@D)
	$(COMPILE.c) -o $@ $<

o/$(MODE)/%.o: %.cc
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%: o/$(MODE)/%.o
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

o/$(MODE)/%.zip.o: %
	@mkdir -p $(dir $@)/.aarch64
	zipobj $(ZIPOBJ_FLAGS) -a x86_64 -o $@ $<
	zipobj $(ZIPOBJ_FLAGS) -a aarch64 -o $(dir $@)/.aarch64/$(notdir $@) $<
