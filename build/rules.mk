#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

LINK.o = $(CXX) $(CCFLAGS) $(LDFLAGS)
COMPILE.c = $(CC) $(CCFLAGS) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.cc = $(CXX) $(CCFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c

o/$(MODE)/%.a:
	$(AR) $(ARFLAGS) $@ $^

o/$(MODE)/%.o: %.c $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) -o $@ $<

o/$(MODE)/%.o: %.cc $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%.o: %.cpp $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%: o/$(MODE)/%.o
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

o/$(MODE)/%.zip.o: % $(COSMOCC)
	@mkdir -p $(dir $@)/.aarch64
	$(ZIPOBJ) $(ZIPOBJ_FLAGS) -a x86_64 -o $@ $<
	$(ZIPOBJ) $(ZIPOBJ_FLAGS) -a aarch64 -o $(dir $@)/.aarch64/$(notdir $@) $<
