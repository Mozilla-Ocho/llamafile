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
