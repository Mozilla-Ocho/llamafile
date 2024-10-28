#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

LINK.o = $(CXX) $(CCFLAGS) $(LDFLAGS)
COMPILE.c = $(CC) $(CCFLAGS) $(CFLAGS) $(CPPFLAGS_) $(CPPFLAGS) $(TARGET_ARCH) -c
COMPILE.cc = $(CXX) $(CCFLAGS) $(CXXFLAGS) $(CPPFLAGS_) $(CPPFLAGS) $(TARGET_ARCH) -c

o/$(MODE)/%.o: %.c $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) -o $@ $<

o/$(MODE)/%.o: o/$(MODE)/%.c $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.c) -o $@ $<

o/$(MODE)/%.o: %.cc $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%.o: %.cc $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%.o: %.cpp $(COSMOCC)
	@mkdir -p $(@D)
	$(COMPILE.cc) -o $@ $<

o/$(MODE)/%.c: %.gperf
	@mkdir -p $(@D)
	build/gperf --output-file=$@ $<

o/$(MODE)/%.a:
	@mkdir -p $(dir $@)/.aarch64
	$(AR) $(ARFLAGS) $@ $^
	$(AR) $(ARFLAGS) $(dir $@)/.aarch64/$(notdir $@) $(foreach x,$^,$(dir $(x)).aarch64/$(notdir $(x)))

o/$(MODE)/%: o/$(MODE)/%.o
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

o/$(MODE)/%.com: o/$(MODE)/%.o
	$(LINK.o) $^ $(LOADLIBES) $(LDLIBS) -o $@

%.runs: %
	$<
	@touch $@

.PRECIOUS: %.1.asc
%.1.asc: %.1
	-MANWIDTH=80 MAN_KEEP_FORMATTING=1 man $< >$@.tmp && mv -f $@.tmp $@
	@rm -f $@.tmp

o/$(MODE)/%.zip.o: % $(COSMOCC)
	@mkdir -p $(dir $@)/.aarch64
	$(ZIPOBJ) $(ZIPOBJ_FLAGS) -a x86_64 -o $@ $<
	$(ZIPOBJ) $(ZIPOBJ_FLAGS) -a aarch64 -o $(dir $@)/.aarch64/$(notdir $@) $<

$(PREFIX)/bin/ape: $(COSMOCC) # cosmocc toolchain setup in restricted ci context 
	$(INSTALL) $(COSMOCC)/bin/ape-$(ARCH).elf $(PREFIX)/bin/ape
	echo ':APE:M::MZqFpD::/usr/bin/ape:' > /proc/sys/fs/binfmt_misc/register
