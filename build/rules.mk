#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

o/$(MODE)/%.a:
	$(AR) $(ARFLAGS) $@ $^

o/$(MODE)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

o/$(MODE)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CCFLAGS) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

o/$(MODE)/%: o/$(MODE)/%.o
	$(CXX) $(CCFLAGS) $(LDFLAGS) $(TARGET_ARCH) $^ $(LOADLIBES) $(LDLIBS) -o $@
