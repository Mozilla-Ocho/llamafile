#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

SRCS = $(foreach x,$(PKGS),$($(x)_SRCS))
HDRS = $(foreach x,$(PKGS),$($(x)_HDRS))
INCS = $(foreach x,$(PKGS),$($(x)_INCS))

o/$(MODE)/depend: $(SRCS) $(HDRS) $(INCS)
	@mkdir -p $(@D)
	$(MKDEPS) -o $@ -r o/$(MODE)/ $(SRCS) $(HDRS) $(INCS)

o/$(MODE)/depend.test: $(SRCS) $(HDRS) $(INCS)
	@mkdir -p $(@D)
	$(MKDEPS) -o $@ -r o/$(MODE)/ $(SRCS) $(HDRS) $(INCS)

$(SRCS):
$(HDRS):
$(INCS):
.DEFAULT:
	@echo
	@echo NOTE: deleting o/$(MODE)/depend because of an unspecified prerequisite: $@
	@echo
	rm -f o/$(MODE)/depend

-include o/$(MODE)/depend
