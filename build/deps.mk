#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#───vi: set et ft=make ts=8 tw=8 fenc=utf-8 :vi───────────────────────┘

SRCS = $(foreach x,$(PKGS),$($(x)_SRCS))
HDRS = $(foreach x,$(PKGS),$($(x)_HDRS))

o/$(MODE)/depend: $(SRCS) $(HDRS)
	$(SHELL) -c 'build/bootstrap/mkdeps.com -o $@ -r o/$(MODE)/ $(SRCS) $(HDRS)'

$(SRCS):
$(HDRS):
.DEFAULT:
	@echo
	@echo NOTE: deleting o/$(MODE)/depend because of an unspecified prerequisite: $@
	@echo
	rm -f o/$(MODE)/depend

-include o//depend
