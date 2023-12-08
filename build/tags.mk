#-*-mode:makefile-gmake;indent-tabs-mode:t;tab-width:8;coding:utf-8-*-┐
#── vi: set noet ft=make ts=8 sw=8 fenc=utf-8 :vi ────────────────────┘

TAGSFLAGS =								\
	-e								\
	-a								\
	--if0=no							\
	--langmap=c:.c.h.i						\
	--line-directives=yes

tags: TAGS HTAGS

TAGS: o/$(MODE)/tags-srcs.txt $(SRCS)
	@rm -f $@
	ctags $(TAGSFLAGS) -L $< -o $@

HTAGS: o/$(MODE)/tags-hdrs.txt $(HDRS) $(INCS)
	@rm -f $@
	build/htags ctags -L $< -o $@

o/$(MODE)/tags-srcs.txt: $(call uniq,$(foreach x,$(SRCS),$(dir $(x))))
	@mkdir -p $(@D)
	$(file >$@) $(foreach x,$(SRCS),$(file >>$@,$(x)))

o/$(MODE)/tags-hdrs.txt: $(call uniq,$(foreach x,$(HDRS) $(INCS),$(dir $(x))))
	@mkdir -p $(@D)
	$(file >$@) $(foreach x,$(HDRS) $(INCS),$(file >>$@,$(x)))
