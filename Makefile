# clean all coursework under a subject
# make -C module-mthm003-analysis-computation clean
#
# clean a single coursework
# make -C module-mthm003-analysis-computation/coursework-1 clean
#
# build a single coursework under a subject
# make -C module-mthm003-analysis-computation
#
# build a single coursework
# make -C module-mthm003-analysis-computation/coursework-1
#
# build everything
# make


SUBJECTS := $(wildcard module-*)

.PHONY: all clean $(SUBJECTS)

all: $(SUBJECTS)

$(SUBJECTS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(SUBJECTS)