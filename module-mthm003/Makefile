COURSEWORKS := $(wildcard coursework-*)

.PHONY: all clean $(COURSEWORKS)

all: $(COURSEWORKS)

$(COURSEWORKS):
	@if [ -f $@/Makefile ]; then \
		$(MAKE) -C $@ $(MAKECMDGOALS); \
	else \
		echo "Skipping $@ (no Makefile)"; \
	fi

clean: $(COURSEWORKS)