COURSEWORKS := $(wildcard coursework-*)

.PHONY: all clean $(COURSEWORKS)

all: $(COURSEWORKS)

$(COURSEWORKS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(COURSEWORKS)
	$(MAKE) -C $(basename $@)