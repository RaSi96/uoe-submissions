SUBJECTS := $(wildcard module-*)

all: $(SUBJECTS)

$(SUBJECTS):
	$(MAKE) -C $@