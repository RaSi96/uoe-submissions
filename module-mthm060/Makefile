FILE   := 11Sep26-writeup
SOURCE := $(FILE).md
BIB    := $(FILE).bib
PDF    := $(FILE).pdf
HEADER := header.tex
CSL    := ieee.csl
SYNTAX := standard-python.theme
IMAGES := $(wildcard images/*)

PANDOC_OPTS := -f markdown+lists_without_preceding_blankline
PANDOC_OPTS += --pdf-engine=xelatex --wrap=none --syntax-highlighting=$(SYNTAX)
PANDOC_OPTS += --filter pandoc-crossref --number-sections --citeproc
PANDOC_OPTS += --csl=$(CSL) --include-in-header=$(HEADER)

.PHONY: all clean

# default target
all: $(PDF)

$(PDF): $(SOURCE) $(BIB) $(IMAGES) $(CSL)
	pandoc $(SOURCE) $(PANDOC_OPTS) -o $(PDF)

clean:
	rm -f $(PDF)