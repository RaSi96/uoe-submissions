FILE   := 31Oct25-submission-CW1
SOURCE := $(FILE).md
BIB    := $(FILE).bib
PDF    := $(FILE).pdf
HEADER := header.tex
CSL    := apa.csl
SYNTAX := standard-python.theme
IMAGES := $(wildcard images/*)

PANDOC_OPTS := -f markdown+lists_without_preceding_blankline
PANDOC_OPTS += --pdf-engine=xelatex --wrap=none --citeproc
PANDOC_OPTS += --csl=$(CSL) --syntax-highlighting=$(SYNTAX) --include-in-header=$(HEADER)

.PHONY: all clean

# default target
all: $(PDF)

$(PDF): $(SOURCE) $(BIB) $(IMAGES) $(CSL)
	pandoc $(SOURCE) $(PANDOC_OPTS) -o $(PDF)

clean:
	rm -f $(PDF)