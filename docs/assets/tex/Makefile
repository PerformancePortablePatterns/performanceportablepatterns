FIGURES := pattern_relation software_relation generator_processor_scan
SVGS := $(FIGURES:%=%.svg)

.PHONY: all
all: $(SVGS)

%.svg: %.tex
	latexmk -lualatex -interaction=nonstopmode -output-directory=intermediate $<
	pdf2svg intermediate/$*.pdf svg/$@
	rm intermediate/*
