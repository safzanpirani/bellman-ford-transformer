ROOT := ..
BUILD := $(ROOT)/tmp/pdfs
FIGURE := figures/saturation-curves.pdf
TARGET := $(ROOT)/output/pdf/exact-bellman-ford-transformer.pdf

.PHONY: all clean

all: $(TARGET)

$(FIGURE): plot_figures.py $(ROOT)/docs/pilot-v4-curves.json
	$(ROOT)/.venv/bin/python plot_figures.py

$(TARGET): main.tex references.bib $(FIGURE)
	mkdir -p $(BUILD) $(dir $(TARGET))
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=tmp/pdfs paper/main.tex
	cd $(ROOT) && BIBINPUTS=.:paper: bibtex tmp/pdfs/main
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=tmp/pdfs paper/main.tex
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=tmp/pdfs paper/main.tex
	cp $(BUILD)/main.pdf $(TARGET)

clean:
	rm -r -- $(BUILD)
