ROOT := ..
BUILD := $(ROOT)/tmp/pdfs
FIGURE := figures/saturation-curves.pdf
OUTDIR := $(ROOT)/output/pdf
TARGET := $(OUTDIR)/exact-bellman-ford-transformer.pdf
TARGET_LM := $(OUTDIR)/exact-bellman-ford-transformer-lmodern.pdf

STYLES := style-times.tex style-lmodern.tex titleblock-times.tex titleblock-lmodern.tex

.PHONY: all times lmodern both clean

all: $(TARGET)
times: $(TARGET)
lmodern: $(TARGET_LM)
both: $(TARGET) $(TARGET_LM)

$(FIGURE): plot_figures.py $(ROOT)/docs/pilot-v4-curves.json
	$(ROOT)/.venv/bin/python plot_figures.py

# $(1) = variant name, $(2) = output pdf.  Each variant builds under its own
# jobname so the two runs never clobber each other's .aux/.bbl.
define BUILD_VARIANT
	mkdir -p $(BUILD) $(OUTDIR)
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
	  -output-directory=tmp/pdfs -jobname=main-$(1) \
	  "\def\paperstyle{$(1)}\input{paper/main.tex}"
	cd $(ROOT) && BIBINPUTS=.:paper: bibtex tmp/pdfs/main-$(1)
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
	  -output-directory=tmp/pdfs -jobname=main-$(1) \
	  "\def\paperstyle{$(1)}\input{paper/main.tex}"
	cd $(ROOT) && pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
	  -output-directory=tmp/pdfs -jobname=main-$(1) \
	  "\def\paperstyle{$(1)}\input{paper/main.tex}"
	cp $(BUILD)/main-$(1).pdf $(2)
endef

$(TARGET): main.tex references.bib $(FIGURE) $(STYLES)
	$(call BUILD_VARIANT,times,$(TARGET))

$(TARGET_LM): main.tex references.bib $(FIGURE) $(STYLES)
	$(call BUILD_VARIANT,lmodern,$(TARGET_LM))

clean:
	rm -r -- $(BUILD)
