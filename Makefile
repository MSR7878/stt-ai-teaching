.PHONY: all clean help pdf list light dark both marp

# Find all .typ files (excluding slides.typ and other partials)
SLIDES := $(filter-out slides.typ, $(wildcard *.typ))
PDF_LIGHT := $(SLIDES:.typ=-light.pdf)
PDF_DARK := $(SLIDES:.typ=-dark.pdf)

# Find all .md files for Marp
MARP_SLIDES := $(wildcard *.md)
MARP_PDF := $(MARP_SLIDES:.md=-marp.pdf)
MARP_HTML := $(MARP_SLIDES:.md=-marp.html)

# Default target - build Marp slides
all: marp
	@echo "✓ All slides built successfully"

# Build Marp slides (PDF and HTML)
marp: $(MARP_PDF) $(MARP_HTML)
	@echo "✓ Marp slides built (PDF and HTML)"

# Build light (regular) PDFs
light: $(PDF_LIGHT)
	@echo "✓ Light theme PDFs built"

# Build dark PDFs (inverted colors from light PDFs)
dark: light $(PDF_DARK)
	@echo "✓ Dark theme PDFs created"

# Build both versions
both: light dark
	@echo "✓ Both light and dark PDFs built"

# Pattern rule for light PDF
%-light.pdf: %.typ slides.typ
	@echo "Compiling $< -> $@"
	@typst compile $< $@

# Pattern rule for dark PDF (color inversion with ghostscript)
%-dark.pdf: %-light.pdf
	@echo "Creating dark version: $< -> $@"
	@if command -v gs >/dev/null 2>&1; then \
		gs -o $@ -sDEVICE=pdfwrite \
		   -c "{1 exch sub}{1 exch sub}{1 exch sub}{1 exch sub} setcolortransfer" \
		   -f $< 2>&1 | grep -v "GPL Ghostscript" | grep -v "Copyright" | grep -v "Processing pages" | grep -v "Page [0-9]" | grep -v "reserved for an Annotation" || true; \
	else \
		echo "Warning: ghostscript (gs) not found. Copying original."; \
		cp $< $@; \
	fi

# Pattern rule for Marp PDF
%-marp.pdf: %.md
	@echo "Building Marp PDF: $< -> $@"
	@marp $< -o $@ --pdf --allow-local-files

# Pattern rule for Marp HTML
%-marp.html: %.md
	@echo "Building Marp HTML: $< -> $@"
	@marp $< -o $@ --html

# List available slides
list:
	@echo "Available slides:"
	@for file in $(SLIDES); do \
		echo "  - $${file%.typ}"; \
	done
	@echo ""
	@echo "Build commands:"
	@echo "  make light     # Build light theme PDFs (default)"
	@echo "  make dark      # Build dark theme PDFs (inverted colors)"
	@echo "  make both      # Build both light and dark PDFs"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -f $(PDF_LIGHT) $(PDF_DARK) $(MARP_PDF) $(MARP_HTML) *.pdf *.html
	@echo "✓ Clean complete"

help:
	@echo "Available targets:"
	@echo "  make marp      # Build Marp slides (PDF and HTML) - default"
	@echo "  make light     # Build Typst light theme PDFs"
	@echo "  make dark      # Build Typst dark theme PDFs"
	@echo "  make both      # Build both Typst light and dark PDFs"
	@echo "  make list      # List available slides"
	@echo "  make clean     # Remove all generated files"
