# Detailed proofs

[Read the PDF](SCOT_proofs.pdf) or inspect the
[original proof source](11_appendix_proofs.tex).

The document contains the full proofs of closure preservation, COT
construction, and exact single-step query answering, followed by the
auxiliary size and cost bounds. It is distributed separately from the
submission manuscript.

## Files

- `11_appendix_proofs.tex`: unchanged proof section from the manuscript.
- `SCOT_proofs.tex`: standalone compilation entry point and notation macros.
- `paper_references.tex`: numbering snapshot for the referenced manuscript
  definitions, statements, algorithms, sections, and equations (2026-11-12).
- `SCOT_proofs.pdf`: compiled proof document.

All references refer to the main manuscript's numbering; the manuscript
itself is not required to compile these files.

## Build

Install a LaTeX distribution with IEEEtran, amsmath, amssymb, amsthm,
microtype, and latexmk. From the repository root, run:

```bash
cd proofs
latexmk -pdf -interaction=nonstopmode -halt-on-error SCOT_proofs.tex
```

Alternatively, run `pdflatex SCOT_proofs.tex` twice in this directory.
The output is `SCOT_proofs.pdf`. The proof source is an input section;
compile `SCOT_proofs.tex` as the document entry point.

## Source integrity

SHA-256 of `11_appendix_proofs.tex`:

```text
367c402d1e0c72ade7aec8a91216dbe1f5c2b842b62e70b329d74442658a2b00
```
