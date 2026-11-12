# Complete SCOT proofs

[Read the PDF](SCOT_proofs.pdf) or [open the standalone LaTeX source](11_appendix_proofs.tex).

The source is a complete document. It defines its notation and semantics,
describes the procedures used in the proofs, states each formal result, and
resolves every internal reference locally. The submission manuscript is not
needed to read or compile it.

## Build

Install a LaTeX distribution with IEEEtran, amsmath, amssymb, amsthm,
microtype, balance, hyperref, and latexmk. From the `proofs` directory, run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error 11_appendix_proofs.tex
```

The output is `11_appendix_proofs.pdf`. `SCOT_proofs.tex` remains as a
compatibility entry point and includes the same document. The tracked
`SCOT_proofs.pdf` is compiled from the standalone source.

`paper_references.tex` is a legacy numbering snapshot from the previous
wrapper and is no longer used by either build entry point.
