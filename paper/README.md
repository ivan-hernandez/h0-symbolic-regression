# arXiv submission

## Files to upload
- `paper.tex`
- `aastex631.cls`
- `h0_summary.png`

## Compilation (macOS)
```bash
# If you don't have MacTeX:
# Download from https://tug.org/mactex/

# Compile twice for cross-references
pdflatex paper.tex
pdflatex paper.tex
```

## Compilation (Linux)
```bash
sudo apt-get install texlive-publishers texlive-latex-extra
pdflatex paper.tex
pdflatex paper.tex
```

## arXiv submission
1. Go to https://arxiv.org/submit
2. Upload paper.tex, aastex631.cls, and h0_summary.png
3. Select "Astrophysics of Galaxies" (astro-ph.GA) or
   "Cosmology and Nongalactic Astrophysics" (astro-ph.CO)
4. Process and preview before submitting
