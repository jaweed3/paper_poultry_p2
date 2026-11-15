# Modify DOC variable then
# modify head to latex, latex_full, pdflatex, pdflatex_full
# which ever you prefer.
# "make pdf" command will convert dvi file to pdf file.

DOC=sample
#head: pdflatex
head: latex_full



pdflatex:
	pdflatex $(DOC)

pdflatex_full:
	make pdflatex
	make bib
	make pdflatex
	make pdflatex



latex:
	latex $(DOC)

latex_full:
	make latex
	make bib
	make latex
	make latex

pdf:
	dvipdfmx $(DOC).dvi

bib:
	bibtex $(DOC)



clean:
	rm -f *.aux *.blg *.log

distclean: clean
	rm -f $(DOC).dvi $(DOC).pdf
