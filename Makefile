tp.html : source/tp.ipynb
	jupyter nbconvert --to html --execute source/tp.ipynb --output-dir res