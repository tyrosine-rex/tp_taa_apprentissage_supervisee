tp.html : source/tp.ipynb
	jupyter nbconvert --ExecutePreprocessor.kernel_name=taa --to html --execute source/tp.ipynb --output-dir res
