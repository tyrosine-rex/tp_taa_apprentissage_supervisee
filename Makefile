res/tp_final.html : source/tp_final.ipynb
	jupyter nbconvert --ExecutePreprocessor.kernel_name=taa_tp_env --to html --execute source/tp_final.ipynb --output-dir res
