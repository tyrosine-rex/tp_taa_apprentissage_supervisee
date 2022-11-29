# tp_taa_apprentissage_supervisee
tp universitaire, pratique de technique d'apprentissage automatique, précisément des techniques supervisées avec le module scikit-learn

# setup env 

First, you need to have a conda distribution (anaconda3, miniforge3, mamba, ...).

Next, create a new environment from requirement file. This new env is called 'taa_tp_env'.

```bash
conda env create -f ./env/requirement.yml
```

Then, activate 'taa_tp_env' and make it available to Jupyter with 'ipykernel' module.

```bash
conda activate taa_tp_env
python -m ipykernel install --user --name taa_tp_env
```

# run it 

just type this

```bash
make
```

OR do this

```bash
jupyter nbconvert --ExecutePreprocessor.kernel_name=taa_tp_env --to html --execute source/tp_final.ipynb --output-dir res
```

A html version of the results is create inside ./res directory
