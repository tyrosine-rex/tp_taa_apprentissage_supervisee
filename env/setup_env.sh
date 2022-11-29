#! /bin/bash

conda env create -f env/requirement.yml
conda activate taa_tp_env
python -m ipykernel install --user --name taa_tp_env
