#!/bin/bash

python -m code.inverting_iv.main --underlying=./code/data/nifty-daily.csv --div_yields=./code/data/nifty-div-yields.csv --bhavcopies=./code/data/nse_bhavs/
python -m code.ssvi.main --data_reserve=./code/inverting_iv/artefacts/ --expiries_start="2019-10-03" --expiries_end="2019-10-31"
python -m code.funvol.option_metrics.main --data_reserve=./code/inverting_iv/artefacts/
python -m code.funvol.legendre.main --data_reserve=./code/funvol/option_metrics/artefacts/ --n_o=4
python -m code.funvol.fpca.main --data_reserve=./code/funvol/legendre/artefacts
python -m code.funvol.detrended_price.main --data_reserve=./code/data/nifty-daily.csv

python -m code.funvol.neural_sde.main --data_reserve=./code/funvol/fpca/artefacts --detrended_price_reserve=./code/funvol/detrended_price/artefacts --wing=ce
python -m code.funvol.neural_sde.main --data_reserve=./code/funvol/fpca/artefacts --detrended_price_reserve=./code/funvol/detrended_price/artefacts --wing=pe

