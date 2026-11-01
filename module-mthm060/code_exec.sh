#!/usr/bin/env bash

set -e

invert_iv() {
    python -m code.inverting_iv.main \
        --underlying=./code/data/nifty-daily.csv \
        --div_yields=./code/data/nifty-div-yields.csv \
        --bhavcopies=./code/data/nse_bhavs/
}

ssvi() {
    python -m code.ssvi.main \
        --data_reserve=./code/inverting_iv/artefacts/ \
        --expiries_start="2019-10-03" \
        --expiries_end="2019-10-31"
}

option_metrics() {
    python -m code.funvol.option_metrics.main \
        --data_reserve=./code/inverting_iv/artefacts/
}

legendre() {
    python -m code.funvol.legendre.main \
        --data_reserve=./code/funvol/option_metrics/artefacts/ \
        --n_o=4
}

funvol_fpca() {
    python -m code.funvol.fpca.main \
        --data_reserve=./code/funvol/legendre/artefacts
}

detrend_price() {
    python -m code.funvol.detrended_price.main \
        --data_reserve=./code/data/nifty-daily.csv
}

neural_sde_ce() {
    python -m code.funvol.neural_sde.main \
        --data_reserve=./code/funvol/fpca/artefacts \
        --detrended_price_reserve=./code/funvol/detrended_price/artefacts \
        --wing=ce
}

neural_sde_pe() {
    python -m code.funvol.neural_sde.main \
        --data_reserve=./code/funvol/fpca/artefacts \
        --detrended_price_reserve=./code/funvol/detrended_price/artefacts \
        --wing=pe
}

# Execute all pipeline steps
run_all() {
    invert_iv
    ssvi
    option_metrics
    legendre
    funvol_fpca
    detrended_price
    neural_sde_ce
    neural_sde_pe
}

# Command-line argument handling
if [ $# -eq 0 ]; then
    run_all
else
    case "$1" in
        --invert_iv)
            invert_iv
            ;;
        --ssvi)
            ssvi
            ;;
        --option_metrics)
            option_metrics
            ;;
        --legendre)
            legendre
            ;;
        --funvol_fpca)
            funvol_fpca
            ;;
        --detrend_price)
            detrend_price
            ;;
        --neural_sde_ce)
            neural_sde_ce
            ;;
        --neural_sde_pe)
            neural_sde_pe
            ;;
        --help|-h)
            echo "Usage: ./code_exec.sh [OPTION]"
            echo ""
            echo "Options:"
            echo "  --invert_iv      Run implied volatility inversion"
            echo "  --ssvi           Run SSVI"
            echo "  --option_metrics Run option metrics"
            echo "  --legendre       Run Legendre transformation"
            echo "  --funvol_fpca    Run functional PCA"
            echo "  --detrend_price  Run detrended price calculation"
            echo "  --neural_sde_ce  Run Neural SDE (call wing)"
            echo "  --neural_sde_pe  Run Neural SDE (put wing)"
            echo "  --help           Show this help"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run ./code_exec.sh --help for available options."
            exit 1
            ;;
    esac
fi
