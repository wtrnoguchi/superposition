#!/bin/bash
./run_collect_data.sh
./run_training.sh
./run_analysis.sh exp1
./run_analysis_exp2.sh
./run_analysis.sh exp3
./run_regression.sh