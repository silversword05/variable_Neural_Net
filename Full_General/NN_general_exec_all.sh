#!/bin/bash
python3 Create_data_file.py
python3 NN_general_train.py
python3 NN_general_test.py
python3 conf_mtx_NN_general.py