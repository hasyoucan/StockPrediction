#! /bin/bash

TARGET=",Nikkei225.csv"
DATA_FILES=",Nikkei225.csv ,TOPIX.csv ,6501.csv"

Product/LSTM/Main.py -g -p -t $TARGET $DATA_FILES
