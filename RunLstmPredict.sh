#! /bin/bash

TARGET=",Nikkei225.txt"
DATA_FILES=",Nikkei225.txt ,TOPIX.txt ,6501.txt"

Product/LSTM/Main.py -g -p -t $TARGET $DATA_FILES
