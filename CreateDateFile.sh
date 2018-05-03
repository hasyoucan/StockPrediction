#! /bin/bash

IN_FILES=,*.txt
OUT_FILE=,date.txt

cat $IN_FILES | awk '{print $1}' | sort | uniq > $OUT_FILE
