#! /bin/bash


target_file='targets.txt'
scraper='scraper/Scraper.py'

cat $target_file | while read line; do
    if [[ $line =~ ^\# ]]; then
        continue
    fi

    name=$(echo $line | awk '{print $1}')
    file_name=$(echo $line | awk '{print $2}')

    python $scraper $name $file_name
done
