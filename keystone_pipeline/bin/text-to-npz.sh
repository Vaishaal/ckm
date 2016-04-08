#!/bin/bash

if [ $# -ne 1 ];
then
  echo "Usage: text-to-npz.sh <output-path>"
  exit 1
fi


read infile

# Get the file to tmp dir
pushd /home/eecs/vaishaal >/dev/null


features=$infile
featuresnpz="$infile".npz

hadoop fs -copyToLocal $features . 2>/dev/null
echo "COPIED FILE"
python ../../text-to-npz.py $features  2>/dev/null

rm -rf $features
hadoop fs -copyFromLocal $featuresnpz $1/ 2>/dev/null
rm $featuresnpz
popd >/dev/null
