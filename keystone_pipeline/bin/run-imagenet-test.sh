#!/bin/bash

FWDIR="$(cd `dirname $0`/..; pwd)"

export SPARK_HOME=/root/spark
export KEYSTONE_MEM=150g

DATE=`date +"%Y_%m_%d_%H_%M_%S"`

$FWDIR/bin/run-pipeline-yarn.sh pipelines.CKMImageNetTest $FWDIR/target/scala-2.10/ckm-assembly-0.1.jar "$@" 2>&1  | tee /home/eecs/vaishaal/imagenet-ckm-test-$DATE.log
