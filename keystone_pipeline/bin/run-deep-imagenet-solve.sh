#!/bin/bash

FWDIR="$(cd `dirname $0`/..; pwd)"

DATE=`date +"%Y_%m_%d_%H_%M_%S"`

$FWDIR/bin/run-pipeline-yarn.sh pipelines.CKMDeepImageNetSolve $FWDIR/target/scala-2.10/ckm-assembly-0.1.jar "$@" 2>&1 | tee /home/eecs/vaishaal/deep-ckm-solve-$DATE.log

