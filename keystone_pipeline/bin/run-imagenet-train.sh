#!/bin/bash

FWDIR="$(cd `dirname $0`/..; pwd)"

$FWDIR/bin/run-pipeline-yarn.sh pipelines.CKMImageNetTrain target/scala-2.10/ckm-assembly-0.1.jar "$@"
