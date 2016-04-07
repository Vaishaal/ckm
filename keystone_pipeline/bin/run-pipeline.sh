#!/bin/bash
# Figure out where we are.
FWDIR="$(cd `dirname $0`; pwd)"

CLASS=$1
shift
JARFILE=$1
shift

# Figure out where the Scala framework is installed
FWDIR="$(cd `dirname $0`/..; pwd)"

if [ -z "$1" ]; then
  echo "Usage: run-main.sh <class> [<args>]" >&2
  exit 1
fi

if [ -z "$OMP_NUM_THREADS" ]; then
    export OMP_NUM_THREADS=1 # added as we were nondeterministically running into an openblas race condition 
fi  

echo "automatically setting OMP_NUM_THREADS=$OMP_NUM_THREADS"

ASSEMBLYJAR="/mnt/ckm/keystone_pipeline/target/scala-2.10/ckm-assembly-0.1-deps.jar"


if [[ -z "$SPARK_HOME" ]]; then
    echo "SPARK_HOME is not set, running pipeline locally, FWDIR=$FWDIR"

  $FWDIR/bin/run-main.sh $CLASS "$@"
#elif [[ -z "$KEYSTONE_HOME" ]]; then
#  echo "KEYSTONE_HOME is not set"
else
  echo "RUNNING ON THE CLUSTER" 
  # TODO: Figure out a way to pass in either a conf file / flags to spark-submit
  KEYSTONE_MEM=${KEYSTONE_MEM:-1g}
  export KEYSTONE_MEM

  # Set some commonly used config flags on the cluster
  $SPARK_HOME/bin/spark-submit \
    --deploy-mode client \
    --class $CLASS \
    --driver-class-path $JARFILE:$ASSEMBLYJAR \
    --conf spark.executor.extraClassPath=$JARFILE:$ASSEMBLYJAR \
    --conf spark.executor.cores=16 \
    --conf spark.hadoop.mapred.min.split.size=2000000000 \
    --driver-memory 100g \
    --conf spark.executorEnv.OMP_NUM_THREADS=1\
    --conf spark.driver.maxResultSize=0 \
    --conf spark.executor.memory=100g \
    --jars $ASSEMBLYJAR \
    $JARFILE \
    "$@"
fi

#    --conf spark.eventLog.compress=true \
#    --conf spark.eventLog.enabled=true \

           
