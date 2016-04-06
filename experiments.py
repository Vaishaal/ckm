import argparse

from yaml import load, dump
from itertools import groupby

import features_pb2
from pandas import DataFrame
import glob 

import logging
import collections
import os
import subprocess

from tabulate import tabulate
import time

'''
Driver for Convolutional Kernel Machine experiments
'''

def main():
    # First parse arguments for the CKM
    parser = argparse.ArgumentParser(description='Convolutional Kernel Machine')
    parser.add_argument('config', help='path to config file for experiment')
    args = parser.parse_args()
    exp = parse_experiment(args.config)

    conf_file = open("/tmp/ckn_" + str(time.time()), "w+")
    dump(exp, conf_file)
    conf_file.close()

    logging.info('Experiment mode: {0}'.format(exp.get("mode")))
    if (exp.get("mode") == "python"):
        results = python_run(exp)
    elif (exp.get("mode") == "scala"):
        results = scala_run(exp, args.config)
    if (not (results is None)):
        print results.to_csv(header=True)

def flatten_dict(d, parent_key='', sep='_'):

    ''' Borrowed from:
    http://stackoverflow.com/questions/6027558/flatten-nested-python-dictionaries-compressing-keys
   '''
    items = []
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, k, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def python_run(exp):
    pass

def scala_run(exp, yaml_path):
    start_time = time.time()
    expid = exp["expid"]
    config_yaml = yaml_path
    env = os.environ.copy()
    # sanity check before running the process

    # if not os.path.isdir(outdir):
    #     raise ValueError("output dir must exist")

    logfile = expid + ".spark.log"
    # if os.path.exists(logfile) and output_sanity_check:
    #     raise ValueError("output dir has logfile, should be empty")
    pipelineClass="pipelines.CKM"
    pipelineJar = "/mnt/ckm/keystone_pipeline/target/scala-2.10/ckm-assembly-0.1.jar"
    if not os.path.exists(pipelineJar):
        raise ValueError("Cannot find pipeline jar")

    # basically capturing the popen output and saving it to disk  and
    # printing it to screen are a multithreaded nightmare, so we let
    # tee do the work for us
    yarn = exp.get("yarn")
    if (yarn):
        p = subprocess.Popen(" ".join(["./keystone_pipeline/bin/run-pipeline-yarn.sh", pipelineClass,
                                   pipelineJar, config_yaml]),
                         shell=True, executable='/bin/bash')
    else:
        p = subprocess.Popen(" ".join(["./keystone_pipeline/bin/run-pipeline.sh", pipelineClass,
                                   pipelineJar, config_yaml]),
                         shell=True, executable='/bin/bash')
    #p = subprocess.Popen(cmd, shell=True, executable='/bin/bash', env = env)
    p.wait()

    if p.returncode != 0:
        raise Exception("invocation terminated with non-zero exit status")
    if (exp["solve"]):
        y_train, y_train_weights, ids_train  = load_scala_results("/tmp/ckm_train_results")
        y_test, y_test_weights, ids_test  = load_scala_results("/tmp/ckm_test_results")
        # TODO: Do more interesting things here
        if exp.get('augment'):
            y_train_weights, y_test_weights = augmented_eval(y_train_weights, y_test_weights, ids_train, ids_test)

        y_train_pred = np.argmax(y_train_weights, axis=1)
        y_test_pred = np.argmax(y_test_weights, axis=1)

        runtime =  time.time() - start_time
        results = compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred)
        results.insert(len(results.columns), "runtime",  runtime)

        if (exp.get("numClasses", 2) >= 5):
            top5_train, top5_test = compute_top5_acc(y_train, y_train_weights, y_test, y_test_weights)
            results.insert(len(results.columns), "top5_train_acc", top5_train)
            results.insert(len(results.columns), "top5_test_acc", top5_test)
        return results
    else:
        return None

def augmented_eval(y_train_weights, y_test_weights, ids_train, ids_test):
    y_train_weights_avg = np.array(map(lambda x: np.average(np.array(map(lambda y: y[1], x[1])), axis=0), groupby(zip(ids_train, y_train_weights), lambda z: z[0])))
    y_test_weights_avg = np.array(map(lambda x: np.average(np.array(map(lambda y: y[1], x[1])), axis=0), groupby(zip(ids_test, y_test_weights), lambda z: z[0])))
    return y_train_weights_avg, y_test_weights_avg

def gen_features(exp, X_train, X_test, seed):
    ckm_run = build_ckm(exp, seed)
    X_train_lift, X_test_lift = ckm_run(X_train, X_test)
    X_train = X_train_lift.reshape(X_train.shape[0], -1)
    X_test = X_test_lift.reshape(X_test.shape[0], -1)
    return X_train, X_test

def save_features_python(X, y, name):
    X = X.reshape(X.shape[0], -1)
    dataset = features_pb2.Dataset()
    dataset.name = name
    for i in range(len(y)):
        datum = dataset.data.add()
        datum.label = int(y[i])
        datum.data.extend(X[i, 0:].tolist())

    f = open(dataset.name + ".bin", "wb")
    f.write(dataset.SerializeToString())
    f.close()

def load_features_python(name):
    dataset = features_pb2.Dataset()
    f = open("{0}.bin".format(name), "rb")
    dataset.ParseFromString(f.read())
    f.close()
    X = []
    y = []
    for datum in dataset.data:
        x_i = list(datum.data)
        y_i = datum.label
        X.append(x_i)
        y.append(y_i)

    return np.array(X), np.array(y)

def load_scala_results(name):
    f = open(name, "r")
    result_lines = f.readlines()
    results = np.array(map(lambda x: map(lambda y: float(y), x.split(",")), result_lines))
    labels = results[:, 1]
    ids = results[:,0]
    weights = results[:, 2:]
    return labels, weights, ids

def load_all_features_from_dir(dirname):
    files = glob.glob(dirname + "/part*")
    all_features = []
    all_labels = []
    i = 1
    for f in files:
        print "Part {0}".format(i)
        features, labels = load_features_from_text(f)
        all_features.extend(features)
        all_labels.extend(labels)
        i += 1
    return np.array(all_features), np.array(all_labels)


def load_features_from_text(fname):
    x_tuples = open(fname).readlines()
    x = map(lambda x: (map(lambda y: float(y), (x[1:-2].split(",")[:-1])), float(x[1:-2].split(",")[-1])), x_tuples)
    features, labels = zip(*x)
    return features, labels

def save_text_features_as_npz(fname_train, fname_test):
    print("Loading Train Features")
    X_train, y_train = load_all_features_from_dir(fname_train)
    print("Loading Test Features")
    X_test, y_test = load_all_features_from_dir(fname_test)
    train_file = open(fname_train + ".npz", "w+")
    test_file = open(fname_test + ".npz", "w+")
    print("Saving Train Features")
    np.savez(train_file, X_train=X_train, y_train=y_train)
    print("Saving Test Features")
    np.savez(test_file, X_test=X_test, y_test=y_test)

def scp_features_to_c78(fname_train, fname_test, path="/work/vaishaal"):
    save_text_features_as_npz(fname_train, fname_test)
    print("Moving features to c78")
    p = subprocess.Popen(" ".join(["scp", fname_train +".npz", "c78.millennium.berkeley.edu:{0}".format(path)]), shell=True, executable='/bin/bash')
    p.wait()
    p = subprocess.Popen(" ".join(["scp", fname_test +".npz", "c78.millennium.berkeley.edu:{0}".format(path)]), shell=True, executable='/bin/bash')
    p.wait()
    if p.returncode != 0:
        raise Exception("invocation terminated with non-zero exit status")



def compute_metrics(exp, y_train, y_train_pred, y_test, y_test_pred):
    return None

def compute_top5_acc(y_train, y_train_weights, y_test, y_test_weights):
    top5_train = y_train_weights.argsort(axis=1)[:,-5:]
    top5_test = y_test_weights.argsort(axis=1)[:,-5:]
    train_res = np.any(np.equal(y_train[:,np.newaxis],top5_train), axis=1)
    test_res = np.any(np.equal(y_test[:,np.newaxis],top5_test), axis=1)
    test_acc = test_res.sum()/(1.0*len(test_res))
    train_acc = train_res.sum()/(1.0*len(train_res))
    return train_acc, test_acc

def build_ckm(exp, seed):
    layers = exp.get("layers")
    filters = exp.get("filters")
    bandwidth = exp.get("bandwidth")
    patch_sizes = exp.get("patch_sizes")
    verbose = exp.get("verbose")
    pool = exp.get("pool")
    channels = exp.get("numChannels", 1)
    def ckm_run(X_train, X_test):
        for i in range(layers):
            if (i == 0 and exp.get("whiten")):
                whiten = True
            else:
                whiten = False
            patch_shape = (patch_sizes[i], patch_sizes[i])
            X_train, X_test =  ckm_apply(X_train, X_test, patch_shape, bandwidth[i], filters[i], pool=pool[i], random_state=(seed+i), whiten=whiten, numChannels=channels)
        return X_train, X_test
    return ckm_run

def parse_experiment(config_file):
    return load(open(config_file))

if __name__ == "__main__":
    main()
