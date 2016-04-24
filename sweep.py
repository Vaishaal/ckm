from experiments import *
import pickle
import yaml
import os

base_exp =  load(open("./keystone_pipeline/imagenet_cluster.exp"))
# 1 Layer sweep
results = []
for reg in [1e-3, 1e-2, 1e-1, 1, 10]:
    exp_copy = base_exp.copy()
    exp["reg"] = reg
    fname = "/tmp/imagenet_reg_sweep_{0}_{1}".format(gamma,  reg)
    sweep_file = open(fname, "w+")
    yaml.dump(exp_copy, sweep_file)
    sweep_file.close()
    os.system("bin/run-imagenet-solve.sh {}".format(fname))

