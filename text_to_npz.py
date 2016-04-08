import sys
from experiments import *
if (len(sys.argv) < 2):
    print "Usage: python text_to_npz.py <featuredir>"
    exit()
dirname = sys.argv[1]
# Below call is very slow
X, Y = load_all_features_from_dir(dirname)
f_handle = open(dirname + ".npz", "w+")
np.savez(f_handle, X=X, y=Y)

