import uproot
import numpy as np

#def write_root(filename, arrays):
#    """Write dictionary of numpy arrays to ROOT file."""
#    with uproot.recreate(filename) as f:
#        f["charmless_fit"] = arrays

def write_root(filename, arrays, tree_name="CharmingFit"):
    """Save a dict of numpy arrays to a ROOT file."""
    with uproot.recreate(filename) as f:
        tree = f.mktree(tree_name, {k: np.float64 for k in arrays})
        tree.extend({k: np.asarray(v, dtype="f8") for k, v in arrays.items()})
