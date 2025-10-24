import uproot
import numpy as np

def write_root(filename, arrays):
    """Write dictionary of numpy arrays to ROOT file."""
    with uproot.recreate(filename) as f:
        f["charmless_fit"] = arrays
