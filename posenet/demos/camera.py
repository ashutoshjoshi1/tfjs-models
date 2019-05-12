import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

def main(data):
    with open('keypoints.pkl', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main(data)
    return "Success"
