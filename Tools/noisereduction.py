import numpy as np

def noise_reduction_mean(x,binsize):
    if np.size(x)%binsize != 0:
        print(f"passed dataset could not be split in requested binsizes \n"
              f"size of passed dataset: {np.size(x)}\n"
              f"binsize: {binsize}")
        return
    else:
        x = np.reshape(x,(int(np.size(x)/binsize),binsize))
        x = np.mean(x,axis=1)
        return x
