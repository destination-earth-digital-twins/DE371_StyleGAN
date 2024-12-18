import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}

def main():

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument("--path", type=str, default="" )
    parser.add_argument( "--filename", type=str, default="" )

    args = parser.parse_args()
    
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16,4))
    for index in range(10):
        filename = args.path+args.filename+f"{index}.npy"
        sample = np.load(filename)[0]
        vmin=np.min(sample[0])
        vmax=np.max(sample[0])
        ax[0].matshow(sample[0], clim=(vmin, vmax), origin="lower")
        ax[0].set_title("rr generated")

        vmin=np.min(sample[1])
        vmax=np.max(sample[1])
        ax[1].matshow(sample[1], clim=(vmin, vmax), origin="lower")
        ax[1].set_title("u generated")
        
        vmin=np.min(sample[2])
        vmax=np.max(sample[2])
        ax[2].matshow(sample[2], clim=(vmin, vmax), origin="lower")
        ax[2].set_title("v generated")
        
        vmin=np.min(sample[3])
        vmax=np.max(sample[3])
        ax[3].matshow(sample[3], clim=(vmin, vmax), origin="lower", cmap="coolwarm")
        ax[3].set_title("t2m generated")

        fig.tight_layout()
        fig.savefig(args.path+f'plots/sample_{index}.png')

        for a in ax:
            a.clear()
        
if __name__ == "__main__":
    main()