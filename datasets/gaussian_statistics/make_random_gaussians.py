import numpy as np 
import os 
import sys
import shutil
import argparse

def random_cov(dim):
    random_X = np.random.normal(size=(dim, dim))
    random_PSD = random_X.T @ random_X 
    _, U = np.linalg.eigh(random_PSD)
    eigvals = np.geomspace(1/2, 2, dim)
    return U @ np.diag(eigvals) @ U.T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=globals()['__doc__']) 
    parser.add_argument('-d', type=int, nargs='+', required=True,  help='Dimensions')
    parser.add_argument('-k', type=int, required=True,  help='Repeats')
    args = parser.parse_args()

    dims = args.d
    k = args.k

    covs = [[ [random_cov(d), random_cov(d)]  for _ in range(k)] for d in dims]
    means = [np.zeros((d,)) for d in dims]

    if(os.path.exists('random_gaussians')):
        affirm = input("CAUTION: This script will overwrite the existing path `random_gaussians`. Do you want to continue? (y/n) ")
        if(affirm == 'y'):
            shutil.rmtree('random_gaussians')
        else:
            sys.exit(0)
    for i, d in enumerate(dims):
        for rep in range(k):
            os.makedirs(f"random_gaussians/d={d}/{rep}")
            np.save(f"random_gaussians/d={d}/{rep}/source_cov.npy", covs[i][rep][0])
            np.save(f"random_gaussians/d={d}/{rep}/target_cov.npy", covs[i][rep][1])

        np.save(f"random_gaussians/d={d}/zero_mean.npy", means[i])
