#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    likelihood = __import__('0-likelihood').likelihood

    P = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    print(likelihood(26, 130, P))
