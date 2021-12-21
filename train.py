""" Train your model """
import argparse # native imports 

import numpy as np # third party imports 

from utils.helper import * # local imports 


def sum(a, b):
    return a + b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', type=int, help='an integer for the accumulator')
    parser.add_argument('-b', type=int, help='an integer for the accumulator')
    args = parser.parse_args()
    a = args.a 
    b = args.b 

    print(sum(a, b))
