import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from model import train
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def a(f):
    mm = []
    with open(f, "r") as q: #read file
        for l in q: #for each line in file
            mm += [c for c in l] #append to mm all charecters in line

    mm = ["<s>", "<s>"] + mm + ["<e>", "<e>"] #add starting and ending tokens to the set of chars related to a line
    return mm, list(set(mm)) #all chars and unique list of chars in a file

def g(x, p): #vectorize a vowel e.g, if 'a' has index 0 in list(set(mm)), then its vector will be 1000000000000000000000000000000
    z = np.zeros(len(p))
    z[p.index(x)] = 1
    return z

def b(u, p): # u= mm, p= list(set(mm))
    gt = [] # stores indecies of vowels i.e. convert alphabetic data (e.g. i, o ...etc.) to numeric (e.g., 4, 9)
    gr = [] # stores vectorized version of context for each vowel occurs in mm
    for v in range(len(u) - 4):
        if u[v+2] not in vowels: #v+2 to skip the two <s> tokens
            continue #if the char is not a vowel, then break and move to the next char
        
        h2 = vowels.index(u[v+2])
        gt.append(h2) #otherwise, store the index (in vowels list) of the vowel u[v+2] in the list of indeces gt
        r = np.concatenate([g(x, p) for x in [u[v], u[v+1], u[v+3], u[v+4]]])
        gr.append(r) #store list of vectorized context in gr i.e. the list of lists (contexts) of lists (vectors).

    return np.array(gr), np.array(gt) #convert gr and gt into numpy arrays 
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", dest="k", type=int, default=200) #no. features
    parser.add_argument("--r", dest="r", type=int, default=100) #no. epochs
    parser.add_argument("m", type=str) #input file (with raw data) path
    parser.add_argument("h", type=str) #output file
    
    args = parser.parse_args()

    q = a(args.m)
    w = b(q[0], q[1])
    t = train(w[0], w[1], q[1], args.k, args.r)

    torch.save(t, args.h) #store model t in output file h 
