
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Split Prong CNN Training File into Train/Test Samples")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-s", "--split", type=float, default=0.8, help="fraction of input to use for training")
args = parser.parse_args()

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_train = rt.TFile(args.infile.replace(".root","_train.root"), "RECREATE")
t_train = t_orig.CloneTree(0)

f_test = rt.TFile(args.infile.replace(".root","_test.root"), "RECREATE")
t_test = t_orig.CloneTree(0)

n_entries = t_orig.GetEntries()
indices = np.arange(n_entries)
random.shuffle(indices)

i = 0
for e in indices:
  if i % 100 == 0:
    print("reached entry %i of %i"%(i,n_entries), flush=True)
  t_orig.GetEntry(e)
  if i <= int(args.split*n_entries):
    t_train.Fill()
  else:
    t_test.Fill()
  i += 1

f_train.cd()
t_train.Write("",rt.TObject.kOverwrite)
f_train.Close()

f_test.cd()
t_test.Write("",rt.TObject.kOverwrite)
f_test.Close()

f_orig.Close()
