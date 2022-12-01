
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Split Prong CNN Training File into Class Balanced Train/Test Samples")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-nT", "--nTrain", type=int, default=10000, help="number of particles per class to write to training file")
parser.add_argument("-nV", "--nTest", type=int, default=2000, help="number of particles per class to write to validation file")
args = parser.parse_args()

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_train = rt.TFile(args.infile.replace(".root","_train.root").replace("largeSample","12KPerClass"), "RECREATE")
t_train = t_orig.CloneTree(0)

f_test = rt.TFile(args.infile.replace(".root","_test.root").replace("largeSample","12KPerClass"), "RECREATE")
t_test = t_orig.CloneTree(0)


def getClass(pid):
  if pid == 11:
    return 0 
  if pid == 22:
    return 1 
  if pid == 13:
    return 2 
  if pid == 211:
    return 3
  if pid == 2212:
    return 4
  return 5

def countersFull(counters, needed):
  for count in counters:
    if count < needed:
      return False
  return True

n_entries = t_orig.GetEntries()
classCountersTrain = [0,0,0,0,0]
classCountersTest = [0,0,0,0,0]


for e in range(n_entries):

  if e % 1000 == 0:
    print("reached entry %i of %i"%(e,n_entries), flush=True)
  t_orig.GetEntry(e)

  if countersFull(classCountersTrain, args.nTrain) and countersFull(classCountersTest, args.nTest):
    break

  classID = getClass(t_orig.pdg)

  if classCountersTrain[classID] < args.nTrain:
    t_train.Fill()
    classCountersTrain[classID] += 1

  elif classCountersTest[classID] < args.nTest:
    t_test.Fill()
    classCountersTest[classID] += 1


f_train.cd()
t_train.Write("",rt.TObject.kOverwrite)
f_train.Close()

f_test.cd()
t_test.Write("",rt.TObject.kOverwrite)
f_test.Close()

f_orig.Close()
