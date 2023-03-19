
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Split Prong CNN Training File into Train/Test Samples")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-nV", "--nVal", type=int, default=2000, help="number of particles per class to write to validation file")
args = parser.parse_args()

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_train = rt.TFile(args.infile.replace(".root","_%iPerClassVal_train.root"%args.nVal), "RECREATE")
t_train = t_orig.CloneTree(0)

f_test = rt.TFile(args.infile.replace(".root","_%iPerClassVal_test.root"%args.nVal), "RECREATE")
t_test = t_orig.CloneTree(0)

f_train_highP = rt.TFile(args.infile.replace("_noPurityCut.root","_%iPerClassVal_train.root"%args.nVal), "RECREATE")
t_train_highP = t_orig.CloneTree(0)

f_test_highP = rt.TFile(args.infile.replace("_noPurityCut.root","_%iPerClassVal_test.root"%args.nVal), "RECREATE")
t_test_highP = t_orig.CloneTree(0)


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

n_entries = t_orig.GetEntries()
classCountersTrain = [0,0,0,0,0]
classCountersTest = [0,0,0,0,0]
classCountersTrainHighP = [0,0,0,0,0]
classCountersTestHighP = [0,0,0,0,0]


for e in range(n_entries):

  if e % 1000 == 0:
    print("reached entry %i of %i"%(e,n_entries), flush=True)
  t_orig.GetEntry(e)

  classID = getClass(abs(t_orig.pdg))
  prongInHighPTest = False

  if t_orig.purity > 0.6:

    if classCountersTestHighP[classID] < args.nVal:
      t_test_highP.Fill()
      classCountersTestHighP[classID] += 1
      prongInHighPTest = True

    else:
      t_train_highP.Fill()
      classCountersTrainHighP[classID] += 1

  if classCountersTest[classID] < args.nVal:
    t_test.Fill()
    classCountersTest[classID] += 1

  elif not prongInHighPTest:
    t_train.Fill()
    classCountersTrain[classID] += 1

print("number of particles per class in total train sample:")
print(classCountersTrain)
print("number of particles per class in high purity train sample:")
print(classCountersTrainHighP)
print("number of particles per class in total test sample:")
print(classCountersTest)
print("number of particles per class in high purity test sample:")
print(classCountersTestHighP)

f_train.cd()
t_train.Write("",rt.TObject.kOverwrite)
f_train.Close()

f_test.cd()
t_test.Write("",rt.TObject.kOverwrite)
f_test.Close()

f_train_highP.cd()
t_train_highP.Write("",rt.TObject.kOverwrite)
f_train_highP.Close()

f_test_highP.cd()
t_test_highP.Write("",rt.TObject.kOverwrite)
f_test_highP.Close()

f_orig.Close()
