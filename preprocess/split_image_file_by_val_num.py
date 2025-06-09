
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Split Prong CNN Training File into Train/Test Samples")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-l", "--islist", action="store_true", default=False, help="If given, treat input as textfile with many events")
parser.add_argument("-nV", "--nVal", type=int, default=2000, help="number of particles per class to write to validation file")
args = parser.parse_args()

#f_orig = rt.TFile(args.infile)
#t_orig = f_orig.Get("ImageTree")

t_orig = rt.TChain("ImageTree")
if not args.islist:
  t_orig.Add( args.infile )
  suffix = "root"
else:
  print("Loading inputlist: ",args.infile)
  suffix = "txt"
  with open(args.infile,"r") as finput:
    ll = finput.readlines()
    numfiles = len(ll)
    ifile = 0
    for l in ll:
      l = l.strip()
      if not os.path.exists(l):
        raise ValueError("Could not load this file")
      if ifile>0 and ifile%10000==0:
        print(f"adding file [{ifile}] of {numfiles}")
      t_orig.Add(l)
      ifile += 1
      
NENTRIES=t_orig.GetEntries()
print("Number of entries in TChain: ",NENTRIES)


f_train = rt.TFile(args.infile.replace(suffix,"_%iPerClassVal_train.root"%args.nVal), "RECREATE")
t_train = t_orig.CloneTree(0)

f_test = rt.TFile(args.infile.replace(suffix,"_%iPerClassVal_test.root"%args.nVal), "RECREATE")
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

n_entries = t_orig.GetEntries()
classCountersTrain = [0,0,0,0,0]
classCountersTest = [0,0,0,0,0]


for e in range(n_entries):

  if e % 1000 == 0:
    print("reached entry %i of %i"%(e,n_entries), flush=True)
  t_orig.GetEntry(e)

  classID = getClass(abs(t_orig.pdg))

  if classCountersTest[classID] < args.nVal:
    t_test.Fill()
    classCountersTest[classID] += 1

  else:
    t_train.Fill()
    classCountersTrain[classID] += 1

print("number of particles per class in train sample:")
print(classCountersTrain)

f_train.cd()
t_train.Write("",rt.TObject.kOverwrite)
f_train.Close()

f_test.cd()
t_test.Write("",rt.TObject.kOverwrite)
f_test.Close()

#f_orig.Close()
