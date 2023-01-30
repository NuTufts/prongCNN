
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Trim Prong CNN File into Class Balanced Samples")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-n", "--nPerClass", type=int, default=10000, help="number of particles per class to write to output file")
parser.add_argument("-o", "--outfile", type=str, default="DEFAULT", help="output file name")
args = parser.parse_args()

if args.outfile == "DEFAULT":
  outname = args.infile.replace(".root","_%iPerClass.root"%args.nPerClass)
else:
  outname = args.outfile

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_out = rt.TFile(outname, "RECREATE")
t_out = t_orig.CloneTree(0)


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
classCounters = [0,0,0,0,0]


for e in range(n_entries):

  if e % 1000 == 0:
    print("reached entry %i of %i"%(e,n_entries), flush=True)
  t_orig.GetEntry(e)

  if countersFull(classCounters, args.nPerClass):
    break

  classID = getClass(abs(t_orig.pdg))

  if classCounters[classID] < args.nPerClass:
    t_out.Fill()
    classCounters[classID] += 1


f_out.cd()
t_out.Write("",rt.TObject.kOverwrite)
f_out.Close()

f_orig.Close()

