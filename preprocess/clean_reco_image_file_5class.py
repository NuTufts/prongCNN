
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Clean Reco Prong CNN Images File")
parser.add_argument("-f", "--infile", required=True, type=str, help="input prongCNN images root file")
parser.add_argument("-pS", "--min5ClassPurity", type=float, default=0.8, help="minimum value for sum of 5-class-particle purity contributions")
parser.add_argument("-pD", "--minDomPartPurity", type=float, default=0.6, help="minimum value for purity contribution of dominant particle")
parser.add_argument("-nH", "--minNHit", type=int, default=10, help="minimum number of hits in any plane")
args = parser.parse_args()

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_out = rt.TFile(args.infile.replace(".root","_cleaned_minHit%i_noSecondaries.root"%args.minNHit), "RECREATE")
t_out = t_orig.CloneTree(0)

def getPIDClass(pid):
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
  if pid == 0:
    return 5
  return 6

for e in range(t_orig.GetEntries()):
  t_orig.GetEntry(e)
  if t_orig.isSecondary == 1:
    continue
  pSum5Class = 0.
  pMax = -1.
  pMaxClass = 7
  for i in range(t_orig.nParticles):
    pidClass = getPIDClass(t_orig.pdgs[i])
    if pidClass < 5:
      pSum5Class += t_orig.purities[i]
    #if t_orig.purities[i] > pMax:
    #  pMax = t_orig.purities[i]
    #  pMaxClass = pidClass
  #if pSum5Class > args.min5ClassPurity and pMax > args.minDomPartPurity and pMaxClass < 5:
  if pSum5Class > args.min5ClassPurity and t_orig.purity > args.minDomPartPurity and getPIDClass(abs(t_orig.pdg)) < 5:
    if t_orig.plane0_nPix >= args.minNHit or t_orig.plane1_nPix >= args.minNHit or t_orig.plane2_nPix >= args.minNHit:
      t_out.Fill()

f_out.cd()
t_out.Write("",rt.TObject.kOverwrite)
f_out.Close()

f_orig.Close()
