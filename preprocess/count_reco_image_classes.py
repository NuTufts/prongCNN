
import os,sys,argparse
import ROOT as rt
import numpy as np
import random
from array import array

parser = argparse.ArgumentParser("Count Reco Prong CNN Image Classes")
parser.add_argument("-f", "--infile", required=True, type=str, help="input prongCNN images root file")
args = parser.parse_args()

f = rt.TFile(args.infile)
t = f.Get("ImageTree")

def getClass(pid, purity):
  if purity < 0.6:
    return 5
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

outFile = rt.TFile("count_reco_image_classes_plots.root","RECREATE")
prongTree = rt.TTree("ProngTree","ProngTree")
puritySum = array('f', [0.])
prongTree.Branch("puritySum", puritySum, 'puritySum/F')
puritySum5Class = array('f', [0.])
prongTree.Branch("puritySum5Class", puritySum5Class, 'puritySum5Class/F')
domPartPDG = array('i', [0])
prongTree.Branch("domPartPDG", domPartPDG, 'domPartPDG/I')
domPartPurity = array('f', [0.])
prongTree.Branch("domPartPurity", domPartPurity, 'domPartPurity/F')
pdg = array('i', [0])
prongTree.Branch("pdg", pdg, 'pdg/I')
purity = array('f',[0.])
prongTree.Branch("purity", purity, 'purity/F')

nProngs = 0
classCounters = [0,0,0,0,0,0]
pidClassCounters = [0,0,0,0,0,0,0]
puritySumLs1 = 0
puritySumEq1 = 0
puritySumGr1 = 0

for e in range(t.GetEntries()):
  t.GetEntry(e)
  classCounters[getClass(t.pdg, t.purity)] += 1
  pSum = 0.
  pSum5Class = 0.
  pMax = -1.
  pMaxPDG = 0
  for i in range(t.nParticles):
    pidClass = getPIDClass(t.pdgs[i])
    pidClassCounters[pidClass] += 1
    pSum += t.purities[i]
    if pidClass < 5:
      pSum5Class += t.purities[i]
    if t.purities[i] > pMax:
      pMax = t.purities[i]
      pMaxPDG = t.pdgs[i]
  if pSum < 1.0 - 1e-6:
    puritySumLs1 += 1
  elif pSum < 1.0 + 1e-6:
    puritySumEq1 += 1
  else:
    puritySumGr1 += 1
  puritySum[0] = pSum
  puritySum5Class[0] = pSum5Class
  domPartPDG[0] = pMaxPDG
  domPartPurity[0] = pMax
  pdg[0] = t.pdg
  purity[0] = t.purity
  prongTree.Fill()
  nProngs += 1

outFile.cd()
prongTree.Write("",rt.TObject.kOverwrite)
outFile.Close()

print("n prongs:", nProngs)
print("n electrons:", classCounters[0])
print("n photons:", classCounters[1])
print("n muons:", classCounters[2])
print("n pions:", classCounters[3])
print("n protons:", classCounters[4])
print("n other:", classCounters[5])
print("n electron contributions:", pidClassCounters[0])
print("n photon contributions:", pidClassCounters[1])
print("n muon contributions:", pidClassCounters[2])
print("n pion contributions:", pidClassCounters[3])
print("n proton contributions:", pidClassCounters[4])
print("n noMC contributions:", pidClassCounters[5])
print("n other contributions:", pidClassCounters[6])
print("n <1 purity sums:", puritySumLs1)
print("n =1 purity sums:", puritySumEq1)
print("n >1 purity sums:", puritySumGr1)

f.Close()
