
import os,sys,argparse
import ROOT as rt

parser = argparse.ArgumentParser("Select images with specified run/subrun/event list")
parser.add_argument("-i", "--infile", required=True, type=str, help="input prongCNN images root file")
parser.add_argument("-o", "--outfile", required=True, type=str, help="output images file name")
parser.add_argument("-nEl", "--nElectrons", type=int, default=0, help="select this many electron images")
parser.add_argument("-nPh", "--nPhotons", type=int, default=0, help="select this many photon images")
parser.add_argument("-nMu", "--nMuons", type=int, default=0, help="select this many muon images")
parser.add_argument("-nPi", "--nPions", type=int, default=0, help="select this many pion images")
parser.add_argument("-nPr", "--nProtons", type=int, default=0, help="select this many proton images")
args = parser.parse_args()

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_out = rt.TFile(args.outfile, "RECREATE")
t_out = t_orig.CloneTree(0)

nElWritten = 0
nPhWritten = 0
nMuWritten = 0
nPiWritten = 0
nPrWritten = 0


for i in range(t_orig.GetEntries()):

  if i % 1000 == 0:
    print("reached entry %i of %i"%(i,t_orig.GetEntries()), flush=True)

  t_orig.GetEntry(i)

  if abs(t_orig.pdg) == 11 and nElWritten < args.nElectrons:
    nElWritten += 1
    t_out.Fill()
    continue

  if abs(t_orig.pdg) == 22 and nPhWritten < args.nPhotons:
    nPhWritten += 1
    t_out.Fill()
    continue

  if abs(t_orig.pdg) == 13 and nMuWritten < args.nMuons:
    nMuWritten += 1
    t_out.Fill()
    continue

  if abs(t_orig.pdg) == 211 and nPiWritten < args.nPions:
    nPiWritten += 1
    t_out.Fill()
    continue

  if abs(t_orig.pdg) == 2212 and nPrWritten < args.nProtons:
    nPrWritten += 1
    t_out.Fill()
    continue

f_out.cd()
t_out.Write("",rt.TObject.kOverwrite)
f_out.Close()

f_orig.Close()

