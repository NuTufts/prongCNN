
import ROOT as rt
import argparse

parser = argparse.ArgumentParser("Find Electron Fragments")
parser.add_argument("-i", "--input", required=True, type=str, help="input prong image file")
args = parser.parse_args()

f = rt.TFile(args.input)
t = f.Get("ImageTree")

for i in range(t.GetEntries()):
  t.GetEntry(i)
  if abs(t.pdg) == 11 and t.purity > 0.6 and t.completeness < 0.01:
    print("(run, sr, e) = (%i, %i, %i),  (vtx, cls) = (%i, %i),  isShower = %i,  bestOtherComp = %f"%(t.run, t.subrun, t.event, t.vertex, t.cluster, t.isShower, t.bestOtherComp))
