
import sys
import argparse
import ROOT as rt

parser = argparse.ArgumentParser("Compare Prong Sample Distributions")
parser.add_argument("-f1", "--file1", required=True, type=str, help="sample 1 prong distributions root file")
parser.add_argument("-f2", "--file2", required=True, type=str, help="sample 2 prong distributions root file")
parser.add_argument("-v1", "--version1", required=True, type=str, help="version tag for file 1")
parser.add_argument("-v2", "--version2", required=True, type=str, help="version tag for file 2")
parser.add_argument("-o", "--outdir", default="./", type=str, help="output file directory")
parser.add_argument("-p", "--plotVars", type=str, nargs="+", default=["completeness", "purity"])
args = parser.parse_args()


def sortHists(hlist):
  iMax = -1
  maxBin = -99.
  i = 0 
  for h in hlist:
    for b in range(h.GetNbinsX()):
      if h.GetBinContent(b+1) > maxBin:
        maxBin = h.GetBinContent(b+1)
        iMax = i 
    i = i+1 
  sortedList = [hlist[iMax]]
  for i in range(len(hlist)):
    if i != iMax:
      sortedList.append(hlist[i])
  return sortedList


rt.TH1.SetDefaultSumw2(rt.kTRUE)
rt.gStyle.SetOptStat(0)

f1 = rt.TFile(args.file1)
f2 = rt.TFile(args.file2)

particles = ["el", "ph", "mu", "pi", "pr"]

h_dummy_1 = rt.TH1F("h_dummy_1","h_dummy_1",10,0,10)
h_dummy_1.SetLineWidth(2)
h_dummy_1.SetLineColor(rt.kBlue)
h_dummy_2 = rt.TH1F("h_dummy_2","h_dummy_2",10,0,10)
h_dummy_2.SetLineWidth(2)
h_dummy_2.SetLineColor(rt.kRed)

leg = rt.TLegend(0.5,0.7,0.7,0.9)
leg.AddEntry(h_dummy_1, args.version1, "l")
leg.AddEntry(h_dummy_2, args.version2, "l")

leg2 = rt.TLegend(0.6,0.4,0.8,0.6)
leg2.AddEntry(h_dummy_1, args.version1, "l")
leg2.AddEntry(h_dummy_2, args.version2, "l")

leg3 = rt.TLegend(0.3,0.7,0.5,0.9)
leg3.AddEntry(h_dummy_1, args.version1, "l")
leg3.AddEntry(h_dummy_2, args.version2, "l")

outFileStem = args.outdir+"prong_sample_comparison_"+args.version1+"_vs_"+args.version2

cnv = rt.TCanvas("cnv","cnv")


for part in particles:
  for var in args.plotVars:

    h1 = f1.Get("h_%s_%s"%(part,var))
    h1.Scale(1.0/h1.Integral())
    h1.GetYaxis().SetTitle("area normalized event count")
    h1.SetLineWidth(2)
    h1.SetLineColor(rt.kBlue)
    h2 = f2.Get("h_%s_%s"%(part,var))
    h2.Scale(1.0/h2.Integral())
    h2.GetYaxis().SetTitle("area normalized event count")
    h2.SetLineWidth(2)
    h2.SetLineColor(rt.kRed)

    h = sortHists([h1,h2])
    h[0].Draw("EHIST")
    h[1].Draw("EHISTSAME")
    leg3.Draw()
    cnv.SaveAs("%s_%s_%s.png"%(outFileStem,part,var))

