
import ROOT as rt
import argparse
from math import pi


parser = argparse.ArgumentParser("Prepare Prong CNN Images Training File")
parser.add_argument("-i", "--input", required=True, type=str, help="input prong image file")
parser.add_argument("-o", "--output", type=str, default="make_prong_sample_plots_output.root", help="output file name")
args = parser.parse_args()


h_el_trueEnergy = rt.TH1F("h_el_trueEnergy", "True Electron Energy for Electron Prongs", 100, 0., 5000.)
h_el_trueTheta = rt.TH1F("h_el_trueTheta", "True Electron Beam Angle for Electron Prongs", 100,0.,180.)
h_el_purity = rt.TH1F("h_el_purity","Electron Prong Purity", 101,0,1.01)
h_el_completeness = rt.TH1F("h_el_completeness","Electron Prong Completeness", 81,0,1.0125)
h_el_purityVsSize = rt.TH2F("h_el_purityVsSize","Electron Prong Purity vs. Size", 100,0,4500, 101,0,1.01)
h_el_completenessVsSize = rt.TH2F("h_el_completenessVsSize","Electron Prong Completeness vs. Size", 100,0,4500, 81,0,1.0125)

h_ph_trueEnergy = rt.TH1F("h_ph_trueEnergy", "True Photon Energy for Photon Prongs", 100, 0., 2000.)
h_ph_trueTheta = rt.TH1F("h_ph_trueTheta", "True Photon Beam Angle for Photon Prongs", 100,0.,180.)
h_ph_purity = rt.TH1F("h_ph_purity","Photon Prong Purity", 101,0,1.01)
h_ph_completeness = rt.TH1F("h_ph_completeness","Photon Prong Completeness", 81,0,1.0125)
h_ph_purityVsSize = rt.TH2F("h_ph_purityVsSize","Photon Prong Purity vs. Size", 100,0,3000, 101,0,1.01)
h_ph_completenessVsSize = rt.TH2F("h_ph_completenessVsSize","Photon Prong Completeness vs. Size", 100,0,3000, 81,0,1.0125)

h_mu_trueEnergy = rt.TH1F("h_mu_trueEnergy", "True Muon Energy for Muon Prongs", 100, 0., 3500.)
h_mu_trueTheta = rt.TH1F("h_mu_trueTheta", "True Muon Beam Angle for Muon Prongs", 100,0.,180.)
h_mu_purity = rt.TH1F("h_mu_purity","Muon Prong Purity", 101,0,1.01)
h_mu_completeness = rt.TH1F("h_mu_completeness","Muon Prong Completeness", 81,0,1.0125)
h_mu_purityVsSize = rt.TH2F("h_mu_purityVsSize","Muon Prong Purity vs. Size", 100,0,2000, 101,0,1.01)
h_mu_completenessVsSize = rt.TH2F("h_mu_completenessVsSize","Muon Prong Completeness vs. Size", 100,0,2000, 81,0,1.0125)

h_pi_trueEnergy = rt.TH1F("h_pi_trueEnergy", "True Pion Energy for Pion Prongs", 100, 0., 2500.)
h_pi_trueTheta = rt.TH1F("h_pi_trueTheta", "True Pion Beam Angle for Pion Prongs", 100,0.,180.)
h_pi_purity = rt.TH1F("h_pi_purity","Pion Prong Purity", 101,0,1.01)
h_pi_completeness = rt.TH1F("h_pi_completeness","Pion Prong Completeness", 81,0,1.0125)
h_pi_purityVsSize = rt.TH2F("h_pi_purityVsSize","Pion Prong Purity vs. Size", 100,0,2000, 101,0,1.01)
h_pi_completenessVsSize = rt.TH2F("h_pi_completenessVsSize","Pion Prong Completeness vs. Size", 100,0,2000, 81,0,1.0125)

h_pr_trueEnergy = rt.TH1F("h_pr_trueEnergy", "True Proton Energy for Proton Prongs", 100, 0., 2500.)
h_pr_trueTheta = rt.TH1F("h_pr_trueTheta", "True Proton Beam Angle for Proton Prongs", 100,0.,180.)
h_pr_purity = rt.TH1F("h_pr_purity","Proton Prong Purity", 101,0,1.01)
h_pr_completeness = rt.TH1F("h_pr_completeness","Proton Prong Completeness", 81,0,1.0125)
h_pr_purityVsSize = rt.TH2F("h_pr_purityVsSize","Proton Prong Purity vs. Size", 100,0,2000, 101,0,1.01)
h_pr_completenessVsSize = rt.TH2F("h_pr_completenessVsSize","Proton Prong Completeness vs. Size", 100,0,2000, 81,0,1.0125)

h_el_trueEnergy.GetXaxis().SetTitle("true energy (MeV)")
h_el_trueEnergy.SetLineWidth(2)
h_el_trueTheta.GetXaxis().SetTitle("true angle with respect to beam (degrees)")
h_el_trueTheta.SetLineWidth(2)
h_el_purity.GetXaxis().SetTitle("purity")
h_el_purity.SetLineWidth(2)
h_el_completeness.GetXaxis().SetTitle("completeness")
h_el_completeness.SetLineWidth(2)
h_el_purityVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_el_purityVsSize.GetYaxis().SetTitle("purity")
h_el_completenessVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_el_completenessVsSize.GetYaxis().SetTitle("completeness")

h_ph_trueEnergy.GetXaxis().SetTitle("true energy (MeV)")
h_ph_trueEnergy.SetLineWidth(2)
h_ph_trueTheta.GetXaxis().SetTitle("true angle with respect to beam (degrees)")
h_ph_trueTheta.SetLineWidth(2)
h_ph_purity.GetXaxis().SetTitle("purity")
h_ph_purity.SetLineWidth(2)
h_ph_completeness.GetXaxis().SetTitle("completeness")
h_ph_completeness.SetLineWidth(2)
h_ph_purityVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_ph_purityVsSize.GetYaxis().SetTitle("purity")
h_ph_completenessVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_ph_completenessVsSize.GetYaxis().SetTitle("completeness")

h_mu_trueEnergy.GetXaxis().SetTitle("true energy (MeV)")
h_mu_trueEnergy.SetLineWidth(2)
h_mu_trueTheta.GetXaxis().SetTitle("true angle with respect to beam (degrees)")
h_mu_trueTheta.SetLineWidth(2)
h_mu_purity.GetXaxis().SetTitle("purity")
h_mu_purity.SetLineWidth(2)
h_mu_completeness.GetXaxis().SetTitle("completeness")
h_mu_completeness.SetLineWidth(2)
h_mu_purityVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_mu_purityVsSize.GetYaxis().SetTitle("purity")
h_mu_completenessVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_mu_completenessVsSize.GetYaxis().SetTitle("completeness")

h_pi_trueEnergy.GetXaxis().SetTitle("true energy (MeV)")
h_pi_trueEnergy.SetLineWidth(2)
h_pi_trueTheta.GetXaxis().SetTitle("true angle with respect to beam (degrees)")
h_pi_trueTheta.SetLineWidth(2)
h_pi_purity.GetXaxis().SetTitle("purity")
h_pi_purity.SetLineWidth(2)
h_pi_completeness.GetXaxis().SetTitle("completeness")
h_pi_completeness.SetLineWidth(2)
h_pi_purityVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_pi_purityVsSize.GetYaxis().SetTitle("purity")
h_pi_completenessVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_pi_completenessVsSize.GetYaxis().SetTitle("completeness")

h_pr_trueEnergy.GetXaxis().SetTitle("true energy (MeV)")
h_pr_trueEnergy.SetLineWidth(2)
h_pr_trueTheta.GetXaxis().SetTitle("true angle with respect to beam (degrees)")
h_pr_trueTheta.SetLineWidth(2)
h_pr_purity.GetXaxis().SetTitle("purity")
h_pr_purity.SetLineWidth(2)
h_pr_completeness.GetXaxis().SetTitle("completeness")
h_pr_completeness.SetLineWidth(2)
h_pr_purityVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_pr_purityVsSize.GetYaxis().SetTitle("purity")
h_pr_completenessVsSize.GetXaxis().SetTitle("max number of pixels in plane")
h_pr_completenessVsSize.GetYaxis().SetTitle("completeness")


f = rt.TFile(args.input)
t = f.Get("ImageTree")


for i in range(t.GetEntries()):

  if i % 10000 == 0:
    print("reached entry %i of %i"%(i, t.GetEntries()))

  t.GetEntry(i)

  nPix = t.plane0_nPix
  if t.plane1_nPix > nPix:
    nPix = t.plane1_nPix
  if t.plane2_nPix > nPix:
    nPix = t.plane2_nPix

  if abs(t.pdg) == 11:
    h_el_trueEnergy.Fill(t.trueEnergy)
    h_el_trueTheta.Fill(t.trueTheta*(180./pi))
    h_el_purity.Fill(t.purity)
    h_el_completeness.Fill(t.completeness)
    h_el_purityVsSize.Fill(nPix, t.purity)
    h_el_completenessVsSize.Fill(nPix, t.completeness)

  if abs(t.pdg) == 22:
    h_ph_trueEnergy.Fill(t.trueEnergy)
    h_ph_trueTheta.Fill(t.trueTheta*(180./pi))
    h_ph_purity.Fill(t.purity)
    h_ph_completeness.Fill(t.completeness)
    h_ph_purityVsSize.Fill(nPix, t.purity)
    h_ph_completenessVsSize.Fill(nPix, t.completeness)

  if abs(t.pdg) == 13:
    h_mu_trueEnergy.Fill(t.trueEnergy)
    h_mu_trueTheta.Fill(t.trueTheta*(180./pi))
    h_mu_purity.Fill(t.purity)
    h_mu_completeness.Fill(t.completeness)
    h_mu_purityVsSize.Fill(nPix, t.purity)
    h_mu_completenessVsSize.Fill(nPix, t.completeness)

  if abs(t.pdg) == 211:
    h_pi_trueEnergy.Fill(t.trueEnergy)
    h_pi_trueTheta.Fill(t.trueTheta*(180./pi))
    h_pi_purity.Fill(t.purity)
    h_pi_completeness.Fill(t.completeness)
    h_pi_purityVsSize.Fill(nPix, t.purity)
    h_pi_completenessVsSize.Fill(nPix, t.completeness)

  if abs(t.pdg) == 2212:
    h_pr_trueEnergy.Fill(t.trueEnergy)
    h_pr_trueTheta.Fill(t.trueTheta*(180./pi))
    h_pr_purity.Fill(t.purity)
    h_pr_completeness.Fill(t.completeness)
    h_pr_purityVsSize.Fill(nPix, t.purity)
    h_pr_completenessVsSize.Fill(nPix, t.completeness)


outFile = rt.TFile(args.output, "RECREATE")

h_el_trueEnergy.Write()
h_el_trueTheta.Write()
h_el_purity.Write()
h_el_completeness.Write()
h_el_purityVsSize.Write()
h_el_completenessVsSize.Write()

h_ph_trueEnergy.Write()
h_ph_trueTheta.Write()
h_ph_purity.Write()
h_ph_completeness.Write()
h_ph_purityVsSize.Write()
h_ph_completenessVsSize.Write()

h_mu_trueEnergy.Write()
h_mu_trueTheta.Write()
h_mu_purity.Write()
h_mu_completeness.Write()
h_mu_purityVsSize.Write()
h_mu_completenessVsSize.Write()

h_pi_trueEnergy.Write()
h_pi_trueTheta.Write()
h_pi_purity.Write()
h_pi_completeness.Write()
h_pi_purityVsSize.Write()
h_pi_completenessVsSize.Write()

h_pr_trueEnergy.Write()
h_pr_trueTheta.Write()
h_pr_purity.Write()
h_pr_completeness.Write()
h_pr_purityVsSize.Write()
h_pr_completenessVsSize.Write()


