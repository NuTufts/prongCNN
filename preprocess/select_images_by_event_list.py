
import os,sys,argparse
import ROOT as rt
import numpy as np
import random

parser = argparse.ArgumentParser("Select images with specified run/subrun/event list")
parser.add_argument("-f", "--infile", required=True, type=str, help="prongCNN images root file")
parser.add_argument("-o", "--outfile", type=str, default="DEFAULT", help="output file name")
args = parser.parse_args()

eventlist = [ #[18520,13,676],
#              [18514,195,9777],
#              [18021,3,166],
#              [18031,383,19184],
#              [18862,415,20765],
#              [18860,124,6248],
#              [18032,515,25798],
#              [18863,152,7650],
#              [16960,184,9249],
#              [18577,320,16019],
#              [16962,75,3769],
#              [16935,300,15039],
              [18527,20,1034] ]#,
#              [18525,115,5767],
#              [18525,115,5767] ]

if args.outfile == "DEFAULT":
  outname = args.infile.replace(".root","_selectedEvents.root")
else:
  outname = args.outfile

f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_out = rt.TFile(outname, "RECREATE")
t_out = t_orig.CloneTree(0)

n_entries = t_orig.GetEntries()
n_written = 0

for e in range(n_entries):

  #if n_written == len(eventlist):
  #  break

  if e % 1000 == 0:
    print("reached entry %i of %i"%(e,n_entries), flush=True)
  t_orig.GetEntry(e)

  event = [t_orig.run, t_orig.subrun, t_orig.event]

  if event in eventlist:
    t_out.Fill()
    n_written += 1


print(f"wrote {n_written} of {len(eventlist)} events")

f_out.cd()
t_out.Write("",rt.TObject.kOverwrite)
f_out.Close()

f_orig.Close()

