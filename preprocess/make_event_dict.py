
import os,sys,argparse
import ROOT as rt
import pickle

parser = argparse.ArgumentParser("Make event dictionary for events in images file(s)")
parser.add_argument("-i", "--inputfiles", required=True, type=str, nargs="+", help="input images file(s)")
parser.add_argument("-o", "--outputfile", required=True, type=str, help="output pickled dictionary file name")
parser.add_argument("--printDict", action="store_true", help="print output dictionary")
args = parser.parse_args()

eventDict = {}

for imgFile in args.inputfiles:
  print("adding events from %s to dictionary"%imgFile)
  f = rt.TFile(imgFile)
  t = f.Get("ImageTree")
  for i in range(t.GetEntries()):
    if i % 1000 == 0:
      print("reached image %i of %i"%(i,t.GetEntries()))
    t.GetEntry(i)
    if t.run in eventDict:
      if t.subrun in eventDict[t.run]:
        if t.event not in eventDict[t.run][t.subrun]:
          eventDict[t.run][t.subrun].append(t.event)
      else:
        eventDict[t.run][t.subrun] = [t.event]
    else:
      eventDict[t.run] = {t.subrun: [t.event]}

for r in eventDict:
  for sr in eventDict[r]:
    eventDict[r][sr].sort()
    if args.printDict:
      print("eventDict[%i][%i] = "%(r,sr), eventDict[r][sr])

outfile = open(args.outputfile, "wb")
pickle.dump(eventDict, outfile)
outfile.close()

