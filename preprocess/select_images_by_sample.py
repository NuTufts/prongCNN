
import os,sys,argparse
import ROOT as rt

parser = argparse.ArgumentParser("Select images with specified run/subrun/event list")
parser.add_argument("-i", "--infile", required=True, type=str, help="input prongCNN images root file")
parser.add_argument("-o", "--outfile", required=True, type=str, help="output images file name")
parser.add_argument("-n", "--ntuple", required=True, type=str, help="select images from events in this gen2 ntuple")
args = parser.parse_args()


eventDict = {}

ntuple = rt.TFile(args.ntuple)
tEvt = ntuple.Get("EventTree")

for i in range(tEvt.GetEntries()):
  tEvt.GetEntry(i)
  eventDict[(tEvt.run,tEvt.subrun,tEvt.event)] = False


f_orig = rt.TFile(args.infile)
t_orig = f_orig.Get("ImageTree")

f_out = rt.TFile(args.outfile, "RECREATE")
t_out = t_orig.CloneTree(0)

nImagesSelected = 0
nEventsSelected = 0
nPrimElectrons = 0
nPrimMuons = 0


for i in range(t_orig.GetEntries()):

  if i % 1000 == 0:
    print("reached entry %i of %i"%(i,t_orig.GetEntries()), flush=True)

  t_orig.GetEntry(i)
  eventID = (t_orig.run,t_orig.subrun,t_orig.event)

  if eventID in eventDict:

    if not eventDict[eventID]:
      nEventsSelected += 1
      eventDict[eventID] = True

    if t_orig.processClass == 0:
      if abs(t_orig.pdg) == 11:
        nPrimElectrons += 1
      if abs(t_orig.pdg) == 13:
        nPrimMuons += 1

    nImagesSelected += 1
    t_out.Fill()


print(f"wrote {nImagesSelected} from {nEventsSelected} events")
print(f"{nPrimElectrons} were for primary electrons and {nPrimMuons} were for primary muons")

f_out.cd()
t_out.Write("",rt.TObject.kOverwrite)
f_out.Close()

f_orig.Close()


