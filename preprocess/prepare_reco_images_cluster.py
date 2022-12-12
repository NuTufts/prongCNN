
import os,sys,argparse
import ROOT as rt
from larlite import larlite
from larlite import larutil
from ublarcvapp import ublarcvapp
from larcv import larcv
from larflow import larflow
from array import array
from math import sqrt as sqrt
from math import acos as acos

parser = argparse.ArgumentParser("Prepare Prong CNN Images Training File")
parser.add_argument("-if", "--recofiles", required=True, type=str, help="text file containing kpsreco file list")
parser.add_argument("-it", "--truthfiles", required=True, type=str, help="text file containing merged_dlreco file list")
parser.add_argument("-ia", "--arrayID", required=True, type=int, help="SLURM array ID")
parser.add_argument("-in", "--nfiles", required=True, type=int, help="number of input files to process")
parser.add_argument("-o", "--outfile", type=str, default="prongCNN_images_file.root", help="output file name")
parser.add_argument("-n", "--pixelWH", type=int, default=512, help="pixel width and height of image")
parser.add_argument("-t", "--pixelThresh", type=float, default=10., help="pixel threshold for image generation")
parser.add_argument("-m", "--minPixelCount", type=int, default=10, help="minimum number of pixels in each plane")
parser.add_argument("-v", "--vertexScoreCut", type=float, default=0.8, help="minimum vertex larmatch keypoint score")
parser.add_argument("-s", "--split", action="store_true", help="split output into training and validation samples")
parser.add_argument("-f", "--splitfrac", type=float, default=0.8, help="fraction of input to use for training if splitting files")
args = parser.parse_args()


sce = larutil.SpaceChargeMicroBooNE()
mcNuVertexer = ublarcvapp.mctools.NeutrinoVertex()
truthTrackSCE = ublarcvapp.mctools.TruthTrackSCE()
truthShowerTrunkSCE = ublarcvapp.mctools.TruthShowerTrunkSCE()


def getFiles(mdlTag, kpsfiles, mdlfiles):
  files = []
  for kpsfile in kpsfiles:
    dlrecofilelist = open(mdlfiles, "r")
    for line in dlrecofilelist:
      dlrecofile = line.replace("\n","")
      samtag = dlrecofile[dlrecofile.find(mdlTag):].replace(mdlTag,"").replace(".root","")
      if samtag in kpsfile:
        files.append([kpsfile, dlrecofile])
        break
    dlrecofilelist.close()
  return files

detCrds = [[0., 256.35], [-116.5, 116.5], [0, 1036.8]]
fidCrds = [ [detCrds[0][0] + 5. , detCrds[0][1] - 5.] ]
fidCrds.append( [detCrds[1][0] + 5. , detCrds[1][1] - 5.] )
fidCrds.append( [detCrds[2][0] + 5. , detCrds[2][1] - 15.] )

def inRange(x, bnd):
  return (x >= bnd[0] and x <= bnd[1])

def isFiducial(p):
  return (inRange(p.X(),fidCrds[0]) and inRange(p.Y(),fidCrds[1]) and inRange(p.Z(),fidCrds[2]))

def getDistance(a, b):
  return sqrt( (a.X() - b.X())**2 + (a.Y() - b.Y())**2 + (a.Z() - b.Z())**2)

def getDistToEdge(p):
  minDist = 1e6
  edgeDists = [p.X() - detCrds[0][0], detCrds[0][1] - p.X(),
               p.Y() - detCrds[1][0], detCrds[1][1] - p.Y(),
               p.Z() - detCrds[2][0], detCrds[2][1] - p.Z()]
  for dist in edgeDists:
    if dist < minDist:
      minDist = dist
  return minDist

def getMCProngParticle(sparseimg_vv, mcpg, mcpm, adc_v):

  particleDict = {}
  trackDict = {}
  totalPixI = 0.

  for p in range(3):
    for pix in sparseimg_vv[p]:
      totalPixI += pix.val
      pixContents = mcpm.getPixContent(p, pix.rawRow, pix.rawCol)
      #for part in pixContents:
      for part in pixContents.particles:
        if abs(part.pdg) in particleDict:
          #particleDict[abs(part.pdg)] += part.pixI
          particleDict[abs(part.pdg)] += pixContents.pixI
        else:
          #particleDict[abs(part.pdg)] = part.pixI
          particleDict[abs(part.pdg)] = pixContents.pixI
        if part.tid in trackDict:
          trackDict[part.tid][2] += pixContents.pixI
        else:
          trackDict[part.tid] = [part.pdg, part.nodeidx, pixContents.pixI]

  maxPartPDG = 0 
  maxPartNID = -1
  maxPartTID = -1
  maxPartI = 0.
  maxPartComp = 0.
  pdglist = []
  puritylist = []

  for part in particleDict:
    pdglist.append(part)
    puritylist.append(particleDict[part]/totalPixI)
    #if particleDict[part] > maxPartI:
    #  maxPartI = particleDict[part]
    #  maxPartPDG = part

  for track in trackDict:
    if trackDict[track][2] > maxPartI:
      maxPartI = trackDict[track][2]
      maxPartPDG = trackDict[track][0]
      maxPartNID = trackDict[track][1]
      maxPartTID = track

  totNodePixI = 0.
  if maxPartI > 0.:
    maxPartNode = mcpg.node_v[maxPartNID]
    if maxPartNode.tid != maxPartTID:
      sys.exit("ERROR: mismatch between node track id from mcpm and mcpg in getMCProngParticle")
    for p in range(3):
      pixels = maxPartNode.pix_vv[p]
      for iP in range(pixels.size()//2):
        row = (pixels[2*iP] - 2400)//6
        col = pixels[2*iP+1]
        totNodePixI += adc_v[p].pixel(row, col)
    if totNodePixI > 0.:
      maxPartComp = maxPartI/totNodePixI

  if maxPartComp > 1.:
    print("ERROR: prong completeness calculated to be >1")

  #return maxPartPDG, maxPartI/totalPixI, pdglist, puritylist
  return maxPartPDG, maxPartTID, totNodePixI, maxPartI/totalPixI, maxPartComp, pdglist, puritylist


def checkCompleteness(flowTriples, adc_v, thrumu_v, prongCluster, cropPt, 
                      mcpm, mcTID, truePixSum, bestComp):
  prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(adc_v, thrumu_v,
   prongCluster, cropPt, args.pixelThresh, args.pixelWH, args.pixelWH)
  matchedSum = 0.
  for p in range(3):
    for pix in prong_vv[p]:
      pixContents = mcpm.getPixContent(p, pix.rawRow, pix.rawCol)
      for part in pixContents.particles:
        if part.tid == mcTID:
          matchedSum += pixContents.pixI
  comp = matchedSum/truePixSum
  if comp > bestComp:
    return comp
  return bestComp


def getBestOtherCompleteness(vertices, vID, tID, sID, flowTriples, adc_v,
                             thrumu_v, mcpm, mcTID, truePartPixSum):
  bestComp = 0.

  for iV, vertex in enumerate(vertices):
    for iT, prongCluster in enumerate(vertex.track_hitcluster_v):
      if iV == vID and iT == tID:
        continue
      cropPt = vertex.track_v[iT].End()
      bestComp = checkCompleteness(flowTriples, adc_v, thrumu_v, prongCluster, cropPt,
       mcpm, mcTID, truePartPixSum, bestComp)

    for iS, prongCluster in enumerate(vertex.shower_v):
      if iV == vID and iS == sID:
        continue
      cropPt = vertex.shower_trunk_v[iS].Vertex()
      bestComp = checkCompleteness(flowTriples, adc_v, thrumu_v, prongCluster, cropPt, 
       mcpm, mcTID, truePartPixSum, bestComp)

  return bestComp


def getTheta(mcstep):
  pmag = sqrt(mcstep.Px()**2 + mcstep.Py()**2 + mcstep.Pz()**2)
  return acos(mcstep.Pz() / pmag)

def getTruePartInfo(ioll, trackid, pdg):
  if pdg == 0:
    return -99., -1., -9999.
  mctracks = ioll.get_data(larlite.data.kMCTrack, "mcreco")
  for mctrack in mctracks:
    if mctrack.TrackID() == trackid and mctrack.PdgCode() == pdg:
      mctrackSCE = truthTrackSCE.applySCE(mctrack)
      return mctrack.Start().E(), getTheta(mctrack.Start()), getDistToEdge(mctrackSCE.Vertex())
  mcshowers = ioll.get_data(larlite.data.kMCShower, "mcreco")
  for mcshower in mcshowers:
    if mcshower.TrackID() == trackid and mcshower.PdgCode() == pdg:
      mcshowerSCE = truthShowerTrunkSCE.applySCE(mcshower)
      return mcshower.Start().E(), getTheta(mcshower.Start()), getDistToEdge(mcshowerSCE.Vertex())
  return -99., -1., -9999.
      

outFile = rt.TFile(args.outfile,"RECREATE")

imageTree = rt.TTree("ImageTree","ImageTree")
nPixels = args.pixelWH*args.pixelWH
run = array('i', [0])
subrun = array('i', [0])
event = array('i', [0])
vertex = array('i', [0])
cluster = array('i', [0])
pdg = array('i', [0])
purity = array('f', [0.])
completeness = array('f', [0.])
bestOtherComp = array('f', [0.])
trueEnergy = array('f', [0.])
trueTheta = array('f', [0.])
trueEdgeDist = array('f', [0.])
nParticles = array('i', [0])
pdgs = array('i', 10*[0])
purities = array('f', 10*[0.])
isShower = array('i', [0])
#isSecondary = array('i', [0])
max_plane_nPix = array('i', [0])
plane0_nPix = array('i', [0])
plane0pix_row = array('i', nPixels*[0])
plane0pix_col = array('i', nPixels*[0])
plane0pix_val = array('f', nPixels*[0.])
plane1_nPix = array('i', [0])
plane1pix_row = array('i', nPixels*[0])
plane1pix_col = array('i', nPixels*[0])
plane1pix_val = array('f', nPixels*[0.])
plane2_nPix = array('i', [0])
plane2pix_row = array('i', nPixels*[0])
plane2pix_col = array('i', nPixels*[0])
plane2pix_val = array('f', nPixels*[0.])
raw_plane0_nPix = array('i', [0])
raw_plane0pix_row = array('i', nPixels*[0])
raw_plane0pix_col = array('i', nPixels*[0])
raw_plane0pix_val = array('f', nPixels*[0.])
raw_plane1_nPix = array('i', [0])
raw_plane1pix_row = array('i', nPixels*[0])
raw_plane1pix_col = array('i', nPixels*[0])
raw_plane1pix_val = array('f', nPixels*[0.])
raw_plane2_nPix = array('i', [0])
raw_plane2pix_row = array('i', nPixels*[0])
raw_plane2pix_col = array('i', nPixels*[0])
raw_plane2pix_val = array('f', nPixels*[0.])
imageTree.Branch("run", run, 'run/I')
imageTree.Branch("subrun", subrun, 'subrun/I')
imageTree.Branch("event", event, 'event/I')
imageTree.Branch("vertex", vertex, 'vertex/I')
imageTree.Branch("cluster", cluster, 'cluster/I')
imageTree.Branch("pdg", pdg, 'pdg/I')
imageTree.Branch("purity", purity, 'purity/F')
imageTree.Branch("completeness", completeness, 'completeness/F')
imageTree.Branch("bestOtherComp", bestOtherComp, 'bestOtherComp/F')
imageTree.Branch("trueEnergy", trueEnergy, 'trueEnergy/F')
imageTree.Branch("trueTheta", trueTheta, 'trueTheta/F')
imageTree.Branch("trueEdgeDist", trueEdgeDist, 'trueEdgeDist/F')
imageTree.Branch("nParticles", nParticles, 'nParticles/I')
imageTree.Branch("pdgs", pdgs, 'pdgs[nParticles]/I')
imageTree.Branch("purities", purities, 'purities[nParticles]/F')
imageTree.Branch("isShower", isShower, 'isShower/I')
#imageTree.Branch("isSecondary", isSecondary, 'isSecondary/I')
imageTree.Branch("max_plane_nPix", max_plane_nPix, 'max_plane_nPix/I')
imageTree.Branch("plane0_nPix", plane0_nPix, 'plane0_nPix/I')
imageTree.Branch("plane0pix_row", plane0pix_row, 'plane0pix_row[plane0_nPix]/I')
imageTree.Branch("plane0pix_col", plane0pix_col, 'plane0pix_col[plane0_nPix]/I')
imageTree.Branch("plane0pix_val", plane0pix_val, 'plane0pix_val[plane0_nPix]/F')
imageTree.Branch("plane1_nPix", plane1_nPix, 'plane1_nPix/I')
imageTree.Branch("plane1pix_row", plane1pix_row, 'plane1pix_row[plane1_nPix]/I')
imageTree.Branch("plane1pix_col", plane1pix_col, 'plane1pix_col[plane1_nPix]/I')
imageTree.Branch("plane1pix_val", plane1pix_val, 'plane1pix_val[plane1_nPix]/F')
imageTree.Branch("plane2_nPix", plane2_nPix, 'plane2_nPix/I')
imageTree.Branch("plane2pix_row", plane2pix_row, 'plane2pix_row[plane2_nPix]/I')
imageTree.Branch("plane2pix_col", plane2pix_col, 'plane2pix_col[plane2_nPix]/I')
imageTree.Branch("plane2pix_val", plane2pix_val, 'plane2pix_val[plane2_nPix]/F')
imageTree.Branch("raw_plane0_nPix", raw_plane0_nPix, 'raw_plane0_nPix/I')
imageTree.Branch("raw_plane0pix_row", raw_plane0pix_row, 'raw_plane0pix_row[raw_plane0_nPix]/I')
imageTree.Branch("raw_plane0pix_col", raw_plane0pix_col, 'raw_plane0pix_col[raw_plane0_nPix]/I')
imageTree.Branch("raw_plane0pix_val", raw_plane0pix_val, 'raw_plane0pix_val[raw_plane0_nPix]/F')
imageTree.Branch("raw_plane1_nPix", raw_plane1_nPix, 'raw_plane1_nPix/I')
imageTree.Branch("raw_plane1pix_row", raw_plane1pix_row, 'raw_plane1pix_row[raw_plane1_nPix]/I')
imageTree.Branch("raw_plane1pix_col", raw_plane1pix_col, 'raw_plane1pix_col[raw_plane1_nPix]/I')
imageTree.Branch("raw_plane1pix_val", raw_plane1pix_val, 'raw_plane1pix_val[raw_plane1_nPix]/F')
imageTree.Branch("raw_plane2_nPix", raw_plane2_nPix, 'raw_plane2_nPix/I')
imageTree.Branch("raw_plane2pix_row", raw_plane2pix_row, 'raw_plane2pix_row[raw_plane2_nPix]/I')
imageTree.Branch("raw_plane2pix_col", raw_plane2pix_col, 'raw_plane2pix_col[raw_plane2_nPix]/I')
imageTree.Branch("raw_plane2pix_val", raw_plane2pix_val, 'raw_plane2pix_val[raw_plane2_nPix]/F')

recolist = []
recofiles = open(args.recofiles,"r")
iF = 0
for line in recofiles:
  if iF < args.arrayID*args.nfiles:
    iF += 1
    continue
  if iF >= (args.arrayID + 1)*args.nfiles:
    break
  recolist.append(line.replace("\n",""))
  iF += 1
recofiles.close()

filepairs = getFiles("merged_dlreco_", recolist, args.truthfiles)


#-------- begin file loop -----------------------------------------------------#
for filepair in filepairs:

  print("reading files: ", filepair)

  try:
    kpsfile = rt.TFile(filepair[0])
    kpst = kpsfile.Get("KPSRecoManagerTree")
    foo = kpst.GetEntries()
  except:
    print("file "+filepair[0]+" is bad, skipping...")
    continue

  try:
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(filepair[1])
    ioll.open()
    foo = ioll.get_entries()
  except:
    print("file "+filepair[1]+" is bad, skipping...")
    continue

  iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward)
  iolcv.add_in_file(filepair[1])
  iolcv.reverse_all_products()
  iolcv.initialize()

  #++++++ begin entry loop ++++++++++++++++++++++++++++++++++++++++++++++++++++=
  for iE in range(ioll.get_entries()):

    kpst.GetEntry(iE)
    ioll.go_to(iE)
    iolcv.read_entry(iE)

    if kpst.run != ioll.run_id() or kpst.subrun != ioll.subrun_id() or kpst.event != ioll.event_id():
      print("EVENTS DON'T MATCH!!!")
      print("truth run/subrun/event: %i/%i/%i"%(ioll.run_id(),ioll.subrun_id(),ioll.event_id()))
      print("reco run/subrun/event: %i/%i/%i"%(kpst.run,kpst.subrun,kpst.event))
      continue

    vertices = kpst.nuvetoed_v
    maxScore = -99.
    foundVertex = False

    for iV in range(vertices.size()):
      if vertices[iV].keypoint_type != 0:
        continue
      if vertices[iV].netNuScore > args.vertexScoreCut and vertices[iV].netNuScore > maxScore:
        maxScore = vertices[iV].netNuScore
        nuVertex = vertices[iV]
        foundVertex = True
        vertex[0] = iV

    if not foundVertex:
      continue

    vtxTVec3 = rt.TVector3(nuVertex.pos[0], nuVertex.pos[1], nuVertex.pos[2])

    if not isFiducial(vtxTVec3):
      continue

    run[0] = kpst.run
    subrun[0] = kpst.subrun
    event[0] = kpst.event

    evtImage2D = iolcv.get_data(larcv.kProductImage2D, "wire")
    csmImage2D = iolcv.get_data(larcv.kProductImage2D, "thrumu")
    adc_v = evtImage2D.Image2DArray()
    thrumu_v = csmImage2D.Image2DArray()

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.set_adc_treename("wire")
    mcpg.buildgraph(iolcv, ioll)

    mcpm = ublarcvapp.mctools.MCPixelPMap()
    mcpm.set_adc_treename("wire")
    #mcpm.buildmap(iolcv, ioll)
    mcpm.buildmap(iolcv, mcpg)

    flowTriples = larflow.prep.FlowTriples()

    #++++++ begin track loop ++++++++++++++++++++++++++++++++++++++++++++++++++=
    for iT in range(nuVertex.track_hitcluster_v.size()):

      cropPt = nuVertex.track_v[iT].End()
      prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(adc_v, thrumu_v,
                  nuVertex.track_hitcluster_v[iT], cropPt, args.pixelThresh, args.pixelWH, args.pixelWH)
      skip = False
      for p in range(3):
        if prong_vv[p].size() < args.minPixelCount:
          skip = True
          break
      if skip:
        continue

      #pdg[0], purity[0], pdglist, puritylist = getMCProngParticle(prong_vv, mcpm)
      pdg[0], trackId, truePixSum, purity[0], completeness[0], pdglist, puritylist = getMCProngParticle(prong_vv, mcpg, mcpm, adc_v)
      bestOtherComp[0] = 0.
      if truePixSum > 0.:
        bestOtherComp[0] = getBestOtherCompleteness(vertices, vertex[0], iT, -1,
         flowTriples, adc_v, thrumu_v, mcpm, trackId, truePixSum)

      nParticles[0] = len(pdglist)
      for iTP in range(len(pdglist)):
        pdgs[iTP] = pdglist[iTP]
        purities[iTP] = puritylist[iTP]

      isShower[0] = 0
      cluster[0] = iT

      trueEnergy[0], trueTheta[0], trueEdgeDist[0] = getTruePartInfo(ioll, trackId, pdg[0])

      iP = 0
      for pix in prong_vv[0]:
        plane0pix_row[iP] = pix.row
        plane0pix_col[iP] = pix.col
        plane0pix_val[iP] = pix.val
        iP += 1
      plane0_nPix[0] = iP
      max_plane_nPix[0] = iP
      iP = 0
      for pix in prong_vv[1]:
        plane1pix_row[iP] = pix.row
        plane1pix_col[iP] = pix.col
        plane1pix_val[iP] = pix.val
        iP += 1
      plane1_nPix[0] = iP
      if iP > max_plane_nPix[0]:
        max_plane_nPix[0] = iP
      iP = 0
      for pix in prong_vv[2]:
        plane2pix_row[iP] = pix.row
        plane2pix_col[iP] = pix.col
        plane2pix_val[iP] = pix.val
        iP += 1
      plane2_nPix[0] = iP
      if iP > max_plane_nPix[0]:
        max_plane_nPix[0] = iP
        
      iP = 0
      for pix in prong_vv[3]:
        raw_plane0pix_row[iP] = pix.row
        raw_plane0pix_col[iP] = pix.col
        raw_plane0pix_val[iP] = pix.val
        iP += 1
      raw_plane0_nPix[0] = iP
      iP = 0
      for pix in prong_vv[4]:
        raw_plane1pix_row[iP] = pix.row
        raw_plane1pix_col[iP] = pix.col
        raw_plane1pix_val[iP] = pix.val
        iP += 1
      raw_plane1_nPix[0] = iP
      iP = 0
      for pix in prong_vv[5]:
        raw_plane2pix_row[iP] = pix.row
        raw_plane2pix_col[iP] = pix.col
        raw_plane2pix_val[iP] = pix.val
        iP += 1
      raw_plane2_nPix[0] = iP
        
      imageTree.Fill()
    #++++++ end track loop +++++++++++++++++++++++++++++++++++++++++++++++++++=

    #++++++ begin shower loop ++++++++++++++++++++++++++++++++++++++++++++++++++=
    for iS in range(nuVertex.shower_v.size()):

      cropPt = nuVertex.shower_trunk_v[iS].Vertex()
      prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(adc_v, thrumu_v, 
                  nuVertex.shower_v[iS], cropPt, args.pixelThresh, args.pixelWH, args.pixelWH)
      skip = False
      for p in range(3):
        if prong_vv[p].size() < args.minPixelCount:
          skip = True
          break
      if skip:
        continue

      #pdg[0], purity[0], pdglist, puritylist = getMCProngParticle(prong_vv, mcpm)
      pdg[0], trackId, truePixSum, purity[0], completeness[0], pdglist, puritylist = getMCProngParticle(prong_vv, mcpg, mcpm, adc_v)
      bestOtherComp[0] = 0.
      if truePixSum > 0.:
        bestOtherComp[0] = getBestOtherCompleteness(vertices, vertex[0], -1, iS,
         flowTriples, adc_v, thrumu_v, mcpm, trackId, truePixSum)

      nParticles[0] = len(pdglist)
      for iTP in range(len(pdglist)):
        pdgs[iTP] = pdglist[iTP]
        purities[iTP] = puritylist[iTP]

      isShower[0] = 1
      cluster[0] = iS

      trueEnergy[0], trueTheta[0], trueEdgeDist[0] = getTruePartInfo(ioll, trackId, pdg[0])

      iP = 0
      for pix in prong_vv[0]:
        plane0pix_row[iP] = pix.row
        plane0pix_col[iP] = pix.col
        plane0pix_val[iP] = pix.val
        iP += 1
      plane0_nPix[0] = iP
      max_plane_nPix[0] = iP
      iP = 0
      for pix in prong_vv[1]:
        plane1pix_row[iP] = pix.row
        plane1pix_col[iP] = pix.col
        plane1pix_val[iP] = pix.val
        iP += 1
      plane1_nPix[0] = iP
      if iP > max_plane_nPix[0]:
        max_plane_nPix[0] = iP
      iP = 0
      for pix in prong_vv[2]:
        plane2pix_row[iP] = pix.row
        plane2pix_col[iP] = pix.col
        plane2pix_val[iP] = pix.val
        iP += 1
      plane2_nPix[0] = iP
      if iP > max_plane_nPix[0]:
        max_plane_nPix[0] = iP
        
      iP = 0
      for pix in prong_vv[3]:
        raw_plane0pix_row[iP] = pix.row
        raw_plane0pix_col[iP] = pix.col
        raw_plane0pix_val[iP] = pix.val
        iP += 1
      raw_plane0_nPix[0] = iP
      iP = 0
      for pix in prong_vv[4]:
        raw_plane1pix_row[iP] = pix.row
        raw_plane1pix_col[iP] = pix.col
        raw_plane1pix_val[iP] = pix.val
        iP += 1
      raw_plane1_nPix[0] = iP
      iP = 0
      for pix in prong_vv[5]:
        raw_plane2pix_row[iP] = pix.row
        raw_plane2pix_col[iP] = pix.col
        raw_plane2pix_val[iP] = pix.val
        iP += 1
      raw_plane2_nPix[0] = iP
        
      imageTree.Fill()
    #++++++ end shower loop +++++++++++++++++++++++++++++++++++++++++++++++++++=

  #++++++ end entry loop ++++++++++++++++++++++++++++++++++++++++++++++++++++=

  ioll.close()
  iolcv.finalize()

#-------- end file loop -----------------------------------------------------#


outFile.cd()
imageTree.Write("",rt.TObject.kOverwrite)
outFile.Close()



if args.split:
  f_orig = rt.TFile(args.outfile)
  t_orig = f_orig.Get("ImageTree")
  
  f_train = rt.TFile(args.outfile.replace(".root","_train.root"), "RECREATE")
  t_train = t_orig.CloneTree(0)
  
  f_test = rt.TFile(args.outfile.replace(".root","_test.root"), "RECREATE")
  t_test = t_orig.CloneTree(0)
  
  n_entries = t_orig.GetEntries()
  for i in range(n_entries):
    t_orig.GetEntry(i)
    if i <= int(args.split*n_entries):
      t_train.Fill()
    else:
      t_test.Fill()
  
  f_train.cd()
  t_train.Write("",rt.TObject.kOverwrite)
  f_train.Close()
  
  f_test.cd()
  t_test.Write("",rt.TObject.kOverwrite)
  f_test.Close()
  
  f_orig.Close()



