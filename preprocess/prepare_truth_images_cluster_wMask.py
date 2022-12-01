
import os,sys,argparse
import ROOT as rt
from larlite import larlite
from larlite import larutil
from ublarcvapp import ublarcvapp
from larcv import larcv
from larflow import larflow
from array import array
from math import sqrt as sqrt

parser = argparse.ArgumentParser("Prepare Prong CNN Images Training File")
parser.add_argument("-if", "--infiles", required=True, type=str, help="corsika paired file list")
parser.add_argument("-ia", "--arrayID", required=True, type=int, help="SLURM array ID")
parser.add_argument("-in", "--nfiles", required=True, type=int, help="number of input files to process")
parser.add_argument("-o", "--outfile", type=str, default="prongCNN_images_file.root", help="output file name")
parser.add_argument("-n", "--pixelWH", type=int, default=512, help="pixel width and height of image")
parser.add_argument("-t", "--pixelThresh", type=float, default=10., help="pixel threshold for image generation")
parser.add_argument("-m", "--minPixelCount", type=int, default=10, help="minimum number of pixels in each plane")
parser.add_argument("-s", "--split", action="store_true", help="split output into training and validation samples")
parser.add_argument("-f", "--splitfrac", type=float, default=0.8, help="fraction of input to use for training if splitting files")
args = parser.parse_args()


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

sce = larutil.SpaceChargeMicroBooNE()
mcNuVertexer = ublarcvapp.mctools.NeutrinoVertex()


outFile = rt.TFile(args.outfile,"RECREATE")

imageTree = rt.TTree("ImageTree","ImageTree")
nPixels = args.pixelWH*args.pixelWH
pdg = array('i', [0])
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
imageTree.Branch("pdg", pdg, 'pdg/I')
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

filepairs = []
files = open(args.infiles,"r")
iF = 0
for line in files:
  if iF < args.arrayID*args.nfiles:
    iF += 1
    continue
  if iF >= (args.arrayID + 1)*args.nfiles:
    break
  filepairs.append(line.replace("\n","").split("   "))
  iF += 1
files.close()


for filepair in filepairs:

  print("reading files: ", filepair)
    
  ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
  ioll.add_in_filename(filepair[1])
  ioll.open()

  iolcv = larcv.IOManager(larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward)
  iolcv.add_in_file(filepair[0])
  iolcv.reverse_all_products()
  iolcv.initialize()


  for iE in range(ioll.get_entries()):

    ioll.go_to(iE)
    iolcv.read_entry(iE)

    mctruth = ioll.get_data(larlite.data.kMCTruth, "generator")
    mcNuPos = mctruth.at(0).GetNeutrino().Nu().Position()
    mcNuVertex = mcNuVertexer.getPos3DwSCE(ioll, sce)
    trueVtxPos = rt.TVector3(mcNuVertex[0], mcNuVertex[1], mcNuVertex[2])

    if not isFiducial(trueVtxPos):
      continue

    gamma_tids = []
    mcshowers = ioll.get_data(larlite.data.kMCShower, "mcreco")
    for mcshower in mcshowers:
      if mcshower.PdgCode() == 22 and getDistance(mcNuPos, mcshower.Start()) < 1e-4 and isFiducial(mcshower.DetProfile()):
        gamma_tids.append(mcshower.TrackID())

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.set_adc_treename("wiremc")
    mcpg.buildgraph(iolcv, ioll)

    evtImage2D = iolcv.get_data(larcv.kProductImage2D, "wiremc")
    adc_v = evtImage2D.Image2DArray()

    flowTriples = larflow.prep.FlowTriples()

    for node in mcpg.node_v:

      if (node.tid == node.mtid and node.origin == 1 and node.process == "primary") or node.tid in gamma_tids:

        if node.E_MeV < 35.:
          continue

        if abs(node.pid) == 2212 and node.E_MeV < 60:
          continue

        #if abs(node.pid) not in [11,211,2212]:
        if abs(node.pid) not in [11,22,13,211,2212]:
          continue

        prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_wMask(adc_v, mcpg, ioll, node.tid, args.pixelThresh, args.pixelWH, args.pixelWH, abs(node.pid) in [11,22])

        skip = False
        for p in range(3):
          if prong_vv[p].size() < args.minPixelCount:
            skip = True
            break
        if skip:
          continue

        pdg[0] = abs(node.pid)

        iP = 0
        for pix in prong_vv[0]:
          plane0pix_row[iP] = pix.row
          plane0pix_col[iP] = pix.col
          plane0pix_val[iP] = pix.val
          iP += 1
        plane0_nPix[0] = iP
        iP = 0
        for pix in prong_vv[1]:
          plane1pix_row[iP] = pix.row
          plane1pix_col[iP] = pix.col
          plane1pix_val[iP] = pix.val
          iP += 1
        plane1_nPix[0] = iP
        iP = 0
        for pix in prong_vv[2]:
          plane2pix_row[iP] = pix.row
          plane2pix_col[iP] = pix.col
          plane2pix_val[iP] = pix.val
          iP += 1
        plane2_nPix[0] = iP
          
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

  ioll.close()
  iolcv.finalize()

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



