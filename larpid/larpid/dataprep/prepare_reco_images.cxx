#include "prepare_reco_images.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <getopt.h>
#include <set>

// ROOT includes
#include "TChain.h"
#include "TBranch.h"

// larlite includes
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/larflow3dhit.h"

// larflow includes  
#include "larflow/PrepFlowMatchData/FlowTriples.h"

namespace larpid {
namespace dataprep {

PrepareRecoImages::PrepareRecoImages() 
    : fOutFile(nullptr), fOutTree(nullptr) {
    // Initialize sparse image vectors
    fSparseRow.resize(3);
    fSparseCol.resize(3);
    fSparseADC.resize(3);
    fSparseRawRow.resize(3);
    fSparseRawCol.resize(3);
    fSparseRawADC.resize(3);

    // build PDG code -> electric charge map (matching Python chargeDict)
    _chargeDict.clear();
    _chargeDict[22] = 0;       // photon
    _chargeDict[11] = -1;      // electron
    _chargeDict[-11] = 1;      // positron
    _chargeDict[13] = -1;      // muon
    _chargeDict[-13] = 1;      // antimuon
    _chargeDict[15] = -1;      // tau
    _chargeDict[-15] = 1;      // antitau
    _chargeDict[12] = 0;       // electron neutrino
    _chargeDict[-12] = 0;      // electron antineutrino
    _chargeDict[14] = 0;       // muon neutrino
    _chargeDict[-14] = 0;      // muon antineutrino
    _chargeDict[16] = 0;       // tau neutrino
    _chargeDict[-16] = 0;      // tau antineutrino
    _chargeDict[211] = 1;      // pi+
    _chargeDict[-211] = -1;    // pi-
    _chargeDict[111] = 0;      // pi0
    _chargeDict[3122] = 0;     // lambda0
    _chargeDict[-3122] = 0;    // antilambda0
    _chargeDict[321] = 1;      // K+
    _chargeDict[-321] = -1;    // K-
    _chargeDict[310] = 0;      // K0S
    _chargeDict[130] = 0;      // K0L
    _chargeDict[3112] = -1;    // sigma-
    _chargeDict[3222] = 1;     // sigma+
    _chargeDict[-3112] = 1;    // antisigma-
    _chargeDict[-3222] = -1;   // antisigma+
    _chargeDict[3322] = 0;     // xi0
    _chargeDict[-3322] = 0;    // antixi0
    _chargeDict[3312] = -1;    // xi-
    _chargeDict[-3312] = 1;    // antixi-
    _chargeDict[2212] = 1;     // proton
    _chargeDict[-2212] = -1;   // antiproton
    _chargeDict[2112] = 0;     // neutron
    _chargeDict[-2112] = 0;    // antineutron
}

PrepareRecoImages::~PrepareRecoImages() {
    if (fOutFile) {
        if (fOutTree) fOutTree->Write();
        fOutFile->Close();
        delete fOutFile;
    }
}

std::vector<std::string> PrepareRecoImages::parseFileList(const std::string& filename) {
    std::vector<std::string> files;
    
    // Check if it's a file list or single file
    if (filename.find(".txt") != std::string::npos || 
        filename.find(".list") != std::string::npos) {
        // It's a list file
        std::ifstream infile(filename);
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty() && line[0] != '#') {
                files.push_back(line);
            }
        }
    } else {
        // Single file
        files.push_back(filename);
    }
    
    return files;
}

std::vector<std::pair<std::string,std::string>> PrepareRecoImages::matchFiles(
    const std::vector<std::string>& recoFiles,
    const std::vector<std::string>& truthFiles) {
    
    std::vector<std::pair<std::string,std::string>> matched;
    
    // Extract SAM tags from filenames for matching
    for (const auto& recoFile : recoFiles) {
        // Extract base name
        size_t pos = recoFile.find_last_of('/');
        std::string basename = (pos != std::string::npos) ? 
                              recoFile.substr(pos+1) : recoFile;
        
        // Extract SAM tags (run_subrun pattern)
        size_t start = basename.find("run");
        if (start == std::string::npos) continue;
        
        size_t end = basename.find("_", start + 10);
        if (end == std::string::npos) end = basename.find(".", start + 10);
        
        std::string samTag = basename.substr(start, end - start);

	std::cout << "samTag: " << samTag << std::endl;
        
        // Find matching truth file
        for (const auto& truthFile : truthFiles) {
            if (truthFile.find(samTag) != std::string::npos) {
                matched.push_back({recoFile, truthFile});
                break;
            }
        }
    }
    
    return matched;
}

bool PrepareRecoImages::isFiducial(const TVector3& vtx, 
                                  float edgeCut, 
                                  float downstreamCut) const {
    // TPC boundaries
    const float xmin = 0.0, xmax = 256.35;
    const float ymin = -116.5, ymax = 116.5;
    const float zmin = 0.0, zmax = 1036.8;
    
    return (vtx.X() > xmin + edgeCut && vtx.X() < xmax - edgeCut &&
            vtx.Y() > ymin + edgeCut && vtx.Y() < ymax - edgeCut &&
            vtx.Z() > zmin + edgeCut && vtx.Z() < zmax - downstreamCut);
}

float PrepareRecoImages::getMinEdgeDist(const TVector3& vtx) const {
    // TPC boundaries
    const float xmin = 0.0, xmax = 256.35;
    const float ymin = -116.5, ymax = 116.5;
    const float zmin = 0.0, zmax = 1036.8;
    
    float min_dist = std::min({
        vtx.X() - xmin,
        xmax - vtx.X(),
        vtx.Y() - ymin,
        ymax - vtx.Y(),
        vtx.Z() - zmin,
        zmax - vtx.Z()
    });
    
    return min_dist;
}

void PrepareRecoImages::setupOutputTree() {

    std::cout << "Prepare output file: " << fConfig.outputFile << std::endl;
  
    fOutFile = new TFile(fConfig.outputFile.c_str(), "RECREATE");
    fOutTree = new TTree("prongdata", "Prong CNN Training Data");
    
    // Event info branches
    fOutTree->Branch("run", &fRun, "run/I");
    fOutTree->Branch("subrun", &fSubrun, "subrun/I");
    fOutTree->Branch("event", &fEvent, "event/I");
    fOutTree->Branch("vtxid", &fVtxID, "vtxid/I");
    fOutTree->Branch("clusterid", &fClusterID, "clusterid/I");
    
    // Truth info branches
    fOutTree->Branch("pdg", &fPDG, "pdg/I");
    fOutTree->Branch("process_class", &fProcessClass, "process_class/I");
    fOutTree->Branch("track_id", &fTrackID, "track_id/I");
    fOutTree->Branch("total_true_pixI", &fTotalTruePixI, "total_true_pixI/F");
    fOutTree->Branch("purity", &fPurity, "purity/F");
    fOutTree->Branch("completeness", &fCompleteness, "completeness/F");
    fOutTree->Branch("best_other_completeness", &fBestOtherCompleteness, "best_other_completeness/F");
    fOutTree->Branch("energy", &fEnergy, "energy/F");
    fOutTree->Branch("angle", &fAngle, "angle/F");
    fOutTree->Branch("min_edge_dist", &fMinEdgeDist, "min_edge_dist/F");
    fOutTree->Branch("pdg_list", &fPDGList);
    fOutTree->Branch("purity_list", &fPurityList);
    fOutTree->Branch("isShower", &fIsShower, "isShower/I");
    fOutTree->Branch("isSecondary", &fIsSecondary, "isSecondary/I");
    
    // Sparse image branches for each plane
    for (int p = 0; p < 3; p++) {
        fOutTree->Branch(Form("plane%d_nPix", p), &fSparseRow[p]);
        fOutTree->Branch(Form("plane%dpix_row", p), &fSparseRow[p]);	
        fOutTree->Branch(Form("plane%dpix_col", p), &fSparseCol[p]);
        fOutTree->Branch(Form("plane%dpix_val", p), &fSparseADC[p]);
        fOutTree->Branch(Form("raw_plane%dpix_row", p), &fSparseRawRow[p]);
        fOutTree->Branch(Form("raw_plane%dpix_col", p), &fSparseRawCol[p]);
        fOutTree->Branch(Form("raw_plane%dpix_val", p), &fSparseRawADC[p]);
    }
}

void PrepareRecoImages::fillOutputTree(const MCProngInfo& info,
                                      const std::vector<std::vector<int>>& sparseRows,
                                      const std::vector<std::vector<int>>& sparseCols,
                                      const std::vector<std::vector<float>>& sparseADCs,
                                      const std::vector<std::vector<int>>& sparseRawRows,
                                      const std::vector<std::vector<int>>& sparseRawCols,
                                      const std::vector<std::vector<float>>& sparseRawADCs,
                                      int run, int subrun, int event,
                                      int vtxid, int clusterid,
                                      int isShower, int isSecondary) {
    // Fill event info
    fRun = run;
    fSubrun = subrun;
    fEvent = event;
    fVtxID = vtxid;
    fClusterID = clusterid;
    
    // Fill truth info
    fPDG = info.pdgCode;
    fProcessClass = info.processClass;
    fTrackID = info.trackID;
    fTotalTruePixI = info.totalTruePixI;
    fPurity = info.purity;
    fCompleteness = info.completeness;
    fBestOtherCompleteness = info.bestOtherCompleteness;
    fEnergy = info.energy;
    fAngle = info.angle;
    fMinEdgeDist = info.minEdgeDist;
    fPDGList = info.pdgList;
    fPurityList = info.purityList;
    fIsShower = isShower;
    fIsSecondary = isSecondary;
    
    // Fill sparse image data
    for (int p = 0; p < 3; p++) {
        // Clear vectors
        fSparseRow[p].clear();
        fSparseCol[p].clear();
        fSparseADC[p].clear();
        fSparseRawRow[p].clear();
        fSparseRawCol[p].clear();
        fSparseRawADC[p].clear();
        
        // Fill prong image data
        if (p < sparseRows.size()) {
            fSparseRow[p] = sparseRows[p];
            fSparseCol[p] = sparseCols[p];
            fSparseADC[p] = sparseADCs[p];
        }
        
        // Fill raw image data
        if (p < sparseRawRows.size()) {
            fSparseRawRow[p] = sparseRawRows[p];
            fSparseRawCol[p] = sparseRawCols[p];
            fSparseRawADC[p] = sparseRawADCs[p];
        }
    }
    
    fOutTree->Fill();
}

void PrepareRecoImages::setupKPSTree(TFile* kpsFile, TTree*& kpsTree,
                                     std::vector<larflow::reco::NuVertexCandidate>*& nuvertex_v,
                                     int& run, int& subrun, int& event) {
    kpsTree = (TTree*)kpsFile->Get("KPSRecoManagerTree");
    if (!kpsTree) {
        throw std::runtime_error("Could not find KPSRecoManagerTree in file");
    }
    
    // Setup branch addresses
    nuvertex_v = nullptr;
    kpsTree->SetBranchAddress("nuvetoed_v", &nuvertex_v);
    kpsTree->SetBranchAddress("run", &run);
    kpsTree->SetBranchAddress("subrun", &subrun);
    kpsTree->SetBranchAddress("event", &event);
}

larflow::reco::NuVertexCandidate* PrepareRecoImages::selectBestVertex(
    const std::vector<larflow::reco::NuVertexCandidate>& vertices) const {
    
    larflow::reco::NuVertexCandidate* bestVertex = nullptr;
    float bestScore = -1.0;
    
    for (const auto& vertex : vertices) {
        // Filter by keypoint type (0 = neutrino vertex)
        if (vertex.keypoint_type != 0) continue;
        
        // Apply vertex score cut
        if (vertex.netNuScore < fConfig.vertexScoreCut) continue;
        
        // Check if this is the best scoring vertex so far
        if (vertex.netNuScore > bestScore) {
            bestScore = vertex.netNuScore;
            bestVertex = const_cast<larflow::reco::NuVertexCandidate*>(&vertex);
        }
    }
    
    return bestVertex;
}

bool PrepareRecoImages::goodTrack(const larlite::track& track) const {
    // Quality cuts from Python code
    if (track.NumberTrajectoryPoints() < 2) return false;
    
    TVector3 vertex(track.Vertex().X(), track.Vertex().Y(), track.Vertex().Z());
    TVector3 end(track.End().X(), track.End().Y(), track.End().Z());
    
    float distance = (end - vertex).Mag();
    if (distance <= 1e-6) return false;
    
    return true;
}

bool PrepareRecoImages::goodShower(const larlite::larflowcluster& shower) const {
    // Basic shower quality cuts - can be expanded
    return shower.size() > 0;
}

bool PrepareRecoImages::checkPixelCounts(const std::vector<std::vector<larflow::prep::CropPixData_t>>& prong_vv) const {
    int goodPlanes = 0;
    
    // Check each wire plane (3 planes)
    for (int plane = 0; plane < 3; plane++) {
        int pixelCount = 0;
        
        // Count pixels above threshold
        if (prong_vv.size() > plane) {
            for (const auto& pixel : prong_vv[plane]) {
                if (pixel.val > fConfig.pixelThreshold) {
                    pixelCount++;
                }
            }
        }
        
        if (pixelCount >= fConfig.minPixelsPerPlane) {
            goodPlanes++;
        }
    }
    
    return goodPlanes >= fConfig.minGoodPlanes;
}

void PrepareRecoImages::extractSparseImage(const std::vector<std::vector<larflow::prep::CropPixData_t>>& prong_vv,
					   std::vector<std::vector<int>>& rows,
					   std::vector<std::vector<int>>& cols, 
					   std::vector<std::vector<float>>& adcs,
					   std::vector<std::vector<int>>& raw_rows,
					   std::vector<std::vector<int>>& raw_cols, 
					   std::vector<std::vector<float>>& raw_adcs ) const
{
    
    // Clear output vectors
    for (int p = 0; p < 3; p++) {
        rows[p].clear();
        cols[p].clear();
        adcs[p].clear();
        raw_rows[p].clear();
        raw_cols[p].clear();
        raw_adcs[p].clear();
    }
    
    // Extract sparse data for each plane
    for (int plane = 0; plane < 3; plane++) {
        if (prong_vv.size() <= plane) continue;
        
        for (const auto& pixel : prong_vv[plane]) {
            if (pixel.val > fConfig.pixelThreshold
		&& pixel.row>=0 && pixel.row<fConfig.imageSize
		&& pixel.col>=0 && pixel.col<fConfig.imageSize		
		) {
	      rows[plane].push_back(pixel.row);
	      cols[plane].push_back(pixel.col);
	      adcs[plane].push_back(pixel.val);
            }
        }
    }

    for (int plane = 3; plane < 6; plane++) {
        if (prong_vv.size() <= plane) continue;
        
        for (const auto& pixel : prong_vv[plane]) {
	  if (pixel.val > fConfig.pixelThreshold
	      && pixel.row>=0 && pixel.row<fConfig.imageSize
	      && pixel.col>=0 && pixel.col<fConfig.imageSize		
	      ) {
	    raw_rows[plane].push_back(pixel.row);
	    raw_cols[plane].push_back(pixel.col);
	    raw_adcs[plane].push_back(pixel.val);
	  }
        }
    }
}

// Main processing function
void PrepareRecoImages::process() {
    // Parse file lists
    std::vector<std::string> recoFiles = parseFileList(fConfig.recoFileList);
    std::vector<std::string> truthFiles = parseFileList(fConfig.truthFileList);

    std::cout << "Number of reco files: " << recoFiles.size() << std::endl;
    std::cout << "Number of truth files: " << truthFiles.size() << std::endl;
    
    // Handle SLURM array job
    if (fConfig.slabID >= 0) {
        if (fConfig.slabID < recoFiles.size()) {
            recoFiles = {recoFiles[fConfig.slabID]};
            truthFiles = {truthFiles[fConfig.slabID]};
        } else {
            std::cerr << "SLURM_ARRAY_TASK_ID " << fConfig.slabID 
                     << " exceeds number of files!" << std::endl;
            return;
        }
    }
    
    // Match files
    std::vector<std::pair<std::string,std::string>> matchedFiles;
    
    if ( recoFiles.size()>1 || truthFiles.size()>1 ) {
      auto matchedFiles = matchFiles(recoFiles, truthFiles);
    }
    else if ( recoFiles.size()==1 && truthFiles.size()==1 ) {
      matchedFiles.push_back( { recoFiles[0], truthFiles[0] } );
    }
    
    if (matchedFiles.empty()) {
      std::cerr << "No matched files found!" << std::endl;
      return;
    }

    std::cout << "Number of matched file pairs: " << matchedFiles.size() << std::endl;
    
    // Setup output
    setupOutputTree();
    
    // Process each file pair
    for (const auto& filePair : matchedFiles) {
        std::cout << "Processing: " << filePair.first << std::endl;
        std::cout << "    Truth: " << filePair.second << std::endl;
        
        // Open KPS reco file (ROOT TFile)
        TFile* kpsFile = TFile::Open(filePair.first.c_str(), "READ");
        if (!kpsFile || kpsFile->IsZombie()) {
            std::cerr << "Error opening KPS reco file: " << filePair.first << std::endl;
            continue;
        }
        
        // Setup KPS tree and branches
        TTree* kpsTree = nullptr;
        std::vector<larflow::reco::NuVertexCandidate>* nuvertex_v = nullptr;
        int run, subrun, event;
        setupKPSTree(kpsFile, kpsTree, nuvertex_v, run, subrun, event);
        
        // Setup truth file access (larlite + larcv)
        larlite::storage_manager io_truth(larlite::storage_manager::kREAD);
        io_truth.add_in_filename(filePair.second);
        io_truth.open();
        
        // Setup larcv IOManager for images
        larcv::IOManager io_larcv(larcv::IOManager::kREAD, "", 
                                 larcv::IOManager::kTickBackward);
        io_larcv.add_in_file(filePair.second);
        io_larcv.initialize();
        
        // Setup FlowTriples for image cropping
        larflow::prep::FlowTriples flowTriples;
        
        // Process events
        int nentries = kpsTree->GetEntries();
        int nprocessed = 0;
        
        for (int entry = 0; entry < nentries; entry++) {
            kpsTree->GetEntry(entry);
            io_truth.go_to(entry);
            io_larcv.read_entry(entry);
            
            std::cout << "Processing entry " << entry << " - Run:" << run 
                     << " Subrun:" << subrun << " Event:" << event << std::endl;
            
            if (!nuvertex_v || nuvertex_v->empty()) {
                std::cout << "No vertices found in entry " << entry << std::endl;
                continue;
            }
            
            // Select best vertex
            larflow::reco::NuVertexCandidate* nuVertex = selectBestVertex(*nuvertex_v);
            if (!nuVertex) {
                std::cout << "No good vertices found in entry " << entry << std::endl;
                continue;
            }
            
            // Check fiducial volume
            TVector3 vtxTVec3(nuVertex->pos[0], nuVertex->pos[1], nuVertex->pos[2]);
            if (!isFiducial(vtxTVec3)) {
                std::cout << "Vertex not in fiducial volume" << std::endl;
                continue;
            }
            
            std::cout << "Selected vertex at (" << nuVertex->pos[0] << ", " 
                     << nuVertex->pos[1] << ", " << nuVertex->pos[2] 
                     << ") with score " << nuVertex->netNuScore << std::endl;

	    // Build MC pixel graph and map for truth matching
	    ublarcvapp::mctools::MCPixelPGraph mcpg;
	    mcpg.set_adc_treename("wire");
	    mcpg.buildgraph(io_larcv, io_truth);
            
	    ublarcvapp::mctools::MCPixelPMap mcpm;
	    mcpm.set_adc_treename("wire");
	    mcpm.buildmap(io_larcv, mcpg);
            
            // Process tracks
            std::cout << "Number of tracks: " << nuVertex->track_v.size() << std::endl;
            for (size_t iT = 0; iT < nuVertex->track_v.size(); iT++) {
                const auto& track = nuVertex->track_v[iT];
                
                // Apply track quality cuts
                if (!goodTrack(track)) {
                    std::cout << "Track " << iT << " failed quality cuts" << std::endl;
                    continue;
                }
                
                // Get crop point (track end)
                TVector3 cropPt(track.End().X(), track.End().Y(), track.End().Z());
                std::cout << "Processing track " << iT << " with end point (" 
                         << cropPt.X() << ", " << cropPt.Y() << ", " << cropPt.Z() << ")" << std::endl;
                
                // Get ADC and thrumu images from larcv
                auto ev_adc = (larcv::EventImage2D*)io_larcv.get_data(larcv::kProductImage2D, "wire");
                auto ev_thrumu = (larcv::EventImage2D*)io_larcv.get_data(larcv::kProductImage2D, "thrumu");
                
                if (!ev_adc || !ev_thrumu) {
                    std::cout << "Could not get image data for track " << iT << std::endl;
                    continue;
                }
                
                const auto& adc_v = ev_adc->Image2DArray();
                const auto& thrumu_v = ev_thrumu->Image2DArray();
                
                if (adc_v.size() < 3 || thrumu_v.size() < 3) {
                    std::cout << "Insufficient image planes for track " << iT << std::endl;
                    continue;
                }
                
                // Get track hit cluster
                if (iT >= nuVertex->track_hitcluster_v.size()) {
                    std::cout << "No hit cluster for track " << iT << std::endl;
                    continue;
                }
                const auto& trackCluster = nuVertex->track_hitcluster_v[iT];
                
                // Generate cropped prong image using FlowTriples
                auto prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(
                    adc_v,                           // ADC images for 3 planes
                    thrumu_v,                        // Threshold muon images for 3 planes
                    trackCluster,                    // Hit cluster for this track
                    cropPt,                          // Crop center point (track end)
                    fConfig.pixelThreshold,          // Pixel threshold (default: 10.0)
                    fConfig.imageSize,               // Image width (default: 512)
                    fConfig.imageSize                // Image height (default: 512)
                );
                
                // Check pixel counts and plane requirements
                if (!checkPixelCounts(prong_vv)) {
                    std::cout << "Track " << iT << " failed pixel count requirements" << std::endl;
                    continue;
                }
                
                // Extract sparse image data
		// There are 6 sparse images.
		// The first three are the prong images, where we only include pixels associated to the track prong.
		// The second is a sparse image for all pixels around the crop point.
                std::vector<std::vector<int>> sparseRows(3), sparseCols(3), sparseRawRows(3), sparseRawCols(3);
                std::vector<std::vector<float>> sparseADCs(3), sparseRawADCs(3);
                extractSparseImage(prong_vv,
				   sparseRows,    sparseCols,    sparseADCs,
				   sparseRawRows, sparseRawCols, sparseRawADCs);				   
                
                // Get MC truth data for truth matching
                auto ev_mctrack = (larlite::event_mctrack*)io_truth.get_data(larlite::data::kMCTrack, "mcreco");
                auto ev_mcshower = (larlite::event_mcshower*)io_truth.get_data(larlite::data::kMCShower, "mcreco");
                
                if (!ev_mctrack || !ev_mcshower) {
                    std::cout << "Could not get MC truth data for track " << iT << std::endl;
                    continue;
                }
                
                
                // Use ADC images already declared above for completeness calculation
                
                // Perform MC truth matching (matching Python function signature)
                MCProngInfo mcInfo = getMCProngParticle(
                    prong_vv,             // Sparse prong image data (CropPixData_t format)
                    mcpg,                 // MC pixel graph
                    mcpm,                 // MC pixel map
                    adc_v,                // ADC images for completeness calculation
                    *ev_mctrack,          // MC track collection
                    *ev_mcshower,         // MC shower collection
                    0,                    // Plane (for bestOtherCompleteness)
                    vtxTVec3              // Vertex position (for edge distance)
                );
                
                mcInfo.minEdgeDist = getMinEdgeDist(cropPt);
                
                // Calculate best other completeness if we have a valid match
                if (mcInfo.totalTruePixI > 0.0) {
                    mcInfo.bestOtherCompleteness = getBestOtherCompleteness(
                        nuvertex_v, 0, iT, -1,  // vID=0 (current vertex), tID=iT, sID=-1 (not a shower)
                        flowTriples, adc_v, thrumu_v, mcpm, 
                        mcInfo.trackID, mcInfo.totalTruePixI);
                }
                                
                // Fill output tree
                int isTrackSecondary = (iT < nuVertex->track_isSecondary_v.size()) ? 
                                       nuVertex->track_isSecondary_v[iT] : 0;
                fillOutputTree(mcInfo, sparseRows, sparseCols, sparseADCs,
                              sparseRawRows, sparseRawCols, sparseRawADCs,
                              run, subrun, event, 0, iT,
                              0, isTrackSecondary);  // 0 for isShower (this is a track)
                
                std::cout << "Successfully processed track " << iT << std::endl;
            }
            
            // Process showers  
            std::cout << "Number of showers: " << nuVertex->shower_v.size() << std::endl;
            for (size_t iS = 0; iS < nuVertex->shower_v.size(); iS++) {
                const auto& shower = nuVertex->shower_v[iS];
                
                // Check shower quality - basic check since we don't have a goodShower equivalent
                if (shower.size() == 0) {
                    std::cout << "Shower " << iS << " has no hits" << std::endl;
                    continue;
                }
                
                // Get crop point (shower trunk vertex)
                if (iS >= nuVertex->shower_trunk_v.size()) {
                    std::cout << "No shower trunk for shower " << iS << std::endl;
                    continue;
                }
                TVector3 cropPt(nuVertex->shower_trunk_v[iS].Vertex().X(),
                               nuVertex->shower_trunk_v[iS].Vertex().Y(),
                               nuVertex->shower_trunk_v[iS].Vertex().Z());
                
                std::cout << "Processing shower " << iS << " with vertex point (" 
                         << cropPt.X() << ", " << cropPt.Y() << ", " << cropPt.Z() << ")" << std::endl;
                
                // Get ADC and thrumu images from larcv
                auto ev_adc = (larcv::EventImage2D*)io_larcv.get_data(larcv::kProductImage2D, "wire");
                auto ev_thrumu = (larcv::EventImage2D*)io_larcv.get_data(larcv::kProductImage2D, "thrumu");
                
                if (!ev_adc || !ev_thrumu) {
                    std::cout << "Could not get image data for shower " << iS << std::endl;
                    continue;
                }
                
                const auto& adc_v = ev_adc->Image2DArray();
                const auto& thrumu_v = ev_thrumu->Image2DArray();
                
                if (adc_v.size() < 3 || thrumu_v.size() < 3) {
                    std::cout << "Insufficient image planes for shower " << iS << std::endl;
                    continue;
                }
                
                // Generate cropped prong image using FlowTriples
                auto prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(
                    adc_v,                           // ADC images for 3 planes
                    thrumu_v,                        // Threshold muon images for 3 planes
                    shower,                          // Hit cluster for this shower
                    cropPt,                          // Crop center point (shower vertex)
                    fConfig.pixelThreshold,          // Pixel threshold (default: 10.0)
                    fConfig.imageSize,               // Image width (default: 512)
                    fConfig.imageSize                // Image height (default: 512)
                );
                
                // Check pixel counts and plane requirements
                if (!checkPixelCounts(prong_vv)) {
                    std::cout << "Shower " << iS << " failed pixel count requirements" << std::endl;
                    continue;
                }
                
                // Extract sparse image data
                std::vector<std::vector<int>> sparseRows(3), sparseCols(3), sparseRawRows(3), sparseRawCols(3);
                std::vector<std::vector<float>> sparseADCs(3), sparseRawADCs(3);
                extractSparseImage(prong_vv,
                                   sparseRows,    sparseCols,    sparseADCs,
                                   sparseRawRows, sparseRawCols, sparseRawADCs);
                
                // Get MC truth data for truth matching
                auto ev_mctrack = (larlite::event_mctrack*)io_truth.get_data(larlite::data::kMCTrack, "mcreco");
                auto ev_mcshower = (larlite::event_mcshower*)io_truth.get_data(larlite::data::kMCShower, "mcreco");
                
                if (!ev_mctrack || !ev_mcshower) {
                    std::cout << "Could not get MC truth data for shower " << iS << std::endl;
                    continue;
                }
                                
                // Perform MC truth matching
                MCProngInfo mcInfo = getMCProngParticle(
                    prong_vv,             // Sparse prong image data (CropPixData_t format)
                    mcpg,                 // MC pixel graph
                    mcpm,                 // MC pixel map
                    adc_v,                // ADC images for completeness calculation
                    *ev_mctrack,          // MC track collection
                    *ev_mcshower,         // MC shower collection
                    0,                    // Plane (for bestOtherCompleteness)
                    vtxTVec3              // Vertex position (for edge distance)
                );
                
                mcInfo.minEdgeDist = getMinEdgeDist(cropPt);
                
                // Calculate best other completeness if we have a valid match
                if (mcInfo.totalTruePixI > 0.0) {
                    mcInfo.bestOtherCompleteness = getBestOtherCompleteness(
                        nuvertex_v, 0, -1, iS,  // vID=0 (current vertex), tID=-1 (not a track), sID=iS
                        flowTriples, adc_v, thrumu_v, mcpm, 
                        mcInfo.trackID, mcInfo.totalTruePixI);
                }
                
                // Fill output tree
                int isShowerSecondary = (iS < nuVertex->shower_isSecondary_v.size()) ?
                                       nuVertex->shower_isSecondary_v[iS] : 0;
                fillOutputTree(mcInfo, sparseRows, sparseCols, sparseADCs,
                              sparseRawRows, sparseRawCols, sparseRawADCs,
                              run, subrun, event, 0, iS,
                              1, isShowerSecondary);  // 1 for isShower
                
                std::cout << "Successfully processed shower " << iS << std::endl;
            }
            
            nprocessed++;
            if (fConfig.maxIterations > 0 && nprocessed >= fConfig.maxIterations) {
                break;
            }
        }
        
        kpsFile->Close();
        delete kpsFile;
        io_truth.close();
        io_larcv.finalize();
    }
    
    // Write output
    if (fOutTree) {
        std::cout << "Writing " << fOutTree->GetEntries() << " entries to output file" << std::endl;
        fOutTree->Write();
    }
}

// Stub implementations for MC matching functions
// These need to be fully implemented based on the Python logic

PrepareRecoImages::MCProngInfo PrepareRecoImages::getMCProngParticle(
    const std::vector<std::vector<larflow::prep::CropPixData_t>>& sparseimg_vv,
    ublarcvapp::mctools::MCPixelPGraph& mcpg,
    ublarcvapp::mctools::MCPixelPMap& mcpm,
    const std::vector<larcv::Image2D>& adc_v,
    const larlite::event_mctrack& mctrack_v,
    const larlite::event_mcshower& mcshower_v,
    int plane,
    const TVector3& vtx3d) {
    
    MCProngInfo info;
    
    // Initialize data structures exactly like Python version
    std::map<int, float> particleDict;  // PDG -> accumulated pixel intensity
    std::map<int, std::vector<float>> trackDict;  // trackID -> [PDG, nodeIndex, accumulated pixI]
    float totalPixI = 0.0;  // Total pixel intensity across all pixels
    
    // Loop through sparse prong image data exactly like Python version
    for (int p = 0; p < 3; p++) {  // Loop over 3 wire planes (U, V, Y)
        if (p >= sparseimg_vv.size()) continue;
        
        int npixels = sparseimg_vv[p].size();  // Number of pixels in this plane
        for (int ipix = 0; ipix < npixels; ipix++) {  // Loop over each pixel
            const auto& pix = sparseimg_vv[p][ipix];  // Get pixel object
            
            // Access pixel coordinates and values exactly like Python
            int rawRow = pix.rawRow;
            int rawCol = pix.rawCol;
            float pixVal = pix.val;
            
            totalPixI += pixVal;  // Accumulate total intensity
            
            // Get MC truth content for this pixel
            auto pixContents = mcpm.getPixContent(p, rawRow, rawCol);
            
            // Loop through particles exactly like Python version:
            // for part in pixContents.particles:
            for (const auto& part : pixContents.particles) {
                // Accumulate by PDG (particle type) - Python logic:
                // if abs(part.pdg) in particleDict:
                //     particleDict[abs(part.pdg)] += pixContents.pixI
                // else:
                //     particleDict[abs(part.pdg)] = pixContents.pixI
                int absPDG = abs(part.pdg);
                if (particleDict.find(absPDG) != particleDict.end()) {
                    particleDict[absPDG] += pixContents.pixI;
                } else {
                    particleDict[absPDG] = pixContents.pixI;
                }
                
                // Track individual particle tracks - Python logic:
                // if part.tid in trackDict:
                //     trackDict[part.tid][2] += pixContents.pixI
                // else:
                //     trackDict[part.tid] = [part.pdg, part.nodeidx, pixContents.pixI]
                if (trackDict.find(part.tid) != trackDict.end()) {
                    trackDict[part.tid][2] += pixContents.pixI;  // Accumulate intensity
                } else {
                    trackDict[part.tid] = std::vector<float>{(float)part.pdg, (float)part.nodeidx, pixContents.pixI};
                }
            }
        }
    }
    
    // Find dominant particle exactly like Python version
    int maxPartPDG = 0;
    int maxPartNID = -1;        // Node ID
    int maxPartTID = -1;        // Track ID
    int maxPartProcClass = -1;  // Process class
    float maxPartI = 0.0;       // Max intensity
    float maxPartComp = 0.0;    // Completeness
    
    // Find track with maximum intensity
    for (const auto& track : trackDict) {
        if (track.second[2] > maxPartI) {
            maxPartI = track.second[2];           // Intensity
            maxPartPDG = (int)track.second[0];    // PDG
            maxPartNID = (int)track.second[1];    // Node index
            maxPartTID = track.first;             // Track ID
        }
    }
    
    // Calculate completeness exactly like Python version
    float totNodePixI = 0.0;
    if (maxPartI > 0.0) {
        auto maxPartNode = mcpg.findTrackID(maxPartTID);
        if (maxPartNode) {
            // Build set of all track IDs in the graph (Python: nodeTIDs)
            std::set<int> nodeTIDs;
            for (const auto& node : mcpg.node_v) {
                nodeTIDs.insert(node.tid);
            }
            
            // Process classification exactly like Python version
            if (maxPartNode->process == "primary") {
                maxPartProcClass = 0;
            } else if (nodeTIDs.find(maxPartNode->mtid) == nodeTIDs.end()) {
                // Mother track ID not in node list
                maxPartProcClass = 1;
            } else if (_chargeDict.find(maxPartNode->mother->pid) == _chargeDict.end()) {
                // Mother PDG not in charge dictionary
                maxPartProcClass = 1;
            } else if (_chargeDict[maxPartNode->mother->pid] == 0) {
                // Mother has neutral charge
                maxPartProcClass = 1;
            } else if (abs(_chargeDict[maxPartNode->mother->pid]) == 1) {
                // Mother has unit charge
                maxPartProcClass = 2;
            } else {
                // Shouldn't happen unless there's a mistake
                maxPartProcClass = -1;
            }
            
            // Calculate true pixel intensity for this particle
	    for (int p=0; p<3; p++) {
	      auto& pixels = maxPartNode->pix_vv[p];
	      for (size_t iP=0; iP<pixels.size()/2; iP++) {
		int row = ( pixels.at(2*iP)-2400 )/6;
		int col = pixels.at(2*iP+1);
		totNodePixI += adc_v.at(p).pixel(row, col);
	      }
	    }

	    if ( totNodePixI > 0.0 ) {
	      maxPartComp = maxPartI/totNodePixI; // Completeness = reco/true
	    }
	}//end of if maxNode found
    }//end of if maxPartI>0.0

    if ( maxPartComp>1.0 ) {
      throw std::runtime_error( "ERROR: prong completeness calculated to be >1");
    }
    
    // Build PDG and purity lists exactly like Python version
    std::vector<int> pdglist;
    std::vector<float> puritylist;
    for (const auto& part : particleDict) {
        pdglist.push_back(part.first);
        puritylist.push_back(part.second / totalPixI);  // Fraction of total intensity
    }
    
    // Fill MCProngInfo struct with all values from Python return
    info.pdgCode = maxPartPDG;
    info.processClass = maxPartProcClass;
    info.trackID = maxPartTID;
    info.totalTruePixI = totNodePixI;
    info.purity = (totalPixI > 0) ? maxPartI / totalPixI : 0.0;
    info.completeness = maxPartComp;
    info.pdgList = pdglist;
    info.purityList = puritylist;
    
    // Best other completeness will be calculated separately in the main loop
    info.bestOtherCompleteness = 0.0;
    
    // Calculate energy and angle from MC truth
    info.energy = 0.0;
    info.angle = 0.0;
    if (maxPartTID >= 0) {
        // Find corresponding MC track or shower
        for (const auto& mctrack : mctrack_v) {
            if (mctrack.TrackID() == maxPartTID) {
                if (mctrack.size() > 0) {
                    info.energy = mctrack[0].E();  // Initial energy
                    TVector3 trackDir(mctrack[0].Px(), mctrack[0].Py(), mctrack[0].Pz());
                    info.angle = trackDir.Theta();
                }
                break;
            }
        }
        
        // Check showers if not found in tracks
        if (info.energy == 0.0) {
            for (const auto& mcshower : mcshower_v) {
                if (mcshower.TrackID() == maxPartTID) {
                    info.energy = mcshower.Start().E();
                    TVector3 showerDir(mcshower.Start().Px(), mcshower.Start().Py(), mcshower.Start().Pz());
                    info.angle = showerDir.Theta();
                    break;
                }
            }
        }
    }
    
    return info;
}

float PrepareRecoImages::checkCompleteness(
    larflow::prep::FlowTriples& flowTriples,
    const std::vector<larcv::Image2D>& adc_v,
    const std::vector<larcv::Image2D>& thrumu_v,
    const larlite::larflowcluster& prongCluster,
    const TVector3& cropPt,
    ublarcvapp::mctools::MCPixelPMap& mcpm,
    int mcTID,
    float truePixSum,
    float bestComp) {
    
    // Make cropped prong image exactly like Python version
    auto prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(
        adc_v, thrumu_v, prongCluster, cropPt, 
        fConfig.pixelThreshold, fConfig.imageSize, fConfig.imageSize);
    
    float matchedSum = 0.0;
    
    // Loop through all 3 planes
    for (int p = 0; p < 3; p++) {
        if (p >= prong_vv.size()) continue;
        
        const auto& prong_v = prong_vv[p];
        
        // Loop through all pixels in this plane
        for (size_t ipix = 0; ipix < prong_v.size(); ipix++) {
            const auto& pix = prong_v.at(ipix);
            
            // Get MC truth content for this pixel
            auto pixContents = mcpm.getPixContent(p, pix.rawRow, pix.rawCol);
            
            // Check if this pixel contains the particle we're looking for
            for (const auto& part : pixContents.particles) {
                if (part.tid == mcTID) {
                    matchedSum += pixContents.pixI;
                }
            }
        }
    }
    
    // Calculate completeness
    float comp = (truePixSum > 0) ? matchedSum / truePixSum : 0.0;
    
    // Return the better of the two completeness values
    return (comp > bestComp) ? comp : bestComp;
}

float PrepareRecoImages::getBestOtherCompleteness(
    const std::vector<larflow::reco::NuVertexCandidate>* vertices,
    int vID, int tID, int sID,
    larflow::prep::FlowTriples& flowTriples,
    const std::vector<larcv::Image2D>& adc_v,
    const std::vector<larcv::Image2D>& thrumu_v,
    ublarcvapp::mctools::MCPixelPMap& mcpm,
    int mcTID,
    float truePartPixSum) {
    
    float bestComp = 0.0;
    
    // Loop through all vertices
    for (size_t iV = 0; iV < vertices->size(); iV++) {
        const auto& vertex = (*vertices)[iV];
        
        // Loop through all tracks in this vertex
        for (size_t iT = 0; iT < vertex.track_hitcluster_v.size(); iT++) {
            // Skip if this is the current track we're checking against
            if (iV == vID && iT == tID) continue;
            
            // Check track quality
            if (!goodTrack(vertex.track_v[iT])) continue;
            
            // Get crop point (track end)
            TVector3 cropPt(vertex.track_v[iT].End().X(),
                           vertex.track_v[iT].End().Y(),
                           vertex.track_v[iT].End().Z());
            
            // Check completeness for this track
            bestComp = checkCompleteness(flowTriples, adc_v, thrumu_v,
                                       vertex.track_hitcluster_v[iT], cropPt,
                                       mcpm, mcTID, truePartPixSum, bestComp);
        }
        
        // Loop through all showers in this vertex
        for (size_t iS = 0; iS < vertex.shower_v.size(); iS++) {
            // Skip if this is the current shower we're checking against
            if (iV == vID && iS == sID) continue;
            
            // Get crop point (shower trunk vertex)
            if (iS >= vertex.shower_trunk_v.size()) continue;
            
            TVector3 cropPt(vertex.shower_trunk_v[iS].Vertex().X(),
                           vertex.shower_trunk_v[iS].Vertex().Y(),
                           vertex.shower_trunk_v[iS].Vertex().Z());
            
            // Check completeness for this shower
            bestComp = checkCompleteness(flowTriples, adc_v, thrumu_v,
                                       vertex.shower_v[iS], cropPt,
                                       mcpm, mcTID, truePartPixSum, bestComp);
        }
    }
    
    return bestComp;
}

} // namespace dataprep
} // namespace larpid

// Main function
int main(int argc, char** argv) {
    larpid::dataprep::PrepareRecoImages processor;
    larpid::dataprep::PrepareRecoImages::Config config;
    
    // Parse command line arguments
    static struct option long_options[] = {
        {"recofiles", required_argument, 0, 'r'},
        {"truthfiles", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {"vertex-score-cut", required_argument, 0, 'v'},
        {"pixel-threshold", required_argument, 0, 'p'},
        {"min-pixels", required_argument, 0, 'm'},
        {"min-planes", required_argument, 0, 'n'},
        {"split", no_argument, 0, 's'},
        {"max-iterations", required_argument, 0, 'i'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "r:t:o:v:p:m:n:si:h", 
                           long_options, &option_index)) != -1) {
        switch (c) {
            case 'r':
                config.recoFileList = optarg;
                break;
            case 't':
                config.truthFileList = optarg;
                break;
            case 'o':
                config.outputFile = optarg;
                break;
            case 'v':
                config.vertexScoreCut = std::stof(optarg);
                break;
            case 'p':
                config.pixelThreshold = std::stoi(optarg);
                break;
            case 'm':
                config.minPixelsPerPlane = std::stoi(optarg);
                break;
            case 'n':
                config.minGoodPlanes = std::stoi(optarg);
                break;
            case 's':
                config.splitTrainTest = true;
                break;
            case 'i':
                config.maxIterations = std::stoi(optarg);
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]\n"
                         << "Options:\n"
                         << "  -r, --recofiles FILE      Reconstruction file list\n"
                         << "  -t, --truthfiles FILE     Truth/DLReco file list\n"
                         << "  -o, --output FILE         Output ROOT file\n"
                         << "  -v, --vertex-score-cut F  Vertex score cut (default: 0.8)\n"
                         << "  -p, --pixel-threshold I   Pixel ADC threshold (default: 10)\n"
                         << "  -m, --min-pixels I        Min pixels per plane (default: 10)\n"
                         << "  -n, --min-planes I        Min good planes (default: 2)\n"
                         << "  -s, --split               Split train/test\n"
                         << "  -i, --max-iterations I    Max events to process\n"
                         << "  -h, --help                Show this help\n";
                return 0;
            default:
                return 1;
        }
    }
    
    // Check for SLURM array job
    const char* slurm_id = std::getenv("SLURM_ARRAY_TASK_ID");
    if (slurm_id) {
        config.slabID = std::stoi(slurm_id);
    }
    
    // Validate required arguments
    if (config.recoFileList.empty() || config.truthFileList.empty() || 
        config.outputFile.empty()) {
        std::cerr << "Error: Missing required arguments\n";
        return 1;
    }
    
    // Run processing
    processor.setConfig(config);
    processor.process();
    
    return 0;
}
