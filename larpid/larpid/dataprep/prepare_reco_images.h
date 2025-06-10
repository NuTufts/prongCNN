#ifndef __LARPID_DATAPREP_PREPARE_RECO_IMAGES_H__
#define __LARPID_DATAPREP_PREPARE_RECO_IMAGES_H__

#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>

// ROOT headers
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"
#include "TChain.h"

// larlite headers
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/DataFormat/mcshower.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/shower.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"

// larflow KPS headers
#include "larflow/Reco/NuVertexCandidate.h"
#include "larflow/Reco/KPCluster.h"

// larutil headers
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

// larcv headers
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// ublarcvapp headers
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "ublarcvapp/MCTools/TruthShowerTrunkSCE.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/MCPixelPMap.h"

// larflow headers
#include "larflow/PrepFlowMatchData/FlowTriples.h"
#include "larflow/PrepFlowMatchData/CropPixData_t.h"

namespace larpid {
namespace dataprep {

  class PrepareRecoImages {
  
  public:
    PrepareRecoImages();
    ~PrepareRecoImages();
    
    // Configuration parameters
    struct Config {
      std::string recoFileList;
      std::string truthFileList;
      std::string outputFile;
      float vertexScoreCut = 0.8;
      int pixelThreshold = 10;
      int minPixelsPerPlane = 10;
      int minGoodPlanes = 2;
      int imageSize = 512;
      bool splitTrainTest = false;
      float trainSplitFraction = 0.9;
      int slabID = -1;
      int maxIterations = 10;
    };
    
    // Set configuration
    void setConfig(const Config& cfg) { fConfig = cfg; }
    
    // Main processing function
    void process();
    
  private:
    // File handling
    std::vector<std::pair<std::string,std::string>> matchFiles(
        const std::vector<std::string>& recoFiles,
        const std::vector<std::string>& truthFiles);
    
    std::vector<std::string> parseFileList(const std::string& filename);
    
    // KPS reco tree handling
    void setupKPSTree(TFile* kpsFile, TTree*& kpsTree, 
                      std::vector<larflow::reco::NuVertexCandidate>*& nuvertex_v,
                      int& run, int& subrun, int& event);
    
    // Vertex selection functions
    larflow::reco::NuVertexCandidate* selectBestVertex(
        const std::vector<larflow::reco::NuVertexCandidate>& vertices) const;
    
    // Track/shower quality functions
    bool goodTrack(const larlite::track& track) const;
    bool goodShower(const larlite::larflowcluster& shower) const;
    
    // Image processing functions
    bool checkPixelCounts(const std::vector<std::vector<larflow::prep::CropPixData_t>>& prong_vv) const;
    void extractSparseImage(const std::vector<std::vector<larflow::prep::CropPixData_t>>& prong_vv,
                           std::vector<std::vector<int>>& rows,
                           std::vector<std::vector<int>>& cols, 
			    std::vector<std::vector<float>>& adcs,
			    std::vector<std::vector<int>>& raw_rows,
			    std::vector<std::vector<int>>& raw_cols, 
			    std::vector<std::vector<float>>& raw_adcs ) const;
			    
    
    // Truth matching functions
    struct MCProngInfo {
      int pdgCode;
      int processClass;  // 0=primary, 1=secondary neutral parent, 2=secondary charged parent
      int trackID;       // maxPartTID
      float totalTruePixI;  // totNodePixI
      float purity;
      float completeness;
      float bestOtherCompleteness;
      float energy;
      float angle;
      float minEdgeDist;
      std::vector<int> pdgList;      // List of all PDG codes found
      std::vector<float> purityList; // List of purities for each PDG
    };
    
    MCProngInfo getMCProngParticle(
        const std::vector<std::vector<larflow::prep::CropPixData_t>>& sparseimg_vv,
        ublarcvapp::mctools::MCPixelPGraph& mcpg,
        ublarcvapp::mctools::MCPixelPMap& mcpm,
        const std::vector<larcv::Image2D>& adc_v,
        const larlite::event_mctrack& mctrack_v,
        const larlite::event_mcshower& mcshower_v,
        int plane,
        const TVector3& vtx3d);
    
    float checkCompleteness(
        larflow::prep::FlowTriples& flowTriples,
        const std::vector<larcv::Image2D>& adc_v,
        const std::vector<larcv::Image2D>& thrumu_v,
        const larlite::larflowcluster& prongCluster,
        const TVector3& cropPt,
        ublarcvapp::mctools::MCPixelPMap& mcpm,
        int mcTID,
        float truePixSum,
        float bestComp);
    
    float getBestOtherCompleteness(
        const std::vector<larflow::reco::NuVertexCandidate>* vertices,
        int vID, int tID, int sID,
        larflow::prep::FlowTriples& flowTriples,
        const std::vector<larcv::Image2D>& adc_v,
        const std::vector<larcv::Image2D>& thrumu_v,
        ublarcvapp::mctools::MCPixelPMap& mcpm,
        int mcTID,
        float truePartPixSum);
    
    // Geometry functions
    bool isFiducial(const TVector3& vtx, 
                    float edgeCut = 5.0, 
                    float downstreamCut = 15.0) const;
    
    float getMinEdgeDist(const TVector3& vtx) const;
    
    // Output handling
    void setupOutputTree();
    void fillOutputTree(const MCProngInfo& info,
                       const std::vector<std::vector<int>>& sparseRows,
                       const std::vector<std::vector<int>>& sparseCols,
                       const std::vector<std::vector<float>>& sparseADCs,
                       const std::vector<std::vector<int>>& sparseRawRows,
                       const std::vector<std::vector<int>>& sparseRawCols,
                       const std::vector<std::vector<float>>& sparseRawADCs,
                       int run, int subrun, int event,
                       int vtxid, int clusterid,
                       int isShower, int isSecondary);
    
    // Member variables
    Config fConfig;
    larutil::SpaceChargeMicroBooNE fSCE;
    std::map<int,int> _chargeDict;
    
    // Output file and tree
    TFile* fOutFile;
    TTree* fOutTree;
    
    // Output tree variables
    int fRun, fSubrun, fEvent, fVtxID, fClusterID;
    int fPDG, fProcessClass, fTrackID;
    float fTotalTruePixI, fPurity, fCompleteness, fBestOtherCompleteness;
    float fEnergy, fAngle, fMinEdgeDist;
    std::vector<int> fPDGList;
    std::vector<float> fPurityList;
    int fIsShower;
    int fIsSecondary;
    
    // Sparse image storage (for each plane)
    std::vector<std::vector<int>> fSparseRow;
    std::vector<std::vector<int>> fSparseCol;
    std::vector<std::vector<float>> fSparseADC;
    std::vector<std::vector<int>> fSparseRawRow;
    std::vector<std::vector<int>> fSparseRawCol;
    std::vector<std::vector<float>> fSparseRawADC;
  };

} // namespace dataprep
} // namespace larpid

#endif
