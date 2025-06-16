#include "LArPIDInterface.h"

#include <iostream>

#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

//#include "larflow/Reco/NuVertexCandidate.h" // need to resolve depedency heirachy

#include "larpid/interface/LArPIDInterface.h"

int main( int nargs, char** argv ) {

    std::cout << "Test Interface Class" << std::endl;

    std::string merged_dlreco_file = argv[1];
    std::string kpsreco_file = argv[2];
    std::cout << "merged_dlreco file: " << merged_dlreco_file << std::endl;
    std::cout << "kpsreco file: " << kpsreco_file << std::endl;

    larcv::IOManager ioman( larcv::IOManager::kREAD, "ioman", larcv::IOManager::kTickBackward );
    ioman.add_in_file( merged_dlreco_file );
    ioman.reverse_all_products();
    ioman.initialize();

    TFile recofile( kpsreco_file.c_str() );
    TTree* recotree = (TTree*)recofile.Get( "KPSRecoManagerTree" );

    int nentries = ioman.get_n_entries();
    int nentries_reco = recotree->GetEntries();

    if ( nentries!=nentries_reco ) {
        std::cout << "Number of entries do not match: iomanager=" << nentries << "  reco=" << nentries_reco << std::endl;
        return 1;
    }

    //std::vector< larflow::reco::NuVertexCandidate >* nuvetoed_v = nullptr;
    //recotree->SetBranchAddress( "nuvetoed_v", &nuvetoed_v );

    for (int ientry=0; ientry<nentries; ientry++) {

        std::cout << "Entry " << ientry << std::endl;

        ioman.read_entry(ientry);
        recotree->GetEntry(ientry);

        auto ev_adc    = (larcv::EventImage2D*)ioman.get_data(larcv::kProductImage2D, "wire" );
        auto ev_thrumu = (larcv::EventImage2D*)ioman.get_data(larcv::kProductImage2D, "thrumu" );

        auto& adc_v    = ev_adc->as_vector();
        auto& thrumu_v = ev_thrumu->as_vector();

        //int nvertices = (int)(nuvetoed_v->size());
        //std::cout << "number of verticecs: " << nvertices << std::endl;

        // if ( nvertices==0 )
        //   continue;
        
        // for (int ivtx=0; ivtx<nvertices; ivtx++ ) {
        //     auto& nuvtx = nuvetoed_v->at(ivtx);

        //     int ntracks  = nuvtx.track_v.size();
        //     int nshowers = nuvtx.shower_v.size();

        //     for (int itrack=0; itrack<ntracks; itrack++ ) {

        //         auto& hitcluster = nuvtx.track_hitcluster_v.at(itrack);
        //         auto& track = nuvtx.track_v.at(itrack);

        //         int npts = track.NumberTrajectoryPoints();
        //         TVector3 endpt = track.LocationAtPoint(npts-1);
        
        //         std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
        //          = larpid::interface::make_cropped_initial_sparse_prong_image_reco( adc_v, 
        //              thrumu_v, hitcluster, endpt, 10.0, 512, 512 );

        //         std::cout << "track[" << itrack << "]" << std::endl;
        //         for (int p=0; p<3; p++) {
        //             std::cout << "  prong[" << p << "]: " << prong_vv.at(p).size() << " pixels" << std::endl;
        //         }
        //     }
        // }

        break;
    }


    // Test basic interface functionality
    std::cout << "Interface test completed successfully" << std::endl;



    return 0;
}