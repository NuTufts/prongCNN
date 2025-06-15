//#include "LArPIDInterface.h"

#include <iostream>

#include "larpid/model/TorchModel.h"

#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/Reco/NuVertexCandidate.h"

#include "larpid/interface/LArPIDInterface.h"

int main( int nargs, char** argv ) {

    std::cout << "Test Interface Class" << std::endl;

    std::string merged_dlreco_file = argv[1];
    std::string kpsreco_file = argv[2];
    std::string model_script_file = argv[3];
    std::string output_file = argv[4];
    bool preserve_shower_pixels = true;
    bool model_debug_mode = false;

    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "[Configuration]" << std::endl;
    std::cout << "  merged_dlreco file: " << merged_dlreco_file << std::endl;
    std::cout << "  kpsreco file: " << kpsreco_file << std::endl;
    std::cout << "  model script file: " << model_script_file << std::endl;
    std::cout << "  output file: " << output_file << std::endl;
    std::cout << "  preserve shower pixels: " << preserve_shower_pixels << std::endl;
    std::cout << "  model debug mode: " << model_debug_mode << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;

    larpid::model::TorchModel model( model_script_file, model_debug_mode );
 
    larcv::IOManager ioman( larcv::IOManager::kREAD, "ioman", larcv::IOManager::kTickBackward );
    ioman.set_verbosity( larcv::msg::kINFO );
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

    std::vector< larflow::reco::NuVertexCandidate >* nuvetoed_v = nullptr;
    recotree->SetBranchAddress( "nuvetoed_v", &nuvetoed_v );

    // output file
    TFile* outfile = new TFile( output_file.c_str(), "recreate" );
    TTree* outtree = new TTree( "larpidTree", "Output of the LArPID CNN on neutrino candidate prongs");

    std::vector< int > nucand_num_vertices;
    std::vector< int > nucand_num_tracks;
    std::vector< int > nucand_num_showers;

    std::vector< std::vector<float> > nucand_track_electron_score; // first index is nucand, second index is over tracks
    std::vector< std::vector<float> > nucand_track_photon_score;   // first index is nucand, second index is over tracks
    std::vector< std::vector<float> > nucand_track_muon_score;     // first index is nucand, second index is over tracks
    std::vector< std::vector<float> > nucand_track_proton_score;   // first index is nucand, second index is over tracks
    std::vector< std::vector<float> > nucand_track_pion_score;     // first index is nucand, second index is over tracks
    std::vector< std::vector<int> >   nucand_track_pid;            // first index is nucand, second index is over tracks

    std::vector< std::vector<float> > nucand_track_purity;         // first index is nucand, second index is over tracks
    std::vector< std::vector<float> > nucand_track_completeness;   // first index is nucand, second index is over tracks

    std::vector< std::vector<float> > nucand_track_process_score;  // first index is nucand, second index is over tracks
    std::vector< std::vector<int> >   nucand_track_process;        // first index is nucand, second index is over tracks

    std::vector< std::vector<float> > nucand_shower_electron_score; // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_photon_score;   // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_muon_score;     // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_proton_score;   // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_pion_score;     // first index is nucand, second index is over showers
    std::vector< std::vector<int> >   nucand_shower_pid;            // first index is nucand, second index is over showers

    std::vector< std::vector<float> > nucand_shower_purity;         // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_completeness;   // first index is nucand, second index is over showers

    std::vector< std::vector<float> > nucand_shower_process_score;  // first index is nucand, second index is over showers
    std::vector< std::vector<int> >   nucand_shower_process;        // first index is nucand, second index is over showers

    for (int ientry=0; ientry<nentries; ientry++) {

        std::cout << "Entry " << ientry << std::endl;

        ioman.read_entry(ientry);
        recotree->GetEntry(ientry);

        // auto ev_adc    = (larcv::EventImage2D*)ioman.get_data(larcv::kProductImage2D, "wire" );
        // auto ev_thrumu = (larcv::EventImage2D*)ioman.get_data(larcv::kProductImage2D, "thrumu" );

        // auto& adc_v    = ev_adc->as_vector();
        // auto& thrumu_v = ev_thrumu->as_vector();

        int nvertices = (int)(nuvetoed_v->size());
        std::cout << "number of neutrino candidates: " << nvertices << std::endl;

        if ( nvertices==0 ) {
          continue;
        }
        
        for (int ivtx=0; ivtx<nvertices; ivtx++ ) {
            auto& nuvtx = nuvetoed_v->at(ivtx);

            int ntracks  = nuvtx.track_v.size();
            int nshowers = nuvtx.shower_v.size();

            std::cout << "NuCandidate[" << ivtx << "]" << std::endl;
            std::cout << " ntracks=" << ntracks << std::endl;
            std::cout << " nshowers=" << nshowers << std::endl;

            for (int itrack=0; itrack<ntracks; itrack++ ) {

                auto& hitcluster = nuvtx.track_hitcluster_v.at(itrack);
                auto& track = nuvtx.track_v.at(itrack);

                int npts = track.NumberTrajectoryPoints();
                TVector3 endpt = track.LocationAtPoint(npts-1);
        
                // std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
                //  = larpid::interface::make_cropped_initial_sparse_prong_image_reco( adc_v, 
                //      thrumu_v, hitcluster, endpt, 10.0, 512, 512 );
                std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
                 = larpid::interface::make_prongCNN_input_sparse_images( ioman, hitcluster, endpt, preserve_shower_pixels );

                std::cout << "track[" << itrack << "]" << std::endl;
                int num_good_planes = 0;
                for (int p=0; p<3; p++) {
                    std::cout << "  prong[" << p << "]: " << prong_vv.at(p).size() << " pixels" << std::endl;
                    if ( prong_vv.at(p).size()>=10 )
                        num_good_planes += 1;
                }

                if ( num_good_planes>=2 ) {
                    larpid::data::ModelOutput output = model.run_inference( prong_vv );
                    std::cout << "  ran model [track prong]: predicted PID = " << output.predictedPID << std::endl;
                }
                else {
                    std::cout << "  did not run model: num_good_planes (" << num_good_planes << ") < " << num_good_planes << std::endl;
                }
            }

            for (int ishower=0; ishower<nshowers; ishower++ ) {

                auto& hitcluster = nuvtx.shower_v.at(ishower);
                int npts = hitcluster.size();
                larlite::track& shower_trunk = nuvtx.shower_trunk_v.at(ishower);
                TVector3 startpt = shower_trunk.LocationAtPoint(0);
        
                // std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
                //  = larpid::interface::make_cropped_initial_sparse_prong_image_reco( adc_v, 
                //      thrumu_v, hitcluster, startpt, 10.0, 512, 512 );
                std::vector< std::vector<larpid::data::CropPixData_t> > prong_vv
                 = larpid::interface::make_prongCNN_input_sparse_images( ioman, hitcluster, startpt, preserve_shower_pixels );

                std::cout << "shower[" << ishower << "] npts=" << npts << std::endl;
                int num_good_planes = 0;
                for (int p=0; p<3; p++) {
                    std::cout << "  prong[" << p << "]: " << prong_vv.at(p).size() << " pixels" << std::endl;
                    if ( prong_vv.at(p).size()>=10 )
                        num_good_planes += 1;
                }

                if ( num_good_planes>=2 ) {

                    larpid::data::ModelOutput output = model.run_inference( prong_vv );
                    std::cout << "Ran model [shower prong]: predicted PID = " << output.predictedPID << std::endl;

                }
            }
        }

    }


    // Test basic interface functionality
    std::cout << "Interface test completed successfully" << std::endl;



    return 0;
}