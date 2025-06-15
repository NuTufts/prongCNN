//#include "LArPIDInterface.h"

#include <iostream>
#include <getopt.h>

#include "larpid/model/TorchModel.h"

#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larflow/Reco/NuVertexCandidate.h"

#include "larpid/interface/LArPIDInterface.h"

void print_usage() {
    std::cout << "Usage: run_larpid [options] <merged_dlreco_file> <kpsreco_file> <model_script_file> <output_file>\n"
              << "\n"
              << "Required arguments:\n"
              << "  merged_dlreco_file   : Path to merged DLGen2 file with wire and shower images\n"
              << "  kpsreco_file         : Path to KPS reconstruction file with neutrino candidates\n"
              << "  model_script_file    : Path to scripted PyTorch model file (.pt)\n"
              << "  output_file          : Path for output ROOT file\n"
              << "\n"
              << "Options:\n"
              << "  -h, --help           : Show this help message\n"
              << "  -s, --no-shower      : Do not preserve shower pixels (default: preserve)\n"
              << "  -d, --debug          : Enable model debug mode (default: disabled)\n"
              << std::endl;
}

int main( int nargs, char** argv ) {

    std::cout << "LArPID CNN Inference Tool" << std::endl;

    // Parse command line arguments
    std::string merged_dlreco_file;
    std::string kpsreco_file;
    std::string model_script_file;
    std::string output_file;
    bool preserve_shower_pixels = true;
    bool model_debug_mode = false;

    // Parse optional arguments
    static struct option long_options[] = {
        {"help",      no_argument,       0, 'h'},
        {"no-shower", no_argument,       0, 's'},
        {"debug",     no_argument,       0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(nargs, argv, "hsd", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                print_usage();
                return 0;
            case 's':
                preserve_shower_pixels = false;
                break;
            case 'd':
                model_debug_mode = true;
                break;
            default:
                print_usage();
                return 1;
        }
    }

    // Check for required positional arguments
    if (optind + 4 != nargs) {
        std::cerr << "Error: Incorrect number of arguments\n" << std::endl;
        print_usage();
        return 1;
    }

    // Get required arguments
    merged_dlreco_file = argv[optind];
    kpsreco_file = argv[optind + 1];
    model_script_file = argv[optind + 2];
    output_file = argv[optind + 3];

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
    TFile* outfile = new TFile( output_file.c_str(), "new" ); // dont overwrite
    TTree* outtree = new TTree( "larpidTree", "Output of the LArPID CNN on neutrino candidate prongs");

    int nucand_num_vertices;
    std::vector< int > nucand_num_tracks;
    std::vector< int > nucand_num_showers;

    // for the track variables below: the first index is over nucand, second index is over tracks
    std::vector< std::vector<int> >   nucand_track_modelrun;       
    std::vector< std::vector<int> >   nucand_track_maxplanepixels;
    std::vector< std::vector<int> >   nucand_track_numgoodplanes;
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

    // for the shower variables below: first index is nucand, second index is over showers
    std::vector< std::vector<int> >   nucand_shower_modelrun;       
    std::vector< std::vector<int> >   nucand_shower_maxplanepixels;
    std::vector< std::vector<int> >   nucand_shower_numgoodplanes;
    std::vector< std::vector<float> > nucand_shower_electron_score; 
    std::vector< std::vector<float> > nucand_shower_photon_score;   // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_muon_score;     // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_proton_score;   // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_pion_score;     // first index is nucand, second index is over showers
    std::vector< std::vector<int> >   nucand_shower_pid;            // first index is nucand, second index is over showers

    std::vector< std::vector<float> > nucand_shower_purity;         // first index is nucand, second index is over showers
    std::vector< std::vector<float> > nucand_shower_completeness;   // first index is nucand, second index is over showers

    std::vector< std::vector<float> > nucand_shower_process_score;  // first index is nucand, second index is over showers
    std::vector< std::vector<int> >   nucand_shower_process;        // first index is nucand, second index is over showers

    // Create TTree branches for output data
    outtree->Branch("nucand_num_vertices", &nucand_num_vertices, "nucand_num_vertices/I");
    outtree->Branch("nucand_num_tracks", &nucand_num_tracks);
    outtree->Branch("nucand_num_showers", &nucand_num_showers);
    
    // Track branches
    outtree->Branch("nucand_track_modelrun",       &nucand_track_modelrun);
    outtree->Branch("nucand_track_maxplanepixels", &nucand_track_maxplanepixels);
    outtree->Branch("nucand_track_numgoodplanes",  &nucand_track_numgoodplanes);
    outtree->Branch("nucand_track_electron_score", &nucand_track_electron_score);
    outtree->Branch("nucand_track_photon_score",   &nucand_track_photon_score);
    outtree->Branch("nucand_track_muon_score",     &nucand_track_muon_score);
    outtree->Branch("nucand_track_proton_score",   &nucand_track_proton_score);
    outtree->Branch("nucand_track_pion_score",     &nucand_track_pion_score);
    outtree->Branch("nucand_track_pid",            &nucand_track_pid);
    outtree->Branch("nucand_track_purity",         &nucand_track_purity);
    outtree->Branch("nucand_track_completeness",   &nucand_track_completeness);
    outtree->Branch("nucand_track_process_score",  &nucand_track_process_score);
    outtree->Branch("nucand_track_process",        &nucand_track_process);
    
    // Shower branches
    outtree->Branch("nucand_shower_modelrun",       &nucand_shower_modelrun);
    outtree->Branch("nucand_shower_maxplanepixels", &nucand_shower_maxplanepixels);
    outtree->Branch("nucand_shower_numgoodplanes",  &nucand_shower_numgoodplanes);
    outtree->Branch("nucand_shower_electron_score", &nucand_shower_electron_score);
    outtree->Branch("nucand_shower_photon_score", &nucand_shower_photon_score);
    outtree->Branch("nucand_shower_muon_score", &nucand_shower_muon_score);
    outtree->Branch("nucand_shower_proton_score", &nucand_shower_proton_score);
    outtree->Branch("nucand_shower_pion_score", &nucand_shower_pion_score);
    outtree->Branch("nucand_shower_pid", &nucand_shower_pid);
    outtree->Branch("nucand_shower_purity", &nucand_shower_purity);
    outtree->Branch("nucand_shower_completeness", &nucand_shower_completeness);
    outtree->Branch("nucand_shower_process_score", &nucand_shower_process_score);
    outtree->Branch("nucand_shower_process", &nucand_shower_process);

    for (int ientry=0; ientry<nentries; ientry++) {

        std::cout << "Entry " << ientry << std::endl;

        ioman.read_entry(ientry);
        recotree->GetEntry(ientry);

        // Clear all branch vectors for new entry
        nucand_num_vertices = 0;
        nucand_num_tracks.clear();
        nucand_num_showers.clear();
        
        nucand_track_modelrun.clear();
        nucand_track_maxplanepixels.clear();
        nucand_track_numgoodplanes.clear();
        nucand_track_electron_score.clear();
        nucand_track_photon_score.clear();
        nucand_track_muon_score.clear();
        nucand_track_proton_score.clear();
        nucand_track_pion_score.clear();
        nucand_track_pid.clear();
        nucand_track_purity.clear();
        nucand_track_completeness.clear();
        nucand_track_process_score.clear();
        nucand_track_process.clear();
        
        nucand_shower_modelrun.clear();
        nucand_shower_maxplanepixels.clear();
        nucand_shower_numgoodplanes.clear();
        nucand_shower_electron_score.clear();
        nucand_shower_photon_score.clear();
        nucand_shower_muon_score.clear();
        nucand_shower_proton_score.clear();
        nucand_shower_pion_score.clear();
        nucand_shower_pid.clear();
        nucand_shower_purity.clear();
        nucand_shower_completeness.clear();
        nucand_shower_process_score.clear();
        nucand_shower_process.clear();

        int nvertices = (int)(nuvetoed_v->size());
        std::cout << "number of neutrino candidates: " << nvertices << std::endl;
        nucand_num_vertices = nvertices;

        if ( nvertices==0 ) {
          outtree->Fill();
          continue;
        }
        
        for (int ivtx=0; ivtx<nvertices; ivtx++ ) {
            auto& nuvtx = nuvetoed_v->at(ivtx);

            int ntracks  = nuvtx.track_v.size();
            int nshowers = nuvtx.shower_v.size();

            std::cout << "NuCandidate[" << ivtx << "]" << std::endl;
            std::cout << " ntracks=" << ntracks << std::endl;
            std::cout << " nshowers=" << nshowers << std::endl;
            
            // Store basic info for this candidate
            nucand_num_tracks.push_back(ntracks);
            nucand_num_showers.push_back(nshowers);
            
            // Initialize vectors for this nucand
            nucand_track_modelrun.push_back(std::vector<int>());
            nucand_track_maxplanepixels.push_back(std::vector<int>());
            nucand_track_numgoodplanes.push_back(std::vector<int>());
            nucand_track_electron_score.push_back(std::vector<float>());
            nucand_track_photon_score.push_back(std::vector<float>());
            nucand_track_muon_score.push_back(std::vector<float>());
            nucand_track_proton_score.push_back(std::vector<float>());
            nucand_track_pion_score.push_back(std::vector<float>());
            nucand_track_pid.push_back(std::vector<int>());
            nucand_track_purity.push_back(std::vector<float>());
            nucand_track_completeness.push_back(std::vector<float>());
            nucand_track_process_score.push_back(std::vector<float>());
            nucand_track_process.push_back(std::vector<int>());
            
            nucand_shower_modelrun.push_back(std::vector<int>());
            nucand_shower_maxplanepixels.push_back(std::vector<int>());
            nucand_shower_numgoodplanes.push_back(std::vector<int>());
            nucand_shower_electron_score.push_back(std::vector<float>());
            nucand_shower_photon_score.push_back(std::vector<float>());
            nucand_shower_muon_score.push_back(std::vector<float>());
            nucand_shower_proton_score.push_back(std::vector<float>());
            nucand_shower_pion_score.push_back(std::vector<float>());
            nucand_shower_pid.push_back(std::vector<int>());
            nucand_shower_purity.push_back(std::vector<float>());
            nucand_shower_completeness.push_back(std::vector<float>());
            nucand_shower_process_score.push_back(std::vector<float>());
            nucand_shower_process.push_back(std::vector<int>());

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
                int maxgoodplane = 0;
                for (int p=0; p<3; p++) {
                    int npixels = prong_vv.at(p).size();
                    std::cout << "  prong[" << p << "]: " << npixels << " pixels" << std::endl;
                    if ( npixels>maxgoodplane )
                        maxgoodplane = npixels;
                    if ( npixels>=10 )
                        num_good_planes += 1;
                }
                nucand_track_numgoodplanes[ivtx].push_back(num_good_planes);
                nucand_track_maxplanepixels[ivtx].push_back(maxgoodplane);

                if ( num_good_planes>=2 ) {
                    larpid::data::ModelOutput output = model.run_inference( prong_vv );
                    std::cout << "  ran model [track prong]: predicted PID = " << output.predictedPID << std::endl;

                    nucand_track_modelrun[ivtx].push_back(1);

                    // Store output into vectors for tracks
                    // Assuming classScores order is: electron, photon, muon, pion, proton
                    if (output.classScores.size() >= 5) {
                        nucand_track_electron_score[ivtx].push_back(output.classScores[0]);
                        nucand_track_photon_score[ivtx].push_back(output.classScores[1]);
                        nucand_track_muon_score[ivtx].push_back(output.classScores[2]);
                        nucand_track_pion_score[ivtx].push_back(output.classScores[3]);
                        nucand_track_proton_score[ivtx].push_back(output.classScores[4]);
                    } else {
                        // Default values if model output is unexpected
                        nucand_track_electron_score[ivtx].push_back(-999.0);
                        nucand_track_photon_score[ivtx].push_back(-999.0);
                        nucand_track_muon_score[ivtx].push_back(-999.0);
                        nucand_track_pion_score[ivtx].push_back(-999.0);
                        nucand_track_proton_score[ivtx].push_back(-999.0);
                    }
                    
                    nucand_track_pid[ivtx].push_back(output.predictedPID);
                    nucand_track_purity[ivtx].push_back(output.purity);
                    nucand_track_completeness[ivtx].push_back(output.completeness);
                    
                    if (output.processScores.size() > 0) {
                        nucand_track_process_score[ivtx].push_back(output.processScores[0]);
                    } else {
                        nucand_track_process_score[ivtx].push_back(-999.0);
                    }
                    nucand_track_process[ivtx].push_back(output.predictedProcess);
                }
                else {
                    std::cout << "  did not run model: num_good_planes (" << num_good_planes << ") < 2" << std::endl;
                    
                    // Store default values for tracks that couldn't be processed
                    nucand_track_modelrun[ivtx].push_back(0);
                    nucand_track_electron_score[ivtx].push_back(-999.0);
                    nucand_track_photon_score[ivtx].push_back(-999.0);
                    nucand_track_muon_score[ivtx].push_back(-999.0);
                    nucand_track_pion_score[ivtx].push_back(-999.0);
                    nucand_track_proton_score[ivtx].push_back(-999.0);
                    nucand_track_pid[ivtx].push_back(-999);
                    nucand_track_purity[ivtx].push_back(-999.0);
                    nucand_track_completeness[ivtx].push_back(-999.0);
                    nucand_track_process_score[ivtx].push_back(-999.0);
                    nucand_track_process[ivtx].push_back(-999);
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
                int maxgoodplane = 0;
                for (int p=0; p<3; p++) {
                    int npixels = prong_vv.at(p).size();
                    std::cout << "  prong[" << p << "]: " <<  npixels << " pixels" << std::endl;
                    if ( npixels>maxgoodplane )
                        maxgoodplane = npixels;
                    if ( npixels>=10 )
                        num_good_planes += 1;
                }
                nucand_shower_numgoodplanes[ivtx].push_back(num_good_planes);
                nucand_shower_maxplanepixels[ivtx].push_back(maxgoodplane);

                if ( num_good_planes>=2 ) {

                    larpid::data::ModelOutput output = model.run_inference( prong_vv );
                    std::cout << "Ran model [shower prong]: predicted PID = " << output.predictedPID << std::endl;
                    nucand_shower_modelrun[ivtx].push_back(1);

                    // Store output into vectors for showers
                    // Assuming classScores order is: electron, photon, muon, pion, proton
                    if (output.classScores.size() >= 5) {
                        nucand_shower_electron_score[ivtx].push_back(output.classScores[0]);
                        nucand_shower_photon_score[ivtx].push_back(output.classScores[1]);
                        nucand_shower_muon_score[ivtx].push_back(output.classScores[2]);
                        nucand_shower_pion_score[ivtx].push_back(output.classScores[3]);
                        nucand_shower_proton_score[ivtx].push_back(output.classScores[4]);
                    } else {
                        // Default values if model output is unexpected
                        nucand_shower_electron_score[ivtx].push_back(-999.0);
                        nucand_shower_photon_score[ivtx].push_back(-999.0);
                        nucand_shower_muon_score[ivtx].push_back(-999.0);
                        nucand_shower_pion_score[ivtx].push_back(-999.0);
                        nucand_shower_proton_score[ivtx].push_back(-999.0);
                    }
                    
                    nucand_shower_pid[ivtx].push_back(output.predictedPID);
                    nucand_shower_purity[ivtx].push_back(output.purity);
                    nucand_shower_completeness[ivtx].push_back(output.completeness);
                    
                    if (output.processScores.size() > 0) {
                        nucand_shower_process_score[ivtx].push_back(output.processScores[0]);
                    } else {
                        nucand_shower_process_score[ivtx].push_back(-999.0);
                    }
                    nucand_shower_process[ivtx].push_back(output.predictedProcess);

                }
                else {
                    std::cout << "  did not run model: num_good_planes (" << num_good_planes << ") < 2" << std::endl;
                    
                    // Store default values for showers that couldn't be processed
                    nucand_shower_modelrun[ivtx].push_back(0);
                    nucand_shower_electron_score[ivtx].push_back(-999.0);
                    nucand_shower_photon_score[ivtx].push_back(-999.0);
                    nucand_shower_muon_score[ivtx].push_back(-999.0);
                    nucand_shower_pion_score[ivtx].push_back(-999.0);
                    nucand_shower_proton_score[ivtx].push_back(-999.0);
                    nucand_shower_pid[ivtx].push_back(-999);
                    nucand_shower_purity[ivtx].push_back(-999.0);
                    nucand_shower_completeness[ivtx].push_back(-999.0);
                    nucand_shower_process_score[ivtx].push_back(-999.0);
                    nucand_shower_process[ivtx].push_back(-999);
                }
            }
        }
        
        // Fill the tree after processing all candidates
        outtree->Fill();

    }

    // Write output file
    outfile->cd();
    outtree->Write();
    outfile->Close();

    std::cout << "LArPID inference completed successfully" << std::endl;
    std::cout << "Output written to: " << output_file << std::endl;

    return 0;
}