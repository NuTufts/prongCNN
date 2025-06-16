#ifndef __LARPID_DATA_NUVERTEX__
#define __LARPID_DATA_NUVERTEX__

#include <string>
#include <vector>

/**
 * @brief A minimized version of larflow::reco::NuVertexCandidate
 * 
 * Meant to store larflowreco output data read from HDF5 file.
 * No ROOT.
 * 
 */

namespace larpid {
namespace data {

    class NuVertex {

    public:

        NuVertex() {};
        virtual ~NuVertex() {};
        
        std::string keypoint_producer;  ///< name of tree containing keypoints used to seed candidates
        int keypoint_index;             ///< index of vertex candidate in container above
        int keypoint_type;              ///< keypoint type
        std::vector<float> pos;         ///< keypoint position
        int row;                        ///< vertex row
        int tick;                       ///< vertex tick
        std::vector<int> col_v;         ///< image columns
        float netScore;                 ///< keypoint score (max score if vertex formed from merged candidates)
        float netNuScore;               ///< keypoint neutrino score (max score, considering only neutrino keypoint types, if formed from merged candidates)

        // TRACK PRONGS AND VARIABLES
        std::vector< std::vector< std::vector<float> > >  track_v;     ///< track candidates: list of 3d points
        std::vector< std::vector< std::vector<float> > >  track_hitcluster_v;  ///< track candidates: list of 3d points
        std::vector<float>           track_len_v;       ///< length of track
        std::vector< std::vector<float> > track_dir_v;  ///< direction of track, using points near vertex
        std::vector<int>             track_isSecondary_v; ///< 1 if added by NuVertexAddSecondaries, 0 otherwise
        
        std::vector< std::vector< std::vector<float> > > shower_v; ///< shower candidates
        std::vector< std::vector< std::vector<float> > > shower_trunk_v;   ///< line for shower trunk for plotting
        std::vector< std::vector<float> > shower_plane_pixsum_vv; ///< pixel sum of showers, a value for each plane
        std::vector< std::vector<float> > shower_plane_mom_vv; ///< energy of showers, a value for each plane
        std::vector< std::vector<float> > shower_plane_dqdx_vv;  ///< dqdx of shower trunk, a value for each plane
        std::vector<int> shower_isSecondary_v; ///< 1 if added by NuVertexAddSecondaries, 0 otherwise

        /** @brief comparator to sort candidates by highest score */
        bool operator<(const NuVertex& rhs) const {
        if ( netScore>rhs.netScore ) return true;
        return false;
        };

    };

}
}

#endif