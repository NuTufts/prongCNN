
#ifndef LARPID_INTERFACE_LARPIDINTERFACE_H
#define LARPID_INTERFACE_LARPIDINTERFACE_H

#include <cmath>

#include "TVector3.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/larflow3dhit.h"

#include "larpid/data/CropPixData_t.h"

namespace larpid {
namespace interface {

  struct ParticleInfo {
    TVector3 cropPoint;
    larlite::larflowcluster larflowProng;
    ParticleInfo() {}
    ParticleInfo(TVector3 pt) : cropPoint(pt) {}
  };

  std::vector< std::vector<larpid::data::CropPixData_t> >
    make_cropped_initial_sparse_prong_image_reco( const std::vector<larcv::Image2D>& adc_v,
                                                  const std::vector<larcv::Image2D>& thrumu_v,
                                                  const larlite::larflowcluster& prong,
                                                  const TVector3& cropCenter,
                                                  float threshold, int rowSpan, int colSpan );

  std::vector< std::vector<larpid::data::CropPixData_t> >
  make_prongCNN_input_sparse_images( larcv::IOManager& iolcv,
                                     const larlite::larflowcluster& prong,
                                     const TVector3& cropCenter, 
                                     bool preserve_shower_pixels=false,
                                     float threshold=10.0, int rowSpan=512, int colSpan=512,
                                     std::string wireimg_treename="wire",
                                     std::string outoftime_treename="thrumu" );

  void getRecoImageBounds( std::vector< std::vector<int> >& imgBounds,
                           const std::vector<larcv::Image2D>& adc_v,
                           const larlite::larflowcluster& prong,
                           const TVector3& cropCenter, int rowSpan, int colSpan );

  void fillProngImagesFromReco( std::vector< std::vector<larpid::data::CropPixData_t> >& sparseimg_vv,
                                const float& threshold,
                                const std::vector<larcv::Image2D>& adc_v,
                                const std::vector<larcv::Image2D>& thrumu_v,
                                const larlite::larflowcluster& prong,
                                const std::vector< std::vector<int> >& imgBounds );

  void fillProngImagesFromRecoAndKeepShowers(std::vector< std::vector<larpid::data::CropPixData_t> >& sparseimg_vv,
                                             const float& threshold,
                                             const std::vector<larcv::Image2D>& adc_v,
                                             const std::vector<larcv::Image2D>& thrumu_v,
                                             const std::vector<const larcv::Image2D*>& showerimg_v,
                                             const larlite::larflowcluster& prong,
                                             const std::vector< std::vector<int> >& imgBounds);

  void fillContextImages( std::vector< std::vector<larpid::data::CropPixData_t> >& sparseimg_vv,
                          const float& threshold,
                          const std::vector<larcv::Image2D>& adc_v,
                          const std::vector<larcv::Image2D>& thrumu_v,
                          const std::vector< std::vector<int> >& imgBounds );
  
  void fillContextImagesAndKeepShowers( std::vector< std::vector<larpid::data::CropPixData_t> >& sparseimg_vv,
                          const float& threshold,
                          const std::vector<larcv::Image2D>& adc_v,
                          const std::vector<larcv::Image2D>& thrumu_v,
                          const std::vector<const larcv::Image2D*>& showerimg_v,
                          const std::vector< std::vector<int> >& imgBounds );


} // namespace interface
} // namespace larpid

#endif  // LARPIDINTERFACE_H

