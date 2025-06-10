#ifndef __LARPID_INTERFACE_CROP_PIX_DATA_T_H__
#define __LARPID_INTERFACE_CROP_PIX_DATA_T_H__

#include <cmath>

namespace larpid {
namespace interface {

  struct CropPixData_t {

    int row; ///< row of pixel in cropped image
    int col; ///< col of pixel in cropped image
    int rawRow; ///< row of pixel in original image
    int rawCol; ///< col of pixel in original image
    float val; ///< value of pixel
    float adc; ///< ADC value (alias for val)
    bool inCrop; ///< pixel is inside crop
    int idx;   ///< index in container
    
    CropPixData_t()
    : row(0),col(0),rawRow(0),rawCol(0),val(0.0),adc(0.0),inCrop(false),idx(0)
    {};
    
    CropPixData_t( int r, int c, int rr, int rc, float v, bool ic)
    : row(r),col(c),rawRow(rr),rawCol(rc),val(v),adc(v),inCrop(ic),idx(0) {};
    
    bool operator==( const CropPixData_t& rhs ) const {
    if(rawRow == rhs.rawRow && rawCol == rhs.rawCol && std::fabs(val - rhs.val) < 1e-3) return true;
      return false;
    };
    
    bool operator<( const CropPixData_t& rhs ) const {
      if (row<rhs.row) return true;
      if ( row==rhs.row ) {
        if ( col<rhs.col ) return true;
        if ( col==rhs.col ) {
          if ( val<rhs.val ) return true;
        }
      }
      return false;
    };
    
  };

}
}
#endif