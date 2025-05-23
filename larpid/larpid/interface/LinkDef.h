/** \defgroup MCTools
 *
 * \brief Algorithms that take in Monte Carlo (i.e. simulation) truth information
 *
 *
 * cint script to generate libraries and python bindings.
 * Declare namespace & classes you defined
 * pragma statement: order matters! Google it ;)
 *
 */
#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace larpid;
#pragma link C++ namespace larpid::interface;
#pragma link C++ struct larpid::interface::CropPixData_t+;
//#pragma link C++ class std::vector<CropPixData_t>+;
//#pragma link C++ class std::vector< std::vector<CropPixData_t> >+;
#pragma link C++ class larpid::interface::ModelOutput+;

#endif




