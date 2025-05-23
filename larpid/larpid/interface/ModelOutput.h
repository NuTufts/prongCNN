#ifndef __LARPID_INTERFACE_MODEL_OUTPUT_H__
#define __LARPID_INTERFACE_MODEL_OUTPUT_H__

namespace larpid {
namespace interface {

  class ModelOutput {
    public:

    int pid, process;
    float completeness, purity;
    float electron_score, photon_score, muon_score, pion_score, proton_score;
    float primary_score, neutralParent_score, chargedParent_score;

    ModelOutput() : pid(0), process(-1), completeness(-1.), purity(-1.), electron_score(-99.),
                    photon_score(-99.), muon_score(-99.), pion_score(-99.), proton_score(-99.),
                    primary_score(-99.), neutralParent_score(-99.), chargedParent_score(-99.) {};

    ModelOutput(int pdg, int proc, float comp, float pur, float el, float ph,
                float mu, float pi, float pr, float prim, float ntrl, float chgd) :
                  pid(pdg), process(proc), completeness(comp), purity(pur), electron_score(el),
                  photon_score(ph), muon_score(mu), pion_score(pi), proton_score(pr),
                  primary_score(prim), neutralParent_score(ntrl), chargedParent_score(chgd) {};

  };

}
}

#endif