#pragma once

#include <psdr/psdr.h>
#include <psdr/core/intersection.h>
#include <psdr/core/records.h>

NAMESPACE_BEGIN(psdr_jit)



template <typename Float_>
struct BSDFSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector3f<ad> wo;
    Float<ad> eta;

    DRJIT_STRUCT(BSDFSample_, pdf, is_valid, wo, eta)
};


PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
public:
    BSDF() {}
    virtual ~BSDF() override {}
    virtual SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;

    SpectrumC evalC(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const {
        return eval(its, wo, active);
    };
    SpectrumD evalD(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const {
        return eval(its, wo, active);
    };

    virtual BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const = 0;
    virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;

    BSDFSampleC sampleC(const IntersectionC &its, const Vector3fC &sample_, MaskC active = true) const {
        return sample(its, sample_, active);
    };
    BSDFSampleD sampleD(const IntersectionD &its, const Vector3fD &sample_, MaskD active = true) const {
        return sample(its, sample_, active);
    };

    virtual FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;

    FloatC pdfC(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const {
        return pdf(its, wo, active);
    };
    FloatD pdfD(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const {
        return pdf(its, wo, active);
    };

    virtual bool anisotropic() const = 0;

    bool m_twoSide = false;

    DRJIT_VCALL_REGISTER(FloatD, BSDF);
                    
PSDR_CLASS_DECL_END(BSDF)

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::BSDF)
    DRJIT_VCALL_METHOD(eval)
    DRJIT_VCALL_METHOD(evalC)
    DRJIT_VCALL_METHOD(evalD)
    DRJIT_VCALL_METHOD(sample)
    DRJIT_VCALL_METHOD(sampleC)
    DRJIT_VCALL_METHOD(sampleD)
    DRJIT_VCALL_METHOD(pdf)
    DRJIT_VCALL_METHOD(pdfC)
    DRJIT_VCALL_METHOD(pdfD)
    DRJIT_VCALL_METHOD(anisotropic)
DRJIT_VCALL_END(psdr_jit::BSDF)
