#pragma once

#include <psdr/psdr.h>
#include <psdr/core/intersection.h>
#include <psdr/core/records.h>

namespace psdr
{



template <typename Float_>
struct BSDFSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector3f<ad> wo;
    Float<ad> eta;

    DRJIT_STRUCT(BSDFSample_, pdf, is_valid, wo, eta)
};


// PSDR_CLASS_DECL_BEGIN(BSDF,, Object)

PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
public:
    BSDF() {}
    virtual ~BSDF() override {}
    virtual SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;

    virtual BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const = 0;
    virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;

    virtual FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;

    virtual bool anisotropic() const = 0;

    DRJIT_VCALL_REGISTER(FloatD, BSDF);
                    
PSDR_CLASS_DECL_END(BSDF)





// PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
// public:
//     virtual ~BSDF() override {}

//     virtual SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const = 0;
//     virtual SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;

//     virtual BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const = 0;
//     virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;

//     virtual FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const = 0;
//     virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;

//     virtual bool anisotropic() const = 0;

//     // DRJIT_VCALL_REGISTER(FloatD, BSDF);

// PSDR_CLASS_DECL_END(BSDF)

} // namespace psdr

// DRJIT_VCALL_TEMPLATE_BEGIN(psdr::BSDF)
//     DRJIT_VCALL_METHOD(eval)
//     DRJIT_VCALL_METHOD(sample)
//     DRJIT_VCALL_METHOD(pdf)
//     DRJIT_VCALL_METHOD(anisotropic)
// DRJIT_VCALL_TEMPLATE_END(psdr::BSDF)

DRJIT_VCALL_BEGIN(psdr::BSDF)
    DRJIT_VCALL_METHOD(eval)
    DRJIT_VCALL_METHOD(sample)
    DRJIT_VCALL_METHOD(pdf)
    DRJIT_VCALL_METHOD(anisotropic)
DRJIT_VCALL_END(psdr::BSDF)
