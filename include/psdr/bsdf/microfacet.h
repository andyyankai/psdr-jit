#pragma once

#include <psdr/core/bitmap.h>
#include "bsdf.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(Microfacet, final, BSDF)
public:
    Microfacet() : m_specularReflectance(0.5f), m_diffuseReflectance(0.5f), m_roughness(0.5f) {}

    Microfacet(const ScalarVector3f &specularRef, const ScalarVector3f &diffuseRef, float roughnessRef) :
        m_specularReflectance(specularRef), m_diffuseReflectance(diffuseRef), m_roughness(roughnessRef) {}

    Microfacet(const char *spec_refl_file, const char *diff_refl_file, const char *roughness_file);

    Microfacet(const Bitmap3fD &spec_refl, const Bitmap3fD &diff_refl, const Bitmap1fD &roughness);

    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override;


    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override;

    bool anisotropic() const override { return false; }

    std::string to_string() const override { return std::string("Microfacet[id=") + m_id + "]"; }

    Bitmap3fD m_specularReflectance,
              m_diffuseReflectance;
    Bitmap1fD m_roughness;

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

PSDR_CLASS_DECL_END(Microfacet)

} // namespace psdr
