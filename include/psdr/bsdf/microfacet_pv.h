#pragma once

#include "bsdf.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(MicrofacetPerVertex, final, BSDF)
public:
    MicrofacetPerVertex(const Vector3fD &spec_refl, const Vector3fD &diff_refl, const Vector1fD &roughness);

    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override;


    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override;

    bool anisotropic() const override { return false; }

    std::string to_string() const override { return std::string("MicrofacetPerVertex[id=") + m_id + "]"; }

    Vector3fD m_specularReflectance,
              m_diffuseReflectance;
    Vector1fD m_roughness;

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active = true) const;

    template <int n, bool ad>
    Vectorf<n, ad> __interpolate(const Intersection<ad> &its, const Vectorf<n, true> &v, Mask<ad> active = true) const;

PSDR_CLASS_DECL_END(MicrofacetPerVertex)

NAMESPACE_END(psdr_jit)
