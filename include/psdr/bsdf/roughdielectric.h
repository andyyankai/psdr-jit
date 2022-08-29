#pragma once

#include "bsdf.h"
#include <psdr/core/bitmap.h>

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(RoughDielectric, final, BSDF)
public:
    RoughDielectric() : m_specular_reflectance(1.0f), m_alpha_u(0.1f), m_alpha_v(0.1f) {
        float intIOR = 1.5f;
        float extIOR = 1.0f;

        m_eta = intIOR / extIOR;
        m_inv_eta = 1.f / m_eta;

        m_anisotropic = false;
    }

    RoughDielectric(float intIOR, float extIOR)
        : m_alpha_u(0.1f), m_alpha_v(0.1f), m_eta(intIOR / extIOR), m_inv_eta(extIOR/intIOR), m_specular_reflectance(1.0f) { m_anisotropic = false; }


    RoughDielectric(const Bitmap1fD &alpha, float intIOR, float extIOR)
        : m_alpha_u(alpha), m_alpha_v(alpha), m_eta(intIOR / extIOR), m_inv_eta(extIOR/intIOR), m_specular_reflectance(1.0f) { m_anisotropic = false; }


    SpectrumC eval(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override;
    BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override;

    BSDFSampleDualC sampleDual(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override;
    BSDFSampleDualD sampleDual(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override;

    FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override;
    FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override;

    bool anisotropic() const override { return m_anisotropic; }

    std::string to_string() const override { return std::string("RoughDielectric[id=") + m_id + "]"; }

    Bitmap1fD m_alpha_u, m_alpha_v;
    Bitmap3fD m_specular_reflectance;
    Bitmap3fD m_specular_transmittance;
    FloatD    m_eta, m_inv_eta;
    bool      m_anisotropic;

protected:
    template <bool ad>
    Spectrum<ad> __eval(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    BSDFSample<ad> __sample(const Intersection<ad>&, const Vector3f<ad>&, Mask<ad>) const;

    template <bool ad>
    Float<ad> __pdf(const Intersection<ad> &, const Vector3f<ad> &, Mask<ad>) const;
PSDR_CLASS_DECL_END(RoughDielectric)

} // namespace psdr
