#include <misc/Exception.h>
#include <psdr/core/warp.h>
#include <psdr/core/bitmap.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/bsdf/ggx.h>

namespace psdr
{
Microfacet::Microfacet(const char *spec_refl_file, const char *diff_refl_file, const char *roughness_file) :
    m_specularReflectance(spec_refl_file), m_diffuseReflectance(diff_refl_file), m_roughness(roughness_file) {}

Microfacet::Microfacet(const Bitmap3fD &spec_refl, const Bitmap3fD &diff_refl, const Bitmap1fD &roughness) :
    m_specularReflectance(spec_refl), m_diffuseReflectance(diff_refl), m_roughness(roughness) {}

SpectrumC Microfacet::eval(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __eval<false>(its, wo, active);
}

SpectrumD Microfacet::eval(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __eval<true>(its, wo, active);
}

template <bool ad>
Spectrum<ad> Microfacet::__eval(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active) const {
    Float<ad> cos_theta_nv = Frame<ad>::cos_theta(its.wi),// view dir
              cos_theta_nl = Frame<ad>::cos_theta(wo);    // light dir

    active &= (cos_theta_nv > 0.f && cos_theta_nl > 0.f);

    Spectrum<ad> diffuse = m_diffuseReflectance.eval<ad>(its.uv) * InvPi;

    Vector3f<ad> H = normalize(its.wi + wo);
    Float<ad> cos_theta_nh = Frame<ad>::cos_theta(H),
              cos_theta_vh = dot(H, its.wi);

    Spectrum<ad> F0 = m_specularReflectance.eval<ad>(its.uv);
    Float<ad> roughness = m_roughness.eval<ad>(its.uv);
    Float<ad> alpha = sqr(roughness);
    Float<ad> k = sqr(roughness + 1.f) / 8.f;

    // GGX NDF term
    Float<ad> tmp = alpha / (cos_theta_nh * cos_theta_nh * (sqr(alpha) - 1.f) + 1.f);
    Float<ad> ggx = tmp * tmp * InvPi;

    // Fresnel term
    Float<ad> coeff = cos_theta_vh * (-5.55473f * cos_theta_vh - 6.8316f);
    Spectrum<ad> fresnel = F0 + (1.f - F0) * pow(2.f, coeff);

    // Geometry term
    Float<ad> smithG1 = cos_theta_nv / (cos_theta_nv * (1.f - k) + k);
    Float<ad> smithG2 = cos_theta_nl / (cos_theta_nl * (1.f - k) + k);
    Float<ad> smithG = smithG1 * smithG2;

    Spectrum<ad> numerator = ggx * smithG * fresnel;
    Float<ad> denominator = 4.f * cos_theta_nl * cos_theta_nv;
    Spectrum<ad> specular = numerator / (denominator + 1e-6f);

    Spectrum<ad> value = (diffuse + specular) * cos_theta_nl;

    return select(active, value, 0.f);
}


BSDFSampleC Microfacet::sample(const IntersectionC &its, const Vector3fC &sample, MaskC active) const {
    return __sample<false>(its, sample, active);
}


BSDFSampleD Microfacet::sample(const IntersectionD &its, const Vector3fD &sample, MaskD active) const {
    return __sample<true>(its, sample, active);
}

template <bool ad>
BSDFSample<ad> Microfacet::__sample(const Intersection<ad>& its, const Vector3f<ad>& sample, Mask<ad> active) const {
    BSDFSample<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha_u = m_roughness.eval<ad>(its.uv);
    Float<ad> alpha_v = m_roughness.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);

    auto m_pair = distr.sample<ad>(its.wi, sample);
    Vector3f<ad> m = std::get<0>(m_pair);
    Float<ad> m_pdf = std::get<1>(m_pair);
    bs.wo = fmsub(Vector3f<ad>(m), 2.f * dot(its.wi, m), its.wi);
    bs.eta = 1.0f; // TODO: Fix later
    bs.pdf = (m_pdf / (4.f * dot(bs.wo, m)));
    bs.is_valid = (cos_theta_i > 0.f && neq(bs.pdf, 0.f) && Frame<ad>::cos_theta(bs.wo) > 0.f) & active;
    return detach(bs);
}

BSDFSampleDualC Microfacet::sampleDual(const IntersectionC &its, const Vector3fC &sample, MaskC active) const {
    return __sampleDual<false>(its, sample, active);
}


BSDFSampleDualD Microfacet::sampleDual(const IntersectionD &its, const Vector3fD &sample, MaskD active) const {
    return __sampleDual<true>(its, sample, active);
}

template <bool ad>
BSDFSampleDual<ad> Microfacet::__sampleDual(const Intersection<ad>& its, const Vector3f<ad>& sample, Mask<ad> active) const {
    BSDFSampleDual<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha_u = m_roughness.eval<ad>(its.uv);
    Float<ad> alpha_v = m_roughness.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);
    Mask<ad> isDual = alpha_u < 0.1;

    auto m_pair = distr.sample<ad>(its.wi, sample);
    Vector3f<ad> m = std::get<0>(m_pair);
    Vector3f<ad> m_flip(-m.x(), -m.y(), m.z());

    bs.pdf1 = std::get<1>(m_pair);
    bs.pdf2 = distr.smith_g1<ad>(its.wi, m_flip) * abs(dot(its.wi, m_flip)) * distr.eval<ad>(m_flip) / abs(Frame<ad>::cos_theta(its.wi));

    bs.pdf1 = 0.5f*(bs.pdf1+bs.pdf2);
    bs.pdf2 = bs.pdf1;

    bs.wo1 = fmsub(Vector3f<ad>(m), 2.f * dot(its.wi, m), its.wi);
    bs.pdf1 /= (4.f * dot(bs.wo1, m));
    bs.is_valid1 = (cos_theta_i > 0.f && neq(bs.pdf1, 0.f) && Frame<ad>::cos_theta(bs.wo1) > 0.f) & active;

    bs.wo2 = fmsub(Vector3f<ad>(m_flip), 2.f * dot(its.wi, m_flip), its.wi);
    bs.pdf2 /= (4.f * dot(bs.wo2, m_flip));
    bs.is_valid2 = (cos_theta_i > 0.f && neq(bs.pdf2, 0.f) && Frame<ad>::cos_theta(bs.wo2) > 0.f) & active;
    bs.isDual = isDual;
    return bs;

}

FloatC Microfacet::pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD Microfacet::pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}

template <bool ad>
Float<ad> Microfacet::__pdf(const Intersection<ad> &its, const Vector3f<ad> &wo, Mask<ad> active) const {
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    Spectrum<ad> m = normalize(wo + its.wi);
    
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
              dot(its.wi, m) > 0.f && dot(wo, m) > 0.f;

    Float<ad> alpha_u = m_roughness.eval<ad>(its.uv);
    Float<ad> alpha_v = m_roughness.eval<ad>(its.uv);
    GGXDistribution distr(alpha_u, alpha_v);

    Float<ad> result = (distr.eval<ad>(m) * distr.smith_g1<ad>(its.wi, m) /
                       (4.f * cos_theta_i));
    return detach(result);
}

} // namespace psdr
