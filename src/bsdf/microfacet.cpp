#include <misc/Exception.h>
#include <psdr/core/warp.h>
#include <psdr/core/bitmap.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/bsdf/ggx.h>

NAMESPACE_BEGIN(psdr_jit)
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
Spectrum<ad> Microfacet::__eval(const Intersection<ad> &_its, const Vector3f<ad> &_wo, Mask<ad> active) const {
    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }



    Float<ad> cos_theta_nv = Frame<ad>::cos_theta(its.wi),// view dir
              cos_theta_nl = Frame<ad>::cos_theta(wo);    // light dir

    active &= (cos_theta_nv > 0.f && cos_theta_nl > 0.f);

    Spectrum<ad> diffuse = m_diffuseReflectance.eval<ad>(its.uv) * InvPi;

    Vector3f<ad> H = normalize(its.wi + wo);
    Float<ad> cos_theta_vh = dot(H, its.wi);

    Spectrum<ad> F0 = m_specularReflectance.eval<ad>(its.uv);
    Float<ad> roughness = m_roughness.eval<ad>(its.uv);
    Float<ad> alpha = sqr(roughness);
    GGXDistribution distr(alpha);

    // GGX NDF term
    Float<ad> ggx = distr.eval<ad>(H);

    // Fresnel term
    Float<ad> coeff = cos_theta_vh * (-5.55473f * cos_theta_vh - 6.8316f);
    Spectrum<ad> fresnel = F0 + (1.f - F0) * pow(2.f, coeff);

    // Geometry term
    Float<ad> smithG = distr.smith_g1<ad>(its.wi, H) * distr.smith_g1<ad>(wo, H);

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
BSDFSample<ad> Microfacet::__sample(const Intersection<ad>& _its, const Vector3f<ad>& sample, Mask<ad> active) const {
    Intersection<ad> its(_its);

    if (m_twoSide) {
        its.wi.z() = drjit::abs(its.wi.z());
    }


    BSDFSample<ad> bs;
    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi);
    Float<ad> alpha = sqr(m_roughness.eval<ad>(its.uv));
    GGXDistribution distr(alpha);

    auto [m, m_pdf] = distr.sample<ad>(its.wi, sample);
    bs.wo = fmsub(Vector3f<ad>(m), 2.f * dot(its.wi, m), its.wi);
    bs.eta = 1.0f; // TODO: Fix later
    bs.pdf = (m_pdf / (4.f * dot(bs.wo, m)));
    bs.is_valid = (cos_theta_i > 0.f && neq(bs.pdf, 0.f) && Frame<ad>::cos_theta(bs.wo) > 0.f) & active;
    return detach(bs);
}


FloatC Microfacet::pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const {
    return __pdf<false>(its, wo, active);
}


FloatD Microfacet::pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const {
    return __pdf<true>(its, wo, active);
}

template <bool ad>
Float<ad> Microfacet::__pdf(const Intersection<ad> &_its, const Vector3f<ad> &_wo, Mask<ad> active) const {

    Intersection<ad> its(_its);
    Vector3f<ad> wo(_wo);

    if (m_twoSide) {
        wo.z() = drjit::mulsign(wo.z(), its.wi.z());
        its.wi.z() = drjit::abs(its.wi.z());
    }

    Float<ad> cos_theta_i = Frame<ad>::cos_theta(its.wi),
              cos_theta_o = Frame<ad>::cos_theta(wo);

    Spectrum<ad> m = normalize(wo + its.wi);
    
    active &= cos_theta_i > 0.f && cos_theta_o > 0.f &&
              dot(its.wi, m) > 0.f && dot(wo, m) > 0.f;

    Float<ad> alpha = sqr(m_roughness.eval<ad>(its.uv));
    GGXDistribution distr(alpha);

    Float<ad> result = (distr.eval<ad>(m) * distr.smith_g1<ad>(its.wi, m) /
                       (4.f * cos_theta_i));
    return detach(result);
}

NAMESPACE_END(psdr_jit)
