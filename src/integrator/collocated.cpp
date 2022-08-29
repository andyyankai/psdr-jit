#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/integrator/collocated.h>

namespace psdr
{

SpectrumC CollocatedIntegrator::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active) const {
    return __Li<false>(scene, ray, active);
}


SpectrumD CollocatedIntegrator::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active) const {
    return __Li<true>(scene, ray, active);
}


template <bool ad>
Spectrum<ad> CollocatedIntegrator::__Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const {
    Intersection<ad> its = scene.ray_intersect<ad>(ray, active);
    Spectrum<ad> result;

    active &= its.is_valid();
    if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
        const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
        result = bsdf->eval(its, its.wi, active)/sqr(its.t);
    } else {
        BSDFArray<ad> bsdf_array = its.shape->bsdf(active);
        result = bsdf_array->eval(its, its.wi, active)/sqr(its.t);
    }

    if constexpr (ad) {
        result *= m_intensity;
    } else {
        result *= detach(m_intensity);
    }

    // result[!active] = Spectrum<ad>(1.0);

    return result;
}

} // namespace psdr
