#include <misc/Exception.h>
#include <psdr/core/cube_distrb.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/transform.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/emitter.h>
#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/perspective.h>
#include <psdr/integrator/path.h>
#include <drjit/loop.h>

NAMESPACE_BEGIN(psdr_jit)

PathTracer::~PathTracer() {}


PathTracer::PathTracer(int max_depth) : m_max_depth(max_depth) {
    PSDR_ASSERT(max_depth >= 0);
}


SpectrumC PathTracer::Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active) const {
    return __Li<false>(scene, sampler, ray, active);
}


SpectrumD PathTracer::Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active) const {
    return __Li<true>(scene, sampler, ray, active);
}

template <bool ad>
Spectrum<ad> PathTracer::__Li(const Scene &scene, Sampler &sampler, const Ray<ad> &_ray, Mask<ad> active) const {
    Ray<ad> curr_ray(_ray);

    Intersection<ad> its = scene.ray_intersect<ad>(curr_ray, active);
    active &= its.is_valid();

    Spectrum<ad> throughput(1.0f);
    Spectrum<ad> eta(1.0f); // Tracks radiance scaling due to index of refraction changes
    Spectrum<ad> result = m_hide_emitters ? zeros<Spectrum<ad>>() : its.Le(active);

    for (int depth=0; depth < m_max_depth; depth++) {
        BSDFArray<ad> bsdf_array = its.shape->bsdf();
        {

            PositionSample<ad> ps = scene.sample_emitter_position<ad>(its.p, sampler.next_2d<ad>(), active);
            Mask<ad> active_direct = active && ps.is_valid && !its.is_emitter(active);
            Vector3f<ad> wod = ps.p - its.p;
            Float<ad> dist_sqr = squared_norm(wod);
            Float<ad> dist = safe_sqrt(dist_sqr);
            wod /= dist;
            Ray<ad> ray1(its.p, wod);
            Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active_direct);
            active_direct &= its1.is_valid();
            active_direct &= (its1.t > dist - ShadowEpsilon) && its1.is_emitter(active_direct);
            Float<ad> cos_val = dot(its1.n, -wod);
            Float<ad> G_val = abs(cos_val) / dist_sqr;
            Spectrum<ad> emitter_val = its1.Le(active);
            Spectrum<ad> bsdf_val2;
            Float<ad> pdf1;

            if constexpr ( ad ) {
                Vector3f<ad> wo_local = its.sh_frame.to_local(wod);
                bsdf_val2 = bsdf_array->evalD(its, wo_local, active_direct);
                bsdf_val2 *= G_val*ps.J/ps.pdf;
                pdf1 = bsdf_array->pdfD(its, wo_local, active_direct);
                pdf1 *= detach(G_val);
            } else {
                Vector3f<ad> wo_local = its.sh_frame.to_local(wod);
                bsdf_val2 = bsdf_array->evalC(its, wo_local, active_direct);
                bsdf_val2 *= G_val*ps.J/ps.pdf;
                pdf1 = bsdf_array->pdfC(its, wo_local, active_direct);
                pdf1 *= G_val;
            }
            Float<ad> weight1 = mis_weight<ad>(ps.pdf, pdf1);
            auto tmp_di = throughput * emitter_val * bsdf_val2 * weight1;
            result[active_direct] += tmp_di;
            
        }

        // Indirect illumination
        {
            BSDFSample<ad> bs;
            if constexpr ( ad ) {
                bs = bsdf_array->sampleD(its, sampler.next_nd<3, ad>(), active);
            } else {
                bs = bsdf_array->sampleC(its, sampler.next_nd<3, ad>(), active);
            }

            Spectrum<ad> bsdf_val;
            Float<ad> pdf0;
            curr_ray = Ray<ad>(its.p, its.sh_frame.to_world(bs.wo));
            Intersection<ad> its1 = scene.ray_intersect<ad, ad>(curr_ray, active);
            active &= bs.is_valid;
            active &= its1.is_valid();
            if constexpr ( ad ) {
                Vector3fD wo = its1.p - its.p;
                wo /= its1.t;
                FloatD cos_val = dot(its1.n, -wo);
                FloatD G_val = abs(cos_val)/sqr(its1.t);
                FloatD J = select(~its1.is_valid(), 1.0f, its1.J);
                G_val = select(~its1.is_valid(), 1.0f, G_val);
                pdf0 = bs.pdf*detach(G_val);
                bsdf_val = select(detach(its1.t) < Epsilon, 0.0f, bsdf_array->evalD(its, its.sh_frame.to_local(wo), active) * G_val * J / pdf0);
            } else {
                FloatC cos_val = dot(its1.n, -curr_ray.d);
                FloatC G_val = abs(cos_val)/sqr(its1.t);
                pdf0 = bs.pdf*G_val;
                auto temp_bsdfa = bsdf_array->evalC(its, bs.wo, active);
                bsdf_val = select(detach(its1.t) < Epsilon, 0.0f, temp_bsdfa / bs.pdf);
                
            }
            Float<ad> weight2 = mis_weight<ad>(pdf0, scene.emitter_position_pdf<ad>(its.p, its1, active)); // MIS weight
            throughput *= bsdf_val;
            eta        *= bs.eta;
            auto tmp_b = its1.Le(active) * throughput * weight2;
            result[active] += tmp_b;
            its = its1;
        }
    }
    return result;



}

NAMESPACE_END(psdr_jit)
