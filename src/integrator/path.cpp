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

namespace psdr
{

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
    // std::cout << "result1" << result << std::endl;
    // return result;
    Int<ad> idx = arange<Int<ad>>(slices<Spectrum<ad>>(result));
    drjit::eval(result); drjit::sync_thread();

// std::cout << "result2" << result << std::endl;
    for (int depth=0; depth < m_max_depth; depth++) {
        BSDFArray<ad> bsdf_array = its.shape->bsdf();
        // Direct illumination
        {
            drjit::eval(); drjit::sync_thread();
            PositionSample<ad> ps = scene.sample_emitter_position<ad>(its.p, sampler.next_2d<ad>(), active);
            Mask<ad> active_direct = active && ps.is_valid;

            Vector3f<ad> wod = ps.p - its.p;
            Float<ad> dist_sqr = squared_norm(wod);
            Float<ad> dist = safe_sqrt(dist_sqr);
            wod /= dist;
            // drjit::eval(result);
// std::cout << "result3" << result << std::endl;
            

            Ray<ad> ray1(its.p, wod);
            Intersection<ad> its1 = scene.ray_intersect<ad, ad>(ray1, active_direct);
            active_direct &= its1.is_valid();
            active_direct &= (its1.t > dist - ShadowEpsilon) && its1.is_emitter(active_direct);

            Float<ad> cos_val = dot(its1.n, -wod);
            Float<ad> G_val = abs(cos_val) / dist_sqr;
            Spectrum<ad> emitter_val = its1.Le(active_direct);

            // std::cout << "emitter_val" << emitter_val << std::endl;
            // std::cout << "result1" << result << std::endl;
            // for (int j=0; j<3; ++j) {
            //     scatter_reduce(ReduceOp::Add, result[j], emitter_val[j], idx);
            // }
            
            // std::cout<< "result2"  << result << std::endl;
            // PSDR_ASSERT(0);
            // return result;
            // result[active_direct] += emitter_val;
            // PSDR_ASSERT(0);

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
                std::cout << its.uv << std::endl;
                std::cout << "bsdf_val" << bsdf_val2 << std::endl;
                // PSDR_ASSERT(0);
                bsdf_val2 *= G_val*ps.J/ps.pdf;

                pdf1 = bsdf_array->pdfC(its, wo_local, active_direct);

                pdf1 *= G_val;
            }
            Float<ad> weight1 = mis_weight<ad>(ps.pdf, pdf1);

            std::cout << "bsdf_val2" << bsdf_val2 << std::endl;
            // // Float<ad> weight1 = 1.0f;
            result[active_direct] += throughput * emitter_val * bsdf_val2 * weight1;

        }


        // // Indirect illumination
        // {
        //     BSDFSample<ad> bs;
        //     if constexpr ( ad ) {
        //         bs = bsdf_array->sampleD(its, sampler.next_nd<3, ad>(), active);
        //     } else {
        //         bs = bsdf_array->sampleC(its, sampler.next_nd<3, ad>(), active);
        //     }
            
        //     Spectrum<ad> bsdf_val;
        //     Float<ad> pdf0;
        //     curr_ray = Ray<ad>(its.p, its.sh_frame.to_world(bs.wo));
        //     Intersection<ad> its1 = scene.ray_intersect<ad, ad>(curr_ray, active);

        //     active &= bs.is_valid;
        //     active &= its1.is_valid();

        //     if constexpr ( ad ) {
        //         Vector3fD wo = its1.p - its.p;
        //         wo /= its1.t;
        //         FloatD cos_val = dot(its1.n, -wo);
        //         FloatD G_val = abs(cos_val)/sqr(its1.t);

        //         FloatD J = select(~its1.is_valid(), 1.0f, its1.J);
        //         G_val = select(~its1.is_valid(), 1.0f, G_val);
        //         pdf0 = bs.pdf*detach(G_val);

        //         bsdf_val = select(detach(its1.t) < Epsilon, 0.0f, bsdf_array->evalD(its, its.sh_frame.to_local(wo), active) * G_val * J / pdf0);
        //     } else {
        //         FloatC cos_val = dot(its1.n, -curr_ray.d);
        //         FloatC G_val = abs(cos_val)/sqr(its1.t);
        //         pdf0 = bs.pdf*G_val;
        //         bsdf_val = select(detach(its1.t) < Epsilon, 0.0f, bsdf_array->evalC(its, bs.wo, active) / bs.pdf);
        //     }
        //     Float<ad> weight2 = mis_weight<ad>(pdf0, scene.emitter_position_pdf<ad>(its.p, its1, active)); // MIS weight
        //     // Float<ad> weight2 = 1.0f;
        //     throughput *= bsdf_val;
        //     active &= throughput > Epsilon;
        //     eta        *= bs.eta;
        //     result[active] += its1.Le(active) * throughput * weight2;
        //     its = its1;
        // }

    }
    std::cout << result << std::endl;
    return result;
}

} // namespace psdr
