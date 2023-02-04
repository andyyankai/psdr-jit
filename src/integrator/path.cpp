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

    drjit::Loop<Mask<ad>> loop("Path Tracer", its, result, throughput, eta, curr_ray, sampler, active);
    loop.set_max_iterations(m_max_depth);
    while (loop(active)) {
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

void PathTracer::render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;
    Vector3fC sample3 = scene.m_samplers[2].next_nd<3, false>();
    std::cout << "render_secondary_edges" << std::endl;

    BoundarySegSampleDirect bss = scene.sample_boundary_segment_direct(sample3);
    MaskC valid = bss.is_valid;

    // _p0 on a face edge, _p2 on an emitter
    const Vector3fC &_p0    = detach(bss.p0);
    Vector3fC       _p2, _dir;
    _p2  = bss.p2;
    _dir = normalize(_p2 - _p0);
    // check visibility between _p0 and _p2
    IntersectionC _its2;
    TriangleInfoD tri_info;
    _its2 = scene.ray_intersect<false>(RayC(_p0, _dir), valid, &tri_info);


    valid &= _its2.is_emitter(valid) && _its2.is_valid() && norm(_its2.p - _p2) < ShadowEpsilon;


    IntersectionC _its1 = scene.ray_intersect<false>(RayC(_p0, -_dir), valid);
    valid &= _its1.is_valid();
    Vector3fC &_p1 = _its1.p;

    // project _p1 onto the image plane and compute the corresponding pixel id
    SensorDirectSampleC sds = (*scene.m_sensors[sensor_id]).sample_direct(_p1);
    valid &= sds.is_valid;

    // trace a camera ray toward _p1 in a differentiable fashion
    RayD camera_ray;
    IntersectionD its1;

    camera_ray = (*scene.m_sensors[sensor_id]).sample_primary_ray(Vector2fD(sds.q));
    its1 = scene.ray_intersect<true, false>(camera_ray, valid);
    valid &= detach(its1.is_valid() )&& norm(detach(its1.p) - _p1) < ShadowEpsilon;


    BSDFArrayD bsdf_array = its1.shape->bsdf();
    if ( scene.m_emitter_env != nullptr ) {
        valid &= detach(neq(bsdf_array, nullptr));
    }

    // calculate base_value
    FloatC      dist    = norm(_p2 - _p1),
                cos2    = abs(dot(_its2.n, -_dir));
    Vector3fC   e       = cross(bss.edge, _dir);
    FloatC      sinphi  = norm(e);
    Vector3fC   proj    = normalize(cross(e, _its2.n));
    FloatC      sinphi2 = norm(cross(_dir, proj));
    FloatC      base_v  = (_its1.t/dist)*(sinphi/sinphi2)*cos2;

    SpectrumC bsdf_val;
    Vector3fC d0 = -detach(camera_ray.d);
    Vector3fC d0_local = _its1.sh_frame.to_local(d0);
    if ( scene.m_bsdfs.size() == 1U || scene.m_meshes.size() == 1U ) {
        const BSDF *bsdf = scene.m_meshes[0]->m_bsdf;
        bsdf_val = bsdf->eval(_its1, d0_local, valid);
    } else {
        BSDFArrayC bsdf_array = _its1.shape->bsdf();
        bsdf_val = bsdf_array->eval(_its1, d0_local, valid);
    }
    // accounting for BSDF's asymmetry caused by shading normals
    FloatC correction = abs((_its1.wi.z()*dot(d0, _its1.n))/(d0_local.z()*dot(_dir, _its1.n)));
    masked(bsdf_val, valid) *= correction;

    SpectrumC value0 = (bsdf_val*_its2.Le(valid)*(base_v*sds.sensor_val/bss.pdf)) & valid;


    Vector3fC n = normalize(cross(_its2.n, proj));
    value0 *= sign(dot(e, bss.edge2))*sign(dot(e, n));

    const Vector3fD &v0 = tri_info.p0,
                    &e1 = tri_info.e1,
                    &e2 = tri_info.e2;

    RayD shadow_ray(its1.p, normalize(bss.p0 - its1.p));
    Vector2fD uv;
    std::tie(uv, std::ignore) = ray_intersect_triangle<true>(v0, e1, e2, shadow_ray);
    Vector3fD u2 = bilinear<true>(detach(v0), detach(e1), detach(e2), uv);

    SpectrumD result_b = (SpectrumD(value0)*dot(Vector3fD(n), u2)) & valid;

    IntC idx_b = select(valid, sds.pixel_idx, -1);
    SpectrumD value_b = result_b - detach(result_b);


    masked(value_b, ~drjit::isfinite<SpectrumD>(value_b)) = 0.f;
    if ( likely(opts.sppse > 1) ) {
        value_b /= static_cast<float>(opts.sppse);
    }

    for (int j=0; j<3; ++j) {
        scatter_reduce(ReduceOp::Add, result[j], value_b[j], IntD(idx_b), idx_b >= 0);
    }

}

NAMESPACE_END(psdr_jit)
