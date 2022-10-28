#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

NAMESPACE_BEGIN(psdr_jit)

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id, int npass) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.width*opts.height;
    IntC idx = arange<IntC>(num_pixels);
    SpectrumC result = __render<false>(scene, sensor_id);
    
    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        log(oss.str().c_str());
    }

    return result;
}


SpectrumD Integrator::renderD(const Scene &scene, int sensor_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    // Interior integral
    SpectrumD result = __render<true>(scene, sensor_id);

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        log(oss.str().c_str());
    }

    return result;
}



template <bool ad>
Spectrum<ad> Integrator::__render(const Scene &scene, int sensor_id) const {
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.width*opts.height;

    Spectrum<ad> result = zeros<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> idx = arange<Int<ad>>(num_samples);
        if ( likely(opts.spp > 1) ) idx /= opts.spp;

        auto [dx, dy] = meshgrid(arange<Float<ad>>(opts.width),arange<Float<ad>>(opts.height));

        Vector2f<ad> samples_base = gather<Vector2f<ad>>(Vector2f<ad>(dx, dy), idx);

        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                /ScalarVector2f(opts.width, opts.height);
        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray);
        masked(value, ~drjit::isfinite<Spectrum<ad>>(value) || drjit::isnan<Spectrum<ad>>(value)) = 0.f;
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], idx);
        }
        if ( likely(opts.spp > 1) ) {
            result /= static_cast<float>(opts.spp);
        }
    }
    
    return result;
}


void Integrator::render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const {
    const RenderOption &opts = scene.m_opts;
    const Sensor *sensor = scene.m_sensors[sensor_id];
    if ( sensor->m_enable_edges ) {
        PrimaryEdgeSample edge_samples = sensor->sample_primary_edge(scene.m_samplers[1].next_1d<false>());
        MaskC valid = (edge_samples.idx >= 0);
        SpectrumC delta_L = Li(scene, scene.m_samplers[1], edge_samples.ray_n, valid) - 
                            Li(scene, scene.m_samplers[1], edge_samples.ray_p, valid);
        SpectrumD value = edge_samples.x_dot_n*SpectrumD(delta_L/edge_samples.pdf);
        masked(value, ~drjit::isfinite<SpectrumD>(value)) = 0.f;
        if ( likely(opts.sppe > 1) ) {
            value /= static_cast<float>(opts.sppe);
        }
        value -= detach(value);

        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], IntD(edge_samples.idx), valid);
        }
    }
}

NAMESPACE_END(psdr_jit)
