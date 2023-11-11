#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

NAMESPACE_BEGIN(psdr_jit)

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id, int seed, IntC pix_id) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    const RenderOption &opts = scene.m_opts;
    int64_t sample_count = static_cast<int64_t>(opts.height)*opts.width*opts.spp;

    if (pix_id != -1 && seed == -1) {
        PSDR_ASSERT_MSG(false, "While using batch rendering, seed must be set!");
    }

    if (seed != -1) {
        if (pix_id == -1) {
            scene.m_samplers[0].seed(arange<UInt64C>(sample_count)+seed);
        } else {
            scene.m_samplers[0].seed(arange<UInt64C>(pix_id.size()*opts.spp)+seed);
        }
    }

    
    const int num_pixels = opts.width*opts.height;
    SpectrumC result;
    if (pix_id == -1) {
        result = __render<false>(scene, sensor_id);
    } else {
        result = __render_batch<false>(scene, sensor_id, pix_id);
    }
    auto end_time = high_resolution_clock::now();
    if ( opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        log(oss.str().c_str());
    }
    drjit::eval(result);
    return result;
}


SpectrumD Integrator::renderD(const Scene &scene, int sensor_id, int seed, IntD pix_id) const {
    if (pix_id != -1 && seed == -1) {
        PSDR_ASSERT_MSG(false, "While using batch rendering, seed must be set!");
    }
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    const RenderOption &opts = scene.m_opts;

    int64_t sample_count = static_cast<int64_t>(opts.height)*opts.width*opts.spp;
    if (seed != -1) {
        if (pix_id == -1) {
            scene.m_samplers[0].seed(arange<UInt64C>(sample_count)+seed);
        } else {
            scene.m_samplers[0].seed(arange<UInt64C>((pix_id.size()*opts.spp))+seed);
        }
    }
    if ( opts.sppe > 0 && seed != -1) {
        scene.m_samplers[1].seed(arange<UInt64C>(sample_count)+seed);
    }
    if ( opts.sppse > 0 && seed != -1) {
        scene.m_samplers[2].seed(arange<UInt64C>(sample_count)+seed);
    }

    // Interior integral
    SpectrumD result;
    if (pix_id == -1) {
        result = __render<true>(scene, sensor_id);
    } else {
        result = __render_batch<true>(scene, sensor_id, pix_id);
    }

    // Boundary integral
    if ( likely(scene.m_opts.sppe > 0) ) {
        render_primary_edges(scene, sensor_id, result);
    }

    if ( likely(scene.m_opts.sppse > 0) ) {
        render_secondary_edges(scene, sensor_id, result);
    }
    
    auto end_time = high_resolution_clock::now();
    if ( scene.m_opts.log_level ) {
        std::stringstream oss;
        oss << "Rendered in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        log(oss.str().c_str());
    }
    drjit::eval(result);
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


template <bool ad>
Spectrum<ad> Integrator::__render_batch(const Scene &scene, int sensor_id, Int<ad> pix_id) const {
    // std::cout << "using batch renderer" << std::endl;
    PSDR_ASSERT_MSG(scene.is_ready(), "Input scene must be configured!");
    PSDR_ASSERT_MSG(sensor_id >= 0 && sensor_id < scene.m_num_sensors, "Invalid sensor id!");
    const RenderOption &opts = scene.m_opts;
    const int num_pixels = pix_id.size();


    Spectrum<ad> result = zeros<Spectrum<ad>>(num_pixels);
    if ( likely(opts.spp > 0) ) {
        int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
        PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

        Int<ad> repeat_idx = arange<Int<ad>>(num_samples) / opts.spp;
        Vector2f<ad> sampled_pixels(pix_id % opts.width, pix_id / opts.width);

        Vector2f<ad> samples_base = gather<Vector2f<ad>>(sampled_pixels, repeat_idx);
        // std::cout << samples_base << std::endl;
        // std::cout << scene.m_samplers[0].next_2d<ad>() << std::endl;
        Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
                                /ScalarVector2f(opts.width, opts.height);

        Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);


        Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray);
        masked(value, ~drjit::isfinite<Spectrum<ad>>(value) || drjit::isnan<Spectrum<ad>>(value)) = 0.f;
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], repeat_idx);
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
