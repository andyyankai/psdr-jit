#include <chrono>
#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/scene/scene.h>
#include <psdr/sensor/sensor.h>
#include <psdr/integrator/integrator.h>

namespace psdr
{

SpectrumC Integrator::renderC(const Scene &scene, int sensor_id, int npass) const {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    const RenderOption &opts = scene.m_opts;
    const int num_pixels = opts.width*opts.height;
    IntC idx = arange<IntC>(num_pixels);
    SpectrumC result = zeros<SpectrumC>(num_pixels);
    for (int i=0; i<npass; ++i) {
        SpectrumC value = __render<false>(scene, sensor_id) / static_cast<float>(npass);
        // masked(value, ~enoki::isfinite<SpectrumC>(value)) = 0.f;
        for (int j=0; j<3; ++j) {
            scatter_reduce(ReduceOp::Add, result[j], value[j], idx);
        }
    }
    
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
    // if ( likely(opts.spp > 0) ) {
    //     int64_t num_samples = static_cast<int64_t>(num_pixels)*opts.spp;
    //     PSDR_ASSERT(num_samples <= std::numeric_limits<int>::max());

    //     Int<ad> idx = arange<Int<ad>>(num_samples);
    //     if ( likely(opts.spp > 1) ) idx /= opts.spp;

    //     Vector2f<ad> data_base = meshgrid(arange<Float<ad>>(opts.width),arange<Float<ad>>(opts.height));

        

    //     Vector2f<ad> samples_base = gather<Vector2f<ad>>(data_base, idx);

    //     Vector2f<ad> samples = (samples_base + scene.m_samplers[0].next_2d<ad>())
    //                             /ScalarVector2f(opts.width, opts.height);
    //     Ray<ad> camera_ray = scene.m_sensors[sensor_id]->sample_primary_ray(samples);
    //     Spectrum<ad> value = Li(scene, scene.m_samplers[0], camera_ray);
    //     // masked(value, ~enoki::isfinite<Spectrum<ad>>(value)) = 0.f;
    //     for (int j=0; j<3; ++j) {
    //         scatter_reduce(ReduceOp::Add, result[j], value[j], idx);
    //     }
    //     if ( likely(opts.spp > 1) ) {
    //         result /= static_cast<float>(opts.spp);
    //     }
    // }
    
    return result;
}

} // namespace psdr
