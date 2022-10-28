#pragma once

#include <string>
#include <psdr/psdr.h>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Integrator,, Object)
public:
    virtual ~Integrator() {}

    SpectrumC renderC(const Scene &scene, int sensor_id = 0, int npass=1) const;
    SpectrumD renderD(const Scene &scene, int sensor_id = 0) const;

protected:
    virtual SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const = 0;
    virtual SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const = 0;


    virtual void render_primary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const;


    template <bool ad>
    Spectrum<ad> __render(const Scene &scene, int sensor_id) const;
PSDR_CLASS_DECL_END(SamplingIntegrator)

NAMESPACE_END(psdr_jit)
