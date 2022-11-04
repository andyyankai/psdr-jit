#pragma once

#include "integrator.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(PathTracer, final, Integrator)
public:
    PathTracer(int max_depth = 1);
    virtual ~PathTracer();

    bool m_hide_emitters = false; 

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, Sampler &sampler, const Ray<ad> &ray, Mask<ad> active) const;

    void render_secondary_edges(const Scene &scene, int sensor_id, SpectrumD &result) const override;

    int m_max_depth;

PSDR_CLASS_DECL_END(PathTracer)

NAMESPACE_END(psdr_jit)
