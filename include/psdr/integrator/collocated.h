#pragma once

#include "integrator.h"

namespace psdr
{

PSDR_CLASS_DECL_BEGIN(CollocatedIntegrator, final, Integrator)
public:
    CollocatedIntegrator(const FloatD &intensity) : m_intensity(intensity) {}

    FloatD m_intensity;

protected:
    SpectrumC Li(const Scene &scene, Sampler &sampler, const RayC &ray, MaskC active = true) const override;
    SpectrumD Li(const Scene &scene, Sampler &sampler, const RayD &ray, MaskD active = true) const override;

    template <bool ad>
    Spectrum<ad> __Li(const Scene &scene, const Ray<ad> &ray, Mask<ad> active) const;
PSDR_CLASS_DECL_END(CollocatedIntegrator)

} // namespace psdr
