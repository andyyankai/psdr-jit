#pragma once

#include <psdr/psdr.h>
#include <psdr/core/records.h>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Emitter,, Object)
public:
    virtual ~Emitter() override {}

    virtual void configure() = 0;

    // Returns the emitted radiance at its.p in direction its.wi

    virtual SpectrumD eval(const IntersectionD &its, MaskD active = true) const = 0;
    virtual PositionSampleD sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const = 0;
    virtual FloatD sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const = 0;

    bool m_ready = false;
    float m_sampling_weight = 1.f;

    DRJIT_VCALL_REGISTER(FloatD, Emitter);

PSDR_CLASS_DECL_END(Emitter)

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::Emitter)
    DRJIT_VCALL_METHOD(eval)
    DRJIT_VCALL_METHOD(sample_position)
    DRJIT_VCALL_METHOD(sample_position_pdf)
DRJIT_VCALL_END(psdr_jit::Emitter)
