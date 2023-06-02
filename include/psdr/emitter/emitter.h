#pragma once

#include <psdr/psdr.h>
#include <psdr/core/records.h>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Emitter,, Object)
public:
    virtual ~Emitter() override {}

    virtual void configure() = 0;

    // Returns the emitted radiance at its.p in direction its.wi

    virtual SpectrumC eval(const IntersectionC &its, MaskC active = true) const = 0;
    virtual SpectrumD eval(const IntersectionD &its, MaskD active = true) const = 0;

    SpectrumC evalC(const IntersectionC &its, MaskC active = true) const {
        return eval(its, active);
    };
    SpectrumD evalD(const IntersectionD &its, MaskD active = true) const {
        return eval(its, active);
    };

    virtual PositionSampleC sample_position(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active = true) const = 0;
    virtual PositionSampleD sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const = 0;

    PositionSampleC sample_positionC(const Vector3fC &ref_p, const Vector2fC &sample2, MaskC active = true) const {
        return sample_position(ref_p, sample2, active);
    };
    PositionSampleD sample_positionD(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const {
        return sample_position(ref_p, sample2, active);
    };

    virtual FloatD sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const = 0;
    
    bool m_ready = false;
    float m_sampling_weight = 1.f;

    DRJIT_VCALL_REGISTER(FloatD, Emitter);

PSDR_CLASS_DECL_END(Emitter)

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::Emitter)
    DRJIT_VCALL_METHOD(eval)
    DRJIT_VCALL_METHOD(evalC)
    DRJIT_VCALL_METHOD(evalD)
    DRJIT_VCALL_METHOD(sample_position)
    DRJIT_VCALL_METHOD(sample_positionC)
    DRJIT_VCALL_METHOD(sample_positionD)
    DRJIT_VCALL_METHOD(sample_position_pdf)
DRJIT_VCALL_END(psdr_jit::Emitter)
