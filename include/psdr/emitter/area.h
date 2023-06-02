#pragma once

#include <psdr/psdr.h>
#include "emitter.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(AreaLight, final, Emitter)
public:
    AreaLight(const ScalarVector3f &radiance) : m_radiance(radiance) {}
    AreaLight(const ScalarVector3f &radiance, const Mesh *mesh) : m_radiance(radiance), m_mesh(mesh) {}
    AreaLight(const Mesh *mesh) : m_mesh(mesh) {}

    void configure() override;

    SpectrumD eval(const IntersectionD &its, MaskD active = true) const override;
    PositionSampleD sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active = true) const override;
    FloatD sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active = true) const override;

    std::string to_string() const override;

    SpectrumD m_radiance;
    const Mesh *m_mesh;

PSDR_CLASS_DECL_END(AreaLight)

NAMESPACE_END(psdr_jit)
