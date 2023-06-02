#include <misc/Exception.h>
#include <psdr/core/frame.h>
#include <psdr/core/intersection.h>
#include <psdr/shape/mesh.h>
#include <psdr/emitter/area.h>

NAMESPACE_BEGIN(psdr_jit)

void AreaLight::configure() {
    PSDR_ASSERT((m_mesh != nullptr) && m_mesh->m_ready);
    m_sampling_weight = m_mesh->m_total_area*
        rgb2luminance<false>(detach(m_radiance))[0];
    m_ready = true;
}

SpectrumD AreaLight::eval(const IntersectionD &its, MaskD active) const {
    PSDR_ASSERT(m_ready);
    return select(active && FrameD::cos_theta(its.wi) > 0.f, m_radiance, 0.f);
}


PositionSampleD AreaLight::sample_position(const Vector3fD &ref_p, const Vector2fD &sample2, MaskD active) const {
    return m_mesh->sample_position(sample2, active);
}


FloatD AreaLight::sample_position_pdf(const Vector3fD &ref_p, const IntersectionD &its, MaskD active) const {
    return m_sampling_weight*its.shape->sample_position_pdf(its, active);
}

std::string AreaLight::to_string() const {
    std::ostringstream oss;
    oss << "AreaLight[radiance = " << m_radiance << ", sampling_weight = " << m_sampling_weight << "]";
    return oss.str();
}

NAMESPACE_END(psdr_jit)
