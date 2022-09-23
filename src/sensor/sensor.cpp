#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/sensor/sensor.h>

NAMESPACE_BEGIN(psdr_jit)

void Sensor::configure(bool cache=true) {
    if (cache) drjit::make_opaque(m_to_world_left, m_to_world_raw, m_to_world_right);

    m_aspect = static_cast<float>(m_resolution.x())/m_resolution.y();
    Matrix4fD m_to_world = m_to_world_left * m_to_world_raw * m_to_world_right;
    PSDR_ASSERT_MSG(std::abs(det(Matrix3fD(m_to_world))[0] - 1.f) < Epsilon,
                    "Sensor transformation should not involve scaling!");
}

NAMESPACE_END(psdr_jit)
