#pragma once

#include <psdr/core/frame.h>
#include "sensor.h"

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(PerspectiveCamera, final, Sensor)
public:
    PerspectiveCamera(float fov_x, float near, float far) : m_fov_x(fov_x), m_near_clip(near), m_far_clip(far), m_use_intrinsic(false) {}
    PerspectiveCamera(float fx, float fy, float cx, float cy, float near, float far) 
        : m_fx(fx), m_fy(fy), m_cx(cx), m_cy(cy), m_near_clip(near), m_far_clip(far), m_use_intrinsic(true) {};

    void configure(bool cache) override;

    RayD sample_primary_ray(const Vector2fD &samples) const override;

    SensorDirectSampleD sample_direct(const Vector3fD &p) const override;

    PrimaryEdgeSample sample_primary_edge(const FloatC &sample1) const override;

    std::string to_string() const override;

    float       m_fov_x,
                m_fx,
                m_fy,
                m_cx,
                m_cy,
                m_near_clip,
                m_far_clip;
    Matrix4fD   m_sample_to_camera, m_camera_to_sample,
                m_world_to_sample, m_sample_to_world;

    bool        m_use_intrinsic;

    Vector3fD   m_camera_pos, m_camera_dir;
    FloatD      m_inv_area;
PSDR_CLASS_DECL_END(PerspectiveCamera)

NAMESPACE_END(psdr_jit)
