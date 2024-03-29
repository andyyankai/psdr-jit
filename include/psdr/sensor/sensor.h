#pragma once

#include <psdr/psdr.h>
#include <psdr/edge/edge.h>
#include <psdr/core/pmf.h>
#include <psdr/core/records.h>

NAMESPACE_BEGIN(psdr_jit)

template <typename Float_>
struct SensorDirectSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector2f<ad>    q;
    Int<ad>         pixel_idx;
    Float<ad>       sensor_val;

    DRJIT_STRUCT(SensorDirectSample_, pdf, is_valid, q, pixel_idx, sensor_val)
};

PSDR_CLASS_DECL_BEGIN(Sensor,, Object)
public:
    virtual ~Sensor() override {}

    virtual void configure(bool cache);

    virtual RayC sample_primary_ray(const Vector2fC &samples) const = 0;
    virtual RayD sample_primary_ray(const Vector2fD &samples) const = 0;

    virtual SensorDirectSampleC sample_direct(const Vector3fC &p) const = 0;

    virtual PrimaryEdgeSample sample_primary_edge(const FloatC &sample1) const = 0;

    inline void set_transform(const Matrix4fD &mat, bool set_left = true) {
        if ( set_left ) {
            m_to_world_left = mat;
        } else {
            m_to_world_right = mat;
        }
    }

    inline void append_transform(const Matrix4fD &mat, bool append_left = true) {
        if ( append_left ) {
            m_to_world_left = mat*m_to_world_left;
        } else {
            m_to_world_right *= mat;
        }
    }

    ScalarVector2i          m_resolution;
    float                   m_aspect;

    // Matrix4fD               m_to_world = identity<Matrix4fD>();
    Matrix4fD               m_to_world_raw   = identity<Matrix4fD>(),
                            m_to_world_left  = identity<Matrix4fD>(),
                            m_to_world_right = identity<Matrix4fD>();

    const Scene             *m_scene = nullptr;

    // Properties for primary edge sampling

    bool                    m_enable_edges = false;
    PrimaryEdgeInfo         m_edge_info;
    DiscreteDistribution    m_edge_distrb;
PSDR_CLASS_DECL_END(Sensor)

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::Sensor)
    DRJIT_VCALL_METHOD(sample_primary_ray)
    DRJIT_VCALL_METHOD(sample_primary_edge)
DRJIT_VCALL_END(psdr_jit::Sensor)
