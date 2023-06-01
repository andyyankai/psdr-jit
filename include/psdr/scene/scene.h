#pragma once

#include <unordered_map>
#include <psdr/psdr.h>
#include <psdr/edge/edge.h>
#include <psdr/shape/mesh.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/bsdf/microfacet.h>
#include <psdr/bsdf/normalmap.h>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Scene, final, Object)
    friend class SceneLoader;

public:
    using ParamMap = std::unordered_map<std::string, const Object&>;

    Scene();
    ~Scene() override;

    void load_file(const char *file_name, bool auto_configure = true);
    void load_string(const char *scene_xml, bool auto_configure = true);

    void configure(std::vector<int> active_sensor=std::vector<int>());

    void add_Sensor(Sensor* sensor);
    void add_EnvironmentMap(const char *fname, ScalarMatrix4f to_world, float scale);
    void add_BSDF(BSDF* bsdf, const char *bsdf_id, bool twoSide = false);
    void add_normalmap_BSDF(NormalMap* bsdf1, Microfacet* bsdf2, const char *bsdf_id, bool twoSide = false);

    void add_Mesh(const char *fname, Matrix4fC transform, const char *bsdf_id, Emitter* emitter);

    bool is_ready() const;

    template <bool ad, bool path_space = false>
    std::pair<Intersection<ad>, TriangleInfoD> boundary_ray_intersect(const Ray<ad> &ray, Mask<ad> active = true) const;


    template <bool ad, bool path_space = false>
    Intersection<ad> ray_intersect(const Ray<ad> &ray, Mask<ad> active = true) const;

    template <bool ad>
    PositionSample<ad> sample_emitter_position(const Vector3f<ad> &ref_p, const Vector2f<ad> &sample, Mask<ad> active = true) const;

    template <bool ad>
    Float<ad> emitter_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active = true) const;

    BoundarySegSampleDirect sample_boundary_segment_direct(const Vector3fC &sample3, MaskC active = true) const;

    bool has_envmap() {
        if (m_emitter_env == nullptr)
            return false;
        return true;
    }

    int seed = 0;

    // std::string to_string() const override;

    int                     m_num_sensors;
    std::vector<Sensor*>    m_sensors;

    std::vector<Emitter*>   m_emitters;
    EnvironmentMap          *m_emitter_env;
    EmitterArrayD           m_emitters_cuda;
    DiscreteDistribution    *m_emitters_distrb;

    std::vector<BSDF*>      m_bsdfs;

    int                     m_num_meshes;
    std::vector<Mesh*>      m_meshes;
    MeshArrayD              m_meshes_cuda;

    // Scene bounding box
    Vector3fC               m_lower, m_upper;

    ParamMap                m_param_map;

    RenderOption            m_opts;

// protected:
    TriangleInfoD           m_triangle_info;
    TriangleUVD             m_triangle_uv;
    MaskD                   m_triangle_face_normals;
    bool                    m_has_bound_mesh;


    SecondaryEdgeInfo       m_sec_edge_info;
    DiscreteDistribution    *m_sec_edge_distrb;


    bool                    m_loaded;
    Scene_OptiX             *m_optix;

PSDR_CLASS_DECL_END(Scene)

NAMESPACE_END(psdr_jit)
