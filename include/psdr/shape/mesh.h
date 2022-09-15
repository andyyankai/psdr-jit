#pragma once


#include <psdr/psdr.h>
#include <psdr/core/records.h>
#include <psdr/edge/edge.h>
#include <map>
#include <array>
#include <vector>

NAMESPACE_BEGIN(psdr_jit)

PSDR_CLASS_DECL_BEGIN(Mesh, final, Object)
public:
    Mesh() = default;
    ~Mesh() override;

    void load(const char *fname, bool verbose = false);
    void configure();
    void prepare_optix_buffers();

    inline void set_transform(const Matrix4fD &mat, bool set_left = true) {
        if ( set_left ) {
            m_to_world_left = mat;
        } else {
            m_to_world_right = mat;
        }
        m_ready = false;
    }

    inline void append_transform(const Matrix4fD &mat, bool append_left = true) {
        if ( append_left ) {
            m_to_world_left = mat*m_to_world_left;
        } else {
            m_to_world_right *= mat;
        }
        m_ready = false;
    }

    PositionSampleC sample_position(const Vector2fC &sample2, MaskC active = true) const;
    PositionSampleD sample_position(const Vector2fD &sample2, MaskD active = true) const;

    FloatC sample_position_pdf(const IntersectionC &its, MaskC active = true) const;
    FloatD sample_position_pdfD(const IntersectionD &its, MaskD active = true) const;

    MaskC get_obj_mask(std::string obj_name) const{
        if (m_id != "") {
            if (m_id == obj_name) {
                return MaskC(1);
            } else {
                return MaskC(0);
            }
        } else {
            int obj_id = stoi(obj_name);
            if (m_mesh_id == obj_id) {
                return MaskC(1);
            } else {
                return MaskC(0);
            }
        }
    };

    IntC get_obj_id() const{
        return IntC(m_mesh_id+1);
    };

    const BSDF* bsdf() const {
        return m_bsdf;
    }

    const Emitter* emitter() const {
        return m_emitter;
    }

    void dump(const char *fname) const;


    std::string to_string() const override;

    int                 m_mesh_id = -1;

    bool                m_ready = false;

    bool                m_use_face_normals = false,
                        m_has_uv = false;

    bool                m_enable_edges = false; // we disable boundary term for now

    EdgeSortOption      m_edge_sort;

    // Indicates if the mesh creates primiary and secondary edges

    Matrix4fD           m_to_world_raw   = identity<Matrix4fD>(),
                        m_to_world_left  = identity<Matrix4fD>(),
                        m_to_world_right = identity<Matrix4fD>();

    const BSDF*         m_bsdf = nullptr;
    const Emitter*      m_emitter = nullptr;

    // ref<BSDF> mm_bsdr;

    int                 m_num_vertices = 0,
                        m_num_faces = 0;

    IntC                m_cut_position;

    Vector2iD           m_valid_edge_indices;

    Vector3fD           m_vertex_positions_raw,
                        m_vertex_normals_raw;

    Vector2fD           m_vertex_uv;

    Vector3fD           m_vertex_positions;

    Vector3iD           m_face_indices,
                        m_face_uv_indices;

    // 0, 1: storing vertex indices of the end points
    // 2, 3: storing face indices sharing each edge
    // 4   : storing the third vertex of face with index stored in [2]
    Vectori<5, true>    m_edge_indices;

    float               m_total_area, m_inv_total_area;

    // For position sampling
    DiscreteDistribution *m_face_distrb = nullptr;

    // Temporary triangle info for Scene::configure()
    TriangleInfoD       *m_triangle_info = nullptr;
    TriangleUVD         *m_triangle_uv = nullptr;
    SecondaryEdgeInfo   *m_sec_edge_info = nullptr;

    // For OptiX ray tracing
    FloatC              m_vertex_buffer;
    IntC                m_face_buffer;

    DRJIT_VCALL_REGISTER(FloatD, Mesh);

protected:
    template <bool ad>
    PositionSample<ad> __sample_position(const Vector2f<ad>&, Mask<ad>) const;
PSDR_CLASS_DECL_END(Mesh)

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::Mesh)
    // DRJIT_VCALL_GETTER(m_bsdf, const Emitter *)
    // DRJIT_VCALL_GETTER(m_emitter, const typename Class::Emitter *)

    // DRJIT_VCALL_GETTER(flags, uint32_t)
    DRJIT_VCALL_METHOD(bsdf)
    DRJIT_VCALL_METHOD(emitter)
    DRJIT_VCALL_METHOD(get_obj_mask)
    DRJIT_VCALL_METHOD(get_obj_id)
    DRJIT_VCALL_METHOD(sample_position)
    DRJIT_VCALL_METHOD(sample_position_pdf)
    DRJIT_VCALL_METHOD(sample_position_pdfD)
DRJIT_VCALL_END(psdr_jit::Mesh)
