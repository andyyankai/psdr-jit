#include <chrono>
#include <misc/Exception.h>

#include <psdr/core/ray.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/pmf.h>
#include <psdr/bsdf/bsdf.h>
#include <psdr/emitter/envmap.h>
#include <psdr/sensor/perspective.h>
#include <psdr/shape/mesh.h>

#include <psdr/scene/scene_optix.h>
#include <psdr/scene/scene_loader.h>
#include <psdr/scene/scene.h>

NAMESPACE_BEGIN(psdr_jit)

Scene::Scene() {
    m_has_bound_mesh = false;
    m_loaded = false;
    m_samplers = new Sampler[3];
    m_optix = new Scene_OptiX();
    m_emitter_env = nullptr;
    m_emitters_distrb = new DiscreteDistribution();
}


Scene::~Scene() {
    for ( Sensor*  s : m_sensors  ) delete s;
    for ( Emitter* e : m_emitters ) delete e;
    for ( BSDF*    b : m_bsdfs    ) delete b;
    for ( Mesh*    m : m_meshes   ) delete m;

    delete[]    m_samplers;
    delete      m_optix;
    delete      m_emitters_distrb;
    // delete      m_sec_edge_distrb;

}


void Scene::load_file(const char *file_name, bool auto_configure) {
    SceneLoader::load_from_file(file_name, *this);
    if ( auto_configure ) configure();
}


void Scene::load_string(const char *scene_xml, bool auto_configure) {
    SceneLoader::load_from_string(scene_xml, *this);
    if ( auto_configure ) configure();
}

void Scene::configure() {
    PSDR_ASSERT_MSG(m_loaded, "Scene not loaded yet!");
    PSDR_ASSERT(m_num_sensors == static_cast<int>(m_sensors.size()));
    PSDR_ASSERT(m_num_meshes == static_cast<int>(m_meshes.size()));

    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();

    // Seed samplers
    if ( m_opts.spp  > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.spp;
        if ( m_samplers[0].m_sample_count != sample_count )
            m_samplers[0].seed(arange<UInt64C>(sample_count)+seed);
    }
    if ( m_opts.sppe > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.sppe;
        if ( m_samplers[1].m_sample_count != sample_count )
            m_samplers[1].seed(arange<UInt64C>(sample_count)+seed);
    }
    if ( m_opts.sppse > 0 ) {
        int64_t sample_count = static_cast<int64_t>(m_opts.height)*m_opts.width*m_opts.sppse;
        if ( m_samplers[2].m_sample_count != sample_count )
            m_samplers[2].seed(arange<UInt64C>(sample_count)+seed);
    }

    // Preprocess meshes
    PSDR_ASSERT_MSG(!m_meshes.empty(), "Missing meshes!");
    std::vector<int> face_offset, edge_offset, cut_offset;
    face_offset.reserve(m_num_meshes + 1);
    face_offset.push_back(0);
    edge_offset.reserve(m_num_meshes + 1);
    edge_offset.push_back(0);

    cut_offset.reserve(m_num_meshes + 1);
    cut_offset.push_back(0);

    m_lower = full<Vector3fC>(std::numeric_limits<float>::max());
    m_upper = full<Vector3fC>(std::numeric_limits<float>::min());
    for ( Mesh *mesh : m_meshes ) {
        mesh->configure();
        face_offset.push_back(face_offset.back() + mesh->m_num_faces);
        edge_offset.push_back(edge_offset.back());
        cut_offset.push_back(cut_offset.back());
        for ( int i = 0; i < 3; ++i ) {
            m_lower[i] = minimum(m_lower[i], min(detach(mesh->m_vertex_positions[i])));
            m_upper[i] = maximum(m_upper[i], max(detach(mesh->m_vertex_positions[i])));
        }
    }

    

    // Preprocess sensors
    PSDR_ASSERT_MSG(!m_sensors.empty(), "Missing sensor!");
    std::vector<size_t> num_edges;
    if ( m_opts.sppe > 0 ) num_edges.reserve(m_sensors.size());
    for ( Sensor *sensor : m_sensors ) {
        sensor->m_resolution = ScalarVector2i(m_opts.width, m_opts.height);
        sensor->m_scene = this;
        sensor->configure();
        if ( m_opts.sppe > 0 ) num_edges.push_back(sensor->m_edge_distrb.m_size);

        for ( int i = 0; i < 3; ++i ) {
            if ( PerspectiveCamera *camera = dynamic_cast<PerspectiveCamera *>(sensor) ) {
                m_lower[i] = minimum(m_lower[i], detach(camera->m_camera_pos[i]));
                m_upper[i] = maximum(m_upper[i], detach(camera->m_camera_pos[i]));
            }
        }
    }

    

    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        oss << "AABB: [lower = " << m_lower << ", upper = " << m_upper << "]";
        log(oss.str().c_str());

        if ( m_opts.sppe > 0 ) {
            std::stringstream oss;
            oss << "(" << num_edges[0];
            for ( size_t i = 1; i < num_edges.size(); ++i ) oss << ", " << num_edges[i];
            oss << ") primary edges initialized.";
            log(oss.str().c_str());
        }
    }




    // Handling env. lighting
    if ( m_emitter_env != nullptr && !m_has_bound_mesh ) {
        FloatC margin = min((m_upper - m_lower)*0.05f);
        m_lower -= margin; m_upper += margin;

        m_emitter_env->m_lower = m_lower;
        m_emitter_env->m_upper = m_upper;

        // Adding a bounding mesh
        std::array<std::vector<float>, 3> lower_, upper_;
        copy_cuda_array<float, 3>(m_lower, lower_);
        copy_cuda_array<float, 3>(m_upper, upper_);

        float vtx_data[3][8];
        for ( int i = 0; i < 8; ++i )
            for ( int j = 0; j < 3; ++j )
                vtx_data[j][i] = (i & (1 << j)) ? upper_[j][0] : lower_[j][0];

        const int face_data[3][12] = {
            0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 4, 4,
            1, 3, 5, 7, 3, 7, 5, 4, 2, 6, 7, 6,
            3, 2, 7, 3, 7, 6, 1, 5, 6, 4, 5, 7
        };

        Mesh *bound_mesh = new Mesh();
        bound_mesh->m_num_vertices = 8;
        bound_mesh->m_num_faces = 12;
        bound_mesh->m_use_face_normals = true;
        bound_mesh->m_enable_edges = false;
        bound_mesh->m_bsdf = nullptr;
        bound_mesh->m_emitter = m_emitter_env;
        for ( int i = 0; i < 3; ++i ) {
            bound_mesh->m_vertex_positions_raw[i] = load<FloatD>(vtx_data[i], 8);
            bound_mesh->m_face_indices[i] = load<IntD>(face_data[i], 12);
        }
        bound_mesh->configure();
        m_meshes.push_back(bound_mesh);

        face_offset.push_back(face_offset.back() + 12);
        edge_offset.push_back(edge_offset.back());

        ++m_num_meshes;
        m_has_bound_mesh = true;
        if ( m_opts.log_level > 0 ) {
            log("Bounding mesh added for environmental lighting.");
        }
    }


    // Preprocess emitters
    if ( !m_emitters.empty() ) {
        std::vector<float> weights;
        weights.reserve(m_emitters.size());
        for ( Emitter *e : m_emitters ) {
            e->configure();
            weights.push_back(e->m_sampling_weight);
        }
        m_emitters_distrb->init(load<FloatC>(weights.data(), weights.size()));

        float inv_total_weight = rcp(m_emitters_distrb->m_sum)[0];
        for ( Emitter *e : m_emitters ) {
            e->m_sampling_weight *= inv_total_weight;
        }
    }


    // PSDR_ASSERT(0);


    // Initialize CUDA arrays
    if ( !m_emitters.empty() ) {
        m_emitters_cuda = load<EmitterArrayD>(m_emitters.data(), m_emitters.size());

    }
    m_meshes_cuda = load<MeshArrayD>(m_meshes.data(), m_num_meshes);

    // Generate global triangle arrays
    m_triangle_info = empty<TriangleInfoD>(face_offset.back());
    m_triangle_uv = zeros<TriangleUVD>(face_offset.back());
    m_triangle_face_normals = empty<MaskD>(face_offset.back());
    for ( int i = 0; i < m_num_meshes; ++i ) {
        const Mesh &mesh = *m_meshes[i];
        const IntD idx = arange<IntD>(mesh.m_num_faces) + face_offset[i];
        scatter(m_triangle_info, *mesh.m_triangle_info, idx);
        scatter(m_triangle_face_normals, MaskD(mesh.m_use_face_normals), idx);
        if ( mesh.m_has_uv ) {
            scatter(m_triangle_uv, *mesh.m_triangle_uv, idx);
        }
    }

    // Initialize OptiX

    m_optix->configure(m_meshes);

    // Cleanup
    for ( int i = 0; i < m_num_meshes; ++i ) {
        Mesh *mesh = m_meshes[i];

        if ( mesh->m_emitter == nullptr ) {
            // std::cout << "clean emitter" << std::endl;
            PSDR_ASSERT(mesh->m_triangle_info != nullptr);
            delete mesh->m_triangle_info;
            mesh->m_triangle_info = nullptr;

            if ( mesh->m_triangle_uv != nullptr ) {
                delete mesh->m_triangle_uv;
                mesh->m_triangle_uv = nullptr;
            }
        }
    }

    auto end_time = high_resolution_clock::now();
    if ( m_opts.log_level > 0 ) {
        std::stringstream oss;
        oss << "Configured in " << duration_cast<duration<double>>(end_time - start_time).count() << " seconds.";
        log(oss.str().c_str());
    }
}


bool Scene::is_ready() const {
    return (m_opts.spp   == 0 || m_samplers[0].is_ready()) &&
           (m_opts.sppe  == 0 || m_samplers[1].is_ready()) &&
           (m_opts.sppse == 0 || m_samplers[2].is_ready()) &&
           m_optix->is_ready();
}


template <bool ad, bool path_space>
Intersection<ad> Scene::ray_intersect(const Ray<ad> &ray, Mask<ad> active) const {
    static_assert(ad || !path_space);
    Intersection<ad> its;

    Vector2i<ad> idx = m_optix->ray_intersect<ad>(ray, active);


    TriangleUV<ad>      tri_uv_info;
    Mask<ad>            face_normal_mask;



    // TODO: FIX
    Vector3f<ad> tri_info_p0;
    Vector3f<ad> tri_info_e1; 
    Vector3f<ad> tri_info_e2; 
    Vector3f<ad> tri_info_n0; 
    Vector3f<ad> tri_info_n1; 
    Vector3f<ad> tri_info_n2; 


    if constexpr ( ad ) {
        its.n = gather<Vector3f<ad>>(m_triangle_info.face_normal, idx[1], active);
        tri_uv_info = gather<TriangleUVD>(m_triangle_uv, idx[1], active);
        face_normal_mask = gather<MaskD>(m_triangle_face_normals, idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(m_triangle_info.p0, idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(m_triangle_info.e1, idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(m_triangle_info.e2, idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(m_triangle_info.n0, idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(m_triangle_info.n1, idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(m_triangle_info.n2, idx[1], active);

    } else {
        its.n = gather<Vector3f<ad>>(detach(m_triangle_info.face_normal), idx[1], active);

        tri_uv_info = gather<TriangleUVC>(detach(m_triangle_uv), idx[1], active);
        face_normal_mask = gather<MaskC>(detach(m_triangle_face_normals), idx[1], active);

        tri_info_p0 = gather<Vector3f<ad>>(detach(m_triangle_info.p0), idx[1], active);
        tri_info_e1 = gather<Vector3f<ad>>(detach(m_triangle_info.e1), idx[1], active);
        tri_info_e2 = gather<Vector3f<ad>>(detach(m_triangle_info.e2), idx[1], active);
        tri_info_n0 = gather<Vector3f<ad>>(detach(m_triangle_info.n0), idx[1], active);
        tri_info_n1 = gather<Vector3f<ad>>(detach(m_triangle_info.n1), idx[1], active);
        tri_info_n2 = gather<Vector3f<ad>>(detach(m_triangle_info.n2), idx[1], active);


    }
    its.J = 1.f;

    const Vector3f<ad> &vertex0 = tri_info_p0, &edge1 = tri_info_e1, &edge2 = tri_info_e2;

    // calculate dudp
    Vector3f<ad> dp0 = edge1,
                 dp1 = edge2;
    Vector2f<ad> uv0 = tri_uv_info[0],
                 uv1 = tri_uv_info[1],
                 uv2 = tri_uv_info[2];
    Vector2f<ad> duv0 = uv1 - uv0,
                 duv1 = uv2 - uv0;
    Float<ad> det     = fmsub(duv0.x(), duv1.y(), duv0.y() * duv1.x()),
              inv_det = rcp(det);
    Mask<ad>  valid_dp = neq(det, 0.f);

    if constexpr ( !ad || path_space ) {
        // Path-space formulation
        const Vector2fC &uv = m_optix->m_its.uv;
        Vector3f<ad> sh_n = normalize(bilinear<ad>(tri_info_n0,
                                                   tri_info_n1 - tri_info_n0,
                                                   tri_info_n2 - tri_info_n0,
                                                   uv));
        masked(sh_n, face_normal_mask) = its.n;

        if constexpr ( ad ) {
            its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        } else {
            its.shape = gather<MeshArrayC>(detach(m_meshes_cuda), idx[0], active);
        }
        its.p = bilinear<ad>(vertex0, edge1, edge2, uv);

        Vector3f<ad> dir = its.p - ray.o;
        its.t = norm(dir);
        dir /= its.t;
        its.uv = bilinear2<ad>(tri_uv_info[0],
                               tri_uv_info[1] - tri_uv_info[0],
                               tri_uv_info[2] - tri_uv_info[0],
                               uv);

        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-dir);


    } else {
        // Standard (solid-angle) formulation
        auto [uv, t] = ray_intersect_triangle<true>(vertex0, edge1, edge2, ray);
        Vector3fD sh_n = normalize(bilinear<true>(tri_info_n0,
                                                  tri_info_n1 - tri_info_n0,
                                                  tri_info_n2 - tri_info_n0,
                                                  uv));

        masked(sh_n, face_normal_mask) = its.n;

        its.shape = gather<MeshArrayD>(m_meshes_cuda, idx[0], active);
        its.p = ray(t);
        its.t = t;
        its.uv = bilinear2<true>(tri_uv_info[0],
                                 tri_uv_info[1] - tri_uv_info[0],
                                 tri_uv_info[2] - tri_uv_info[0],
                                 uv);
        // calculate dudp
        its.dp_du = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_dv = zeros<Vector3f<ad>>(slices(sh_n));
        its.dp_du[valid_dp] = fmsub( duv1.y(), dp0, duv0.y() * dp1) * inv_det;
        its.dp_dv[valid_dp] = fnmadd(duv1.x(), dp0, duv0.x() * dp1) * inv_det;
        its.sh_frame = Frame<ad>(sh_n);
        its.sh_frame.s[valid_dp] = normalize(fnmadd(its.sh_frame.n, dot(its.sh_frame.n, its.dp_du), its.dp_du));
        its.sh_frame.t[valid_dp] = cross(its.sh_frame.n, its.sh_frame.s);
        its.wi = its.sh_frame.to_local(-ray.d);

    }

    drjit::eval(its);
    return its;
}


template <bool ad>
PositionSample<ad> Scene::sample_emitter_position(const Vector3f<ad> &ref_p, const Vector2f<ad> &_sample2, Mask<ad> active) const {
    PSDR_ASSERT_MSG(!m_emitters.empty(), "No Emitter!");

    if ( m_emitters.size() == 1U ) {
        return m_emitters[0]->sample_position(ref_p, _sample2, active);
    } else {
        PositionSample<ad> result;
        Vector2f<ad> sample2 = _sample2;
        auto [emitter_index, emitter_pdf] = m_emitters_distrb->sample_reuse<ad>(sample2.y());

        EmitterArray<ad> emitter_arr;
        if constexpr ( ad ) {
            emitter_arr = gather<EmitterArrayD>(m_emitters_cuda, IntD(emitter_index), active);
            result = emitter_arr->sample_positionD(ref_p, sample2, active);
        } else {
            emitter_arr = gather<EmitterArrayC>(detach(m_emitters_cuda), emitter_index, active);
            result = emitter_arr->sample_positionC(ref_p, sample2, active);
        }
        result.pdf *= emitter_pdf;

        return result;
    }

    
    
}


template <bool ad>
Float<ad> Scene::emitter_position_pdf(const Vector3f<ad> &ref_p, const Intersection<ad> &its, Mask<ad> active) const {
    if constexpr (ad) {
        return its.shape->emitter()->sample_position_pdfD(ref_p, its, active);
    } else {
        return its.shape->emitter()->sample_position_pdfC(ref_p, its, active);
    }
    
}

// Explicit instantiations
template IntersectionC Scene::ray_intersect<false, false>(const RayC&, MaskC) const;
template IntersectionD Scene::ray_intersect<true , false>(const RayD&, MaskD) const;
template IntersectionD Scene::ray_intersect<true , true >(const RayD&, MaskD) const;

template PositionSampleC Scene::sample_emitter_position<false>(const Vector3fC&, const Vector2fC&, MaskC) const;
template PositionSampleD Scene::sample_emitter_position<true >(const Vector3fD&, const Vector2fD&, MaskD) const;

template FloatC Scene::emitter_position_pdf<false>(const Vector3fC&, const IntersectionC&, MaskC) const;
template FloatD Scene::emitter_position_pdf<true >(const Vector3fD&, const IntersectionD&, MaskD) const;

NAMESPACE_END(psdr_jit)
