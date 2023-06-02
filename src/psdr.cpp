#include <psdr/psdr.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
// #include <pybind11/stl_bind.h>

#include <psdr/core/ray.h>
#include <psdr/core/frame.h>
#include <psdr/core/bitmap.h>
#include <psdr/core/intersection.h>
#include <psdr/core/sampler.h>
#include <psdr/core/records.h>
#include <psdr/core/warp.h>
#include <psdr/core/transform.h>
#include <psdr/core/records.h>

#include <psdr/edge/edge.h>

#include <psdr/sensor/sensor.h>

#include <psdr/sensor/perspective.h>
#include <psdr/sensor/orthographic.h>

#include <psdr/bsdf/bsdf.h>
#include <psdr/core/pmf.h>

#include <psdr/emitter/emitter.h>

#include <psdr/emitter/area.h>
#include <psdr/emitter/envmap.h>

#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>


namespace py = pybind11;
using namespace py::literals;
using namespace psdr_jit;

using namespace drjit;

PYBIND11_MODULE(psdr_jit, m) {
    py::module::import("drjit");
    py::module::import("drjit.cuda");
    py::module::import("drjit.cuda.ad");

    // jit_set_flag(JitFlag::LoopRecord, false);
    // jit_set_flag(JitFlag::VCallRecord, false);


    m.doc() = "Path-space differentiable renderer";
    m.attr("__name__") = "psdr_jit";

    m.attr("ShadowEpsilon") = psdr_jit::ShadowEpsilon;
    m.attr("Epsilon") = psdr_jit::Epsilon;
    m.attr("InvPi") = psdr_jit::InvPi;
    

    m.def("ray_intersect_triangleC", &ray_intersect_triangle<false>);
    m.def("ray_intersect_triangleD", &ray_intersect_triangle<true>);

    m.def("bilinearC", &bilinear<false>);
    m.def("bilinearD", &bilinear<true>);

    m.def("mis_weightC", &mis_weight<false>);
    m.def("mis_weightD", &mis_weight<true>);

    m.def("square_to_cosine_hemisphere", &warp::square_to_cosine_hemisphere<true>);
    m.def("square_to_cosine_hemisphere_pdf", &warp::square_to_cosine_hemisphere_pdf<true>);

    // Core classes

    py::class_<Object>(m, "Object")
        .def("type_name", &Object::type_name)
        .def_readonly("id", &Object::m_id)
        .def("__repr__", &Object::to_string);

    // py::class_<ref<Mesh>>(m, "Meshptr");


    py::class_<RenderOption>(m, "RenderOption")
        .def(py::init<>())
        .def(py::init<int, int, int>(), "width"_a, "height"_a, "spp/sppe"_a)
        .def(py::init<int, int, int, int>(), "width"_a, "height"_a, "spp"_a, "sppe"_a)
        .def(py::init<int, int, int, int, int>(), "width"_a, "height"_a, "spp"_a, "sppe"_a, "sppse"_a)
        .def_readwrite("width", &RenderOption::width)
        .def_readwrite("height", &RenderOption::height)
        .def_readwrite("spp", &RenderOption::spp)
        .def_readwrite("sppe", &RenderOption::sppe)
        .def_readwrite("sppse", &RenderOption::sppse)
        .def_readwrite("log_level", &RenderOption::log_level)
        .def("__repr__",
            [](const RenderOption &ro) {
                std::stringstream oss;
                oss << "[width: " << ro.width << ", height: " << ro.height
                    << ", spp: " << ro.spp << ", sppe: " << ro.sppe << ", sppse: " << ro.sppse
                    << ", log_level: " << ro.log_level << "]";
                return oss.str();
            }
        );

    py::class_<PrimaryEdgeSample>(m, "PrimaryEdgeSample")
        .def_readwrite("x_dot_n", &PrimaryEdgeSample::x_dot_n)
        .def_readwrite("ray_n", &PrimaryEdgeSample::ray_n)
        .def_readwrite("ray_p", &PrimaryEdgeSample::ray_p)
        .def_readwrite("pdf", &PrimaryEdgeSample::pdf)
        .def_readwrite("idx", &PrimaryEdgeSample::idx);

    py::class_<RayC>(m, "RayC")
        .def(py::init<>())
        .def(py::init<const Vector3fC &, const Vector3fC &>())
        .def("reversed", &RayC::reversed)
        .def_readwrite("o", &RayC::o)
        .def_readwrite("d", &RayC::d);

    py::class_<RayD>(m, "RayD")
        .def(py::init<>())
        .def(py::init<const Vector3fD &, const Vector3fD &>())
        .def("reversed", &RayD::reversed)
        .def_readwrite("o", &RayD::o)
        .def_readwrite("d", &RayD::d);

    py::class_<TriangleInfoD>(m, "TriangleInfoD")
        .def_readwrite("p0", &TriangleInfoD::p0)
        .def_readwrite("e1", &TriangleInfoD::e1)
        .def_readwrite("e2", &TriangleInfoD::e2);

    py::class_<FrameC>(m, "FrameC")
        .def(py::init<>())
        .def(py::init<const Vector3fC &>())
        .def("cos_theta", &FrameC::cos_theta)
        
        .def("to_world", &FrameC::to_world)
        .def("to_local", &FrameC::to_local)
        .def_readwrite("s", &FrameC::s)
        .def_readwrite("t", &FrameC::t)
        .def_readwrite("n", &FrameC::n);

    py::class_<FrameD>(m, "FrameD")
        .def(py::init<>())
        .def(py::init<const Vector3fD &>())
        .def("cos_theta", &FrameD::cos_theta)
        .def("to_world", &FrameD::to_world)
        .def("to_local", &FrameD::to_local)
        .def_readwrite("s", &FrameD::s)
        .def_readwrite("t", &FrameD::t)
        .def_readwrite("n", &FrameD::n);

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("seed", &Sampler::seed)
        .def("next_1d", &Sampler::next_1d<false>)
        .def("next_2d", &Sampler::next_2d<false>)
        .def("next_3d", &Sampler::next_3d<false>);

    py::class_<BoundarySegSampleDirect>(m, "BoundarySegSampleDirect")
        .def_readwrite("pdf", &BoundarySegSampleDirect::pdf)
        .def_readwrite("p0", &BoundarySegSampleDirect::p0)
        .def_readwrite("edge", &BoundarySegSampleDirect::edge)
        .def_readwrite("edge2", &BoundarySegSampleDirect::edge2)
        .def_readwrite("p2", &BoundarySegSampleDirect::p2)
        .def_readwrite("is_valid", &BoundarySegSampleDirect::is_valid);

    py::class_<DiscreteDistribution>(m, "DiscreteDistribution")
        .def(py::init<>())
        .def("init", &DiscreteDistribution::init)
        .def("sample", &DiscreteDistribution::sample)
        .def_readonly("sum", &DiscreteDistribution::m_sum)
        .def("pmf", &DiscreteDistribution::pmf);


    py::class_<Bitmap1fD>(m, "Bitmap1fD")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<int, int, const FloatD&>())
        .def(py::init<const char*>())
        .def("load_openexr", &Bitmap1fD::load_openexr)
        .def("eval", &Bitmap1fD::eval<true>, "uv"_a, "flip_v"_a = true)
        .def_readwrite("resolution", &Bitmap1fD::m_resolution)
        .def_readwrite("data", &Bitmap1fD::m_data)
        .def_readwrite("translate", &Bitmap1fD::m_trans)
        .def_readwrite("rotate", &Bitmap1fD::m_rot)
        .def_readwrite("scale", &Bitmap1fD::m_scale);

    py::class_<Bitmap3fD>(m, "Bitmap3fD")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<int, int, const SpectrumD&>())
        .def(py::init<const char*>())
        .def("load_openexr", &Bitmap3fD::load_openexr)
        .def("eval", &Bitmap3fD::eval<true>, "uv"_a, "flip_v"_a = true)
        .def_readwrite("resolution", &Bitmap3fD::m_resolution)
        .def_readwrite("data", &Bitmap3fD::m_data)
        .def_readwrite("translate", &Bitmap3fD::m_trans)
        .def_readwrite("rotate", &Bitmap3fD::m_rot)
        .def_readwrite("scale", &Bitmap3fD::m_scale);


    py::class_<InteractionD>(m, "Interaction")
        .def("is_valid", &InteractionD::is_valid)
        .def_readonly("wi", &InteractionD::wi)
        .def_readonly("p", &InteractionD::p)
        .def_readonly("t", &InteractionD::t);

    py::class_<IntersectionD, InteractionD>(m, "Intersection")
        .def("is_emitter", &IntersectionD::is_emitter)
        .def("Le", &IntersectionD::Le)
        .def("get_bsdf", &IntersectionD::get_bsdf)
        .def_readonly("shape", &IntersectionD::shape)
        .def_readonly("n", &IntersectionD::n)
        .def_readonly("sh_frame", &IntersectionD::sh_frame)
        .def_readonly("uv", &IntersectionD::uv)
        .def_readonly("J", &IntersectionD::J);

 
    {
        py::object dr       = py::module_::import("drjit"),
                   dr_array = dr.attr("ArrayBase");
        py::class_<BSDFArrayD> cls(m, "BSDFArray", dr_array);

        cls.def("eval",
                [](BSDFArrayD bsdf, IntersectionD its, Vector3fD wi, MaskD active) {
                    return bsdf->eval(its, wi, active);
                });
    };

    // MeshArray
    {
        py::object dr       = py::module_::import("drjit"),
                   dr_array = dr.attr("ArrayBase");
        py::class_<MeshArrayD> cls(m, "MeshArrayD", dr_array);

        cls.def("diffuse",
                [](MeshArrayD shape, IntersectionD its, MaskD active) {
                    return shape->bsdf()->eval_type(its, active);
                });
        cls.def("bsdf",
                [](MeshArrayD shape) {
                    return shape->bsdf();
                });
        cls.def("bsdf_eval",
                [](MeshArrayD shape, IntersectionD its, Vector3fD wi, MaskD active) {
                    return shape->bsdf()->eval(its, wi, active);
                });
        cls.def("bsdf_sample",
                [](MeshArrayD shape, IntersectionD its, Vector3fD rnd, MaskD active) {
                    return shape->bsdf()->sample(its, rnd, active);
                });
        cls.def("bsdf_pdf",
                [](MeshArrayD shape, IntersectionD its, Vector3fD rnd, MaskD active) {
                    return shape->bsdf()->pdf(its, rnd, active);
                });
        cls.def("bsdf_valid",
                [](MeshArrayD shape) {
                    return neq(shape->bsdf(), nullptr);
                });
    };
    {
        py::object dr       = py::module_::import("drjit"),
                   dr_array = dr.attr("ArrayBase");
        py::class_<MeshArrayC> cls(m, "MeshArrayC", dr_array);

        cls.def("diffuse",
                [](MeshArrayC shape, IntersectionC its, MaskC active) {
                    return shape->bsdf()->eval_type(its, active);
                });
        cls.def("bsdf",
                [](MeshArrayC shape) {
                    return shape->bsdf();
                });
        cls.def("bsdf_eval",
                [](MeshArrayC shape, IntersectionC its, Vector3fC wi, MaskC active) {
                    return shape->bsdf()->eval(its, wi, active);
                });
        cls.def("bsdf_sample",
                [](MeshArrayC shape, IntersectionC its, Vector3fC rnd, MaskC active) {
                    return shape->bsdf()->sample(its, rnd, active);
                });
        cls.def("bsdf_pdf",
                [](MeshArrayC shape, IntersectionC its, Vector3fC rnd, MaskC active) {
                    return shape->bsdf()->pdf(its, rnd, active);
                });
        cls.def("bsdf_valid",
                [](MeshArrayC shape) {
                    return neq(shape->bsdf(), nullptr);
                });
    };

    py::class_<SampleRecordC>(m, "SampleRecordC")
        .def_readonly("pdf", &SampleRecordC::pdf)
        .def_readonly("is_valid", &SampleRecordC::is_valid);

    py::class_<SampleRecordD>(m, "SampleRecordD")
        .def_readonly("pdf", &SampleRecordD::pdf)
        .def_readonly("is_valid", &SampleRecordD::is_valid);

    py::class_<PositionSampleC, SampleRecordC>(m, "PositionSampleC")
        .def_readonly("p", &PositionSampleC::p)
        .def_readonly("J", &PositionSampleC::J);

    py::class_<PositionSampleD, SampleRecordD>(m, "PositionSampleD")
        .def_readonly("p", &PositionSampleD::p)
        .def_readonly("J", &PositionSampleD::J);

    py::class_<BSDF, PyBSDF /* <--- trampoline*/>(m, "BSDF")
        .def(py::init<>())
        .def_readwrite("id", &BSDF::m_id)
        .def("__repr__", &BSDF::to_string)
        .def("to_string", &BSDF::to_string)
        .def("eval", &BSDF::eval)
        .def("anisotropic", &BSDF::anisotropic)
        .def("test_vir", &BSDF::test_vir);
    // Shapes

    py::class_<Mesh, Object>(m, "Mesh")
        .def(py::init<>())
        .def("load_raw", &Mesh::load_raw, "v"_a,  "f"_a, "verbose"_a = false)
        .def("load", &Mesh::load, "filename"_a, "verbose"_a = false)
        .def("configure", &Mesh::configure)
        .def("set_transform", &Mesh::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Mesh::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fC&, MaskC>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fD&, MaskD>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def_readonly("num_vertices", &Mesh::m_num_vertices)
        .def_readonly("num_faces", &Mesh::m_num_faces)
        .def_readonly("bsdf", &Mesh::m_bsdf)
        .def("bsdf_array", &Mesh::bsdf)
        .def_readwrite("to_world", &Mesh::m_to_world_raw)
        .def_readwrite("to_world_right", &Mesh::m_to_world_right)
        .def_readwrite("to_world_left", &Mesh::m_to_world_left)
        .def_readwrite("vertex_positions_T", &Mesh::m_vertex_positions, "Object-space vertex positions")
        .def_readwrite("vertex_positions", &Mesh::m_vertex_positions_raw, "Object-space vertex positions")
        .def_readonly("vertex_normals", &Mesh::m_vertex_normals_raw, "Object-space vertex normals")
        .def_readwrite("vertex_uv", &Mesh::m_vertex_uv)
        .def_readwrite("face_indices", &Mesh::m_face_indices)
        .def_readwrite("face_uv_indices", &Mesh::m_face_uv_indices)
        .def_readwrite("valid_edge_indices", &Mesh::m_valid_edge_indices)
        .def_readwrite("use_face_normal", &Mesh::m_use_face_normals)
        .def_readwrite("enable_edges", &Mesh::m_enable_edges) // TODO
        .def("edge_indices", [](const Mesh &mesh) { return head<4>(mesh.m_edge_indices); })
        .def("dump", &Mesh::dump);


    py::class_<Emitter, Object>(m, "Emitter");

    py::class_<AreaLight, Emitter>(m, "AreaLight")
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<const ScalarVector3f&, const Mesh*>())
        .def_readwrite("radiance", &AreaLight::m_radiance);

    py::class_<EnvironmentMap, Emitter>(m, "EnvironmentMap")
        .def(py::init<const char *>())
        .def_readonly("to_world", &EnvironmentMap::m_to_world_raw)
        .def("set_transform", &EnvironmentMap::set_transform)
        .def_readwrite("radiance", &EnvironmentMap::m_radiance)
        .def_readwrite("scale", &EnvironmentMap::m_scale);
        
    py::class_<Sensor, Object>(m, "Sensor")
        .def("sample_primary_edge", &Sensor::sample_primary_edge)
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_primary_ray", &Sensor::sample_primary_ray)
        .def_readwrite("enable_edges", &Sensor::m_enable_edges)
        .def_readwrite("to_world", &Sensor::m_to_world_raw)
        .def_readwrite("to_world_left", &Sensor::m_to_world_left)
        .def_readwrite("to_world_right", &Sensor::m_to_world_right);



    py::class_<PerspectiveCamera, Sensor>(m, "PerspectiveCamera")
        .def(py::init<float, float, float>())
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_direct", &Sensor::sample_direct, "vec"_a)
        .def_readonly("world_to_sample", &PerspectiveCamera::m_world_to_sample)
        .def_readwrite("to_world", &PerspectiveCamera::m_to_world_raw)
        .def_readwrite("to_world_left", &Sensor::m_to_world_left)
        .def_readwrite("to_world_right", &Sensor::m_to_world_right);

    py::class_<OrthographicCamera, Sensor>(m, "OrthographicCamera")
        .def(py::init<float, float>())
        .def("set_transform", &Sensor::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Sensor::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_direct", &Sensor::sample_direct, "vec"_a)
        .def_readonly("world_to_sample", &OrthographicCamera::m_world_to_sample)
        .def_readwrite("to_world", &OrthographicCamera::m_to_world_raw)
        .def_readwrite("to_world_left", &Sensor::m_to_world_left)
        .def_readwrite("to_world_right", &Sensor::m_to_world_right);

    py::class_<SensorDirectSample_<FloatD>, SampleRecord_<FloatD>>(m, "SensorDirectSample")
        .def_readwrite("q", &SensorDirectSample_<FloatD>::q)
        .def_readwrite("pixel_idx", &SensorDirectSample_<FloatD>::pixel_idx)
        .def_readwrite("sensor_val", &SensorDirectSample_<FloatD>::sensor_val)
        .def_readwrite("is_valid", &SensorDirectSample_<FloatD>::is_valid);

    py::class_<BSDFSampleD>(m, "BSDFSample")
        .def(py::init<>())
        .def_readwrite("pdf", &BSDFSampleD::pdf)
        .def_readwrite("is_valid", &BSDFSampleD::is_valid)
        .def_readwrite("eta", &BSDFSampleD::eta)
        .def_readwrite("wo", &BSDFSampleD::wo);

    py::class_<Scene, Object>(m, "Scene")
        .def(py::init<>())
        .def("has_envmap", &Scene::has_envmap)
        .def("sample_boundary_segment_direct", &Scene::sample_boundary_segment_direct)
        .def("add_Sensor", &Scene::add_Sensor, "Add Sensor")
        .def("add_EnvironmentMap", &Scene::add_EnvironmentMap, "Add EnvironmentMap")
        .def("add_Mesh", &Scene::add_Mesh, "Add Mesh")
        .def("add_BSDF", &Scene::add_BSDF, "Add BSDf", "bsdf"_a, "name"_a, "twoSide"_a = false)
        .def("sample_emitter_position", &Scene::sample_emitter_position<true>)

        .def_readonly("sensor", &Scene::m_sensors)

        .def("boundary_ray_intersect", &Scene::boundary_ray_intersect<false>)
        .def("boundary_ray_intersectAD", &Scene::boundary_ray_intersect<true>)
        .def("boundary_ray_intersectADAD", &Scene::boundary_ray_intersect<true, true>)

        .def("emitter_position_pdfC", &Scene::emitter_position_pdf<false>)
        .def("emitter_position_pdfD", &Scene::emitter_position_pdf<true>)

        .def("ray_intersect", &Scene::ray_intersect<false>)
        .def("ray_intersectAD", &Scene::ray_intersect<true>)
        .def("ray_intersectADAD", &Scene::ray_intersect<true, true>)

        .def("configure", &Scene::configure, "active_sensor"_a=std::vector<int>())
        .def_readwrite("opts", &Scene::m_opts, "Render options")
        .def_readwrite("seed", &Scene::seed, "Sample seed")
        .def_readonly("num_sensors", &Scene::m_num_sensors)
        .def_readonly("num_meshes", &Scene::m_num_meshes)
        .def_readonly("param_map", &Scene::m_param_map, "Parameter map");
}
