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


#include <psdr/bsdf/bsdf.h>
#include <psdr/bsdf/diffuse.h>
#include <psdr/bsdf/ggx.h>
#include <psdr/bsdf/roughconductor.h>
#include <psdr/bsdf/roughdielectric.h>
#include <psdr/bsdf/microfacet.h>

#include <psdr/bsdf/normalmap.h>

#include <psdr/core/pmf.h>

#include <psdr/emitter/emitter.h>

#include <psdr/emitter/area.h>
// #include <psdr/emitter/envmap.h>

#include <psdr/shape/mesh.h>
#include <psdr/scene/scene.h>
#include <psdr/scene/scene_loader.h>


#include <psdr/integrator/integrator.h>
#include <psdr/integrator/field.h>
#include <psdr/integrator/collocated.h>
#include <psdr/integrator/path.h>

#include <psdr/jit_optix_test.h>


namespace py = pybind11;
using namespace py::literals;
using namespace psdr_jit;

using namespace drjit;

void optix_jit_test() {
    jit_test aa;;
    aa.trace_ray();
}

void drjit_test() {
    jit_init((uint32_t)JitBackend::CUDA);

    FloatD a = arange<FloatD>(10);
    enable_grad(a);

    FloatD b = a * 2.f;
    std::cout << "val: " << b << std::endl;

    backward(b);
    std::cout << "grad: " << grad(a) << std::endl;


    Matrix4fD           m_to_world_raw = identity<Matrix4fD>();
    std::cout << m_to_world_raw << std::endl;

}


void drjit_memory() {
    FloatC a(1.f, 2.f, 3.f, 4.f);
    std::vector<float> b;
    b.resize(a.size());
    // std::cout << a.slices() << std::endl;


    std::cout << slices<Vector3fC>(a) << std::endl;
    drjit::store(b.data(), a);
}

PYBIND11_MODULE(psdr_jit, m) {
    py::module::import("drjit");
    py::module::import("drjit.cuda");
    py::module::import("drjit.cuda.ad");

    m.doc() = "Path-space differentiable renderer";
    m.attr("__name__") = "psdr_jit";

    m.def("optix_jit_test", &optix_jit_test);

    m.def("drjit_test", &drjit_test);

    m.def("drjit_memory", &drjit_memory);

    // Core classes

    py::class_<Object>(m, "Object")
        .def("type_name", &Object::type_name)
        .def_readonly("id", &Object::m_id)
        .def("__repr__", &Object::to_string);

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
    py::class_<EdgeSortOption>(m, "EdgeSortOption")
        .def(py::init<>())
        .def_readwrite("enable_sort", &EdgeSortOption::enable_sort)
        .def_readwrite("local_angle", &EdgeSortOption::local_angle)
        .def_readwrite("global_angle", &EdgeSortOption::global_angle)
        .def_readwrite("min_global_step", &EdgeSortOption::min_global_step)
        .def_readwrite("max_depth", &EdgeSortOption::max_depth);

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

    py::class_<FrameC>(m, "FrameC")
        .def(py::init<>())
        .def(py::init<const Vector3fC &>())
        .def_readwrite("s", &FrameC::s)
        .def_readwrite("t", &FrameC::t)
        .def_readwrite("n", &FrameC::n);

    py::class_<FrameD>(m, "FrameD")
        .def(py::init<>())
        .def(py::init<const Vector3fD &>())
        .def_readwrite("s", &FrameD::s)
        .def_readwrite("t", &FrameD::t)
        .def_readwrite("n", &FrameD::n);

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("seed", &Sampler::seed)
        .def("next_1d", &Sampler::next_1d<false>)
        .def("next_2d", &Sampler::next_2d<false>);

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
        .def_readwrite("data", &Bitmap1fD::m_data);
        // .def_readwrite("to_world", &Bitmap1fD::m_to_world_raw)
        // .def_readwrite("to_world_left", &Bitmap1fD::m_to_world_left)
        // .def_readwrite("to_world_right", &Bitmap1fD::m_to_world_right)
        // .def("set_transform", &Bitmap1fD::set_transform, "mat"_a, "set_left"_a = true)
        // .def("append_transform", &Bitmap1fD::append_transform, "mat"_a, "append_left"_a = true);

    py::class_<Bitmap3fD>(m, "Bitmap3fD")
        .def(py::init<>())
        .def(py::init<const ScalarVector3f&>())
        .def(py::init<int, int, const SpectrumD&>())
        .def(py::init<const char*>())
        .def("load_openexr", &Bitmap3fD::load_openexr)
        .def("eval", &Bitmap3fD::eval<true>, "uv"_a, "flip_v"_a = true)
        .def_readwrite("resolution", &Bitmap3fD::m_resolution)
        .def_readwrite("data", &Bitmap3fD::m_data);
        // .def_readwrite("to_world", &Bitmap3fD::m_to_world_raw)
        // .def_readwrite("to_world_left", &Bitmap3fD::m_to_world_left)
        // .def_readwrite("to_world_right", &Bitmap3fD::m_to_world_right)
        // .def("set_transform", &Bitmap3fD::set_transform, "mat"_a, "set_left"_a = true)
        // .def("append_transform", &Bitmap3fD::append_transform, "mat"_a, "append_left"_a = true);


    
    py::class_<InteractionC>(m, "InteractionC")
        .def("is_valid", &InteractionC::is_valid)
        .def_readonly("wi", &InteractionC::wi)
        .def_readonly("p", &InteractionC::p)
        .def_readonly("t", &InteractionC::t);

    py::class_<InteractionD>(m, "InteractionD")
        .def("is_valid", &InteractionD::is_valid)
        .def_readonly("wi", &InteractionD::wi)
        .def_readonly("p", &InteractionD::p)
        .def_readonly("t", &InteractionD::t);

    py::class_<IntersectionC, InteractionC>(m, "IntersectionC")
        .def_readonly("shape", &IntersectionC::shape)
        .def_readonly("n", &IntersectionC::n)
        .def_readonly("sh_frame", &IntersectionC::sh_frame)
        .def_readonly("uv", &IntersectionC::uv)
        .def_readonly("J", &IntersectionC::J);

    py::class_<IntersectionD, InteractionD>(m, "IntersectionD")
        .def_readonly("shape", &IntersectionD::shape)
        .def_readonly("n", &IntersectionD::n)
        .def_readonly("sh_frame", &IntersectionD::sh_frame)
        .def_readonly("uv", &IntersectionD::uv)
        .def_readonly("J", &IntersectionD::J);

 
    // Records

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


    py::class_<BSDF, Object>(m, "BSDF")
        .def("anisotropic", &BSDF::anisotropic);

    py::class_<NormalMap, BSDF>(m, "NormalMapBSDF")
        .def_readwrite("nested_bsdf", &NormalMap::m_bsdf)
        .def_readwrite("normal_map", &NormalMap::m_nmap);

    py::class_<Diffuse, BSDF>(m, "DiffuseBSDF")
        //.def(py::init<>())
        //.def(py::init<const ScalarVector3f&>())
        //.def(py::init<const char*>())
        //.def(py::init<const Bitmap3fD&>())
        .def_readwrite("reflectance", &Diffuse::m_reflectance);

    py::class_<RoughConductor, BSDF>(m, "RoughConductorBSDF")
        .def_readwrite("alpha_u", &RoughConductor::m_alpha_u)
        .def_readwrite("alpha_v", &RoughConductor::m_alpha_v)
        .def_readwrite("eta", &RoughConductor::m_eta)
        .def_readwrite("k", &RoughConductor::m_k)
        .def_readwrite("specular_reflectance", &RoughConductor::m_specular_reflectance);

    py::class_<RoughDielectric, BSDF>(m, "RoughDielectricBSDF");
    // Sensors

    py::class_<Microfacet, BSDF>(m, "MicrofacetBSDF")
        .def_readwrite("roughness", &Microfacet::m_roughness)
        .def_readwrite("diffuseReflectance", &Microfacet::m_diffuseReflectance)
        .def_readwrite("specularReflectance", &Microfacet::m_specularReflectance);


    // Shapes

    py::class_<Mesh, Object>(m, "Mesh")
        .def(py::init<>())
        .def("load", &Mesh::load, "filename"_a, "verbose"_a = false)
        .def("configure", &Mesh::configure)
        .def("set_transform", &Mesh::set_transform, "mat"_a, "set_left"_a = true)
        .def("append_transform", &Mesh::append_transform, "mat"_a, "append_left"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fC&, MaskC>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def("sample_position", py::overload_cast<const Vector2fD&, MaskD>(&Mesh::sample_position, py::const_), "sample2"_a, "active"_a = true)
        .def_readonly("num_vertices", &Mesh::m_num_vertices)
        .def_readonly("num_faces", &Mesh::m_num_faces)
        .def_readonly("bsdf", &Mesh::m_bsdf)
        .def_readwrite("to_world", &Mesh::m_to_world_raw)
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
        .def(py::init<const ScalarVector3f&, const Mesh*>());

    // py::class_<EnvironmentMap, Emitter>(m, "EnvironmentMap")
    //     .def(py::init<const char *>())
    //     .def_readonly("to_world", &EnvironmentMap::m_to_world_raw)
    //     .def("set_transform", &EnvironmentMap::set_transform)
    //     .def_readwrite("radiance", &EnvironmentMap::m_radiance)
    //     .def_readwrite("scale", &EnvironmentMap::m_scale);

    py::class_<Scene, Object>(m, "Scene")
        .def(py::init<>())
        .def("load_file", &Scene::load_file, "file_name"_a, "auto_configure"_a = true)
        .def("load_string", &Scene::load_string, "scene_xml"_a, "auto_configure"_a = true)
        .def("configure", &Scene::configure)
        .def("configure2", &Scene::configure2)
        .def_readwrite("opts", &Scene::m_opts, "Render options")
        .def_readwrite("seed", &Scene::seed, "Sample seed")
        .def_readonly("num_sensors", &Scene::m_num_sensors)
        .def_readonly("num_meshes", &Scene::m_num_meshes)
        .def_readonly("param_map", &Scene::m_param_map, "Parameter map");



    py::class_<Integrator, Object>(m, "Integrator")
        .def("renderC", &Integrator::renderC, "scene"_a, "sensor_id"_a = 0, "npass"_a = 1)
        .def("renderD", &Integrator::renderD, "scene"_a, "sensor_id"_a = 0);

    py::class_<FieldExtractionIntegrator, Integrator>(m, "FieldExtractionIntegrator")
        .def(py::init<char*>());


    py::class_<CollocatedIntegrator, Integrator>(m, "CollocatedIntegrator")
        .def_readwrite("m_intensity", &CollocatedIntegrator::m_intensity)
        .def(py::init<const FloatD &>());

    py::class_<PathTracer, Integrator>(m, "PathTracer")
        .def(py::init<int>(), "max_depth"_a = 1)
        .def_readwrite("hide_emitters", &PathTracer::m_hide_emitters);


}
