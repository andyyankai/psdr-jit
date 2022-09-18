#include <misc/Exception.h>
#include <psdr/core/ray.h>
#include <psdr/fwd.h>


// #include <psdr/scene/optix.h>
#include <psdr/scene/scene_optix.h>
#include <psdr/shape/mesh.h>
#include <drjit-core/optix.h>
// #include <optix.h>
// #include <optix_function_table_definition.h>
// #include <optix_stubs.h>
#include <nanothread/nanothread.h>


#include <psdr/optix/ptx.h>
#include <cuda/psdr_jit.h>
#include <iomanip>

#include <cuda_runtime.h>


NAMESPACE_BEGIN(psdr_jit)

using OptixResult = int;
using OptixProgramGroupKind = int;
using OptixCompileDebugLevel = int;
using OptixDeviceContext = void*;
using OptixLogCallback = void (*)(unsigned int, const char *, const char *, void *);
using OptixTask = void*;


using CUdeviceptr            = void*;
using CUstream               = void*;
using OptixPipeline          = void *;
using OptixModule            = void *;
using OptixProgramGroup      = void *;
using OptixResult            = int;
using OptixTraversableHandle = unsigned long long;
using OptixBuildOperation    = int;
using OptixBuildInputType    = int;
using OptixVertexFormat      = int;
using OptixIndicesFormat     = int;
using OptixTransformFormat   = int;
using OptixAccelPropertyType = int;
using OptixProgramGroupKind  = int;

// =====================================================
//            Commonly used OptiX constants
// =====================================================

#define OPTIX_PROGRAM_GROUP_KIND_RAYGEN               0x2421

#define OPTIX_BUILD_INPUT_TYPE_TRIANGLES         0x2141
#define OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES 0x2142
#define OPTIX_BUILD_INPUT_TYPE_INSTANCES         0x2143
#define OPTIX_BUILD_OPERATION_BUILD              0x2161

#define OPTIX_GEOMETRY_FLAG_NONE           0
#define OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT 1

#define OPTIX_INDICES_FORMAT_UNSIGNED_INT3 0x2103
#define OPTIX_VERTEX_FORMAT_FLOAT3         0x2121
#define OPTIX_SBT_RECORD_ALIGNMENT         16ull
#define OPTIX_SBT_RECORD_HEADER_SIZE       32

#define OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0
#define OPTIX_COMPILE_OPTIMIZATION_DEFAULT       0
#define OPTIX_COMPILE_OPTIMIZATION_LEVEL_0       0x2340
#define OPTIX_COMPILE_DEBUG_LEVEL_NONE           0x2350
#define OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL        0x2351

#define OPTIX_BUILD_FLAG_ALLOW_COMPACTION  2
#define OPTIX_BUILD_FLAG_PREFER_FAST_TRACE 4
#define OPTIX_PROPERTY_TYPE_COMPACTED_SIZE 0x2181

#define OPTIX_EXCEPTION_FLAG_NONE           0
#define OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW 1
#define OPTIX_EXCEPTION_FLAG_TRACE_DEPTH    2
#define OPTIX_EXCEPTION_FLAG_USER           4
#define OPTIX_EXCEPTION_FLAG_DEBUG          8

#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY 0
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS 1
#define OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING (1u << 1)

#define OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM   (1 << 0)
#define OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE (1 << 31)

#define OPTIX_PROGRAM_GROUP_KIND_MISS      0x2422
#define OPTIX_PROGRAM_GROUP_KIND_HITGROUP  0x2424

#define OPTIX_INSTANCE_FLAG_NONE              0
#define OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM (1u << 6)

#define OPTIX_RAY_FLAG_NONE                   0
#define OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT (1u << 2)
#define OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT     (1u << 3)

#define OPTIX_MODULE_COMPILE_STATE_COMPLETED 0x2364


/// Stores information about a Shape on the Optix side
struct OptixHitGroupData {
    /// Shape id in Dr.Jit's pointer registry
    uint32_t shape_registry_id;
    /// Pointer to the memory region of Shape data (e.g. \c OptixSphereData )
    void* data;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using MissSbtRecord     = EmptySbtRecord;
using HitGroupSbtRecord = SbtRecord<OptixHitGroupData>;


// template <typename T>
// struct Record
// {
//     __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
//     T data;
// };

// typedef Record<RayGenData>   RayGenRecord;
// typedef Record<MissData>     MissRecord;
// typedef Record<HitGroupData> HitGroupRecord;


// =====================================================
//          Commonly used OptiX data structures
// =====================================================

struct OptixMotionOptions {
    unsigned short numKeys;
    unsigned short flags;
    float timeBegin;
    float timeEnd;
};

struct OptixAccelBuildOptions {
    unsigned int buildFlags;
    OptixBuildOperation operation;
    OptixMotionOptions motionOptions;
};

struct OptixAccelBufferSizes {
    size_t outputSizeInBytes;
    size_t tempSizeInBytes;
    size_t tempUpdateSizeInBytes;
};

struct OptixBuildInputTriangleArray {
    const CUdeviceptr* vertexBuffers;
    unsigned int numVertices;
    OptixVertexFormat vertexFormat;
    unsigned int vertexStrideInBytes;
    CUdeviceptr indexBuffer;
    unsigned int numIndexTriplets;
    OptixIndicesFormat indexFormat;
    unsigned int indexStrideInBytes;
    CUdeviceptr preTransform;
    const unsigned int* flags;
    unsigned int numSbtRecords;
    CUdeviceptr sbtIndexOffsetBuffer;
    unsigned int sbtIndexOffsetSizeInBytes;
    unsigned int sbtIndexOffsetStrideInBytes;
    unsigned int primitiveIndexOffset;
    OptixTransformFormat transformFormat;
};

struct OptixBuildInput {
    OptixBuildInputType type;
    union {
        OptixBuildInputTriangleArray triangleArray;
        char pad[1024];
    };
};

struct OptixPayloadType {
    unsigned int numPayloadValues;
    const unsigned int *payloadSemantics;
};

struct OptixModuleCompileOptions {
    int maxRegisterCount;
    int optLevel;
    int debugLevel;
    const void *boundValues;
    unsigned int numBoundValues;
    unsigned int numPayloadTypes;
    OptixPayloadType *payloadTypes;
};

struct OptixPipelineCompileOptions {
    int usesMotionBlur;
    unsigned int traversableGraphFlags;
    int numPayloadValues;
    int numAttributeValues;
    unsigned int exceptionFlags;
    const char* pipelineLaunchParamsVariableName;
    unsigned int usesPrimitiveTypeFlags;
};

struct OptixAccelEmitDesc {
    CUdeviceptr result;
    OptixAccelPropertyType type;
};

struct OptixProgramGroupSingleModule {
    OptixModule module;
    const char* entryFunctionName;
};

struct OptixProgramGroupHitgroup {
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
};

struct OptixProgramGroupDesc {
    OptixProgramGroupKind kind;
    unsigned int flags;

    union {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupHitgroup hitgroup;
    };
};

struct OptixProgramGroupOptions {
    OptixPayloadType *payloadType;
};

struct OptixShaderBindingTable {
    CUdeviceptr raygenRecord;
    CUdeviceptr exceptionRecord;
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
};

struct OptixConfig {
    OptixDeviceContext context;
    OptixPipelineCompileOptions pipeline_compile_options;
    OptixModule module;
    OptixProgramGroup program_groups[3];
    uint32_t pipeline_jit_index;
};


struct PathTracerState
{
    OptixDeviceContext             context                  = 0;
    OptixTraversableHandle         gas_handle               = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer      = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices               = 0;
    OptixModule                    ptx_module               = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline                 = 0;
    OptixProgramGroup              raygen_prog_group        = 0;
    OptixProgramGroup              radiance_miss_group      = 0;
    OptixProgramGroup              radiance_hit_group       = 0;
    cudaStream_t                       stream                   = 0;
    Params                         params;
    Params*                        d_params                 = nullptr;
    OptixShaderBindingTable        sbt                      = {};
};

void Intersection_OptiX::reserve(int64_t size) {
    PSDR_ASSERT(size > 0);
    if ( size != m_size ) {
        m_size = size;
        triangle_id = empty<IntC>(size);
        shape_id = empty<IntC>(size);
        uv = empty<Vector2fC>(size);
    }
}


Scene_OptiX::Scene_OptiX() {
    m_accel = nullptr;
}

void optix_release( PathTracerState& state )
{

    // OptixResult (*optixProgramGroupDestroy)(OptixProgramGroup) = nullptr;
    // OptixResult (*optixPipelineDestroy)(OptixPipeline) = nullptr;
    // OptixResult (*optixModuleDestroy)(OptixModule) = nullptr;
    // OptixResult (*optixDeviceContextDestroy)(OptixDeviceContext) = nullptr;


    // #define LOOKUP(name) name = (decltype(name)) jit_optix_lookup(#name)
    // LOOKUP(optixProgramGroupDestroy);
    // LOOKUP(optixPipelineDestroy);
    // LOOKUP(optixModuleDestroy);
    // LOOKUP(optixDeviceContextDestroy);
    // #undef LOOKUP

    // jit_optix_check( optixPipelineDestroy( state.pipeline ) );
    // jit_optix_check( optixProgramGroupDestroy( state.radiance_miss_group ) );
    // jit_optix_check( optixProgramGroupDestroy( state.radiance_hit_group ) );
    // jit_optix_check( optixModuleDestroy( state.ptx_module ) );
    // jit_optix_check( optixDeviceContextDestroy( state.context ) );

    // jit_free(state.pipeline);
    // jit_free(state.radiance_miss_group);
    // jit_free(state.radiance_hit_group);
    // jit_free(state.ptx_module);
    // jit_free(state.context);
    jit_free( reinterpret_cast<void*>( state.sbt.missRecordBase ) ) ;
    jit_free( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) );
    jit_free( reinterpret_cast<void*>(state.d_gas_output_buffer));
    jit_free( reinterpret_cast<void*>( state.d_params ) );
}


Scene_OptiX::~Scene_OptiX() {
    if ( m_accel != nullptr ) {
        optix_release(*(PathTracerState *) m_accel);
        delete m_accel;
    }
}




void Scene_OptiX::configure(const std::vector<Mesh *> &meshes) {
    std::cout << "building optix" << std::endl;
    // drjit::sync_thread(); drjit::eval();
    // PSDR_ASSERT(!meshes.empty());
    size_t num_meshes = meshes.size();
    
    if ( m_accel == nullptr ) {
        std::vector<int> face_offset(num_meshes + 1);
        face_offset[0] = 0;
        for ( size_t i = 0; i < num_meshes; ++i ) {
            face_offset[i + 1] = face_offset[i] + meshes[i]->m_num_faces;
        }
        
        m_accel = new PathTracerState();
        PathTracerState &state = *(PathTracerState *) m_accel;


        OptixModuleCompileOptions module_compile_options { };
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;




        state.context = jit_optix_context();
        state.pipeline_compile_options.usesMotionBlur     = false;
        state.pipeline_compile_options.numPayloadValues   = 6;
        state.pipeline_compile_options.numAttributeValues = 2; // the minimum legal value
        state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";



        const std::string ptx = psdr_jit::getPtxString( OPTIX_SAMPLE_NAME, (std::string(PSDR_CUDA_FILE) + ".cu").c_str() );

        char optix_log[2048];
        size_t optix_log_size = sizeof(optix_log);
        auto check_log = [&](int rv) {
            if (rv) {
                fprintf(stderr, "\tLog: %s%s", optix_log,
                        optix_log_size > sizeof(optix_log) ? "<TRUNCATED>" : "");
                jit_optix_check(rv);
            }
        };

        OptixResult (*optixModuleCreateFromPTXWithTasks)(OptixDeviceContext,
                                                const OptixModuleCompileOptions *,
                                                const OptixPipelineCompileOptions *,
                                                const char *, size_t,
                                                char *, size_t *,
                                                OptixModule *,
                                                OptixTask *firstTask);

        OptixResult (*optixProgramGroupCreate)(OptixDeviceContext,
                                       const OptixProgramGroupDesc *,
                                       unsigned int,
                                       const OptixProgramGroupOptions *, char *,
                                       size_t *, OptixProgramGroup *) = nullptr;


        OptixResult (*optixAccelComputeMemoryUsage)(OptixDeviceContext,
          const OptixAccelBuildOptions *, const OptixBuildInput *, unsigned int,
          OptixAccelBufferSizes *);


        OptixResult (*optixAccelBuild)(OptixDeviceContext, CUstream, const OptixAccelBuildOptions *,
          const OptixBuildInput *, unsigned int, CUdeviceptr, size_t, CUdeviceptr,
          size_t, OptixTraversableHandle *, const OptixAccelEmitDesc *, unsigned int);

        OptixResult (*optixAccelCompact)(OptixDeviceContext, CUstream, OptixTraversableHandle,
            CUdeviceptr, size_t, OptixTraversableHandle *);

        #define LOOKUP(name) name = (decltype(name)) jit_optix_lookup(#name)
        LOOKUP(optixModuleCreateFromPTXWithTasks);
        LOOKUP(optixProgramGroupCreate);
        LOOKUP(optixAccelComputeMemoryUsage);
        LOOKUP(optixAccelBuild);
        LOOKUP(optixAccelCompact);
        #undef LOOKUP

        OptixTask task;
        

        optixModuleCreateFromPTXWithTasks(
                state.context,
                &module_compile_options,
                &state.pipeline_compile_options,
                (const char *)ptx.c_str(),
                (size_t)ptx.size(),
                optix_log,
                &optix_log_size,
                &state.ptx_module,
                &task
                );

        OptixProgramGroupOptions  program_group_options = {};

        {
            OptixProgramGroupDesc raygen_prog_group_desc    = {};
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = state.ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__psdr_rg";

            optixProgramGroupCreate(
                        state.context, &raygen_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        optix_log,
                        &optix_log_size,
                        &state.raygen_prog_group
                        );
        }

        {
            OptixProgramGroupDesc moe_miss_prog_group_desc  = {};
            moe_miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            moe_miss_prog_group_desc.miss.module            = state.ptx_module;
            moe_miss_prog_group_desc.miss.entryFunctionName = "__miss__psdr_ms";
            optixProgramGroupCreate(
                        state.context, &moe_miss_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        optix_log, &optix_log_size,
                        &state.radiance_miss_group
                        );
        }

        {
            OptixProgramGroupDesc hit_prog_group_desc        = {};
            hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH            = state.ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__psdr_ch";
            optixProgramGroupCreate(
                        state.context,
                        &hit_prog_group_desc,
                        1,  // num program groups
                        &program_group_options,
                        optix_log,
                        &optix_log_size,
                        &state.radiance_hit_group
                        );
        }
        
        uint32_t triangle_input_flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
        std::vector<void*> vertex_buffer_ptrs(num_meshes);
        std::vector<OptixBuildInput> build_inputs(num_meshes);
        for ( size_t i = 0; i < num_meshes; ++i ) {
            const Mesh *mesh = meshes[i];

            build_inputs[i].type                                = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            build_inputs[i].triangleArray.vertexFormat          = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_inputs[i].triangleArray.vertexStrideInBytes   = 3*sizeof(float);
            build_inputs[i].triangleArray.numVertices           = static_cast<uint32_t>(slices<FloatC>(mesh->m_vertex_buffer)/3);
            vertex_buffer_ptrs[i]                               = (void*)(mesh->m_vertex_buffer.data());
            build_inputs[i].triangleArray.vertexBuffers         = &vertex_buffer_ptrs[i];

            build_inputs[i].triangleArray.indexFormat           = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            build_inputs[i].triangleArray.numIndexTriplets      = static_cast<uint32_t>(slices(mesh->m_face_buffer)/3);
            build_inputs[i].triangleArray.indexBuffer           = (void*)(mesh->m_face_buffer.data());
            build_inputs[i].triangleArray.indexStrideInBytes    = 3*sizeof(int);

            build_inputs[i].triangleArray.flags                 = triangle_input_flags;
            build_inputs[i].triangleArray.numSbtRecords         = 1;
        }

        int num_build_inputs = static_cast<int>(build_inputs.size());
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys = 0;

        drjit::sync_thread();

        OptixAccelBufferSizes buffer_sizes;
        jit_optix_check( optixAccelComputeMemoryUsage(
                    state.context,
                    &accel_options,
                    build_inputs.data(),
                    num_build_inputs,
                    &buffer_sizes
                    ) );

        void* d_temp_buffer = jit_malloc(AllocType::Device, buffer_sizes.tempSizeInBytes);
        void* output_buffer = jit_malloc(AllocType::Device, buffer_sizes.outputSizeInBytes + 8);

        OptixAccelEmitDesc emit_property = {};
        emit_property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = (CUdeviceptr)((char*)output_buffer + buffer_sizes.outputSizeInBytes);

        jit_optix_check( optixAccelBuild(
                    state.context,
                    0,
                    &accel_options,
                    build_inputs.data(),
                    num_build_inputs,
                    (CUdeviceptr)d_temp_buffer,
                    buffer_sizes.tempSizeInBytes,
                    (CUdeviceptr)output_buffer,
                    buffer_sizes.outputSizeInBytes,
                    &state.gas_handle,
                    &emit_property,
                    1
                    ) );

        jit_free(d_temp_buffer);

        size_t compact_size;
        jit_memcpy(JitBackend::CUDA,
                   &compact_size,
                   (void*)emit_property.result,
                   sizeof(size_t));

        if (compact_size < buffer_sizes.outputSizeInBytes) {
            void* compact_buffer = jit_malloc(AllocType::Device, compact_size);
            jit_optix_check(optixAccelCompact(
                state.context,
                (CUstream) jit_cuda_stream(),
                state.gas_handle,
                (CUdeviceptr)compact_buffer,
                compact_size,
                &state.gas_handle
            ));
            jit_free(output_buffer);
            output_buffer = compact_buffer;
        }

        // build CUDA stream
        state.params.handle         = state.gas_handle;

        if (state.d_gas_output_buffer)
            jit_free(reinterpret_cast<void*>(state.d_gas_output_buffer));
        state.d_gas_output_buffer = (CUdeviceptr)output_buffer;


    }
        // =====================================================
        //  Shader Binding Table generation
        // =====================================================

    // void build_accel(PathTracerState& state, const std::vector<OptixBuildInput>& triangle_input)

    // build_accel(*m_accel, build_inputs);
    // PSDR_ASSERT(0);
}


bool Scene_OptiX::is_ready() const {
    return m_accel != nullptr;
}


template <bool ad>
Vector2i<ad> Scene_OptiX::ray_intersect(const Ray<ad> &ray, Mask<ad> &active) const {

    OptixResult (*optixLaunch)(OptixPipeline, CUstream, CUdeviceptr, size_t,
                       const OptixShaderBindingTable *, unsigned int,
                       unsigned int, unsigned int) = nullptr;


    #define LOOKUP(name) name = (decltype(name)) jit_optix_lookup(#name)
    LOOKUP(optixLaunch);
    #undef LOOKUP

    std::cout << "optix ray trace" << std::endl;
    PathTracerState &s = *(PathTracerState *) m_accel;
    Vector3fC ray_o, ray_d;
    if constexpr (ad) {
        ray_o = detach(ray.o);
        ray_d = detach(ray.d);
    } else {
        ray_o = ray.o;
        ray_d = ray.d;
    }
    
    // FloatC ray_mint(0.f), ray_time(10.0);
    // FloatC ray_maxt;

    // ray_maxt = drjit::Largest<FloatC>;

    // using UInt64 = drjit::uint64_array_t<FloatC>;


    // int m_accel_handle = drjit::opaque<int>(s.gas_handle);

    // std::cout << ray_o.x().index() << std::endl;
    // std::cout << ray_o.y().index() << std::endl;
    // std::cout << ray_o.z().index() << std::endl;
    // std::cout << ray_d.x().index() << std::endl;
    // std::cout << ray_d.y().index() << std::endl;
    // std::cout << ray_d.z().index() << std::endl;

    // uint32_t trace_args[] {
    //     m_accel_handle.index(),
    //     ray_o.x().index(), ray_o.y().index(), ray_o.z().index(),
    //     ray_d.x().index(), ray_d.y().index(), ray_d.z().index(),
    //     ray_mint.index(), ray_maxt.index(), ray_time.index(),
    //     ray_mask.index(), ray_flags.index(),
    //     sbt_offset.index(), sbt_stride.index(),
    //     miss_sbt_index.index(), payload_hit.index()
    // };

    // jit_optix_ray_trace(sizeof(trace_args) / sizeof(uint32_t),
    //                     trace_args, active.index());


    const int m = static_cast<int>(slices<Vector3fC>(ray_o));
    m_its.reserve(m);

    // ray_d.x().index();

    s.params.ray_o_x         = ray_o.x().data();
    s.params.ray_o_y         = ray_o.y().data();
    s.params.ray_o_z         = ray_o.z().data();

    s.params.ray_d_x         = ray_d.x().data();
    s.params.ray_d_y         = ray_d.y().data();
    s.params.ray_d_z         = ray_d.z().data();

    s.params.ray_tmax        = ray.tmax.data();
    s.params.tri_index       = m_its.triangle_id.data();
    s.params.shape_index     = m_its.shape_id.data();
    s.params.barycentric_u   = m_its.uv.x().data();
    s.params.barycentric_v   = m_its.uv.y().data();

        cudaMemcpyAsync(
            reinterpret_cast<void*>( &s.d_params ),
            &s.params, sizeof( Params ),
            cudaMemcpyHostToDevice, s.stream
    );

    std::cout << ray.tmax[0];
        optixLaunch(
            s.pipeline,
            s.stream,
            reinterpret_cast<CUdeviceptr>( s.d_params ),
            sizeof( Params ),
            &s.sbt,
            m,              // launch size
            1,              // launch height
            1               // launch depth
    );
    // CUDA_SYNC_CHECK();
    std::cout << "\b\b\b";
    active &= (m_its.shape_id >= 0) && (m_its.triangle_id >= 0);
    std::cout << m_its.shape_id << std::endl;


    // PSDR_ASSERT(0);
    return Vector2i<ad>(m_its.shape_id, m_its.triangle_id);
}


// Explicit instantiations
template Vector2iC Scene_OptiX::ray_intersect<false>(const RayC &ray, MaskC &active) const;
template Vector2iD Scene_OptiX::ray_intersect<true>(const RayD &ray, MaskD &active) const;

NAMESPACE_END(psdr_jit)
