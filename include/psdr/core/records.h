#pragma once

#include <drjit/struct.h>
#include "ray.h"
#include "intersection.h"

NAMESPACE_BEGIN(psdr_jit)

template <typename Float_>
struct DiscreteRecord_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;
    Float<ad>       rnd;
    Float<ad>       pdf;
    Int<ad>        idx;

    DRJIT_STRUCT(DiscreteRecord_, rnd, pdf, idx)
};

template <typename Float_>
struct SampleRecord_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;

    Float<ad>       pdf;
    Mask<ad>        is_valid;

    DRJIT_STRUCT(SampleRecord_, pdf, is_valid)
};

template <typename Float_>
struct PositionSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)

    Vector3f<ad>    p, n;
    Float<ad>       J;

    DRJIT_STRUCT(PositionSample_, pdf, is_valid, p, n, J)
};


struct BoundarySegSampleDirect : public SampleRecord_<FloatC> {
    PSDR_IMPORT_BASE(SampleRecord_<FloatC>, pdf, is_valid)
    
    // Sample point on a face edge
    Vector3fD           p0;
    Vector3fC           edge, edge2;

    // Sample point on an emitter
    Vector3fC           p2; // for indirect, p2 is a direction
};

NAMESPACE_END(psdr_jit)
