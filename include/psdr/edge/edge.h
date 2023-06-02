#pragma once

#include <psdr/psdr.h>
#include <psdr/core/ray.h>
#include <psdr/core/pmf.h>

NAMESPACE_BEGIN(psdr_jit)

/********************************************
* Primary edge info types
********************************************/

struct PrimaryEdgeSample {
    FloatD  x_dot_n;
    IntC    idx;
    RayC    ray_n, ray_p;
    FloatC  pdf;
};


template <typename Float_>
struct PrimaryEdgeInfo_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;
    Vector2f<ad>            p0, p1;
    Vector2f<ad>            edge_normal;
    Float<ad>               edge_length;

    DRJIT_STRUCT(PrimaryEdgeInfo_, p0, p1, edge_normal, edge_length)
};

using PrimaryEdgeInfo = PrimaryEdgeInfo_<FloatD>;


/********************************************
 * Secondary edge info
 ********************************************/

template <typename Float_>
struct SecondaryEdgeInfo_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;

    // p0 and (p0 + e1) are the two endpoints of the edge
    Vector3f<ad>            p0, e1;

    // n0 and n1 are the normals of the two faces sharing the edge
    Vector3f<ad>            n0, n1;

    // p2 is the third vertex of the face with normal n0
    Vector3f<ad>            p2;

    Mask<ad>                is_boundary;

    int size() {
        return is_boundary.size();
    }

    DRJIT_STRUCT(SecondaryEdgeInfo_, p0, e1, n0, n1, p2, is_boundary)
};

using SecondaryEdgeInfo = SecondaryEdgeInfo_<FloatD>;



NAMESPACE_END(psdr_jit)

