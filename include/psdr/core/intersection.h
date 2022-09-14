#pragma once

#include "frame.h"

NAMESPACE_BEGIN(psdr_jit)

template <typename Float_>
struct Interaction_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;

    template <typename Float1_>
    inline Interaction_(const Interaction_<Float1_> &in) : wi(in.wi), p(in.p), t(in.t) {}

    virtual Mask<ad> is_valid() const = 0;

    Vector3f<ad> wi, p;
    Float<ad> t = Infinity;

    DRJIT_STRUCT(Interaction_, wi, p, t)
};


template <typename Float_>
struct Intersection_ : public Interaction_<Float_> {
    PSDR_IMPORT_BASE(Interaction_<Float_>, ad, wi, p, t)

    Mask<ad> is_valid() const override {
        return neq(shape, nullptr);
    }

    inline Mask<ad> is_emitter(Mask<ad> active) const {
        return neq(shape->emitter(), nullptr);
    }

    inline Spectrum<ad> Le(Mask<ad> active) const {
        if constexpr (ad) {
            return shape->emitter()->evalD(*this, active);
        } else {
            return shape->emitter()->evalC(*this, active);
        }
        
    }

    MeshArray<ad>       shape;

    Vector3f<ad>        n;                  // geometric normal

    /// Position partials wrt. the UV parameterization
    Vector3f<ad>        dp_du, dp_dv;

    Frame<ad>           sh_frame;           // shading frame

    Vector2f<ad>        uv;
    Float<ad>           J;                  // Jacobian determinant for material-form reparam

    DRJIT_STRUCT(Intersection_, wi, p, t, shape, n, dp_du, dp_dv, sh_frame, uv, J)
};

NAMESPACE_END(psdr_jit)
