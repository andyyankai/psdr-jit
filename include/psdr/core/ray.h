#pragma once

#include <psdr/psdr.h>
#include <misc/Exception.h>

NAMESPACE_BEGIN(psdr_jit)

template <typename Float_>
struct Ray_ {
    static constexpr bool ad = std::is_same_v<Float_, FloatD>;

    inline Ray_(const Vector3f<ad> &o, const Vector3f<ad> &d, const Float<ad> &tmax) : o(o), d(d), tmax(tmax) {
    }
    inline Ray_(const Vector3f<ad> &o, const Vector3f<ad> &d) : o(o), d(d) {
        tmax = full<Float<ad>>(Infinity, slices<Vector3f<ad>>(d));
    }

    inline Ray_ reversed() const { return Ray_<Float_>(o, -d, tmax); }

    inline int size() const {
        return tmax.size();
    }

    Vector3f<ad> operator() (const Float<ad> &t) const { return fmadd(d, t, o); }

    Vector3f<ad> o, d;
    Float<ad> tmax;

    DRJIT_STRUCT(Ray_, o, d, tmax);
};

NAMESPACE_END(psdr_jit)
