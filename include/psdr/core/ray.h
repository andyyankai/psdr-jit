#pragma once

#include <psdr/psdr.h>
#include <misc/Exception.h>

NAMESPACE_BEGIN(psdr_jit)

template <bool ad>
struct Ray {
    inline Ray() = default;

    inline Ray(const Vector3f<ad> &o, const Vector3f<ad> &d, const Float<ad> &tmax) : o(o), d(d), tmax(tmax) {
        drjit::eval(o);
        drjit::eval(d);
        drjit::eval(tmax);

    }

    inline Ray(const Vector3f<ad> &o, const Vector3f<ad> &d) : o(o), d(d) {
        PSDR_ASSERT(o.size() == d.size());
        PSDR_ASSERT(slices<Vector3f<ad>>(o) == slices<Vector3f<ad>>(d));
        drjit::eval(o);
        drjit::eval(d);

        tmax = full<Float<ad>>(Infinity, slices<Vector3f<ad>>(d));
        drjit::eval(tmax);
    }

    template <bool ad1> inline Ray(const Ray<ad1> &ray) : o(ray.o), d(ray.d), tmax(ray.tmax) {


}

    Vector3f<ad> o, d;
    Float<ad> tmax;

    Vector3f<ad> operator() (const Float<ad> &t) const { return fmadd(d, t, o); }

    inline size_t size() {
        return slices<Vector3f<ad>>(o);
    }

    inline Ray reversed() const { return Ray(o, -d, tmax); }

    // DRJIT_STRUCT(Ray, o, d, tmax)


};


inline RayC detach(const RayD &ray) {
    return RayC(detach(ray.o), detach(ray.d), detach(ray.tmax));
}



NAMESPACE_END(psdr_jit)
