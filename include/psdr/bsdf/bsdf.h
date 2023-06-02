#pragma once

#include <psdr/psdr.h>
#include <psdr/core/intersection.h>
#include <psdr/core/records.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

NAMESPACE_BEGIN(psdr_jit)



template <typename Float_>
struct BSDFSample_ : public SampleRecord_<Float_> {
    PSDR_IMPORT_BASE(SampleRecord_<Float_>, ad, pdf, is_valid)
    Vector3f<ad> wo;
    Float<ad> eta;

    DRJIT_STRUCT(BSDFSample_, pdf, is_valid, wo, eta)
};


PSDR_CLASS_DECL_BEGIN(BSDF,, Object)
public:
    BSDF() {}
    virtual ~BSDF() override {}
    virtual SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;
    virtual SpectrumD eval_type(const IntersectionD &its, MaskD active = true) const = 0;
    virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;
    virtual bool anisotropic() const = 0;
    virtual int test_vir() = 0;
    std::string to_string() const override {
        std::stringstream oss;
        oss << type_name();
        if ( m_id != "" ) oss << "[id=" << m_id << "]";
        return oss.str();
    }

    bool m_twoSide = false;

    DRJIT_VCALL_REGISTER(FloatD, BSDF);
                    
PSDR_CLASS_DECL_END(BSDF)


    class PyBSDF : public BSDF {
    public:
        PyBSDF() : BSDF() { }

        SpectrumD eval(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override {
            PYBIND11_OVERRIDE_PURE(SpectrumD, BSDF, eval, its, wo, active);
        }

        SpectrumD eval_type(const IntersectionD &its, MaskD active = true) const override {
            // PYBIND11_OVERRIDE_PURE(SpectrumD, BSDF, eval_type, its, active);
            return SpectrumD(0.1f,0.1f,0.9f);
        }
        BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override {
            PYBIND11_OVERRIDE_PURE(BSDFSampleD, BSDF, sample, its, sample, active);
        }
        FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override {
            PYBIND11_OVERRIDE_PURE(FloatD, BSDF, pdf, its, wo, active);
        }
        std::string to_string() const override {
            PYBIND11_OVERRIDE_PURE(std::string, BSDF, to_string);
        } ;

        bool anisotropic() const override {
            PYBIND11_OVERRIDE_PURE(bool, BSDF, anisotropic);
        } ;

        int test_vir() override {
            PYBIND11_OVERRIDE_PURE(int, BSDF, test_vir);
        };

    };

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::BSDF)
    DRJIT_VCALL_METHOD(eval_type)
    DRJIT_VCALL_METHOD(eval)
    DRJIT_VCALL_METHOD(sample)
    DRJIT_VCALL_METHOD(pdf)
    DRJIT_VCALL_METHOD(anisotropic)
DRJIT_VCALL_END(psdr_jit::BSDF)
