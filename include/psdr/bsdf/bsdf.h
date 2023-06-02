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
    virtual SpectrumC evalC(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const = 0;
    virtual SpectrumD evalD(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const = 0;

    virtual SpectrumD eval_type(const IntersectionD &its, MaskD active = true) const = 0;

    virtual BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const = 0;
    virtual BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const = 0;

    BSDFSampleC sampleC(const IntersectionC &its, const Vector3fC &sample_, MaskC active = true) const {
        return sample(its, sample_, active);
    };
    BSDFSampleD sampleD(const IntersectionD &its, const Vector3fD &sample_, MaskD active = true) const {
        return sample(its, sample_, active);
    };

    virtual FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const = 0;
    virtual FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const = 0;

    FloatC pdfC(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const {
        return pdf(its, wo, active);
    };
    FloatD pdfD(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const {
        return pdf(its, wo, active);
    };

    virtual bool anisotropic() const = 0;

    virtual int test_vir() = 0;

    std::string to_string() const override{
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

        SpectrumC evalC(const IntersectionC &its, const Vector3fC &wo, MaskC active = true) const override {
            PYBIND11_OVERRIDE_PURE(SpectrumC, BSDF, evalC, its, wo, active);
            // return SpectrumC(0.1f,0.1f,0.9f);
        }

        SpectrumD evalD(const IntersectionD &its, const Vector3fD &wo, MaskD active = true) const override {
            PYBIND11_OVERRIDE_PURE(SpectrumD, BSDF, evalD, its, wo, active);
            // return SpectrumD(0.1f,0.1f,0.9f);
        }

        SpectrumD eval_type(const IntersectionD &its, MaskD active = true) const override {
            // PYBIND11_OVERRIDE_PURE(SpectrumD, BSDF, eval_type, its, active);
            return SpectrumD(0.1f,0.1f,0.9f);
        }

        BSDFSampleC sample(const IntersectionC &its, const Vector3fC &sample, MaskC active = true) const override {
            // PYBIND11_OVERRIDE_PURE(BSDFSampleC, BSDF, sample, sample, active);
            BSDFSampleC bss;
            return bss;
        }
        BSDFSampleD sample(const IntersectionD &its, const Vector3fD &sample, MaskD active = true) const override {
            // PYBIND11_OVERRIDE_PURE(BSDFSampleD, BSDF, sample, sample, active);
            BSDFSampleD bss;
            return bss;
        }

        FloatC pdf(const IntersectionC &its, const Vector3fC &wo, MaskC active) const override {
            // PYBIND11_OVERRIDE_PURE(FloatC, BSDF, pdf, its, wo, active);
            return FloatC(1.0f);
        }
        FloatD pdf(const IntersectionD &its, const Vector3fD &wo, MaskD active) const override {
            // PYBIND11_OVERRIDE_PURE(FloatD, BSDF, pdf, its, wo, active);
            return FloatD(1.0f);
        }

        std::string to_string() const override {
            // PYBIND11_OVERRIDE_PURE(std::string, BSDF, to_string);
            return std::string("CustomBSDF[id=") + m_id + "]";
            // return false;
        } ;

        bool anisotropic() const override {
            PYBIND11_OVERRIDE_PURE(bool, BSDF, anisotropic);
            // return false;
        } ;

        int test_vir() override {
            PYBIND11_OVERRIDE_PURE(int, BSDF, test_vir);
            // return 1;
        };

    };

NAMESPACE_END(psdr_jit)

DRJIT_VCALL_BEGIN(psdr_jit::BSDF)
    DRJIT_VCALL_METHOD(eval_type)
    DRJIT_VCALL_METHOD(evalC)
    DRJIT_VCALL_METHOD(evalD)
    DRJIT_VCALL_METHOD(sample)
    DRJIT_VCALL_METHOD(sampleC)
    DRJIT_VCALL_METHOD(sampleD)
    DRJIT_VCALL_METHOD(pdf)
    DRJIT_VCALL_METHOD(pdfC)
    DRJIT_VCALL_METHOD(pdfD)
    DRJIT_VCALL_METHOD(anisotropic)
DRJIT_VCALL_END(psdr_jit::BSDF)
