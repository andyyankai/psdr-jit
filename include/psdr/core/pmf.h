#pragma once

#include <psdr/psdr.h>
#include <psdr/core/records.h>
#include <misc/Exception.h>
#include <drjit/math.h>
#include <drjit/util.h>
#include <drjit/array_constants.h>

namespace psdr
{

inline FloatC compute_cdf(const FloatC &pmf) {
    size_t size = pmf.size();

    const ScalarFloat *ptr_pmf = pmf.data();

    if (size == 0)
        PSDR_ASSERT_MSG(0, "DiscreteDistribution: empty distribution!");

    std::vector<ScalarFloat> cdf(size);
    ScalarVector2u m_valid = (uint32_t) -1;
    double sum = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        double value = (double) *ptr_pmf++;
        sum += value;
        cdf[i] = (ScalarFloat) sum;

        if (value < 0.0) {
            PSDR_ASSERT_MSG(0, "DiscreteDistribution: entries must be non-negative!");
        } else if (value > 0.0) {
            // Determine the first and last wavelength bin with nonzero density
            if (m_valid.x() == (uint32_t) -1)
                m_valid.x() = i;
            m_valid.y() = i;
        }
    }
    FloatC m_cdf = load<FloatC>(cdf.data(), size);
    return m_cdf;
}

struct DiscreteDistribution {
    DiscreteDistribution() = default;

    void init(const FloatC &pmf);

    std::pair<IntC, FloatC> sample(const FloatC &samples) const;

    template <bool ad>
    std::pair<IntC, FloatC> sample_reuse(Float<ad> &samples) const;

    const FloatC &pmf() const { return m_pmf_normalized; }
    const FloatC &cmf() const { return m_cmf_normalized; }

    int     m_size;
    FloatC  m_sum;

// protected:
    FloatC  m_pmf, m_pmf_normalized, m_cmf, m_cmf_normalized;
};

PSDR_CLASS_DECL_BEGIN(DDistribution,, Object)
public:
    virtual ~DDistribution() override {}

    virtual void init(const FloatC &pmf) = 0;
    virtual DiscreteRecordC sample_reuse(FloatC &samples) const = 0;
    virtual DiscreteRecordC sample_reuse(FloatD &samples) const = 0;
    virtual const FloatC &pmf() const = 0;
    virtual const FloatC &cmf() const = 0;

PSDR_CLASS_DECL_END(DDistribution)


PSDR_CLASS_DECL_BEGIN(_DiscreteDistribution, final, DDistribution)
public:
    void init(const FloatC &pmf) override {
        m_size = static_cast<int>((pmf.size()));
        m_sum = sum(pmf);
        m_pmf = pmf;
        m_cmf = compute_cdf(m_pmf);
        m_pmf_normalized = pmf/m_sum;
        m_cmf_normalized = m_cmf/m_sum;
    }

    DiscreteRecordC sample_reuse(FloatC &_samples) const override {
        FloatC samples(_samples);
        DiscreteRecordC result;
        if ( unlikely(m_size == 1) ) {
            PSDR_ASSERT(0);
            result.pdf = full<FloatC>(1.f);
            result.idx = zeros<IntC>();
            return result;
        }
        samples *= m_sum;
        IntC idx;

        idx = binary_search<IntC>(
            0, m_size - 1, [&](IntC i) DRJIT_INLINE_LAMBDA { return gather<FloatC>(m_cmf, i) < samples; }
        );
        samples -= gather<FloatC>(m_cmf, idx - 1, idx > 0);
        FloatC pmf = gather<FloatC>(m_pmf, idx);
        masked(samples, pmf > 0.f) /= pmf;
        samples = clamp(samples, 0.f, 1.f);

        result.rnd = samples;
        result.pdf = pmf/m_sum;
        result.idx = idx;
        return result;
    }

    DiscreteRecordC sample_reuse(FloatD &samples) const override {

        DiscreteRecordC result;
        if ( unlikely(m_size == 1) ) {
            PSDR_ASSERT(0);
            result.pdf = full<FloatC>(1.f);
            result.idx = zeros<IntC>();
            return result;
        }
        samples *= m_sum;
        IntC idx;
        idx = binary_search<IntC>(
            0, m_size - 1, [&](IntC i) DRJIT_INLINE_LAMBDA { return gather<FloatC>(m_cmf, i) < detach(samples); }
        );
        samples -= gather<FloatC>(m_cmf, idx - 1, idx > 0);
        FloatC pmf = gather<FloatC>(m_pmf, idx);
        masked(samples, pmf > 0.f) /= pmf;
        samples = clamp(samples, 0.f, 1.f);

        result.rnd = detach(samples);
        result.pdf = pmf/m_sum;
        result.idx = idx;
        return result;
    }


    const FloatC &pmf() const override { return m_pmf_normalized; }
    const FloatC &cmf() const override { return m_cmf_normalized; }

    int     m_size;
    FloatC  m_sum;
    FloatC  m_pmf, m_pmf_normalized, m_cmf, m_cmf_normalized;

PSDR_CLASS_DECL_END(_DiscreteDistribution)

} // namespace psdr
