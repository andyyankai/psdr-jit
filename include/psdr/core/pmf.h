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
        std::cout << value << std::endl;
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

} // namespace psdr
