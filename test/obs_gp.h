/*
 * GPisMap - Online Continuous Mapping using Gaussian Process Implicit Surfaces
 * https://github.com/leebhoram/GPisMap
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of any FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License v3 for more details.
 *
 * You should have received a copy of the GNU General Public License v3
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-3.0.html.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 *          Huang Zonghao<ac@hzh.io>
 */

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#define DEFAULT_OBSGP_SCALE_PARAM double(0.5)
#define DEFAULT_OBSGP_NOISE_PARAM double(0.01)
#define DEFAULT_OBSGP_OVERLAP_SZ  6
#define DEFAULT_OBSGP_GROUP_SZ    20
#define DEFAULT_OBSGP_MARGIN      double(0.0175)

typedef Eigen::MatrixX<double> EMatrixX;
typedef Eigen::VectorX<double> EVectorX;
typedef Eigen::RowVectorX<double> ERowVectorX;

// This class builds a GP regressor using the Ornstein-Uhlenbeck covariance function.
// NOTE: See covFnc.h)
class GPou {
#if defined(BUILD_TEST)
public:
#endif
    EMatrixX m_x_;
    EMatrixX m_l_;
    EVectorX m_alpha_;  // inv(m_k_+Kx) * y

    const double m_scale_ = DEFAULT_OBSGP_SCALE_PARAM;
    const double m_noise_ = DEFAULT_OBSGP_NOISE_PARAM;
    bool m_trained_ = false;

public:
    GPou() = default;

    [[maybe_unused]] void
    Reset() {
        m_trained_ = false;
    }

    [[nodiscard]] bool
    IsTrained() const {
        return m_trained_;
    }

    [[maybe_unused]] int
    GetNumSamples() {
        return (int) m_x_.cols();
    }

    void
    Train(const EMatrixX &xt, const EVectorX &f);

    void
    Test(const EMatrixX &xt, EVectorX &f, EVectorX &var);
};

// This is a base class to build a partitioned GP regressor, holding multiple local GPs using GPou.
class ObsGp {
#if defined(BUILD_TEST)
public:
#else
protected:
#endif
    bool m_trained_{};

    std::vector<std::shared_ptr<GPou>> m_gps_;  // pointer to the local GPs

public:
    ObsGp() = default;

    virtual ~ObsGp() = default;

    [[nodiscard]] bool
    IsTrained() const {
        return m_trained_;
    }

    virtual void
    Reset();

    virtual void
    Train(double xt[], double f[], int n[]) = 0;

    virtual void
    Test(const EMatrixX &xt, EVectorX &val, EVectorX &var) = 0;
};

typedef struct ObsGpParam_t {
    // Note:
    // ObsGp is implemented to use the Ornstein-Uhlenbeck covariance function,
    // which has a form of k(r)=exp(-r/l) (See covFnc.h)
    double scale;  // the m_scale_ parameter l
    double noise;  // the m_noise_ parameter of the measurement
    // currently use a constant value
    // could be potentially modified to have heteroscedastic m_noise_
    // Note:
    // ObsGp is implemented to have overlapping partitioned GPs.
    double margin;  // used to decide if valid m_range_
    // (don't use if too close to boundary
    //  because the derivatives are hard to sample)
    int overlap;     // the overlapping parameters: number of samples to overlap
    int group_size;  // the number of samples to group together
    // (the actual group GetSize will be (group_size+overlap)
    ObsGpParam_t() = default;

    ObsGpParam_t(double s, double n, double m, int ov, int gsz)
        : scale(s),
          noise(n),
          margin(m),
          overlap(ov),
          group_size(gsz) {}
} ObsGpParam;

// This class implements ObsGp for 1D input.
class ObsGp1D : public ObsGp {
#if defined(BUILD_TEST)
public:
#endif
    int m_n_group_{};  // number of local GPs
    int m_n_samples_;  // number of total input points.

    std::vector<double> m_range_;  // partitioned m_range_ for test

    const ObsGpParam m_param_ = {DEFAULT_OBSGP_SCALE_PARAM, DEFAULT_OBSGP_NOISE_PARAM, DEFAULT_OBSGP_MARGIN, DEFAULT_OBSGP_OVERLAP_SZ, DEFAULT_OBSGP_GROUP_SZ};

public:
    ObsGp1D()
        : m_n_samples_(0) {}

    void
    Reset() override;

    // NOTE: In 1D, it must be f_out > 0.
    void
    Train(double xt[], double f_out[], int n[]) override;

    void
    Test(const EMatrixX &xt, EVectorX &val, EVectorX &var) override;
};
