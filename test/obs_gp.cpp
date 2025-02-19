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

#include "obs_gp.h"

#include "cov_fnc.h"

#include <Eigen/Cholesky>

///////////////////////////////////////////////////////////
// GPou
///////////////////////////////////////////////////////////

void
GPou::Train(const EMatrixX &xt, const EVectorX &f) {
    // int dim = xt.rows();
    int n = static_cast<int>(xt.cols());

    if (n > 0) {
        m_x_ = xt;
        EMatrixX k = OrnsteinUhlenbeck(xt, m_scale_, m_noise_);
        // GPIS paper: eq(10)
        m_l_ = k.llt().matrixL();
        m_alpha_ = f;
        m_l_.triangularView<Eigen::Lower>().solveInPlace(m_alpha_);
        m_l_.transpose().triangularView<Eigen::Upper>().solveInPlace(m_alpha_);

        m_trained_ = true;
    }
}

void
GPou::Test(const EMatrixX &xt, EVectorX &f, EVectorX &var) {

    EMatrixX k = OrnsteinUhlenbeck(m_x_, xt, m_scale_);
    f = k.transpose() * m_alpha_;
    // calculate m_l_.inv() * k
    m_l_.triangularView<Eigen::Lower>().solveInPlace(k);
    // calculate k.transpose() * m_l_.inv().transpose() * m_l_.inv() * k
    k = k.array().pow(2);
    EVectorX v = k.colwise().sum();
    var = 1 - v.head(xt.cols()).array();
}

///////////////////////////////////////////////////////////
// ObsGp
///////////////////////////////////////////////////////////

void
ObsGp::Reset() {
    m_trained_ = false;
    m_gps_.clear();
}

///////////////////////////////////////////////////////////
// ObsGp 1D
///////////////////////////////////////////////////////////

void
ObsGp1D::Reset() {
    ObsGp::Reset();
    m_range_.clear();
    m_n_samples_ = 0;
}

void
ObsGp1D::Train(double xt[], double f_out[], int n[]) {
    Reset();

    if ((n[0] > 0) && (xt != nullptr)) {
        m_n_samples_ = n[0];
        m_n_group_ = m_n_samples_ / (m_param_.group_size) + 1;

        m_range_.push_back(xt[0]);
        for (int i = 0; i < (m_n_group_ - 1); i++) {
            // Make sure there are enough overlap

            if (i < m_n_group_ - 2) {
                int i1 = i * m_param_.group_size;
                int i2 = i1 + m_param_.group_size + m_param_.overlap;

                m_range_.push_back(xt[i2 - m_param_.overlap / 2]);

                Eigen::Map<ERowVectorX> x(xt + i1, m_param_.group_size + m_param_.overlap);
                Eigen::Map<EVectorX> f(f_out + i1, m_param_.group_size + m_param_.overlap);
                // Train each gp group
                std::shared_ptr<GPou> g(new GPou());
                g->Train(x, f);

                m_gps_.push_back(std::move(g));

            } else {  // the last two groups split in half
                // the second to last
                int i1 = i * m_param_.group_size;
                int i2 = i1 + (m_n_samples_ - i1 + m_param_.overlap) / 2;
                m_range_.push_back(xt[i2 - m_param_.overlap / 2]);

                Eigen::Map<ERowVectorX> x(xt + i1, i2 - i1);
                Eigen::Map<EVectorX> f(f_out + i1, i2 - i1);
                std::shared_ptr<GPou> g(new GPou());
                g->Train(x, f);
                m_gps_.push_back(std::move(g));
                i++;

                // the last one
                i1 = i1 + (m_n_samples_ - i1 - m_param_.overlap) / 2;
                i2 = m_n_samples_ - 1;
                m_range_.push_back(xt[i2]);
                new (&x) Eigen::Map<ERowVectorX>(xt + i1, i2 - i1 + 1);
                new (&f) Eigen::Map<EVectorX>(f_out + i1, i2 - i1 + 1);

                std::shared_ptr<GPou> gp_last(new GPou());
                gp_last->Train(x, f);
                m_gps_.push_back(std::move(gp_last));
            }
        }

        m_trained_ = true;
    }
}

void
ObsGp1D::Test(const EMatrixX &xt, EVectorX &val, EVectorX &var) {

    if (!IsTrained()) { return; }

    auto dim = xt.rows();
    auto n = xt.cols();

    if (dim == 1) {
        double lim_l = (*(m_range_.begin()) + m_param_.margin);
        double lim_r = (*(m_range_.end() - 1) - m_param_.margin);
        for (int k = 0; k < n; k++) {

            EVectorX f = val.segment(k, 1);
            EVectorX v = var.segment(k, 1);
            var(k) = 1e6;
            // find the corresponding group
            if (xt(0, k) >= lim_l && xt(0, k) <= lim_r) {  // in-between
                int j = 0;
                for (auto it = (m_range_.begin() + 1); it != m_range_.end(); ++it, j++) {
                    if (xt(0, k) >= *(it - 1) && xt(0, k) <= *it) {
                        // and test
                        if (m_gps_[j]->IsTrained()) {
                            m_gps_[j]->Test(xt.block(0, k, 1, 1), f, v);
                            val(k) = f(0);
                            var(k) = v(0);
                        }
                        break;
                    }
                }
            }
        }
    }
}
