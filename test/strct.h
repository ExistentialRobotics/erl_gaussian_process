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

#include "erl_common/float_type.hpp"
#include "params.h"
#include <memory>
#include <vector>

enum NodeType {
    NONE = 0, HIT = 1, FREE, CLUSTER
};

template<typename T>
struct Point {
    T x;
    T y;

    Point(T _x, T _y) {
        x = _x;
        y = _y;
    }

    Point() {
        x = 0;
        y = 0;
    }
};

template<typename T>
struct Point3 {
    T x;
    T y;
    T z;

    Point3(T _x, T _y, T _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    Point3() {
        x = 0;
        y = 0;
        z = 0;
    }
};

class Node {
#if defined(ERL_BUILD_TEST)
    public:
#endif
    Point<double> pos;
    Point<double> grad;
    double val;
    double pose_sig;
    double grad_sig;
    NodeType nt;

public:

    Node(Point<double> _pos, double _val, double _pose_sig, Point<double> _grad, double _grad_sig, NodeType n);

    explicit Node(Point<double> _pos, NodeType _nt = NodeType::NONE);

    Node();

    void
    updateData(double _val, double _pose_sig, Point<double> _grad, double _grad_sig, NodeType n);

    void
    UpdateNoise(double _pose_sig, double _grad_sig);

    const Point<double> &
    getPos() { return pos; }

    const Point<double> &
    getGrad() { return grad; }

    [[nodiscard]] double
    GetPosX() const { return pos.x; }

    [[nodiscard]] double
    GetPosY() const { return pos.y; }

    [[nodiscard]] double
    GetGradX() const { return grad.x; }

    [[nodiscard]] double
    GetGradY() const { return grad.y; }

    [[nodiscard]] double
    GetVal() const { return val; }

    [[nodiscard]] double
    GetPosNoise() const { return pose_sig; };

    [[nodiscard]] double
    GetGradNoise() const { return grad_sig; };

    NodeType
    getType() { return nt; }
};

class Node3 {
    Point3<double> pos;
    Point3<double> grad;
    double val;
    double pose_sig;
    double grad_sig;
    NodeType nt;

public:

    Node3(Point3<double> _pos, double _val, double _pose_sig, Point3<double> _grad, double _grad_sig, NodeType n = NodeType::NONE);

    Node3(Point3<double> _pos, NodeType _nt = NodeType::NONE);

    Node3();

    void
    updateData(double _val, double _pose_sig, Point3<double> _grad, double _grad_sig, NodeType n = NodeType::NONE);

    void
    UpdateNoise(double _pose_sig, double _grad_sig);

    const Point3<double> &
    GetPos() { return pos; }

    const Point3<double> &
    GetGrad() { return grad; }

    double
    getPosX() { return pos.x; }

    double
    getPosY() { return pos.y; }

    double
    getPosZ() { return pos.z; }

    double
    getGradX() { return grad.x; }

    double
    getGradY() { return grad.y; }

    double
    getGradZ() { return grad.z; }

    double
    getVal() { return val; }

    double
    GetPosNoise() { return pose_sig; };

    double
    GetGradNoise() { return grad_sig; };

    NodeType
    GetType() { return nt; }
};

//////////////////////////////////////////////////////////////////////////
// Parameters

// Observation GP
typedef struct obsGPparam_ {
    // Npte:
    // ObsGp is implemented to use the Ornstein-Uhlenbeck covariance function,
    // which has a form of k(r)=exp(-r/l) (See covFnc.h)
    double scale;            // the m_scale_ parameter l
    double noise;            // the m_noise_ parameter of the measurement
    // currently use a constant value
    // could be potentially modified to have heteroscedastic m_noise_
    // Note:
    // ObsGp is implemented to have overlapping partitioned GPs.
    double margin;           // used to decide if valid m_range_
    // (don't use if too close to boundary
    //  because the derivates are hard to sample)
    int overlap;          // the overlapping parameters: number of samples to overlap
    int group_size;       // the number of samples to group together
    // (the actual group size will be (group_size+overlap)
    obsGPparam_() {}

    obsGPparam_(double s, double n, double m, int ov, int gsz) :
        scale(s),
        noise(n),
        margin(m),
        overlap(ov),
        group_size(gsz) {}
} obsGPparam;

// GPIS (SDF)
typedef struct onGPISparam_ {
    // Note:
    // OnlineGPIS is implemented to use the Matern class covariance function with (nu=2/3),
    // which has a form of k(r)=(1+sqrt(3)*r/l)exp(-sqrt(3)*r/l) (See covFnc.h)
    double scale;            // the m_scale_ parameter l
    double noise;            // the default m_noise_ parameter of the measurement
    // currently use heteroscedastic m_noise_ acoording to a m_noise_ model
    double noise_deriv;      // the default m_noise_ parameter of the derivative measurement
    // currently use a m_noise_ model by numerical computation.
    onGPISparam_() {}

    onGPISparam_(double s, double n, double nd) : scale(s), noise(n), noise_deriv(nd) {}
} onGPISparam;

// QuadTree (2D) and OcTree (3D)
typedef struct tree_param_ {
    double initroot_halfleng;
    double min_halfleng;         // minimum (leaf) resolution of tree
    double min_halfleng_sqr;
    double max_halfleng;         // maximum (root) resolution of tree
    double max_halfleng_sqr;
    double cluster_halfleng;    // the resolution of GP clusters
    double cluster_halfleng_sqr;
public:
    tree_param_() : initroot_halfleng(DEFAULT_TREE_INIT_ROOT_HALFLENGTH),
                    min_halfleng(DEFAULT_TREE_MIN_HALFLENGTH),
                    min_halfleng_sqr(DEFAULT_TREE_MIN_HALFLENGTH * DEFAULT_TREE_MIN_HALFLENGTH),
                    max_halfleng(DEFAULT_TREE_MAX_HALFLENGTH),
                    max_halfleng_sqr(DEFAULT_TREE_MAX_HALFLENGTH * DEFAULT_TREE_MAX_HALFLENGTH),
                    cluster_halfleng(DEFAULT_TREE_CLUSTER_HALFLENGTH),
                    cluster_halfleng_sqr(DEFAULT_TREE_CLUSTER_HALFLENGTH * DEFAULT_TREE_CLUSTER_HALFLENGTH) {}

    tree_param_(double mi, double ma, double ini, double c) :
        initroot_halfleng(ini),
        min_halfleng(mi),
        min_halfleng_sqr(mi * mi),
        max_halfleng(ma),
        max_halfleng_sqr(ma * ma),
        cluster_halfleng(c),
        cluster_halfleng_sqr(c * c) {}
} tree_param;
