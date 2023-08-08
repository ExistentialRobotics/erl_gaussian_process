#pragma once

#include <functional>
#include <memory>

#include "erl_covariance/covariance.hpp"

namespace erl::gaussian_process {
    using namespace common;
    using namespace covariance;

    /**
     * VanillaGaussianProcess implements the standard Gaussian Process
     */
    class VanillaGaussianProcess {

    public:
        // structure for holding the parameters
        struct Setting : public Yamlable<Setting> {
            std::shared_ptr<Covariance::Setting> kernel = []() -> std::shared_ptr<Covariance::Setting> {
                auto setting = std::make_shared<Covariance::Setting>();
                setting->type = Covariance::Type::kOrnsteinUhlenbeck;
                setting->alpha = 1.;
                setting->scale = 0.5;
                setting->scale_mix = 1.;
                return setting;
            }();
            bool auto_normalize = false;
        };

#if defined(BUILD_TEST)
    public:
#else
    private:
#endif
        Eigen::MatrixXd m_mat_x_train_;
        Eigen::MatrixXd m_mat_l_;
        Eigen::VectorXd m_vec_alpha_;
        double m_mean_ = 0.;
        double m_std_ = 0.;
        bool m_trained_ = false;
        std::shared_ptr<Setting> m_setting_;
        std::shared_ptr<Covariance> m_kernel_;

    public:
        explicit VanillaGaussianProcess(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)) {}

        [[nodiscard]] inline bool
        IsTrained() const {
            return m_trained_;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        inline void
        Reset() {
            m_trained_ = false;
            m_kernel_ = nullptr;
        }

        void
        Train(Eigen::MatrixXd mat_x_train, const Eigen::Ref<const Eigen::VectorXd> &vec_y, const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y);

        void
        Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out) const;
    };
}  // namespace erl::gaussian_process

namespace YAML {
    template<>
    struct convert<erl::gaussian_process::VanillaGaussianProcess::Setting> {
        inline static Node
        encode(const erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
            Node node;
            node["kernel"] = *setting.kernel;
            node["auto_normalize"] = setting.auto_normalize;
            return node;
        }

        inline static bool
        decode(const Node &node, erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
            if (!node.IsMap()) { return false; }
            *setting.kernel = node["kernel"].as<erl::covariance::Covariance::Setting>();
            setting.auto_normalize = node["auto_normalize"].as<bool>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::gaussian_process::VanillaGaussianProcess::Setting &setting) {
        out << BeginMap;
        out << Key << "kernel" << Value << *setting.kernel;
        out << Key << "auto_normalize" << Value << setting.auto_normalize;
        out << EndMap;
        return out;
    }
}  // namespace YAML
