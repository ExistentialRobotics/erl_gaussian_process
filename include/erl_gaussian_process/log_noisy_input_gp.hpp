#pragma once

#include <utility>
#include <vector>

#include "noisy_input_gp.hpp"

namespace erl::gaussian_process {

    class LogNoisyInputGaussianProcess : public NoisyInputGaussianProcess {

    public:
        struct Setting : public NoisyInputGaussianProcess::Setting {
            double log_lambda = 10.0;
            double edf_threshold = 0.1;
        };

    private:
        Eigen::MatrixXd m_mat_log_l_;
        Eigen::VectorXd m_vec_log_alpha_;  // for logGPIS inference: determine the distance
        std::shared_ptr<Setting> m_setting_;
        std::shared_ptr<Covariance> m_kernel_;

    public:
        static std::shared_ptr<LogNoisyInputGaussianProcess>
        Create();
        static std::shared_ptr<LogNoisyInputGaussianProcess>
        Create(std::shared_ptr<Setting> setting);

        void
        Reset() override {
            NoisyInputGaussianProcess::Reset();
            m_kernel_ = nullptr;
        }

        void
        Train(
            Eigen::MatrixXd mat_x_train,
            Eigen::VectorXb vec_grad_flag,
            const Eigen::Ref<const Eigen::VectorXd> &vec_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_f,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) override;

        void
        Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out)
            const override;

    protected:
        LogNoisyInputGaussianProcess();
        explicit LogNoisyInputGaussianProcess(std::shared_ptr<Setting> setting);
    };
}  // namespace erl::gaussian_process

namespace YAML {
    template<>
    struct convert<erl::gaussian_process::LogNoisyInputGaussianProcess::Setting> {
        inline static Node
        encode(const erl::gaussian_process::LogNoisyInputGaussianProcess::Setting &setting) {
            Node node;
            node["kernel"] = setting.kernel;
            node["log_lambda"] = setting.log_lambda;
            node["edf_threshold"] = setting.edf_threshold;
            return node;
        }

        inline static bool
        decode(const Node &node, erl::gaussian_process::LogNoisyInputGaussianProcess::Setting &setting) {
            if (!node.IsMap()) { return false; }
            setting.kernel = node["kernel"].as<decltype(setting.kernel)>();
            setting.log_lambda = node["log_lambda"].as<double>();
            setting.edf_threshold = node["edf_threshold"].as<double>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::gaussian_process::LogNoisyInputGaussianProcess::Setting &setting) {
        out << BeginMap;
        out << Key << "kernel" << Value << setting.kernel;
        out << Key << "log_lambda" << Value << setting.log_lambda;
        out << Key << "edf_threshold" << Value << setting.edf_threshold;
        out << EndMap;
        return out;
    }
}  // namespace YAML
