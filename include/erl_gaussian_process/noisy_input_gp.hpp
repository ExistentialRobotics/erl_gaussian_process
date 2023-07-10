#pragma once

#include <utility>
#include <vector>

#include "erl_covariance/covariance.hpp"
#include "vanilla_gp.hpp"

namespace erl::gaussian_process {

    class NoisyInputGaussianProcess {

    public:
        struct Setting : public common::Yamlable<Setting> {
            std::shared_ptr<Covariance::Setting> kernel = []() -> std::shared_ptr<Covariance::Setting> {
                auto setting = std::make_shared<Covariance::Setting>();
                setting->type = Covariance::Type::kMatern32;
                setting->alpha = 1.;
                setting->scale = 1.2;
                return setting;
            }();
        };

#if defined(BUILD_TEST)
    public:
        Eigen::VectorXd m_vec_y_;
        Eigen::MatrixXd m_mat_k_train_;
        Eigen::VectorXd m_vec_sigma_x_;
        Eigen::VectorXd m_vec_sigma_grad_;
#else
    protected:
#endif
        Eigen::MatrixXd m_mat_x_train_;
        Eigen::VectorXb m_vec_grad_flag_;
        Eigen::MatrixXd m_mat_l_;
        Eigen::VectorXd m_vec_alpha_;
        bool m_trained_ = false;
        double m_three_over_scale_square_ = 0.;  // for computing normal variance
        std::shared_ptr<Setting> m_setting_;
        std::shared_ptr<Covariance> m_kernel_;

    public:
        static std::shared_ptr<NoisyInputGaussianProcess>
        Create();
        static std::shared_ptr<NoisyInputGaussianProcess>
        Create(std::shared_ptr<Setting> setting);

        [[maybe_unused]] [[nodiscard]] inline bool
        IsTrained() const {
            return m_trained_;
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        inline virtual void
        Reset() {
            m_trained_ = false;
            m_kernel_ = nullptr;
        }

        [[maybe_unused]] [[nodiscard]] inline long
        GetNumSamples() const {
            return m_mat_x_train_.cols();
        }

        virtual void
        Train(
            Eigen::MatrixXd mat_x_train,
            Eigen::VectorXb vec_grad_flag,
            const Eigen::Ref<const Eigen::VectorXd> &vec_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_f,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad);

        virtual void
        Test(const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test, Eigen::Ref<Eigen::VectorXd> vec_f_out, Eigen::Ref<Eigen::VectorXd> vec_var_out) const;

        virtual ~NoisyInputGaussianProcess() = default;

    protected:
        NoisyInputGaussianProcess();
        explicit NoisyInputGaussianProcess(std::shared_ptr<Setting> setting);
    };
}  // namespace erl::gaussian_process

namespace YAML {

    template<>
    struct convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting> {
        inline static Node
        encode(const erl::gaussian_process::NoisyInputGaussianProcess::Setting &setting) {
            Node node;
            node["kernel"] = *setting.kernel;
            return node;
        }

        inline static bool
        decode(const Node &node, erl::gaussian_process::NoisyInputGaussianProcess::Setting &setting) {
            if (!node.IsMap()) { return false; }
            *setting.kernel = node["kernel"].as<erl::gaussian_process::Covariance::Setting>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::gaussian_process::NoisyInputGaussianProcess::Setting &setting) {
        out << YAML::BeginMap;
        out << YAML::Key << "kernel" << YAML::Value << *setting.kernel;
        out << YAML::EndMap;
        return out;
    }
}  // namespace YAML
