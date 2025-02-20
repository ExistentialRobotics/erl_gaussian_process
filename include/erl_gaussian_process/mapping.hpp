#pragma once

#include "init.hpp"

#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::gaussian_process {
    enum class MappingType { kIdentity = 0, kInverse = 1, kInverseSqrt = 2, kExp = 3, kLog = 4, kTanh = 5, kSigmoid = 6, kUnknown = 7 };
}

template<>
struct YAML::convert<erl::gaussian_process::MappingType> {
    static Node
    encode(const erl::gaussian_process::MappingType &type);

    static bool
    decode(const Node &node, erl::gaussian_process::MappingType &type);
};

namespace erl::gaussian_process {

    template<typename Dtype>
    class Mapping {

    public:
        struct Setting : common::Yamlable<Setting> {
            MappingType type = MappingType::kUnknown;
            Dtype scale = 1.0;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        std::shared_ptr<Setting> m_setting_;

    public:
        std::function<Dtype(Dtype)> map;
        std::function<Dtype(Dtype)> inv;

        static std::shared_ptr<Mapping>
        Create();

        static std::shared_ptr<Mapping>
        Create(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

    private:
        Mapping();

        explicit Mapping(std::shared_ptr<Setting> setting);
    };

#include "mapping.tpp"

    using Mapping_d = Mapping<double>;
    using Mapping_f = Mapping<float>;
}  // namespace erl::gaussian_process

template<>
struct YAML::convert<erl::gaussian_process::Mapping_d::Setting> : erl::gaussian_process::Mapping_d::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gaussian_process::Mapping_f::Setting> : erl::gaussian_process::Mapping_f::Setting::YamlConvertImpl {};
