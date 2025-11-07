#pragma once

#include "init.hpp"

#include "erl_common/enum_parse.hpp"
#include "erl_common/yaml.hpp"

#include <functional>
#include <memory>

namespace erl::gaussian_process {
    enum class MappingType {
        kIdentity = 0,
        kInverse = 1,
        kInverseSqrt = 2,
        kExp = 3,
        kLog = 4,
        kTanh = 5,
        kSigmoid = 6,
        kUnknown = 7
    };
}

ERL_REFLECT_ENUM_SCHEMA(
    erl::gaussian_process::MappingType,
    8,
    ERL_REFLECT_ENUM_MEMBER("identity", erl::gaussian_process::MappingType::kIdentity),
    ERL_REFLECT_ENUM_MEMBER("inverse", erl::gaussian_process::MappingType::kInverse),
    ERL_REFLECT_ENUM_MEMBER("inverse_sqrt", erl::gaussian_process::MappingType::kInverseSqrt),
    ERL_REFLECT_ENUM_MEMBER("exp", erl::gaussian_process::MappingType::kExp),
    ERL_REFLECT_ENUM_MEMBER("log", erl::gaussian_process::MappingType::kLog),
    ERL_REFLECT_ENUM_MEMBER("tanh", erl::gaussian_process::MappingType::kTanh),
    ERL_REFLECT_ENUM_MEMBER("sigmoid", erl::gaussian_process::MappingType::kSigmoid),
    ERL_REFLECT_ENUM_MEMBER("unknown", erl::gaussian_process::MappingType::kUnknown))
ERL_PARSE_ENUM(erl::gaussian_process::MappingType, 8);

namespace erl::gaussian_process {

    template<typename Dtype>
    class Mapping {

    public:
        struct Setting : public common::Yamlable<Setting> {
            MappingType type = MappingType::kUnknown;
            Dtype scale = 1.0;

            ERL_REFLECT_SCHEMA(
                Setting,
                ERL_REFLECT_MEMBER(Setting, type),
                ERL_REFLECT_MEMBER(Setting, scale));
        };

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

    using MappingD = Mapping<double>;
    using MappingF = Mapping<float>;

    extern template class Mapping<double>;
    extern template class Mapping<float>;
}  // namespace erl::gaussian_process
