#include "erl_gaussian_process/mapping.hpp"

#include <cmath>

YAML::Node
YAML::convert<erl::gaussian_process::MappingType>::encode(
    const erl::gaussian_process::MappingType &type) {
    Node node;
    using namespace erl::gaussian_process;
    switch (type) {
        case MappingType::kIdentity: {
            node = "kIdentity";
            break;
        }
        case MappingType::kInverse: {
            node = "kInverse";
            break;
        }
        case MappingType::kInverseSqrt: {
            node = "kInverseSqrt";
            break;
        }
        case MappingType::kExp: {
            node = "kExp";
            break;
        }
        case MappingType::kLog: {
            node = "kLog";
            break;
        }
        case MappingType::kTanh: {
            node = "kTanh";
            break;
        }
        case MappingType::kSigmoid: {
            node = "kSigmoid";
            break;
        }
        case MappingType::kUnknown:
        default: {
            node = "kUnknown";
        }
    }
    return node;
}

bool
YAML::convert<erl::gaussian_process::MappingType>::decode(
    const Node &node,
    erl::gaussian_process::MappingType &type) {
    if (!node.IsScalar()) { return false; }
    auto type_name = node.as<std::string>();
    using namespace erl::gaussian_process;
    if (type_name == "kIdentity") {
        type = MappingType::kIdentity;
    } else if (type_name == "kInverse") {
        type = MappingType::kInverse;
    } else if (type_name == "kInverseSqrt") {
        type = MappingType::kInverseSqrt;
    } else if (type_name == "kExp") {
        type = MappingType::kExp;
    } else if (type_name == "kLog") {
        type = MappingType::kLog;
    } else if (type_name == "kTanh") {
        type = MappingType::kTanh;
    } else if (type_name == "kSigmoid") {
        type = MappingType::kSigmoid;
    } else {
        type = MappingType::kUnknown;
    }

    return true;
}

namespace erl::gaussian_process {

    template<typename Dtype>
    YAML::Node
    Mapping<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, type);
        ERL_YAML_SAVE_ATTR(node, setting, scale);
        return node;
    }

    template<typename Dtype>
    bool
    Mapping<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, type);
        ERL_YAML_LOAD_ATTR(node, setting, scale);
        return true;
    }

    template<typename Dtype>
    std::shared_ptr<Mapping<Dtype>>
    Mapping<Dtype>::Create() {
        return std::shared_ptr<Mapping>(new Mapping());
    }

    template<typename Dtype>
    std::shared_ptr<Mapping<Dtype>>
    Mapping<Dtype>::Create(std::shared_ptr<Setting> setting) {
        return std::shared_ptr<Mapping>(new Mapping(std::move(setting)));
    }

    template<typename Dtype>
    Mapping<Dtype>::Mapping()
        : Mapping(std::make_shared<Setting>()) {}

    template<typename Dtype>
    Mapping<Dtype>::Mapping(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        switch (m_setting_->type) {
            case MappingType::kIdentity: {
                map = [](const Dtype x) { return x; };
                inv = map;
                break;
            }
            case MappingType::kInverse: {
                map = [](const Dtype x) { return 1.0f / x; };
                inv = map;
                break;
            }
            case MappingType::kInverseSqrt: {
                map = [](const Dtype x) { return 1.0f / std::sqrt(x); };
                inv = [](const Dtype y) { return 1.0f / (y * y); };
                break;
            }
            case MappingType::kExp: {
                map = [&](const Dtype x) { return std::exp(-m_setting_->scale * x); };
                inv = [&](const Dtype y) { return -std::log(y) / m_setting_->scale; };
                break;
            }
            case MappingType::kLog: {
                map = [&](const Dtype x) { return std::log(m_setting_->scale * x); };
                inv = [&](const Dtype y) { return std::exp(y) / m_setting_->scale; };
                break;
            }
            case MappingType::kTanh: {
                map = [&](const Dtype x) { return std::tanh(m_setting_->scale * x); };
                inv = [&](const Dtype y) { return std::atanh(y) / m_setting_->scale; };
                break;
            }
            case MappingType::kSigmoid: {
                map = [&](const Dtype x) -> Dtype {
                    return 1.0f / (1.0f + std::exp(-m_setting_->scale * x));
                };
                inv = [&](const Dtype y) -> Dtype {
                    if (y >= 1.0f) {
                        return std::numeric_limits<Dtype>::infinity() / m_setting_->scale;
                    }
                    if (y <= 0.0f) {
                        return -std::numeric_limits<Dtype>::infinity() / m_setting_->scale;
                    }
                    return std::log(y / (1.0f - y)) / m_setting_->scale;
                };
                break;
            }
            case MappingType::kUnknown:
            default:
                throw std::logic_error("Mapping type is kUnknown, which is unexpected.");
        }
    }

    template class Mapping<double>;
    template class Mapping<float>;
}  // namespace erl::gaussian_process
