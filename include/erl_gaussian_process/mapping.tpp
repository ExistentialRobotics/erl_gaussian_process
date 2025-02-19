#pragma once

#include <cmath>

template<typename Dtype>
YAML::Node
Mapping<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
    YAML::Node node;
    node["type"] = setting.type;
    node["scale"] = setting.scale;
    return node;
}

template<typename Dtype>
bool
Mapping<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
    if (!node.IsMap()) { return false; }
    setting.type = node["type"].as<MappingType>();
    setting.scale = node["scale"].as<Dtype>();
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
            map = [](const Dtype x) { return 1. / x; };
            inv = map;
            break;
        }
        case MappingType::kInverseSqrt: {
            map = [](const Dtype x) { return 1. / std::sqrt(x); };
            inv = [](const Dtype y) { return 1. / (y * y); };
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
            map = [&](const Dtype x) -> Dtype { return 1. / (1. + std::exp(-m_setting_->scale * x)); };
            inv = [&](const Dtype y) -> Dtype {
                if (y >= 1.) { return std::numeric_limits<Dtype>::infinity() / m_setting_->scale; }
                if (y <= 0.) { return -std::numeric_limits<Dtype>::infinity() / m_setting_->scale; }
                return std::log(y / (1. - y)) / m_setting_->scale;
            };
            break;
        }
        case MappingType::kUnknown:
        default:
            throw std::logic_error("Mapping type is kUnknown, which is unexpected.");
    }
}
