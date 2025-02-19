#include "erl_gaussian_process/mapping.hpp"

YAML::Node
YAML::convert<erl::gaussian_process::MappingType>::encode(const erl::gaussian_process::MappingType &type) {
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
YAML::convert<erl::gaussian_process::MappingType>::decode(const Node &node, erl::gaussian_process::MappingType &type) {
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
