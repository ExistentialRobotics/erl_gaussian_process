#include "erl_common/pybind11.hpp"
#include "erl_gaussian_process/mapping.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

void
BindMapping(const py::module &m) {
    auto py_mapping = py::class_<Mapping, std::shared_ptr<Mapping>>(m, "Mapping");

    py::enum_<Mapping::Type>(py_mapping, "Type", py::arithmetic(), "Type of mapping.")
        .value(Mapping::GetTypeName(Mapping::Type::kIdentity), Mapping::Type::kIdentity)
        .value(Mapping::GetTypeName(Mapping::Type::kInverse), Mapping::Type::kInverse)
        .value(Mapping::GetTypeName(Mapping::Type::kInverseSqrt), Mapping::Type::kInverseSqrt)
        .value(Mapping::GetTypeName(Mapping::Type::kExp), Mapping::Type::kExp)
        .value(Mapping::GetTypeName(Mapping::Type::kTanh), Mapping::Type::kTanh)
        .value(Mapping::GetTypeName(Mapping::Type::kSigmoid), Mapping::Type::kSigmoid)
        .value(Mapping::GetTypeName(Mapping::Type::kUnknown), Mapping::Type::kUnknown)
        .export_values();

    py::class_<Mapping::Setting, std::shared_ptr<Mapping::Setting>>(py_mapping, "Setting")
        .def_readwrite("type", &Mapping::Setting::type)
        .def_readwrite("scale", &Mapping::Setting::scale);

    py_mapping.def(py::init([] { return Mapping::Create(); }))
        .def(py::init([](std::shared_ptr<Mapping::Setting> setting) { return Mapping::Create(std::move(setting)); }), py::arg("setting"))
        .def_property_readonly("setting", &Mapping::GetSetting)
        .def_property_readonly("map", [](const std::shared_ptr<Mapping> &mapping) { return mapping->map; })
        .def_property_readonly("inv", [](const std::shared_ptr<Mapping> &mapping) { return mapping->inv; });
}
