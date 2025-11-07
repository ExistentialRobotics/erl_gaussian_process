#include "erl_common/pybind11.hpp"
#include "erl_common/pybind11_yaml.hpp"
#include "erl_gaussian_process/mapping.hpp"

using namespace erl::common;
using namespace erl::gaussian_process;

template<typename Dtype>
void
BindMappingImpl(const py::module &m, const char *name) {
    using T = Mapping<Dtype>;

    auto py_mapping = py::class_<T, std::shared_ptr<T>>(m, name);
    BindYamlable<decltype(py_mapping), typename T::Setting>(py_mapping, "Setting");

    py_mapping.def(py::init([] { return T::Create(); }))
        .def(
            py::init([](std::shared_ptr<typename T::Setting> setting) {
                return T::Create(std::move(setting));
            }),
            py::arg("setting"))
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly(
            "map",
            [](const std::shared_ptr<T> &mapping) { return mapping->map; })
        .def_property_readonly("inv", [](const std::shared_ptr<T> &mapping) {
            return mapping->inv;
        });
}

void
BindMapping(const py::module &m) {
    BindYamlableEnum<decltype(m), MappingType, 8>(m, "MappingType");
    BindMappingImpl<double>(m, "MappingD");
    BindMappingImpl<float>(m, "MappingF");
}
