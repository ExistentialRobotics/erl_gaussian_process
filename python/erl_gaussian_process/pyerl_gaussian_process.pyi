from enum import IntEnum
from typing import Callable
from typing import Tuple
from typing import overload

import numpy as np
import numpy.typing as npt

from erl_common.yaml import YamlableBase
from erl_covariance import Covariance

__all__ = [
    "VanillaGaussianProcess",
    "Mapping",
    "LidarGaussianProcess1D",
    "NoisyInputGaussianProcess",
]

class VanillaGaussianProcess:
    class Setting(YamlableBase):
        kernel: Covariance.Setting
        auto_normalize: bool

        def __init__(self: VanillaGaussianProcess.Setting): ...

    def __init__(self: VanillaGaussianProcess, setting: Setting): ...
    @property
    def is_trained(self: VanillaGaussianProcess) -> bool: ...
    @property
    def setting(self: VanillaGaussianProcess) -> Setting: ...
    def reset(self: VanillaGaussianProcess) -> None: ...
    def train(
        self: VanillaGaussianProcess,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        sigma_y: npt.NDArray[np.float64],
    ) -> None: ...
    def test(
        self: VanillaGaussianProcess, xt: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class Mapping:
    class Type(IntEnum):
        kIdentity = 0
        kInverse = 1
        kInverseSqrt = 2
        kExp = 3
        kTanh = 4
        kSigmoid = 5
        kUnknown = 6

    class Setting(YamlableBase):
        type: Mapping.Type
        scale: float
    @overload
    def __init__(self: Mapping): ...
    @overload
    def __init__(self: Mapping, setting: Mapping.Setting): ...

    setting: Mapping.Setting
    map: Callable[[float], float]
    inv: Callable[[float], float]

class LidarGaussianProcess1D:
    class TrainBuffer:
        class Setting(YamlableBase):
            valid_angle_min: float
            valid_angle_max: float
            valid_range_min: float
            valid_range_max: float
            mapping: Mapping.Setting

            def __init__(self: LidarGaussianProcess1D.TrainBuffer.Setting): ...

        def __len__(self: LidarGaussianProcess1D.TrainBuffer) -> int: ...

        angles: npt.NDArray[np.float64]
        distances: npt.NDArray[np.float64]
        mapped_distances: npt.NDArray[np.float64]
        local_directions: npt.NDArray[np.float64]
        xy_locals: npt.NDArray[np.float64]
        global_directions: npt.NDArray[np.float64]
        xy_globals: npt.NDArray[np.float64]
        max_distance: float
        position: npt.NDArray[np.float64]
        rotation: npt.NDArray[np.float64]

    class Setting(YamlableBase):
        group_size: int
        overlap_size: int
        boundary_margin: float
        init_variance: float
        sensor_range_var: float
        max_valid_range_var: float
        occ_test_temperature: float
        train_buffer: LidarGaussianProcess1D.TrainBuffer.Setting
        gp: VanillaGaussianProcess.Setting

        def __init__(self: LidarGaussianProcess1D.Setting): ...

    def __init__(self: LidarGaussianProcess1D, setting: Setting): ...
    @property
    def is_trained(self: LidarGaussianProcess1D) -> bool: ...
    @property
    def setting(self: LidarGaussianProcess1D) -> Setting: ...
    @property
    def train_buffer(self: LidarGaussianProcess1D) -> TrainBuffer: ...
    def global_to_local_so2(
        self: LidarGaussianProcess1D, vec_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_so2(
        self: LidarGaussianProcess1D, vec_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def global_to_local_se2(
        self: LidarGaussianProcess1D, vec_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_se2(
        self: LidarGaussianProcess1D, vec_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def reset(self: LidarGaussianProcess1D) -> None: ...
    def train(
        self: LidarGaussianProcess1D,
        angles: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
        pose: npt.NDArray[np.float64],
    ) -> None: ...
    def test(
        self: LidarGaussianProcess1D, thetas: npt.NDArray[np.float64], un_map: bool = True
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def compute_occ(self: LidarGaussianProcess1D, angle: float, r: float) -> Tuple[bool, float, float, float]: ...

class NoisyInputGaussianProcess:
    class Setting(YamlableBase):
        kernel: Covariance.Setting

        def __init__(self: NoisyInputGaussianProcess.Setting): ...

    def __init__(self: NoisyInputGaussianProcess, setting: Setting): ...
    @property
    def is_trained(self: NoisyInputGaussianProcess) -> bool: ...
    @property
    def setting(self: NoisyInputGaussianProcess) -> Setting: ...
    def reset(self: NoisyInputGaussianProcess) -> None: ...
    def train(
        self: NoisyInputGaussianProcess,
        mat_x_train: npt.NDArray[np.float64],
        vec_grad_flag: npt.NDArray[np.bool_],
        vec_y: npt.NDArray[np.float64],
        vec_sigma_x: npt.NDArray[np.float64],
        vec_sigma_y: npt.NDArray[np.float64],
        vec_sigma_grad: npt.NDArray[np.float64],
    ) -> None: ...
    def test(
        self: NoisyInputGaussianProcess, mat_x_test: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
