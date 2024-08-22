from enum import IntEnum
from typing import Callable
from typing import Tuple
from typing import overload

import numpy as np
import numpy.typing as npt
from erl_common.yaml import YamlableBase

from erl_covariance import Covariance
from erl_geometry import LidarFrame2D
from erl_geometry import RangeSensorFrame3D

__all__ = [
    "VanillaGaussianProcess",
    "Mapping",
    "LidarGaussianProcess2D",
    "NoisyInputGaussianProcess",
    "RangeSensorGaussianProcess3D",
]

class VanillaGaussianProcess:
    class Setting(YamlableBase):
        kernel_type: str
        kernel: Covariance.Setting
        max_num_samples: int
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
        mat_x_train: npt.NDArray[np.float64],
        vec_y: npt.NDArray[np.float64],
        vac_var_y: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(
        self: VanillaGaussianProcess, mat_x_test: npt.NDArray[np.float64]
    ) -> Tuple[bool, npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

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

class LidarGaussianProcess2D:
    class Setting(YamlableBase):
        group_size: int
        overlap_size: int
        margin: float
        init_variance: float
        sensor_range_var: float
        max_valid_range_var: float
        occ_test_temperature: float
        lidar_frame: LidarFrame2D.Setting
        gp: VanillaGaussianProcess.Setting
        mapping: Mapping.Setting

        def __init__(self: LidarGaussianProcess2D.Setting): ...

    def __init__(self: LidarGaussianProcess2D, setting: Setting): ...
    @property
    def is_trained(self: LidarGaussianProcess2D) -> bool: ...
    @property
    def setting(self: LidarGaussianProcess2D) -> Setting: ...
    @property
    def gps(self: LidarGaussianProcess2D) -> list[VanillaGaussianProcess]: ...
    @property
    def angle_partitions(self: LidarGaussianProcess2D) -> list[Tuple[int, int, float, float]]: ...
    @property
    def lidar_frame(self: LidarGaussianProcess2D) -> LidarFrame2D: ...
    def global_to_local_so2(
        self: LidarGaussianProcess2D, dir_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_so2(
        self: LidarGaussianProcess2D, dir_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def global_to_local_se2(
        self: LidarGaussianProcess2D, xy_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_se2(
        self: LidarGaussianProcess2D, xy_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def reset(self: LidarGaussianProcess2D) -> None: ...
    def train(
        self: LidarGaussianProcess2D,
        rotation: npt.NDArray[np.float64],
        translation: npt.NDArray[np.float64],
        ranges: npt.NDArray[np.float64],
        repartition_on_hit_rays: bool,
    ) -> None: ...
    def test(
        self: LidarGaussianProcess2D,
        angles: npt.NDArray[np.float64],
        angles_are_local: bool,
        un_map: bool,
        parallel: bool,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def compute_occ(self: LidarGaussianProcess2D, angle: float, r: float) -> Tuple[bool, float, float, float]: ...

class NoisyInputGaussianProcess:
    class Setting(YamlableBase):
        kernel_type: str
        kernel: Covariance.Setting
        max_num_samples: int

        def __init__(self: NoisyInputGaussianProcess.Setting): ...

    def __init__(self: NoisyInputGaussianProcess, setting: Setting): ...
    @property
    def is_trained(self: NoisyInputGaussianProcess) -> bool: ...
    @property
    def setting(self: NoisyInputGaussianProcess) -> Setting: ...
    def reset(self: NoisyInputGaussianProcess, max_num_samples: int, x_dim: int) -> None: ...
    @property
    def num_train_samples(self: NoisyInputGaussianProcess) -> int: ...
    @property
    def num_train_samples_with_grad(self: NoisyInputGaussianProcess) -> int: ...
    @property
    def kernel(self: NoisyInputGaussianProcess) -> Covariance: ...
    @property
    def x_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def y_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def grad_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def grad_flag(self: NoisyInputGaussianProcess) -> npt.NDArray[np.bool_]: ...
    @property
    def var_x_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def var_y_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def var_grad_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def k_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def alpha(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def cholesky_k_train(self: NoisyInputGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def memory_usage(self: NoisyInputGaussianProcess) -> int: ...
    def train(
        self: NoisyInputGaussianProcess,
        mat_x_train: npt.NDArray[np.float64],
        vec_grad_flag: npt.NDArray[np.bool_],
        vec_y: npt.NDArray[np.float64],
        vec_var_x: npt.NDArray[np.float64],
        vec_var_y: npt.NDArray[np.float64],
        vec_var_grad: npt.NDArray[np.float64],
    ) -> None: ...
    def test(
        self: NoisyInputGaussianProcess, mat_x_test: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class RangeSensorGaussianProcess3D:
    class Setting(YamlableBase):
        row_group_size: int
        row_overlap_size: int
        row_margin: int
        col_group_size: int
        col_overlap_size: int
        col_margin: int
        init_variance: float
        sensor_range_var: float
        max_valid_range_var: float
        occ_test_temperature: float
        range_sensor_frame_type: str
        range_sensor_frame: RangeSensorFrame3D.Setting
        gp: VanillaGaussianProcess.Setting
        mapping: Mapping.Setting

        def __init__(self: RangeSensorGaussianProcess3D.Setting): ...

    def __init__(self: RangeSensorGaussianProcess3D, setting: Setting): ...
    @property
    def is_trained(self: RangeSensorGaussianProcess3D) -> bool: ...
    @property
    def setting(self: RangeSensorGaussianProcess3D) -> Setting: ...
    @property
    def gps(self: RangeSensorGaussianProcess3D) -> list[list[VanillaGaussianProcess]]: ...
    @property
    def row_partitions(self: RangeSensorGaussianProcess3D) -> list[Tuple[int, int, float, float]]: ...
    @property
    def col_partitions(self: RangeSensorGaussianProcess3D) -> list[Tuple[int, int, float, float]]: ...
    @property
    def range_sensor_frame(self: RangeSensorGaussianProcess3D) -> RangeSensorFrame3D: ...
    @property
    def mapping(self: RangeSensorGaussianProcess3D) -> Mapping: ...
    def global_to_local_so3(
        self: RangeSensorGaussianProcess3D, dir_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_so3(
        self: RangeSensorGaussianProcess3D, dir_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def global_to_local_se3(
        self: RangeSensorGaussianProcess3D, xyz_global: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def local_to_global_se3(
        self: RangeSensorGaussianProcess3D, xyz_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def compute_frame_coords(
        self: RangeSensorGaussianProcess3D, xyz_frame: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def reset(self: RangeSensorGaussianProcess3D) -> None: ...
    def store_data(
        self: RangeSensorGaussianProcess3D,
        rotation: npt.NDArray[np.float64],
        translation: npt.NDArray[np.float64],
        ranges: npt.NDArray[np.float64],
    ) -> None: ...
    def train(
        self: RangeSensorGaussianProcess3D,
        rotation: npt.NDArray[np.float64],
        translation: npt.NDArray[np.float64],
        ranges: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(
        self: RangeSensorGaussianProcess3D,
        directions: npt.NDArray[np.float64],
        directions_are_local: bool,
        un_map: bool,
        parallel: bool,
    ) -> Tuple[bool, npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
