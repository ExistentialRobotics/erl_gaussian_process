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
        kernel_setting_type: str
        kernel: Covariance.Setting
        max_num_samples: int

        def __init__(self: VanillaGaussianProcess.Setting): ...

    class TestResult:
        num_test: int
        k_test: np.ndarray
        @overload
        def get_mean(self, y_index: int, parallel: bool) -> np.ndarray: ...
        @overload
        def get_mean(self, index: int, y_index: int) -> float: ...
        @overload
        def get_variance(self, parallel: bool) -> np.ndarray: ...
        @overload
        def get_variance(self, index: int) -> float: ...

    class TrainSet:
        x_dim: int
        y_dim: int
        num_samples: int
        x: np.ndarray
        y: np.ndarray
        var: np.ndarray

    def __init__(self: VanillaGaussianProcess, setting: Setting): ...
    @property
    def is_trained(self: VanillaGaussianProcess) -> bool: ...
    @property
    def setting(self: VanillaGaussianProcess) -> Setting: ...
    def reset(self: VanillaGaussianProcess) -> None: ...
    def train(
        self: VanillaGaussianProcess,
        mat_x_train: np.ndarray,
        mat_y_train: np.ndarray,
        vac_var_y: np.ndarray,
    ) -> bool: ...
    def test(self: VanillaGaussianProcess, mat_x_test: np.ndarray) -> TestResult: ...

class VanillaGaussianProcessD(VanillaGaussianProcess): ...
class VanillaGaussianProcessF(VanillaGaussianProcess): ...

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
        partition_on_hit_rays: bool
        symmetric_partitions: bool
        group_size: int
        overlap_size: int
        margin: float
        init_variance: float
        sensor_range_var: float
        max_valid_range_var: float
        occ_test_temperature: float
        sensor_frame: LidarFrame2D.Setting
        gp: VanillaGaussianProcess.Setting
        mapping: Mapping.Setting

        def __init__(self: LidarGaussianProcess2D.Setting): ...

    class TestResult:
        num_test: int
        def get_ktest(self, index: int): ...
        @overload
        def get_mean(self, parallel: bool) -> tuple[bool, np.ndarray]: ...
        @overload
        def get_mean(self, index: int) -> tuple[bool, float]: ...
        @overload
        def get_variance(self, parallel: bool) -> tuple[bool, np.ndarray]: ...
        @overload
        def get_variance(self, index: int) -> tuple[bool, float]: ...

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
    def sensor_frame(self: LidarGaussianProcess2D) -> LidarFrame2D: ...
    def global_to_local_so2(self: LidarGaussianProcess2D, dir_global: np.ndarray) -> np.ndarray: ...
    def local_to_global_so2(self: LidarGaussianProcess2D, dir_local: np.ndarray) -> np.ndarray: ...
    def global_to_local_se2(self: LidarGaussianProcess2D, xy_global: np.ndarray) -> np.ndarray: ...
    def local_to_global_se2(self: LidarGaussianProcess2D, xy_local: np.ndarray) -> np.ndarray: ...
    def reset(self: LidarGaussianProcess2D) -> None: ...
    def partition_on_angles(self: LidarGaussianProcess2D) -> None: ...
    def partition_on_hit_rays(self: LidarGaussianProcess2D) -> None: ...
    def train(
        self: LidarGaussianProcess2D,
        rotation: np.ndarray,
        translation: np.ndarray,
        ranges: np.ndarray,
    ) -> None: ...
    def search_partition(self: LidarGaussianProcess2D, angle_local: float) -> int: ...
    def test(
        self: LidarGaussianProcess2D,
        angles: np.ndarray,
        angles_are_local: bool,
        un_map: bool,
    ) -> TestResult: ...
    def compute_occ(self: LidarGaussianProcess2D, angle_local: float, r: float) -> Tuple[bool, float, float]: ...

class LidarGaussianProcess2Dd(LidarGaussianProcess2D): ...
class LidarGaussianProcess2Df(LidarGaussianProcess2D): ...

class NoisyInputGaussianProcess:
    class Setting(YamlableBase):
        kernel_type: str
        kernel_setting_type: str
        kernel: Covariance.Setting
        max_num_samples: int
        no_gradient_observation: bool

        def __init__(self: NoisyInputGaussianProcess.Setting): ...

    class TestResult:
        num_test: int
        k_test: np.ndarray
        @overload
        def get_mean(self, y_index: int, parallel: bool) -> np.ndarray: ...
        @overload
        def get_mean(self, index: int, y_index: int) -> float: ...
        @overload
        def get_gradient(self, y_index: int, parallel: bool) -> np.ndarray: ...
        @overload
        def get_gradient(self, index: int, y_index: int) -> np.ndarray: ...
        @overload
        def get_mean_variance(self, parallel: bool) -> np.ndarray: ...
        @overload
        def get_mean_variance(self, index: int) -> float: ...
        @overload
        def get_gradient_variance(self, parallel: bool) -> np.ndarray: ...
        @overload
        def get_gradient_variance(self, index: int) -> np.ndarray: ...
        @overload
        def get_variance(self, parallel: bool) -> np.ndarray: ...
        @overload
        def get_variance(self, index: int) -> np.ndarray: ...

    class TrainSet:
        x_dim: int
        y_dim: int
        num_samples: int
        num_samples_with_grad: int
        x: np.ndarray
        y: np.ndarray
        grad: np.ndarray
        var_x: np.ndarray
        var_y: np.ndarray
        var_grad: np.ndarray
        grad_flag: np.ndarray

    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def using_reduced_rank_kernel(self) -> bool: ...
    kernel_origin: np.ndarray
    def reset(self, max_num_samples: int, x_dim: int, y_dim: int) -> None: ...
    @property
    def kernel(self) -> Covariance: ...
    @property
    def train_set(self) -> TrainSet: ...
    @property
    def k_train(self) -> np.ndarray: ...
    @property
    def alpha(self) -> np.ndarray: ...
    @property
    def cholesky_k_train(self) -> np.ndarray: ...
    @property
    def memory_usage(self) -> int: ...
    def update_ktrain(self) -> None: ...
    def train(
        self,
        mat_x_train: np.ndarray,
        mat_y_train: np.ndarray,
        mat_grad_train: np.ndarray,
        vec_grad_flag: npt.NDArray[np.bool_],
        vec_var_x: np.ndarray,
        vec_var_y: np.ndarray,
        vec_var_grad: np.ndarray,
    ) -> None: ...
    def test(self, mat_x_test: np.ndarray) -> TestResult: ...

class NoisyInputGaussianProcessD(NoisyInputGaussianProcess): ...
class NoisyInputGaussianProcessF(NoisyInputGaussianProcess): ...

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
        sensor_frame_type: str
        range_sensor_frame: RangeSensorFrame3D.Setting
        gp: VanillaGaussianProcess.Setting
        mapping: Mapping.Setting

        def __init__(self: RangeSensorGaussianProcess3D.Setting): ...

    class TestResult:
        num_test: int
        def get_ktest(self, index: int): ...
        @overload
        def get_mean(self, parallel: bool) -> tuple[bool, np.ndarray]: ...
        @overload
        def get_mean(self, index: int) -> tuple[bool, float]: ...
        @overload
        def get_variance(self, parallel: bool) -> tuple[bool, np.ndarray]: ...
        @overload
        def get_variance(self, index: int) -> tuple[bool, float]: ...

    def __init__(self, setting: Setting): ...
    @property
    def is_trained(self) -> bool: ...
    @property
    def setting(self) -> Setting: ...
    @property
    def gps(self) -> list[list[VanillaGaussianProcess]]: ...
    @property
    def row_partitions(self) -> list[Tuple[int, int, float, float]]: ...
    @property
    def col_partitions(self) -> list[Tuple[int, int, float, float]]: ...
    @property
    def range_sensor_frame(self) -> RangeSensorFrame3D: ...
    @property
    def mapping(self) -> Mapping: ...
    def global_to_local_so3(self, dir_global: np.ndarray) -> np.ndarray: ...
    def local_to_global_so3(self, dir_local: np.ndarray) -> np.ndarray: ...
    def global_to_local_se3(self, xyz_global: np.ndarray) -> np.ndarray: ...
    def local_to_global_se3(self, xyz_local: np.ndarray) -> np.ndarray: ...
    def compute_frame_coords(self, xyz_frame: np.ndarray) -> np.ndarray: ...
    def reset(self) -> None: ...
    def store_data(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        ranges: np.ndarray,
    ) -> None: ...
    def train(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        ranges: np.ndarray,
    ) -> bool: ...
    def search_partition(self, frame_coords: np.ndarray) -> tuple[int, int]: ...
    def test(
        self,
        directions: np.ndarray,
        directions_are_local: bool,
        un_map: bool,
    ) -> TestResult: ...
    def compute_occ(self, dir_local: np.ndarray, r: float) -> Tuple[bool, float, float]: ...

class RangeSensorGaussianProcess3Dd(RangeSensorGaussianProcess3D): ...
class RangeSensorGaussianProcess3Df(RangeSensorGaussianProcess3D): ...
