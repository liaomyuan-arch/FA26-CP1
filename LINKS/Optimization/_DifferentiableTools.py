import numpy as np
from typing import Optional, Union, List, Callable, Tuple, overload
import jax
from ..Kinematics import MechanismSolver
from ..Kinematics._kin import with_crank
from ..Geometry import CurveEngine
from ._utils import differentiated_distance, differentiated_distance_scaled

class PreprocessedBatch:
    def __init__(self, As, node_types, orders, mappings, valid, is_single):
        self.As = As
        self.node_types = node_types
        self.orders = orders
        self.mappings = mappings
        self.is_single = is_single
        self.valid = valid

class DifferentiableTools:
    def __init__(self,
                 timesteps: Optional[int] = 200,
                 max_size: Optional[int] = 20,
                 scaled = False,
                 device: Optional[Union[jax.Device, str]] = 'cpu'):
        '''
        A class to set up the optimization core for mechanism design.
        Parameters
        ----------
        max_size : int, optional
            Maximum number of joints in the mechanism. Default is 20.
        timesteps : int, optional
            Number of timesteps to use when solving the mechanism. Default is 200.
        scaled : bool, optional
            Whether to use scale-normalized curve distance. Default is False.
        device : Union[jax.Device, str], optional
            The device to use for JAX computations. Can be 'cpu', 'gpu', or a jax.Device instance. Default is 'cpu'.
        '''

        self.max_size = max_size
        self.timesteps = timesteps
        self.scaled = scaled

        if isinstance(device, jax.Device):
            self.device = device
        elif device=='cpu':
            self.device = jax.devices('cpu')[0]
        elif device=='gpu':
            self.device = jax.devices('gpu')[0]
        else:
            raise ValueError("Device must be 'cpu' or 'gpu' or a jax.Device instance.")

        if self.scaled:
            self.distance_function = differentiated_distance_scaled
        else:
            self.distance_function = differentiated_distance

        self.solver = MechanismSolver(max_size=max_size, timesteps=timesteps, is_sorted=False, device=self.device)

    def compile(self):
        self.distance_function = jax.jit(self.distance_function, device=self.device)
    
    def get_preprocessed(self,
                         x0s: Union[np.ndarray, List[np.ndarray]],
                         edges: Union[np.ndarray, List[np.ndarray]],
                         fixed_joints: Union[np.ndarray, List[np.ndarray]]):
        is_single = False
        if not isinstance(x0s, list):
            is_single = True

        As_, x0s_, node_types_, orders, mappings, valid = self.solver.preprocess(
            x0s=x0s,
            edges=edges,
            fixed_nodes=fixed_joints
        )

        return PreprocessedBatch(As_, node_types_, orders, mappings, valid, is_single)

    @overload
    def __call__(self,
                 x0s: np.ndarray,
                 preprocessed: PreprocessedBatch,
                 target_curve: np.ndarray,
                 target_idx: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 start_theta: Optional[float] = 0.0,
                 end_theta: Optional[float] = 2 * np.pi):
        ...

    @overload
    def __call__(self,
                 x0s: Union[np.ndarray, List[np.ndarray]],
                 edges: Union[np.ndarray, List[np.ndarray]],
                 fixed_joints: Union[np.ndarray, List[np.ndarray]],
                 target_curve: np.ndarray,
                 target_idx: Union[np.ndarray, List[np.ndarray]] = None,
                 start_theta: Optional[float] = 0.0,
                 end_theta: Optional[float] = 2 * np.pi):
        ...
        
    def __call__(self, *args, **kwargs):
        # if any item in args or kwargs is PreprocessedBatch
        is_preprocessed = False
        for arg in args:
            if isinstance(arg, PreprocessedBatch):
                is_preprocessed = True
                break
        if not is_preprocessed:
            for key, value in kwargs.items():
                if isinstance(value, PreprocessedBatch):
                    is_preprocessed = True
                    break
        
        if is_preprocessed:
            return self._preproc_call(*args, **kwargs)
        else:
            return self._basic_call(*args, **kwargs)
        
    def _preproc_call(self,
                 x0s: np.ndarray,
                 preprocessed: PreprocessedBatch,
                 target_curve: np.ndarray,
                 target_idx: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 start_theta: Optional[float] = 0.0,
                 end_theta: Optional[float] = 2 * np.pi):
        
        is_single = preprocessed.is_single
        As_ = preprocessed.As
        node_types_ = preprocessed.node_types
        orders = preprocessed.orders
        mappings = preprocessed.mappings
        valid = preprocessed.valid

        # x0s are the free (nodes 2+) positions; prepend the fixed crank to match the
        # full node indexing that orders/mappings (built from a prior preprocess call) use.
        x0s = [with_crank(x0) for x0 in x0s] if not is_single else [with_crank(x0s)]

        x0s_ = np.zeros((len(x0s), self.max_size, 2), dtype=np.float64)
        for i in range(len(x0s)):
            x0s_[i, :len(x0s[i])] = x0s[i][orders[i]]

        if target_idx is None:
            target_idx = [orders[i][-1] for i in range(len(orders))] if not is_single else orders[0][-1]

        if not is_single:
            for i in range(len(target_idx)):
                if target_idx[i] == None:
                    target_idx[i] = orders[i][-1]

        thetas_ = np.linspace(start_theta, end_theta, self.timesteps)

        target_idx_ = np.array([mappings[i][target_idx[i]] for i in range(len(target_idx))] if not is_single else [mappings[0][target_idx]])

        n_mechanisms = As_.shape[0]
        target_curves = target_curve[None].repeat(n_mechanisms, axis=0)

        with jax.default_device(self.device):
            outputs, distance_grads = self.distance_function(
                x0s_,
                As_,
                node_types_,
                thetas_,
                target_idx_,
                target_curves
            )
            distances = np.array(outputs[1])
            distance_grads = np.array(distance_grads)
            distances[~valid] = np.inf
            distance_grads[~valid] = 0.0

            distance_grads = [distance_grads[i][mappings[i]] for i in range(len(distance_grads))]
            # Drop the crank rows: callers only ever see gradients for the free positions they gave us.
            distance_grads = [dg[2:] for dg in distance_grads]

        if np.any(np.isnan(distance_grads)):
            invalids = np.isnan(distance_grads).any(axis=-1).any(axis=-1)
            distances[invalids] = np.inf

        if is_single:
            distances = distances[0]
            distance_grads = distance_grads[0]
            return distances, np.nan_to_num(distance_grads, nan=0.0)

        return distances, [np.nan_to_num(dg, nan=0.0) for dg in distance_grads]

    def _basic_call(self,
                 x0s: Union[np.ndarray, List[np.ndarray]],
                 edges: Union[np.ndarray, List[np.ndarray]],
                 fixed_joints: Union[np.ndarray, List[np.ndarray]],
                 target_curve: np.ndarray,
                 target_idx: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 start_theta: Optional[float] = 0.0,
                 end_theta: Optional[float] = 2 * np.pi):

        is_single = False
        if not isinstance(x0s, list):
            is_single = True

        As_, x0s_, node_types_, orders, mappings, valid = self.solver.preprocess(
            x0s=x0s,
            edges=edges,
            fixed_nodes=fixed_joints
        )
        
        if target_idx is None:
            target_idx = [orders[i][-1] for i in range(len(orders))] if not is_single else orders[0][-1]
        
        if not is_single:
            for i in range(len(target_idx)):
                if target_idx[i] == None:
                    target_idx[i] = orders[i][-1]
        
        thetas_ = np.linspace(start_theta, end_theta, self.timesteps)
        
        target_idx_ = np.array([mappings[i][target_idx[i]] for i in range(len(target_idx))] if not is_single else [mappings[0][target_idx]])
        
        n_mechanisms = As_.shape[0]
        target_curves = target_curve[None].repeat(n_mechanisms, axis=0)
        
        with jax.default_device(self.device):
            outputs, distance_grads = self.distance_function(
                x0s_,
                As_,
                node_types_,
                thetas_,
                target_idx_,
                target_curves
            )
            distances = np.array(outputs[1])
            distance_grads = np.array(distance_grads)
            distances[~valid] = np.inf

            distance_grads[~valid] = 0.0

            distance_grads = [distance_grads[i][mappings[i]] for i in range(len(distance_grads))]
            # Drop the crank rows: callers only ever see gradients for the free positions they gave us.
            distance_grads = [dg[2:] for dg in distance_grads]

            for i in range(len(distance_grads)):
                if is_single:
                    continue
                else:
                    if distance_grads[i].shape != x0s[i].shape:
                        g = np.zeros_like(x0s[i])
                        if distance_grads[i].shape[0] < g.shape[0]:
                            g[:distance_grads[i].shape[0], :distance_grads[i].shape[1]] = distance_grads[i]
                        else:
                            g = distance_grads[i][:g.shape[0], :g.shape[1]]
                        distance_grads[i] = g

        invalids = np.array([np.isnan(distance_grads[i]).any() for i in range(len(distance_grads))])
        distances[invalids] = np.inf

        distance_grads = [np.nan_to_num(grad, nan=0.0) for grad in distance_grads]

        if is_single:
            distances = distances[0]
            distance_grads = distance_grads[0]

        return distances, distance_grads