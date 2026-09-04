import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

if not hasattr(pin, "ReferenceFrame"):
    class _ReferenceFrameCompat:
        LOCAL_WORLD_ALIGNED = 0
        LOCAL = 1

    pin.ReferenceFrame = _ReferenceFrameCompat()

def load_jsonl(jsonl_path):
    """
    load jsonl file
    """
    data = []
    with open(jsonl_path, 'r', encoding='UTF-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def quat_to_rotate6d(q: np.ndarray, scalar_first = True) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :2, :].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :2, :].reshape(q.shape[:-1] + (6,))

def rotvec_to_rotate6d(rotvec: np.ndarray) -> np.ndarray:
    rot = R.from_rotvec(rotvec)
    return rot.as_matrix()[..., :2, :].reshape(rotvec.shape[:-1] + (6,))

def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec)
    if rotvec.shape[-1] != 3:
        raise ValueError("Last dimension must be 3 (got %s)" % (rotvec.shape[-1],))
    return R.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(*rotvec.shape[:-1], 3, 3)

def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotate6d_to_mat(v6)).as_euler('xyz')

def rotate6d_to_mat(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., :3]
    a2 = v6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-2)      # shape (..., 3, 3)
    return rot_mats

def mat_to_rotate6d(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.shape[-2:] != (3, 3):
        raise ValueError("Last two dimension must be (3, 3) (got %s)" % (mat.shape[:-2],))
    return mat[..., :2, :].reshape(mat.shape[:-2] + (6,))

def mat_to_rotvec(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.shape[-2:] != (3, 3):
        raise ValueError("Last two dimension must be (3, 3) (got %s)" % (mat.shape[:-2],))
    return R.from_matrix(mat.reshape(-1, 3, 3)).as_rotvec().reshape(*mat.shape[:-2], 3)

def rotate6d_to_quat(v6: np.ndarray, scalar_first = True) -> np.ndarray:
    return R.from_matrix(rotate6d_to_mat(v6)).as_quat(scalar_first = scalar_first)

def rotate6d_to_rotvec(v6: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotate6d_to_mat(v6)).as_rotvec()

class OnlineStats:
    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.min = np.full(shape, np.inf)
        self.max = np.full(shape, -np.inf)

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    def update_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return

        B = X.shape[0]

        batch_mean = X.mean(axis=0)
        batch_min = X.min(axis=0)
        batch_max = X.max(axis=0)

        # batch M2
        diff = X - batch_mean
        batch_M2 = np.sum(diff * diff, axis=0)

        if self.count == 0:
            # first batch
            self.count = B
            self.mean = batch_mean
            self.M2 = batch_M2
            self.min = batch_min
            self.max = batch_max
            return

        # merge two Welford states
        delta = batch_mean - self.mean
        total = self.count + B

        self.mean += delta * B / total
        self.M2 += batch_M2 + delta * delta * self.count * B / total
        self.count = total

        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)

    def finalize(self):
        var = self.M2 / max(self.count - 1, 1)
        return {
            "mean": self.mean,
            "std": np.sqrt(var),
            "min": self.min,
            "max": self.max,
        }
