import numpy as np


def transform(position, quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-10:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    t = np.eye(4)
    t[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    t[:3, 3] = position
    return t


def relative(ee, obj):
    a = np.asarray(ee)
    b = np.asarray(obj)
    for t in (a, b):
        if t.shape != (4, 4) or not np.isfinite(t).all():
            raise ValueError("invalid SE3")
        if (
            not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1)
        ):
            raise ValueError("invalid rigid rotation")
    t = np.linalg.solve(a, b)
    return np.r_[t[:3, 3], t[:3, 0], t[:3, 1]]  # translation + column-based rotation6D
