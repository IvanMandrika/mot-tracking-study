import numpy as np
import scipy.linalg

CHI2_95_4DOF = 9.4877
STD_WEIGHT_POSITION = 1.0 / 20
STD_WEIGHT_VELOCITY = 1.0 / 160


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    return np.stack([box[..., 0] + w / 2, box[..., 1] + h / 2, w / h, h], axis=-1)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    xyah = np.asarray(xyah, dtype=np.float64)
    w = xyah[..., 2] * xyah[..., 3]
    h = xyah[..., 3]
    return np.stack([xyah[..., 0] - w / 2, xyah[..., 1] - h / 2, xyah[..., 0] + w / 2, xyah[..., 1] + h / 2], axis=-1)


class KalmanFilter:
    def __init__(self) -> None:
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = xyah[3]
        mean = np.concatenate([xyah, np.zeros(4)])
        std = np.array([
            2 * STD_WEIGHT_POSITION * h, 2 * STD_WEIGHT_POSITION * h, 1e-2, 2 * STD_WEIGHT_POSITION * h,
            10 * STD_WEIGHT_VELOCITY * h, 10 * STD_WEIGHT_VELOCITY * h, 1e-5, 10 * STD_WEIGHT_VELOCITY * h,
        ])
        return mean, np.diag(std**2)

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array([
            STD_WEIGHT_POSITION * h, STD_WEIGHT_POSITION * h, 1e-2, STD_WEIGHT_POSITION * h,
            STD_WEIGHT_VELOCITY * h, STD_WEIGHT_VELOCITY * h, 1e-5, STD_WEIGHT_VELOCITY * h,
        ])
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(std**2)

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array([STD_WEIGHT_POSITION * h, STD_WEIGHT_POSITION * h, 1e-1, STD_WEIGHT_POSITION * h])
        return self.H @ mean, self.H @ cov @ self.H.T + np.diag(std**2)

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        chol = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        gain = scipy.linalg.cho_solve(chol, (cov @ self.H.T).T, check_finite=False).T
        new_mean = mean + gain @ (xyah - proj_mean)
        new_cov = cov - gain @ proj_cov @ gain.T
        return new_mean, new_cov

    def gating_distance(self, means: np.ndarray, covs: np.ndarray, xyah: np.ndarray) -> np.ndarray:
        h = means[:, 3]
        std = np.stack([STD_WEIGHT_POSITION * h, STD_WEIGHT_POSITION * h, np.full_like(h, 1e-1),
                        STD_WEIGHT_POSITION * h], axis=1)
        proj_cov = covs[:, :4, :4] + std[:, :, None] ** 2 * np.eye(4)
        d = np.atleast_2d(xyah)[None, :, :] - means[:, None, :4]
        return np.einsum("nmi,nij,nmj->nm", d, np.linalg.inv(proj_cov), d)
