"""小规模对称矩阵的纯标准库线性代数工具。

本模块只服务于二阶矩校正：对测量指标这种低维稠密矩阵做
Jacobi 对称特征分解、半正定/秩诊断、稳定求逆和最近半正定投影。
不引入第三方依赖，也不做大规模数值优化。

约定
----
* 矩阵用 ``list[list[float]]`` 表示，行优先、方阵。
* 特征值升序返回；``vectors[i][k]`` 是第 ``k`` 个特征向量的第 ``i`` 个分量，
  即特征向量按列存放。
* 半正定与秩的阈值都相对于矩阵的最大绝对特征值，因而对单位缩放保持不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class MatrixError(ValueError):
    """矩阵输入不合法或无法完成所需运算。"""


class NotPositiveDefinite(MatrixError):
    """矩阵奇异或非正定，不能唯一求逆。"""

    def __init__(self, size: int, rank: int, min_eigenvalue: float, threshold: float):
        self.size = size
        self.rank = rank
        self.min_eigenvalue = min_eigenvalue
        self.threshold = threshold
        super().__init__(
            f"矩阵不是严格正定：秩 {rank}/{size}，最小特征值 {min_eigenvalue:.6g}，"
            f"判定阈值 {threshold:.6g}；拒绝加 ridge 后继续给出唯一解"
        )


@dataclass(frozen=True)
class MatrixDiagnostics:
    """对称矩阵的诊断结果。"""

    size: int
    symmetric: bool
    eigenvalues: tuple[float, ...]
    min_eigenvalue: float
    max_abs_eigenvalue: float
    psd: bool
    full_rank: bool
    rank: int
    nonpositive_diagonal: tuple[int, ...]
    tolerance: float
    threshold: float

    @property
    def positive_diagonal(self) -> bool:
        return not self.nonpositive_diagonal


def _validate_square(matrix) -> int:
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise MatrixError("需要非空方阵")
    n = len(matrix)
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise MatrixError("需要方阵且每行长度一致")
    return n


def symmetrize(matrix) -> list[list[float]]:
    n = _validate_square(matrix)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = (float(matrix[i][j]) + float(matrix[j][i])) / 2.0
            result[i][j] = value
            result[j][i] = value
    return result


def max_abs_entry(matrix) -> float:
    n = _validate_square(matrix)
    return max(abs(float(matrix[i][j])) for i in range(n) for j in range(n))


def jacobi_eigh(matrix, tolerance: float = 1e-14, max_sweeps: int = 100):
    """对称矩阵的 Jacobi 特征分解，返回 ``(升序特征值, 特征向量矩阵)``。

    输入若只有微小不对称，会先做对称平均；调用方若要求严格对称，
    应先使用 :func:`diagnose_symmetric` 检查 ``symmetric`` 字段。
    """

    n = _validate_square(matrix)
    a = symmetrize(matrix)
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if n == 1:
        return [a[0][0]], vectors

    scale = max(abs(a[i][j]) for i in range(n) for j in range(n))
    if scale == 0.0:
        return [0.0] * n, vectors
    threshold = tolerance * scale

    for _ in range(max_sweeps):
        off = max(
            (abs(a[i][j]) for i in range(n) for j in range(i + 1, n)),
            default=0.0,
        )
        if off <= threshold:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= threshold:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for k in range(n):
                    if k == p or k == q:
                        continue
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[p][k] = a[k][p]
                    a[k][q] = s * akp + c * akq
                    a[q][k] = a[k][q]

                app = a[p][p]
                aqq = a[q][q]
                a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                a[p][q] = 0.0
                a[q][p] = 0.0

                for k in range(n):
                    vkp = vectors[k][p]
                    vkq = vectors[k][q]
                    vectors[k][p] = c * vkp - s * vkq
                    vectors[k][q] = s * vkp + c * vkq

    values = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda k: values[k])
    ordered_values = [values[k] for k in order]
    ordered_vectors = [[vectors[i][k] for k in order] for i in range(n)]
    return ordered_values, ordered_vectors


def diagnose_symmetric(matrix, tolerance: float = 1e-10) -> MatrixDiagnostics:
    """检查对称性并给出半正定、秩与对角诊断。

    对称性阈值相对于最大绝对元素；半正定/秩阈值相对于最大绝对特征值。
    ``psd`` 允许 ``min_eigenvalue >= -threshold``，即舍入范围内的负特征值；
    ``full_rank`` 要求 ``min_eigenvalue > threshold``。
    """

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise MatrixError("tolerance 必须是有限正数")
    n = _validate_square(matrix)
    entry_scale = max_abs_entry(matrix)
    symmetry_scale = entry_scale if entry_scale > 0.0 else 1.0
    symmetric = all(
        abs(float(matrix[i][j]) - float(matrix[j][i])) <= tolerance * symmetry_scale
        for i in range(n)
        for j in range(i + 1, n)
    )

    symmetric_matrix = symmetrize(matrix)
    values, _ = jacobi_eigh(symmetric_matrix)
    max_abs_eigenvalue = max((abs(v) for v in values), default=0.0)
    threshold = tolerance * max_abs_eigenvalue
    min_eigenvalue = values[0]
    nonpositive = tuple(i for i in range(n) if symmetric_matrix[i][i] <= 0.0)
    rank = sum(1 for value in values if value > threshold)

    return MatrixDiagnostics(
        size=n,
        symmetric=symmetric,
        eigenvalues=tuple(values),
        min_eigenvalue=min_eigenvalue,
        max_abs_eigenvalue=max_abs_eigenvalue,
        psd=min_eigenvalue >= -threshold,
        full_rank=min_eigenvalue > threshold,
        rank=rank,
        nonpositive_diagonal=nonpositive,
        tolerance=tolerance,
        threshold=threshold,
    )


def invert_positive_definite(matrix, tolerance: float = 1e-10) -> list[list[float]]:
    """对称正定矩阵求逆；奇异或非正定时抛出 :class:`NotPositiveDefinite`。

    这里绝不添加 ridge 或使用伪逆：秩不足意味着该逆（以及依赖它的偏相关）
    不能由当前数据唯一确定。
    """

    symmetric_matrix = symmetrize(matrix)
    diagnostics = diagnose_symmetric(symmetric_matrix, tolerance)
    if not diagnostics.full_rank:
        raise NotPositiveDefinite(
            diagnostics.size,
            diagnostics.rank,
            diagnostics.min_eigenvalue,
            diagnostics.threshold,
        )
    values, vectors = jacobi_eigh(symmetric_matrix)
    n = diagnostics.size
    inverse = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            total = math.fsum(
                vectors[i][k] * vectors[j][k] / values[k] for k in range(n)
            )
            inverse[i][j] = total
            inverse[j][i] = total
    return inverse


def nearest_positive_semidefinite(matrix, tolerance: float = 1e-10) -> list[list[float]]:
    """Frobenius 范数下最近的对称半正定矩阵（负特征值截断为 0）。

    仅作为对照/诊断使用，不能把投影后的矩阵当作由数据识别出的校正结果。
    """

    symmetric_matrix = symmetrize(matrix)
    values, vectors = jacobi_eigh(symmetric_matrix)
    n = len(values)
    repaired = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            total = math.fsum(
                vectors[i][k] * vectors[j][k] * max(values[k], 0.0)
                for k in range(n)
            )
            repaired[i][j] = total
            repaired[j][i] = total
    return repaired


def frobenius_distance(left, right) -> float:
    n = _validate_square(left)
    if _validate_square(right) != n:
        raise MatrixError("矩阵维度不一致")
    return math.sqrt(
        math.fsum(
            (float(left[i][j]) - float(right[i][j])) ** 2
            for i in range(n)
            for j in range(n)
        )
    )


def submatrix(matrix, indices) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in indices] for i in indices]
