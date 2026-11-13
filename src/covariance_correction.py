"""二阶矩校正：从含已知测量误差的观测协方差还原潜在协方差。

模型（主情形）
--------------
每条合成记录给出观测向量 ``Z_i = X_i + e_i`` 与已知误差协方差 ``E_i``。
假设 ``E[e_i] = 0``、``e_i`` 与潜在 ``X_i`` 独立、不同记录之间独立。
在这些假设下

    Cov(Z) = Cov(X) + E[E_i]

因此用 ``n-1`` 分母的观测样本协方差 ``Sz`` 与 ``mean(E_i)`` 的算术平均做差：

    Sx = Sz - mean_i(E_i)

它就是潜在协方差 ``Cov(X)`` 的无偏估计。注意三点：

* 不是 ``Sz - E_i / n``：误差贡献是各记录误差的**算术平均**，
  ``E_i / n`` 只对应"总误差被均摊"这种不同的记账方式。
* 不能只减对角元：误差之间若有已知相关（``E_i`` 非对角），
  观测协方差中的对应协方差同样被污染，必须整矩阵相减。
* 缺维度的整条记录退出主估计；不用不同成对样本拼凑协方差，
  否则各元素来自不同样本集合，``mean(E_i)`` 也不再对应同一批记录。

``Sx`` 不保证半正定。此时保留原始估计并标记为"不可靠相关"，
不静默加 ridge，也不宣称最近半正定投影"修复了由数据识别出的结果"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from .linear_algebra import (
    MatrixDiagnostics,
    MatrixError,
    NotPositiveDefinite,
    diagnose_symmetric,
    frobenius_distance,
    invert_positive_definite,
    nearest_positive_semidefinite,
    submatrix,
)

UNCERTAINTY_DISCLAIMER = (
    "所有结果仅为误差方差假设下的关联结构估计；相关不等于因果，"
    "投影斜率是线性关联的投影，不代表干预效应。"
)


class CorrectionError(ValueError):
    """校正或派生量无法在当前数据下可靠计算。"""


@dataclass(frozen=True)
class CovarianceResult:
    """校正后的潜在协方差及其可靠性状态。"""

    observed_covariance: tuple[tuple[float, ...], ...]
    mean_error_covariance: tuple[tuple[float, ...], ...]
    potential_covariance: tuple[tuple[float, ...], ...]
    sample_size: int
    observed_diagnostics: MatrixDiagnostics
    potential_diagnostics: MatrixDiagnostics
    correction_error: Optional[Exception] = None

    @property
    def reliable(self) -> bool:
        """潜在矩阵是否半正定且对角为正，可以可靠求相关。"""
        return self.correction_error is None and self.potential_diagnostics.psd and self.potential_diagnostics.positive_diagonal


def matrix_mean(matrices: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    """逐元素算术平均。空输入没有定义，直接报错。"""

    if not matrices:
        raise CorrectionError("没有可用于平均的误差协方差")
    n = len(matrices[0])
    count = len(matrices)
    return [
        [math.fsum(float(matrix[i][j]) for matrix in matrices) / count for j in range(n)]
        for i in range(n)
    ]


def sample_covariance(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """无偏样本协方差（分母 ``n-1``），与总体协方差的记账保持一致。"""

    n = len(vectors)
    if n < 2:
        raise CorrectionError("样本协方差至少需要 2 条完整记录")
    dimension = len(vectors[0])
    means = [math.fsum(float(v[j]) for v in vectors) / n for j in range(dimension)]
    covariance = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i, dimension):
            total = math.fsum(
                (float(v[i]) - means[i]) * (float(v[j]) - means[j]) for v in vectors
            )
            value = total / (n - 1)
            covariance[i][j] = value
            covariance[j][i] = value
    return covariance


def correct_covariance(
    vectors: Sequence[Sequence[float]],
    error_covariances: Sequence[Sequence[Sequence[float]]],
    tolerance: float = 1e-10,
) -> CovarianceResult:
    """计算 ``Sx = Sz - mean(E_i)`` 并诊断结果。"""

    if len(vectors) != len(error_covariances):
        raise CorrectionError("观测向量与误差协方差数量不一致")
    observed = sample_covariance(vectors)
    mean_error = matrix_mean(error_covariances)
    potential = [[observed[i][j] - mean_error[i][j] for j in range(len(observed))] for i in range(len(observed))]

    try:
        potential_diagnostics = diagnose_symmetric(potential, tolerance)
    except MatrixError as exc:  # 正常路径不会发生，保留显式失败
        raise CorrectionError(f"潜在协方差无法诊断：{exc}") from exc
    correction_error = None
    if not potential_diagnostics.psd:
        correction_error = CorrectionError(
            "校正后的潜在协方差不是半正定：最小特征值 "
            f"{potential_diagnostics.min_eigenvalue:.6g} 低于阈值 "
            f"{potential_diagnostics.threshold:.6g}；保留原始矩阵，不宣称可由数据可靠求相关"
        )

    return CovarianceResult(
        observed_covariance=tuple(tuple(row) for row in observed),
        mean_error_covariance=tuple(tuple(row) for row in mean_error),
        potential_covariance=tuple(tuple(row) for row in potential),
        sample_size=len(vectors),
        observed_diagnostics=diagnose_symmetric(observed, tolerance),
        potential_diagnostics=potential_diagnostics,
        correction_error=correction_error,
    )


@dataclass(frozen=True)
class CorrelationResult:
    status: str
    value: Optional[float] = None
    reason: Optional[str] = None


def correlations(matrix_diagnostics: MatrixDiagnostics, matrix) -> list[list[Optional[float]]]:
    """由协方差矩阵求相关；不可靠时整体返回 ``None``，不给可解释系数。"""

    size = matrix_diagnostics.size
    if not (matrix_diagnostics.psd and matrix_diagnostics.positive_diagonal):
        return [[None] * size for _ in range(size)]
    standard = [math.sqrt(float(matrix[i][i])) for i in range(size)]
    result: list[list[Optional[float]]] = [[None] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            value = float(matrix[i][j]) / (standard[i] * standard[j])
            result[i][j] = max(-1.0, min(1.0, value))
    return result


def partial_correlation(
    matrix_diagnostics: MatrixDiagnostics,
    matrix,
    pair: tuple[int, int],
    given: tuple[int, ...],
) -> CorrelationResult:
    """偏相关：只对配置指定的条件维度做条件化，再取精度矩阵的负归一化。

    例如 ``pair=(0, 1), given=(2,)`` 表示相关 ``X1, X2 | X3``，只用
    ``[X1, X2, X3]`` 的 3×3 子矩阵求逆，不隐含条件化其他维度。
    所需子矩阵秩不足时返回 ``unidentified``，绝不加 ridge 后继续算。
    """

    i, j = pair
    if i == j:
        return CorrelationResult(status="invalid", reason="偏相关的两个维度不能相同")
    if i in given or j in given:
        return CorrelationResult(status="invalid", reason="条件维度不能包含查询维度")

    if not matrix_diagnostics.psd:
        return CorrelationResult(status="unreliable", reason="潜在协方差不是半正定，偏相关不可靠")
    if not matrix_diagnostics.positive_diagonal:
        return CorrelationResult(status="unreliable", reason="潜在协方差对角元非正，偏相关不可靠")

    indices = [i, j, *given]
    conditioned = submatrix(matrix, indices)
    conditioned_diagnostics = diagnose_symmetric(conditioned, matrix_diagnostics.tolerance)
    if not conditioned_diagnostics.full_rank:
        return CorrelationResult(
            status="unidentified",
            reason=(
                "偏相关所需的条件化子矩阵秩不足"
                f"（秩 {conditioned_diagnostics.rank}/{len(indices)}，"
                f"最小特征值 {conditioned_diagnostics.min_eigenvalue:.6g}）；"
                "不唯一，拒绝加 ridge 冒充解"
            ),
        )
    try:
        precision = invert_positive_definite(conditioned, matrix_diagnostics.tolerance)
    except NotPositiveDefinite as exc:
        return CorrelationResult(status="unidentified", reason=str(exc))
    denom = math.sqrt(precision[0][0] * precision[1][1])
    if not math.isfinite(denom) or denom == 0.0:
        return CorrelationResult(status="unidentified", reason="精度矩阵对角为零")
    value = -precision[0][1] / denom
    return CorrelationResult(status="ok", value=max(-1.0, min(1.0, value)))


def projection_slope(
    matrix_diagnostics: MatrixDiagnostics,
    matrix,
    response: int,
    predictors: tuple[int, ...],
) -> tuple[str, Optional[list[float]], Optional[str]]:
    """线性投影斜率 ``Cov(Y, Xp) Cov(Xp)^-1``（响应在前）。

    这是线性最小二乘投影斜率，不是因果效应。预测变量协方差秩不足时
    返回不可识别，不加 ridge。
    """

    if not matrix_diagnostics.psd:
        return "unreliable", None, "潜在协方差不是半正定，投影斜率不可靠"
    if not matrix_diagnostics.positive_diagonal:
        return "unreliable", None, "潜在协方差对角元非正，投影斜率不可靠"
    k = len(predictors)
    predictor_index = list(predictors)
    try:
        predictor_block = submatrix(matrix, predictor_index)
        predictor_diag = diagnose_symmetric(predictor_block, matrix_diagnostics.tolerance)
        if not predictor_diag.full_rank:
            return (
                "unidentified",
                None,
                f"预测变量协方差秩不足（秩 {predictor_diag.rank}/{k}），斜率不唯一",
            )
        inverse = invert_positive_definite(predictor_block, matrix_diagnostics.tolerance)
    except NotPositiveDefinite as exc:
        return "unidentified", None, str(exc)
    covariance = [
        float(matrix[response][predictor_index[j]]) for j in range(k)
    ]
    slopes = [
        math.fsum(covariance[t] * inverse[t][j] for t in range(k))
        for j in range(k)
    ]
    return "ok", slopes, None


@dataclass(frozen=True)
class RepairComparison:
    """最近半正定投影，只用于对照，不覆盖原始估计。"""

    matrix: tuple[tuple[float, ...], ...]
    min_eigenvalue: float
    frobenius_distance: float
    note: str = (
        "最近半正定投影仅为对照；它不改变原始估计，也不表示该矩阵可由数据识别。"
    )


def repair_comparison(matrix, tolerance: float = 1e-10) -> RepairComparison:
    repaired = nearest_positive_semidefinite(matrix, tolerance)
    diagnostics = diagnose_symmetric(repaired, tolerance)
    return RepairComparison(
        matrix=tuple(tuple(row) for row in repaired),
        min_eigenvalue=diagnostics.min_eigenvalue,
        frobenius_distance=frobenius_distance(matrix, repaired),
    )
