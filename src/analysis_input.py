"""分析输入的严格解析与校验。

只读取 ``fixtures/observations.json``（虚构的逐条合成测量汇总）与
``fixtures/moments.json``（配置与代数核对参考）。不访问原始个人行为，
也不做任何样本级的外部数据连接。

记录契约
--------
* ``dimensions`` 给出固定维度顺序；所有向量与矩阵按此顺序解释。
* ``z`` 与 ``E`` 必须同时存在且都有 ``len(dimensions)`` 项。
  任一位置为 ``null`` 视为该记录缺维度，整条记录退出主估计。
* 额外字段一律忽略，不参与计算。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .linear_algebra import MatrixError, diagnose_symmetric


class InputError(ValueError):
    """输入结构或取值不符合契约。"""


@dataclass(frozen=True)
class ErrorCovarianceCheck:
    """对一条记录已知误差协方差的校验诊断。"""

    key: str
    symmetric: bool
    psd: bool
    full_rank: bool
    rank: int
    min_eigenvalue: float
    max_abs_eigenvalue: float


@dataclass(frozen=True)
class AnalysisInput:
    dimensions: tuple[str, ...]
    usable_keys: tuple[str, ...]
    usable_vectors: tuple[tuple[float, ...], ...]
    usable_error_covariances: tuple[tuple[tuple[float, ...], ...], ...]
    exclusions: tuple[dict, ...]
    error_checks: tuple[ErrorCovarianceCheck, ...] = field(default=())

    @property
    def dimension(self) -> int:
        return len(self.dimensions)

    @property
    def count(self) -> int:
        return len(self.usable_keys)


def _require_finite_number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} 必须是数字，实际为 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{where} 必须是有限数字，实际为 {value!r}")
    return number


def _record_missing(value) -> bool:
    return value is None


def _parse_vector(value, dimension: int, where: str):
    if not isinstance(value, list):
        raise InputError(f"{where} 必须是数组")
    if len(value) != dimension:
        raise InputError(f"{where} 长度应为 {dimension}，实际为 {len(value)}")
    return tuple(_require_finite_number(item, f"{where}[{i}]") for i, item in enumerate(value))


def _parse_matrix(value, dimension: int, where: str):
    if not isinstance(value, list):
        raise InputError(f"{where} 必须是二维数组")
    if len(value) != dimension:
        raise InputError(f"{where} 行数应为 {dimension}，实际为 {len(value)}")
    rows = []
    for i, row in enumerate(value):
        if not isinstance(row, list) or len(row) != dimension:
            raise InputError(f"{where}[{i}] 必须是长度为 {dimension} 的数组")
        rows.append(
            tuple(_require_finite_number(item, f"{where}[{i}][{j}]") for j, item in enumerate(row))
        )
    return tuple(rows)


def load_records(path, psd_tolerance: float = 1e-10) -> AnalysisInput:
    """读取并校验逐条合成观测，返回可用于主估计的记录与剔除清单。

    ``Error`` 协方差必须对称半正定；不满足者记为不可用并进入剔除清单。
    """

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{source} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise InputError("observations 顶层必须是对象")

    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        raise InputError("dimensions 必须是非空数组")
    if not all(isinstance(name, str) and name for name in dimensions):
        raise InputError("dimensions 每项必须是非空字符串")
    if len(set(dimensions)) != len(dimensions):
        raise InputError("dimensions 不能重复")
    dimension = len(dimensions)

    records = payload.get("records")
    if not isinstance(records, list):
        raise InputError("records 必须是数组")

    usable_keys = []
    usable_vectors = []
    usable_errors = []
    exclusions = []
    checks = []
    seen_keys = set()

    for index, record in enumerate(records):
        where = f"records[{index}]"
        if not isinstance(record, dict):
            raise InputError(f"{where} 必须是对象")
        key = record.get("key", f"#{index}")
        if not isinstance(key, str) or not key:
            raise InputError(f"{where}.key 必须是非空字符串")
        if key in seen_keys:
            raise InputError(f"记录键重复：{key!r}")
        seen_keys.add(key)
        record_where = f"records[{index}]({key})"

        if "z" not in record or "E" not in record:
            raise InputError(f"{record_where} 必须同时包含 z 与 E")
        z_value = record["z"]
        e_value = record["E"]
        if not isinstance(z_value, list):
            raise InputError(f"{record_where}.z 必须是数组")
        if not isinstance(e_value, list):
            raise InputError(f"{record_where}.E 必须是二维数组")
        if len(z_value) != dimension:
            raise InputError(
                f"{record_where}.z 长度应为 {dimension}，实际为 {len(z_value)}；"
                "缺维度请显式写 null，其余维度需保持固定顺序"
            )
        if len(e_value) != dimension:
            raise InputError(f"{record_where}.E 行数应为 {dimension}，实际为 {len(e_value)}")

        if any(_record_missing(item) for item in z_value):
            missing = [dimensions[i] for i, item in enumerate(z_value) if _record_missing(item)]
            exclusions.append(
                {
                    "key": key,
                    "reason": "missing_dimension",
                    "missing_dimensions": missing,
                }
            )
            continue

        vector = _parse_vector(z_value, dimension, f"{record_where}.z")
        matrix = _parse_matrix(e_value, dimension, f"{record_where}.E")
        try:
            diagnostics = diagnose_symmetric(matrix, psd_tolerance)
        except MatrixError as exc:
            raise InputError(f"{record_where}.E 无法诊断：{exc}") from exc

        checks.append(
            ErrorCovarianceCheck(
                key=key,
                symmetric=diagnostics.symmetric,
                psd=diagnostics.psd,
                full_rank=diagnostics.full_rank,
                rank=diagnostics.rank,
                min_eigenvalue=diagnostics.min_eigenvalue,
                max_abs_eigenvalue=diagnostics.max_abs_eigenvalue,
            )
        )
        if not diagnostics.symmetric:
            exclusions.append(
                {
                    "key": key,
                    "reason": "error_covariance_not_symmetric",
                    "detail": "E 与其转置差异超过对称性阈值",
                }
            )
            continue
        if not diagnostics.psd:
            exclusions.append(
                {
                    "key": key,
                    "reason": "error_covariance_not_psd",
                    "detail": {
                        "min_eigenvalue": _clean(diagnostics.min_eigenvalue),
                        "threshold": _clean(diagnostics.threshold),
                    },
                }
            )
            continue

        usable_keys.append(key)
        usable_vectors.append(vector)
        usable_errors.append(matrix)

    if not usable_keys:
        raise InputError("没有可用于主估计的完整记录")

    return AnalysisInput(
        dimensions=tuple(dimensions),
        usable_keys=tuple(usable_keys),
        usable_vectors=tuple(usable_vectors),
        usable_error_covariances=tuple(usable_errors),
        exclusions=tuple(exclusions),
        error_checks=tuple(checks),
    )


def _clean(value: float) -> float:
    """把 -0.0 规整为 0.0，便于稳定输出。"""
    return 0.0 if value == 0.0 else float(value)


@dataclass(frozen=True)
class AnalysisConfig:
    partial_correlations: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    projection_response: str
    projection_predictors: tuple[str, ...]
    psd_tolerance: float
    bootstrap_repeats: int
    seed: int
    repair_comparison: bool


def _parse_index_list(value, dimensions, where: str):
    if not isinstance(value, list):
        raise InputError(f"{where} 必须是数组")
    names = []
    for item in value:
        if not isinstance(item, str) or item not in dimensions:
            raise InputError(f"{where} 含未知维度 {item!r}")
        if item in names:
            raise InputError(f"{where} 含重复维度 {item!r}")
        names.append(item)
    return tuple(names)


def load_config(path, dimensions) -> AnalysisConfig:
    """读取配置：偏相关查询、投影、容差与 bootstrap 参数。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{source} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise InputError("moments 顶层必须是对象")

    queries = payload.get("partial_correlation_queries", [])
    if not isinstance(queries, list):
        raise InputError("partial_correlation_queries 必须是数组")
    parsed_queries = []
    for i, query in enumerate(queries):
        if not isinstance(query, dict):
            raise InputError(f"partial_correlation_queries[{i}] 必须是对象")
        pair = _parse_index_list(query.get("pair"), dimensions, f"partial_correlation_queries[{i}].pair")
        if len(pair) != 2:
            raise InputError(f"partial_correlation_queries[{i}].pair 必须恰有两项")
        given = _parse_index_list(
            query.get("given", []), dimensions, f"partial_correlation_queries[{i}].given"
        )
        parsed_queries.append((pair, given))

    projection = payload.get("projection")
    if not isinstance(projection, dict):
        raise InputError("projection 必须是对象")
    response = projection.get("response")
    if not isinstance(response, str) or response not in dimensions:
        raise InputError("projection.response 必须是已知维度")
    predictors = _parse_index_list(projection.get("predictors"), dimensions, "projection.predictors")
    if not predictors:
        raise InputError("projection.predictors 不能为空")
    if response in predictors:
        raise InputError("projection.response 不能同时是预测变量")

    tolerance = _require_finite_number(
        payload.get("psd_tolerance", 1e-10), "psd_tolerance"
    )
    if tolerance <= 0.0:
        raise InputError("psd_tolerance 必须为正数")

    repeats = payload.get("bootstrap_repeats", 200)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
        raise InputError("bootstrap_repeats 必须是非负整数")

    seed = payload.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError("seed 必须是整数")

    repair = payload.get("repair_comparison", False)
    if not isinstance(repair, bool):
        raise InputError("repair_comparison 必须是布尔值")

    return AnalysisConfig(
        partial_correlations=tuple(parsed_queries),
        projection_response=response,
        projection_predictors=predictors,
        psd_tolerance=tolerance,
        bootstrap_repeats=repeats,
        seed=seed,
        repair_comparison=repair,
    )


def load_reference(path):
    """读取 moments.json 中的代数核对参考（可为空）。"""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    reference = payload.get("reference") if isinstance(payload, dict) else None
    if reference is None:
        return None
    if not isinstance(reference, dict):
        raise InputError("reference 必须是对象")
    observed = _parse_matrix(reference.get("observed_covariance"), 2, "reference.observed_covariance")
    error = _parse_matrix(reference.get("mean_error_covariance"), 2, "reference.mean_error_covariance")
    return observed, error
