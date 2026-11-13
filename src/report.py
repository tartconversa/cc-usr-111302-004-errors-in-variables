"""二阶矩校正的诊断汇总与整记录 bootstrap。"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional, Sequence

from .analysis_input import AnalysisConfig, AnalysisInput, load_config, load_records, load_reference
from .covariance_correction import (
    CorrectionError,
    CovarianceResult,
    UNCERTAINTY_DISCLAIMER,
    correlations,
    correct_covariance,
    partial_correlation,
    projection_slope,
    repair_comparison,
)
from .linear_algebra import MatrixDiagnostics, diagnose_symmetric


def _clean_matrix(matrix) -> Optional[list[list[float]]]:
    if matrix is None:
        return None
    return [
        [0.0 if float(value) == 0.0 else float(value) for value in row] for row in matrix
    ]


def _clean_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else (0.0 if float(value) == 0.0 else float(value))


def diagnostics_dict(diagnostics: MatrixDiagnostics) -> dict:
    return {
        "symmetric": diagnostics.symmetric,
        "positive_semidefinite": diagnostics.psd,
        "full_rank": diagnostics.full_rank,
        "rank": diagnostics.rank,
        "min_eigenvalue": float(diagnostics.min_eigenvalue),
        "max_abs_eigenvalue": float(diagnostics.max_abs_eigenvalue),
        "nonpositive_diagonal_indices": list(diagnostics.nonpositive_diagonal),
        "tolerance": diagnostics.tolerance,
        "threshold": diagnostics.threshold,
    }


def _index_map(dimensions: Sequence[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(dimensions)}


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    """线性插值分位数；空列表返回 ``None``。"""

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability 必须在 [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _metric_record(samples: Sequence[float], failures: dict[str, int], attempts: int) -> dict:
    interval = None
    if samples:
        interval = [_percentile(samples, 0.025), _percentile(samples, 0.975)]
    return {
        "attempts": attempts,
        "successful": len(samples),
        "failed": attempts - len(samples),
        "failure_reasons": dict(sorted(failures.items())),
        "interval": interval,
        "interval_scope": "仅成功重复的条件分位数" if samples else None,
        "successful_only": len(samples) != attempts,
    }


def _increment(reasons: dict[str, int], reason: str) -> None:
    reasons[reason] = reasons.get(reason, 0) + 1


def _pair_name(dimensions: Sequence[str], i: int, j: int) -> str:
    return f"correlation:{dimensions[i]},{dimensions[j]}"


def _partial_name(pair: Sequence[str], given: Sequence[str]) -> str:
    return f"partial:{pair[0]},{pair[1]}|{','.join(given) if given else '(none)'}"


def _main_metrics(input_data: AnalysisInput, config: AnalysisConfig, correction: CovarianceResult):
    index = _index_map(input_data.dimensions)
    pairs = [
        (i, j) for i in range(input_data.dimension) for j in range(i + 1, input_data.dimension)
    ]
    correlation_results = {}
    for i, j in pairs:
        name = _pair_name(input_data.dimensions, i, j)
        correlation_results[name] = correlations(
            correction.potential_diagnostics, correction.potential_covariance
        )[i][j]

    partial_results = {}
    for pair, given in config.partial_correlations:
        name = _partial_name(pair, given)
        pair_indices = (index[pair[0]], index[pair[1]])
        given_indices = tuple(index[item] for item in given)
        partial_results[name] = partial_correlation(
            correction.potential_diagnostics,
            correction.potential_covariance,
            pair_indices,
            given_indices,
        )

    response = index[config.projection_response]
    predictor_indices = tuple(index[name] for name in config.projection_predictors)
    projection_status, slopes, projection_reason = projection_slope(
        correction.potential_diagnostics,
        correction.potential_covariance,
        response,
        predictor_indices,
    )
    projection = {
        "status": projection_status,
        "response": config.projection_response,
        "predictors": list(config.projection_predictors),
        "slopes": slopes,
        "reason": projection_reason,
        "meaning": "线性投影斜率（Cov(Y, Xp) Cov(Xp)^-1），不是因果效应",
    }
    return correlation_results, partial_results, projection


def bootstrap(input_data: AnalysisInput, config: AnalysisConfig, correction: CovarianceResult) -> dict:
    """整记录 bootstrap：每次同时抽 ``Z_i`` 与其 ``E_i``，失败次数单列。"""

    repeats = config.bootstrap_repeats
    result: dict = {
        "repeats": repeats,
        "seed": config.seed,
        "resampling_unit": "整条记录（Z_i 与 E_i 同抽）",
        "denominator": "n-1",
        "correction_failures": 0,
        "correction_failure_reasons": {},
        "metrics": {},
        "note": (
            "校正失败的重复单列且不剔除后重算；任何区间都只是成功子集的条件分位数，"
            "不代表总体稳定性。点估计以全样本校正结果为准。"
        ),
    }
    if repeats == 0:
        result["status"] = "disabled"
        return result

    vectors = input_data.usable_vectors
    errors = input_data.usable_error_covariances
    n = len(vectors)
    if n < 2:
        result["status"] = "impossible"
        result["reason"] = "完整记录少于 2 条，无法计算 n-1 分母的样本协方差"
        return result
    result["status"] = "ok"

    rng = random.Random(config.seed)
    index = _index_map(input_data.dimensions)
    pairs = [(i, j) for i in range(input_data.dimension) for j in range(i + 1, input_data.dimension)]

    correlation_samples = {_pair_name(input_data.dimensions, i, j): [] for i, j in pairs}
    correlation_failures = {name: {} for name in correlation_samples}
    partial_samples = {}
    partial_failures = {}
    for pair, given in config.partial_correlations:
        name = _partial_name(pair, given)
        partial_samples[name] = []
        partial_failures[name] = {}
    projection_samples = {
        f"projection:{config.projection_response}~{predictor}": []
        for predictor in config.projection_predictors
    }
    projection_failures = {name: {} for name in projection_samples}

    response = index[config.projection_response]
    predictor_names = config.projection_predictors
    predictor_indices = tuple(index[item] for item in predictor_names)

    for _ in range(repeats):
        chosen = rng.choices(range(n), k=n)
        sampled_vectors = [vectors[i] for i in chosen]
        sampled_errors = [errors[i] for i in chosen]
        try:
            replicate = correct_covariance(sampled_vectors, sampled_errors, config.psd_tolerance)
        except CorrectionError as exc:
            result["correction_failures"] += 1
            _increment(result["correction_failure_reasons"], str(exc))
            continue
        if replicate.correction_error is not None:
            result["correction_failures"] += 1
            _increment(result["correction_failure_reasons"], "潜在协方差未通过半正定或对角检查")
            continue

        diagnostics = replicate.potential_diagnostics
        matrix = replicate.potential_covariance
        got = correlations(diagnostics, matrix)
        for i, j in pairs:
            name = _pair_name(input_data.dimensions, i, j)
            if got[i][j] is None:
                _increment(correlation_failures[name], "校正后的相关不可靠")
            else:
                correlation_samples[name].append(got[i][j])

        for pair, given in config.partial_correlations:
            name = _partial_name(pair, given)
            outcome = partial_correlation(
                diagnostics,
                matrix,
                (index[pair[0]], index[pair[1]]),
                tuple(index[item] for item in given),
            )
            if outcome.status == "ok" and outcome.value is not None:
                partial_samples[name].append(outcome.value)
            else:
                _increment(partial_failures[name], outcome.reason or outcome.status)

        status, slopes, reason = projection_slope(diagnostics, matrix, response, predictor_indices)
        if status == "ok" and slopes is not None:
            for position, predictor in enumerate(predictor_names):
                projection_samples[f"projection:{config.projection_response}~{predictor}"].append(
                    slopes[position]
                )
        else:
            for predictor in predictor_names:
                _increment(
                    projection_failures[f"projection:{config.projection_response}~{predictor}"],
                    reason or status,
                )

    metrics = {}
    for name, samples in correlation_samples.items():
        metrics[name] = _metric_record(samples, correlation_failures[name], repeats)
    for name, samples in partial_samples.items():
        metrics[name] = _metric_record(samples, partial_failures[name], repeats)
    for name, samples in projection_samples.items():
        metrics[name] = _metric_record(samples, projection_failures[name], repeats)
    result["metrics"] = metrics
    return result


def _reference_check(observed, error, dimensions: Sequence[str], tolerance: float) -> dict:
    """直接使用给定矩阵做代数核对，不把夹具样本协方差当作答案。"""

    potential = [
        [float(observed[i][j]) - float(error[i][j]) for j in range(2)] for i in range(2)
    ]
    diagnostics = diagnose_symmetric(potential, tolerance)
    correlation_matrix = correlations(diagnostics, potential)
    projection_status, slopes, reason = projection_slope(diagnostics, potential, 1, (0,))
    first = dimensions[0] if dimensions else "x1"
    second = dimensions[1] if len(dimensions) > 1 else "x2"
    return {
        "note": (
            "代数核对：直接使用给定的观测协方差与平均误差协方差，"
            "不是夹具样本协方差的估计结果。"
        ),
        "observed_covariance": _clean_matrix(observed),
        "mean_error_covariance": _clean_matrix(error),
        "corrected_covariance": _clean_matrix(potential),
        "correlation": _clean_optional(correlation_matrix[0][1]),
        "projection": {
            "response": second,
            "predictor": first,
            "status": projection_status,
            "slopes": slopes,
            "reason": reason,
            "meaning": f"{second} 对 {first} 的线性投影斜率",
        },
    }


def build_report(observations_path, moments_path) -> dict:
    """读取合成输入、执行校正、生成完整的诊断报告。

    输出只包含汇总矩与诊断，不读取也不输出任何原始个人行为记录。
    """

    observations = Path(observations_path)
    moments = Path(moments_path)

    raw_payload = json.loads(observations.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise CorrectionError("observations 顶层必须是对象")
    hint_dimensions = raw_payload.get("dimensions")
    if not isinstance(hint_dimensions, list):
        raise CorrectionError("observations.dimensions 必须是数组")
    records_total = len(raw_payload.get("records", []))

    config = load_config(moments, hint_dimensions)
    input_data = load_records(observations, config.psd_tolerance)
    if input_data.count < 2:
        raise CorrectionError("主估计至少需要 2 条完整记录（n-1 分母）")

    correction = correct_covariance(
        input_data.usable_vectors,
        input_data.usable_error_covariances,
        config.psd_tolerance,
    )
    _, partial_results, projection = _main_metrics(
        input_data, config, correction
    )
    reliable = correction.reliable

    report: dict = {
        "tool": "errors-in-variables second-moment correction",
        "inputs": {
            "observations": str(observations),
            "moments": str(moments),
            "dimensions": list(input_data.dimensions),
            "records_total": records_total,
            "records_used": input_data.count,
            "exclusions": list(input_data.exclusions),
            "records_used_note": "只用完整记录；未用不同成对样本拼凑协方差",
        },
        "error_covariance_checks": {
            "rule": "每条 E_i 必须对称半正定；不满足者整条退出主估计",
            "all_symmetric": all(check.symmetric for check in input_data.error_checks),
            "all_psd": all(check.psd for check in input_data.error_checks),
            "records": [
                {
                    "key": check.key,
                    "symmetric": check.symmetric,
                    "positive_semidefinite": check.psd,
                    "rank": check.rank,
                    "min_eigenvalue": _clean_optional(check.min_eigenvalue),
                    "max_abs_eigenvalue": _clean_optional(check.max_abs_eigenvalue),
                }
                for check in input_data.error_checks
            ],
        },
        "correction": {
            "formula": "Sx = Sz - mean_i(E_i)",
            "denominator": "n-1",
            "sample_size": correction.sample_size,
            "observed_covariance": _clean_matrix(correction.observed_covariance),
            "mean_error_covariance": _clean_matrix(correction.mean_error_covariance),
            "potential_covariance": _clean_matrix(correction.potential_covariance),
            "status": "ok" if reliable else "unreliable",
            "reasons": [] if reliable else [str(correction.correction_error)],
            "diagnostics": {
                "observed": diagnostics_dict(correction.observed_diagnostics),
                "potential": diagnostics_dict(correction.potential_diagnostics),
            },
            "matrix_retained_on_failure": not reliable,
            "ridge_added": False,
        },
        "correlations": {
            "status": "ok" if reliable else "unreliable",
            "dimensions": list(input_data.dimensions),
            "matrix": _clean_matrix(
                correlations(correction.potential_diagnostics, correction.potential_covariance)
            )
            if reliable
            else None,
            "note": "对称；不可靠时为 null，不给出可解释的相关系数",
        },
        "partial_correlations": [
            {
                "pair": list(pair),
                "given": list(given),
                "status": outcome.status,
                "value": _clean_optional(outcome.value),
                "reason": outcome.reason,
            }
            for (pair, given), outcome in zip(
                config.partial_correlations, partial_results.values()
            )
        ],
        "projection": projection,
        "reference_check": None,
        "repair_comparison": None,
        "bootstrap": bootstrap(input_data, config, correction),
        "assumptions": [
            "每条记录满足 Z_i = X_i + e_i，e_i 均值为 0",
            "测量误差 e_i 与潜在 X_i 独立",
            "不同记录之间的误差独立，误差协方差已知",
            "误差贡献等于各记录误差协方差的算术平均，因此 Sx = Sz - mean_i(E_i)，不是减去 E_i/n",
            "不能只减对角元，否则会忽略误差之间的已知相关",
            "缺失维度按整条记录剔除；协方差元素来自同一批完整记录",
        ],
        "disclaimer": UNCERTAINTY_DISCLAIMER,
    }

    reference = load_reference(moments)
    if reference is not None:
        observed, error = reference
        report["reference_check"] = _reference_check(
            observed, error, input_data.dimensions, config.psd_tolerance
        )

    if config.repair_comparison and (
        not correction.potential_diagnostics.psd
        or not correction.potential_diagnostics.positive_diagonal
    ):
        comparison = repair_comparison(
            correction.potential_covariance, config.psd_tolerance
        )
        report["repair_comparison"] = {
            "matrix": _clean_matrix(comparison.matrix),
            "min_eigenvalue": _clean_optional(comparison.min_eigenvalue),
            "frobenius_distance": _clean_optional(comparison.frobenius_distance),
            "note": comparison.note,
            "overrides_primary_estimate": False,
        }

    return report
