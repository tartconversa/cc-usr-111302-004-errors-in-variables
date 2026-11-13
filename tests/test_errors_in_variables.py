"""二阶矩校正的端到端与代数核对测试。

所有数据均为虚构合成值，不含真实个人行为。
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.analysis_input import InputError, load_records
from src.covariance_correction import (
    CorrectionError,
    correct_covariance,
    partial_correlation,
    projection_slope,
    repair_comparison,
    sample_covariance,
)
from src.linear_algebra import diagnose_symmetric, invert_positive_definite, NotPositiveDefinite
from src.report import build_report

ROOT = Path(__file__).resolve().parents[1]


def write_inputs(directory: Path, records, repeats=0, repair=True, given=(), projection=None):
    observations = {
        "dimensions": ["x1", "x2"],
        "records": records,
    }
    queries = [{"pair": ["x1", "x2"], "given": list(given)}]
    moments = {
        "reference": {
            "observed_covariance": [[4, 2], [2, 9]],
            "mean_error_covariance": [[1, 0.5], [0.5, 5]],
        },
        "partial_correlation_queries": queries,
        "projection": projection or {"response": "x2", "predictors": ["x1"]},
        "psd_tolerance": 1e-10,
        "bootstrap_repeats": repeats,
        "seed": 7,
        "repair_comparison": repair,
    }
    obs_path = directory / "observations.json"
    mom_path = directory / "moments.json"
    obs_path.write_text(json.dumps(observations), encoding="utf-8")
    mom_path.write_text(json.dumps(moments), encoding="utf-8")
    return obs_path, mom_path


class ReferenceAlgebraTests(unittest.TestCase):
    """题面给出的代数核对：不是夹具样本的估计答案。"""

    OBSERVED = [[4.0, 2.0], [2.0, 9.0]]
    ERROR = [[1.0, 0.5], [0.5, 5.0]]
    EXPECTED = [[3.0, 1.5], [1.5, 4.0]]

    def test_corrected_covariance_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(
                Path(directory),
                # 占位记录：样本协方差不必等于参考值，参考核对独立进行
                [{"key": "r", "z": [0, 0], "E": [[0.1, 0], [0, 0.1]]},
                 {"key": "s", "z": [1, 1], "E": [[0.2, 0], [0, 0.2]]}],
            )
            report = build_report(obs, mom)
        check = report["reference_check"]
        self.assertEqual(check["observed_covariance"], self.OBSERVED)
        self.assertEqual(check["mean_error_covariance"], self.ERROR)
        self.assertEqual(check["corrected_covariance"], self.EXPECTED)

    def test_correlation_is_one_point_five_over_sqrt_twelve(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(
                Path(directory),
                [{"key": "r", "z": [0, 0], "E": [[0.1, 0], [0, 0.1]]},
                 {"key": "s", "z": [1, 1], "E": [[0.2, 0], [0, 0.2]]}],
            )
            report = build_report(obs, mom)
        expected = 1.5 / math.sqrt(3.0 * 4.0)
        self.assertAlmostEqual(report["reference_check"]["correlation"], expected, places=12)
        self.assertAlmostEqual(expected, 0.4330127018922193, places=15)

    def test_projection_slope_is_one_half(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(
                Path(directory),
                [{"key": "r", "z": [0, 0], "E": [[0.1, 0], [0, 0.1]]},
                 {"key": "s", "z": [1, 1], "E": [[0.2, 0], [0, 0.2]]}],
            )
            report = build_report(obs, mom)
        self.assertAlmostEqual(report["reference_check"]["projection"]["slopes"][0], 0.5, places=12)


class CorrectionFormulaTests(unittest.TestCase):
    def test_arithmetic_mean_of_error_covariances_with_correlated_errors(self):
        records = [
            {"key": "a", "z": [0, 0], "E": [[2, 1], [1, 2]]},
            {"key": "b", "z": [2, 4], "E": [[4, 0], [0, 3]]},
        ]
        vectors = [tuple(r["z"]) for r in records]
        errors = [tuple(tuple(row) for row in r["E"]) for r in records]
        result = correct_covariance(vectors, errors)
        # Sz 用 n-1 分母：偏差 (-1,-2),(1,2) -> [[2,4],[4,8]]
        observed = sample_covariance(vectors)
        self.assertEqual(observed, [[2.0, 4.0], [4.0, 8.0]])
        mean_error = [[3.0, 0.5], [0.5, 2.5]]
        self.assertEqual([list(row) for row in result.mean_error_covariance], mean_error)
        self.assertEqual(
            [list(row) for row in result.potential_covariance],
            [[observed[i][j] - mean_error[i][j] for j in range(2)] for i in range(2)],
        )
        # 只减对角会得到 (2-3, 4), (4, 8-2.5) = (-1,4),(4,5.5)，明显不同
        self.assertNotEqual(
            result.potential_covariance[0][1],
            observed[0][1] - 0.0,
        )
        # 除以 n 的错误写法也明显不同
        self.assertNotEqual(
            result.potential_covariance[0][0],
            observed[0][0] - result.mean_error_covariance[0][0] / 2,
        )

    def test_denominator_is_n_minus_one(self):
        vectors = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        covariance = sample_covariance(vectors)
        # 总体分母为 3 时方差是 2/3；n-1 分母是 1
        self.assertAlmostEqual(covariance[0][0], 1.0, places=15)
        self.assertAlmostEqual(covariance[1][1], 0.0, places=15)

    def test_unit_scaling_invariance(self):
        records = [
            {"key": "a", "z": [1, 2], "E": [[0.1, 0.02], [0.02, 0.2]]},
            {"key": "b", "z": [2, 5], "E": [[0.2, 0.01], [0.01, 0.1]]},
            {"key": "c", "z": [4, 3], "E": [[0.1, 0], [0, 0.1]]},
        ]
        base = correct_covariance(
            [tuple(r["z"]) for r in records],
            [tuple(tuple(row) for row in r["E"]) for r in records],
        )
        scale = 10.0
        scaled_records = [
            {"key": r["key"], "z": [scale * v for v in r["z"]],
             "E": [[scale * scale * v for v in row] for row in r["E"]]}
            for r in records
        ]
        scaled = correct_covariance(
            [tuple(r["z"]) for r in scaled_records],
            [tuple(tuple(row) for row in r["E"]) for r in scaled_records],
        )
        corr_base = [[base.potential_covariance[i][j] /
                      math.sqrt(base.potential_covariance[i][i] * base.potential_covariance[j][j])
                      for j in range(2)] for i in range(2)]
        corr_scaled = [[scaled.potential_covariance[i][j] /
                        math.sqrt(scaled.potential_covariance[i][i] * scaled.potential_covariance[j][j])
                        for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(corr_base[i][j], corr_scaled[i][j], places=12)
        self.assertTrue(base.potential_diagnostics.psd and scaled.potential_diagnostics.psd)
        # 尺度不变性不依赖绝对特征值阈值
        self.assertAlmostEqual(
            base.potential_diagnostics.min_eigenvalue * scale * scale,
            scaled.potential_diagnostics.min_eigenvalue,
            places=8,
        )


class UnreliablePotentialTests(unittest.TestCase):
    def test_noise_larger_than_signal_keeps_matrix_and_flags_correlation(self):
        records = [
            {"key": "a", "z": [0, 0], "E": [[10, 0], [0, 10]]},
            {"key": "b", "z": [1, 1], "E": [[10, 0], [0, 10]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(Path(directory), records, repeats=0, repair=True)
            report = build_report(obs, mom)
        correction = report["correction"]
        self.assertEqual(correction["status"], "unreliable")
        self.assertEqual(correction["potential_covariance"], [[-9.5, 0.5], [0.5, -9.5]])
        self.assertEqual(correction["matrix_retained_on_failure"], True)
        self.assertFalse(correction["diagnostics"]["potential"]["positive_semidefinite"])
        self.assertIsNone(report["correlations"]["matrix"])
        self.assertEqual(report["correlations"]["status"], "unreliable")
        self.assertFalse(report["partial_correlations"][0]["status"] == "ok")
        self.assertFalse(report["projection"]["status"] == "ok")
        # 对照投影存在但不覆盖原始估计
        self.assertIsNotNone(report["repair_comparison"])
        self.assertFalse(report["repair_comparison"]["overrides_primary_estimate"])
        self.assertNotEqual(
            report["repair_comparison"]["matrix"],
            correction["potential_covariance"],
        )
        repaired = diagnose_symmetric(report["repair_comparison"]["matrix"])
        self.assertTrue(repaired.psd)

    def test_singular_potential_matrix_refuses_ridge_for_partial(self):
        records = [
            {"key": "a", "z": [0, 0], "E": [[1, 1], [1, 1]]},
            {"key": "b", "z": [2, 2], "E": [[1, 1], [1, 1]]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(Path(directory), records, repeats=0)
            report = build_report(obs, mom)
        potential = report["correction"]["potential_covariance"]
        self.assertEqual(potential, [[1.0, 1.0], [1.0, 1.0]])
        diagnostics = diagnose_symmetric(potential)
        self.assertTrue(diagnostics.psd)
        self.assertEqual(diagnostics.rank, 1)
        self.assertFalse(diagnostics.full_rank)
        # 偏相关必须对条件化子矩阵求逆；秩不足时拒绝给出唯一值
        self.assertEqual(report["partial_correlations"][0]["status"], "unidentified")
        # 投影只对预测变量块求逆，单变量块在这里满秩，斜率可以给出
        self.assertEqual(report["projection"]["status"], "ok")
        self.assertEqual(report["projection"]["slopes"], [1.0])
        self.assertTrue(report["correction"]["ridge_added"] is False)


class InputHandlingTests(unittest.TestCase):
    RECORDS = json.loads((ROOT / "fixtures/observations.json").read_text(encoding="utf-8"))

    def test_missing_dimension_excluded_whole_record(self):
        loaded = load_records(ROOT / "fixtures/observations.json")
        self.assertEqual(loaded.count, 4)
        self.assertEqual(loaded.exclusions[0]["key"], "missing")
        self.assertEqual(loaded.exclusions[0]["reason"], "missing_dimension")
        # 未被剔除的记录全部参与同一批协方差，而不是成对样本拼凑
        self.assertEqual(set(loaded.usable_keys), {"a", "b", "c", "d"})

    def test_malformed_record_raises_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "dimensions": ["x1", "x2"],
                "records": [{"key": "a", "z": [1], "E": [[1, 0], [0, 1]]}],
            }), encoding="utf-8")
            with self.assertRaises(InputError):
                load_records(path)

    def test_asymmetric_error_covariance_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asym.json"
            path.write_text(json.dumps({
                "dimensions": ["x1", "x2"],
                "records": [
                    {"key": "a", "z": [0, 0], "E": [[1, 2], [0, 1]]},
                    {"key": "b", "z": [1, 1], "E": [[1, 0], [0, 1]]},
                    {"key": "c", "z": [2, 0], "E": [[1, 0], [0, 1]]},
                ],
            }), encoding="utf-8")
            loaded = load_records(path)
        self.assertEqual(loaded.usable_keys, ("b", "c"))
        self.assertEqual(loaded.exclusions[0]["reason"], "error_covariance_not_symmetric")

    def test_non_psd_error_covariance_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neg.json"
            path.write_text(json.dumps({
                "dimensions": ["x1", "x2"],
                "records": [
                    {"key": "a", "z": [0, 0], "E": [[1, 2], [2, 1]]},
                    {"key": "b", "z": [1, 1], "E": [[1, 0], [0, 1]]},
                    {"key": "c", "z": [2, 0], "E": [[1, 0], [0, 1]]},
                ],
            }), encoding="utf-8")
            loaded = load_records(path)
        self.assertEqual(loaded.usable_keys, ("b", "c"))
        self.assertEqual(loaded.exclusions[0]["reason"], "error_covariance_not_psd")

    def test_all_records_unusable_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "none.json"
            path.write_text(json.dumps({
                "dimensions": ["x1", "x2"],
                "records": [{"key": "a", "z": [None, 1], "E": [[1, 0], [0, 1]]}],
            }), encoding="utf-8")
            with self.assertRaises(InputError):
                load_records(path)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_reports_failures_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(
                Path(directory),
                json.loads((ROOT / "fixtures/observations.json").read_text(encoding="utf-8"))["records"],
                repeats=40,
            )
            report = build_report(obs, mom)
        boot = report["bootstrap"]
        self.assertEqual(boot["status"], "ok")
        self.assertGreater(boot["correction_failures"], 0)
        self.assertEqual(boot["correction_failures"], sum(boot["correction_failure_reasons"].values()))
        for metric in boot["metrics"].values():
            self.assertEqual(metric["attempts"], 40)
            self.assertEqual(metric["successful"] + metric["failed"], 40)
            self.assertTrue(metric["successful_only"])
            self.assertEqual(metric["interval_scope"], "仅成功重复的条件分位数")

    def test_bootstrap_is_seed_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(
                Path(directory),
                json.loads((ROOT / "fixtures/observations.json").read_text(encoding="utf-8"))["records"],
                repeats=25,
            )
            first = build_report(obs, mom)
            second = build_report(obs, mom)
        self.assertEqual(first["bootstrap"], second["bootstrap"])

    def test_bootstrap_resamples_record_with_its_own_error_matrix(self):
        # 用 spy 检查每次重抽的 (Z_i, E_i) 都来自同一条原始记录
        import src.report as report_module

        observations = json.loads((ROOT / "fixtures/observations.json").read_text(encoding="utf-8"))
        usable = [
            (tuple(r["z"]), tuple(tuple(row) for row in r["E"]))
            for r in observations["records"]
            if all(v is not None for v in r["z"])
        ]
        seen_pairs = []
        original = report_module.correct_covariance

        def spy(vectors, errors, tolerance):
            for vector, error in zip(vectors, errors):
                seen_pairs.append((tuple(vector), tuple(tuple(row) for row in error)))
            return original(vectors, errors, tolerance)

        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(Path(directory), observations["records"], repeats=10)
            report_module.correct_covariance = spy
            try:
                report = build_report(obs, mom)
            finally:
                report_module.correct_covariance = original
        self.assertEqual(report["bootstrap"]["repeats"], 10)
        allowed = set(usable)
        for pair in seen_pairs:
            self.assertIn(pair, allowed, "重抽必须把 Z_i 与它自己的 E_i 一起抽出")

    def test_zero_repeats_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            obs, mom = write_inputs(Path(directory), InputHandlingTests.RECORDS["records"], repeats=0)
            report = build_report(obs, mom)
        self.assertEqual(report["bootstrap"]["status"], "disabled")


class LinearAlgebraTests(unittest.TestCase):
    def test_invert_singular_refuses(self):
        with self.assertRaises(NotPositiveDefinite):
            invert_positive_definite([[1.0, 1.0], [1.0, 1.0]])

    def test_inverse_is_accurate(self):
        matrix = [[3.0, 1.5], [1.5, 4.0]]
        inverse = invert_positive_definite(matrix)
        product = [[sum(matrix[i][k] * inverse[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        self.assertAlmostEqual(product[0][0], 1.0, places=12)
        self.assertAlmostEqual(product[0][1], 0.0, places=12)
        self.assertAlmostEqual(product[1][0], 0.0, places=12)
        self.assertAlmostEqual(product[1][1], 1.0, places=12)

    def test_nearest_psd_does_not_add_negative_eigenvalues(self):
        source = [[-9.5, 0.5], [0.5, -9.5]]
        comparison = repair_comparison(source)
        self.assertTrue(diagnose_symmetric(comparison.matrix).psd)
        self.assertGreater(comparison.frobenius_distance, 0.0)


class FixtureReportTests(unittest.TestCase):
    def test_fixture_report_uses_only_complete_records(self):
        report = build_report(ROOT / "fixtures/observations.json", ROOT / "fixtures/moments.json")
        self.assertEqual(report["inputs"]["records_total"], 5)
        self.assertEqual(report["inputs"]["records_used"], 4)
        self.assertEqual(report["inputs"]["exclusions"][0]["key"], "missing")
        self.assertEqual(report["correction"]["denominator"], "n-1")
        self.assertEqual(report["correction"]["status"], "ok")
        self.assertIn("Sx = Sz - mean_i(E_i)", report["correction"]["formula"])
        self.assertIn("不等于因果", report["disclaimer"])

    def test_fixture_sample_covariance_differs_from_reference(self):
        # 参考核对不是要求少量夹具样本的协方差恰好等于它
        report = build_report(ROOT / "fixtures/observations.json", ROOT / "fixtures/moments.json")
        self.assertNotEqual(
            report["correction"]["observed_covariance"],
            report["reference_check"]["observed_covariance"],
        )


if __name__ == "__main__":
    unittest.main()
