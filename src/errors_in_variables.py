"""二阶矩校正命令入口：读取合成汇总测量并输出诊断报告。

用法::

    python -m src.errors_in_variables fixtures/observations.json fixtures/moments.json
    python -m src.errors_in_variables fixtures/observations.json fixtures/moments.json --output report.json

只读取上面两个汇总文件；报告只含矩、诊断与剔除清单，不含原始个人行为。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis_input import InputError
from .covariance_correction import CorrectionError
from .report import build_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="在已知误差协方差假设下校正二阶矩并输出可识别性诊断"
    )
    parser.add_argument("observations", help="逐条合成观测（虚构汇总测量）")
    parser.add_argument("moments", help="配置与代数核对参考")
    parser.add_argument("--output", help="可选：把 JSON 报告写到该路径，而不是标准输出")
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON 缩进空格数（默认 2）"
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.observations, args.moments)
    except (InputError, CorrectionError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, ensure_ascii=False, indent=args.indent, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {target}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
