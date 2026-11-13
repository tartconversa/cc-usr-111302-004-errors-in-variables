# 已知测量误差下的协方差校正与可识别诊断

本项目保存虚构数据共享及产品运营的合成表格与二阶矩校正工具。读取器只解析文件并显示记录数；校正工具在显式误差假设下做统计推断。所有地区、机构、产品与人数均为合成。

## 字段与单位

observations每条记录含同维z向量及已知误差矩阵E；moments中的矩阵参考专供代数核对，不是夹具样本的估计答案。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 运行已有读取器

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.moment_reader fixtures/observations.json
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 二阶矩校正

    python -m src.errors_in_variables fixtures/observations.json fixtures/moments.json
    python -m src.errors_in_variables fixtures/observations.json fixtures/moments.json --output .workspace-output/report.json

主情形假设每条合成记录满足 Z_i = X_i + e_i，误差均值 0、与潜在 X_i 独立，记录之间误差独立；E_i 已知且可异方差。用 n-1 分母的观测协方差 Sz 减去各记录误差协方差的算术平均：

    Sx = Sz - mean_i(E_i)

不是减去 E_i/n，也不能只减对角元。缺维度（显式 null）的记录整条退出主估计并记入剔除清单；不用不同成对样本拼凑协方差。每条 E_i 先核验对称半正定，不满足者同样整条退出。

校正后的 Sx 若不为半正定（或对角非正），原始矩阵照常保留并标记为不可靠，不输出可解释的相关、偏相关或投影斜率，也不加 ridge；最近半正定投影只作对照字段出现，不覆盖原始估计，也不表示由数据识别。偏相关只对配置指定的 ``given`` 维度条件化，所需子矩阵秩不足时返回 ``unidentified``。

bootstrap 按整记录重抽（每次同时抽 Z_i 与其 E_i），校正失败次数单列，任何区间都是成功子集的条件分位数，不代表总体稳定。输出中的相关是关联结构，不是因果结论；该校正完全依赖独立误差与已知误差协方差假设。

## 固定容器

    docker build -t errors-in-variables:local .
    docker run --rm errors-in-variables:local

容器镜像以完整digest固定，默认仅检查输入契约；实现新增分析功能所需依赖由开发者明确登记。新增校正工具只用标准库，未登记新依赖。

