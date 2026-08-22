#!/usr/bin/env python
"""
生成最终完整对比评估报告（HTML）
整合所有方案的评估结果，生成带图表的完整报告
"""
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(eval_dir: Path):
    """加载所有评估结果"""
    results = {}

    # 速度/接触/滑移结果
    full_path = eval_dir / "full_results.json"
    if full_path.exists():
        with open(full_path) as f:
            results["full"] = json.load(f)

    # 光流法结果
    of_path = eval_dir / "optical_flow_results.json"
    if of_path.exists():
        with open(of_path) as f:
            results["optical_flow"] = json.load(f)

    # RGB 位姿结果
    rgb_path = eval_dir / "pose_rgb_results.json"
    if rgb_path.exists():
        with open(rgb_path) as f:
            results["pose_rgb"] = json.load(f)

    # RGBD 位姿结果
    rgbd_path = eval_dir / "pose_rgbd_results.json"
    if rgbd_path.exists():
        with open(rgbd_path) as f:
            results["pose_rgbd"] = json.load(f)

    return results


def generate_report(results: dict, output_path: Path):
    """生成 HTML 报告"""
    # 速度估计数据
    vel_methods = []
    if "full" in results:
        for name, data in results["full"].items():
            if "velocity" in data:
                vel_methods.append({
                    "name": name,
                    "rmse": data["velocity"]["linear_velocity_rmse_m_s"],
                    "mae": data["velocity"]["linear_velocity_mae_m_s"],
                    "category": "位姿序列" if name in ["finite_diff", "savgol", "kalman"] else "其他"
                })
    if "optical_flow" in results:
        for name, data in results["optical_flow"].items():
            label = "稠密光流 (Farneback)" if "farneback" in name else "稀疏光流 (LK)"
            vel_methods.append({
                "name": label,
                "rmse": data["linear_velocity_rmse_m_s"],
                "mae": data["linear_velocity_mae_m_s"],
                "category": "纯视觉光流"
            })
    vel_methods.sort(key=lambda x: x["rmse"])

    # 接触检测数据
    con_methods = []
    if "full" in results:
        for name, data in results["full"].items():
            if "contact" in data:
                con_methods.append({
                    "name": name,
                    "f1": data["contact"]["F1"],
                    "precision": data["contact"]["precision"],
                    "recall": data["contact"]["recall"],
                    "accuracy": data["contact"]["accuracy"],
                })
    con_methods.sort(key=lambda x: -x["f1"])

    # 滑移检测数据
    slip_methods = []
    if "full" in results:
        for name, data in results["full"].items():
            if "slip" in data:
                slip_methods.append({
                    "name": name,
                    "f1": data["slip"]["F1"],
                    "precision": data["slip"]["precision"],
                    "recall": data["slip"]["recall"],
                    "accuracy": data["slip"]["accuracy"],
                })
    slip_methods.sort(key=lambda x: -x["f1"])

    # 位姿估计数据
    pose_methods = []
    if "pose_rgb" in results:
        r = results["pose_rgb"]
        pose_methods.append({
            "name": "RGB (ResNet18)",
            "add_mm": r["add_mean_mm"],
            "proj_px": r["projection_error_mean_px"],
            "acc_20mm": r["add_20mm_accuracy"],
        })
    if "pose_rgbd" in results:
        r = results["pose_rgbd"]
        pose_methods.append({
            "name": "RGBD (ResNet18)",
            "add_mm": r["add_mean_mm"],
            "proj_px": r["projection_error_mean_px"],
            "acc_20mm": r["add_20mm_accuracy"],
        })
    pose_methods.sort(key=lambda x: x["add_mm"])

    # 生成速度表行
    vel_rows = ""
    for i, m in enumerate(vel_methods):
        rank_class = f"rank-{i+1}" if i < 3 else "rank-n"
        vel_rows += f"""        <tr>
          <td><span class="rank-badge {rank_class}">{i+1}</span></td>
          <td>{m['name']}</td>
          <td>{m['rmse']:.4f}</td>
          <td>{m['mae']:.4f}</td>
          <td>{m['category']}</td>
        </tr>\n"""

    # 接触表行
    con_rows = ""
    for i, m in enumerate(con_methods):
        rank_class = f"rank-{i+1}" if i < 3 else "rank-n"
        con_rows += f"""        <tr>
          <td><span class="rank-badge {rank_class}">{i+1}</span></td>
          <td>{m['name']}</td>
          <td>{m['f1']:.3f}</td>
          <td>{m['accuracy']:.3f}</td>
          <td>{m['precision']:.3f}</td>
          <td>{m['recall']:.3f}</td>
        </tr>\n"""

    # 滑移表行
    slip_rows = ""
    for i, m in enumerate(slip_methods):
        rank_class = f"rank-{i+1}" if i < 3 else "rank-n"
        slip_rows += f"""        <tr>
          <td><span class="rank-badge {rank_class}">{i+1}</span></td>
          <td>{m['name']}</td>
          <td>{m['f1']:.3f}</td>
          <td>{m['accuracy']:.3f}</td>
          <td>{m['precision']:.3f}</td>
          <td>{m['recall']:.3f}</td>
        </tr>\n"""

    # 位姿表行
    pose_rows = ""
    for i, m in enumerate(pose_methods):
        rank_class = f"rank-{i+1}" if i < 3 else "rank-n"
        pose_rows += f"""        <tr>
          <td><span class="rank-badge {rank_class}">{i+1}</span></td>
          <td>{m['name']}</td>
          <td>{m['add_mm']:.2f}</td>
          <td>{m['proj_px']:.1f}</td>
          <td>{m['acc_20mm']*100:.1f}%</td>
        </tr>\n""" if pose_methods else ""

    pose_section = ""
    if pose_methods:
        pose_section = f"""
<section>
  <h2>位姿估计方案对比</h2>
  <p>在 MuJoCo 评估集上测试位姿估计模型的 ADD 和投影误差。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>排名</th>
          <th>方法</th>
          <th>ADD 均值 (mm)</th>
          <th>投影误差 (px)</th>
          <th>ADD &lt; 20mm</th>
        </tr>
      </thead>
      <tbody>
{pose_rows}      </tbody>
    </table>
  </div>
  <figure class="chart-figure">
    <figcaption>图 4：位姿估计 ADD 对比（越低越好）</figcaption>
    <div id="chart-pose" style="width:100%;min-height:300px"></div>
  </figure>
</section>
"""

    total_methods = len(vel_methods) + len(con_methods) + len(slip_methods) + len(pose_methods)

    html = f"""<!-- Generated by Trae Work -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>抓取感知方案对比评估报告 - 完整版</title>
<style>
  :root {{
    --bg: #f8fafc;
    --bg2: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --rule: #e2e8f0;
    --accent: #3b82f6;
    --accent2: #10b981;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif;
    background: var(--bg);
    color: var(--ink);
    line-height: 1.7;
    font-size: 15px;
  }}
  .container {{ max-width: 1080px; margin: 0 auto; padding: 3rem 2rem; }}
  header.report-header {{
    text-align: center;
    padding: 3rem 0 2.5rem;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 2.5rem;
  }}
  header.report-header h1 {{
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  header.report-header .subtitle {{ color: var(--muted); font-size: 1rem; }}
  header.report-header .meta {{
    margin-top: 1.25rem;
    display: flex;
    justify-content: center;
    gap: 2rem;
    font-size: 0.85rem;
    color: var(--muted);
    flex-wrap: wrap;
  }}
  header.report-header .meta span strong {{ color: var(--ink); font-weight: 600; }}
  section {{ margin-bottom: 3rem; }}
  h2 {{
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent);
    display: inline-block;
  }}
  h3 {{ font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 0.75rem; }}
  p {{ margin-bottom: 1rem; }}
  mark.key {{ background: none; color: var(--accent); font-weight: 600; }}
  .summary-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
  }}
  .summary-card {{
    background: var(--bg2);
    border: 1px solid var(--rule);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
  }}
  .summary-card .number {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .summary-card .label {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.25rem; }}
  .summary-card .unit {{ font-size: 0.8rem; color: var(--muted); font-weight: 400; }}
  .table-wrap {{
    overflow-x: auto;
    overflow-y: auto;
    max-height: 600px;
    margin: 1rem 0;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; background: var(--bg2); }}
  thead th {{
    background: var(--accent);
    color: white;
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 1;
  }}
  tbody td {{ padding: 0.65rem 1rem; border-bottom: 1px solid var(--rule); }}
  tbody tr:hover {{ background: #f1f5f9; }}
  tbody tr:nth-child(1) td {{ background: #ecfdf5; font-weight: 600; }}
  .rank-badge {{
    display: inline-block;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    text-align: center;
    line-height: 24px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
  }}
  .rank-1 {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); }}
  .rank-2 {{ background: linear-gradient(135deg, #94a3b8, #64748b); }}
  .rank-3 {{ background: linear-gradient(135deg, #d97706, #92400e); }}
  .rank-n {{ background: var(--muted); }}
  .chart-figure {{
    margin: 1.5rem 0;
    background: var(--bg2);
    border: 1px solid var(--rule);
    border-radius: 10px;
    padding: 1.25rem;
  }}
  .chart-figure figcaption {{
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.75rem;
  }}
  footer {{
    margin-top: 4rem;
    padding-top: 2rem;
    border-top: 1px solid var(--rule);
    font-size: 0.85rem;
    color: var(--muted);
  }}
  @media (max-width: 768px) {{
    .container {{ padding: 1.5rem 1rem; }}
    header.report-header h1 {{ font-size: 1.5rem; }}
    .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<div class="container">

<header class="report-header">
  <h1>抓取感知方案对比评估报告</h1>
  <div class="subtitle">基于 MuJoCo 物理仿真真值的多方案基线对比 · 完整版</div>
  <div class="meta">
    <span><strong>方案数:</strong> {total_methods} 种</span>
    <span><strong>评估集:</strong> 18 场景 × 300 帧</span>
    <span><strong>GT 来源:</strong> MuJoCo 物理仿真</span>
    <span><strong>日期:</strong> 2026-08-19</span>
  </div>
</header>

<!-- 执行摘要 -->
<section>
  <h2>执行摘要</h2>
  <p>
    本报告在统一的 MuJoCo 物理仿真评估集上，对 <mark class="key">位姿估计、速度估计、接触检测、滑移检测</mark> 四大感知任务的多种 baseline 方案进行了系统性对比评估。
    所有方案均以 MuJoCo 输出的物理真值为评判标准，确保公平可比。
  </p>
  <div class="summary-cards">
    <div class="summary-card">
      <div class="number">{total_methods}<span class="unit"> 种方案</span></div>
      <div class="label">参与对比的 baseline</div>
    </div>
    <div class="summary-card">
      <div class="number">4<span class="unit"> 大任务</span></div>
      <div class="label">位姿 / 速度 / 接触 / 滑移</div>
    </div>
    <div class="summary-card">
      <div class="number">5,400<span class="unit"> 帧</span></div>
      <div class="label">评估数据量</div>
    </div>
    <div class="summary-card">
      <div class="number">6<span class="unit"> 种物体</span></div>
      <div class="label">YCB 代表性物体</div>
    </div>
  </div>
</section>

<!-- 位姿估计 -->
{pose_section}
<!-- 速度估计 -->
<section>
  <h2>速度估计方案对比</h2>
  <p>对比 {len(vel_methods)} 种速度估计方法，含位姿序列方法和纯视觉光流方法。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>排名</th>
          <th>方法</th>
          <th>RMSE (m/s)</th>
          <th>MAE (m/s)</th>
          <th>类别</th>
        </tr>
      </thead>
      <tbody>
{vel_rows}      </tbody>
    </table>
  </div>
  <figure class="chart-figure">
    <figcaption>图 1：速度估计 RMSE 对比（越低越好）</figcaption>
    <div id="chart-velocity" style="width:100%;min-height:350px"></div>
  </figure>
</section>

<!-- 接触检测 -->
<section>
  <h2>接触检测方案对比</h2>
  <p>对比 {len(con_methods)} 种接触检测方法，含力觉和纯视觉方案。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>排名</th>
          <th>方法</th>
          <th>F1</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody>
{con_rows}      </tbody>
    </table>
  </div>
  <figure class="chart-figure">
    <figcaption>图 2：接触检测 F1 对比（越高越好）</figcaption>
    <div id="chart-contact" style="width:100%;min-height:350px"></div>
  </figure>
</section>

<!-- 滑移检测 -->
<section>
  <h2>滑移检测方案对比</h2>
  <p>对比 {len(slip_methods)} 种滑移检测方法，仅在接触帧上评估。</p>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>排名</th>
          <th>方法</th>
          <th>F1</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
        </tr>
      </thead>
      <tbody>
{slip_rows}      </tbody>
    </table>
  </div>
  <figure class="chart-figure">
    <figcaption>图 3：滑移检测 F1 对比（越高越好）</figcaption>
    <div id="chart-slip" style="width:100%;min-height:350px"></div>
  </figure>
</section>

<footer>
  <p><strong>说明：</strong>所有方案在同一 MuJoCo 物理仿真评估集上测试，以物理引擎输出的 GT 为评判标准。</p>
  <p style="margin-top: 0.5rem;">评估代码：<code>scripts/full_evaluation.py</code> · <code>scripts/eval_pose_on_mujoco.py</code></p>
</footer>

</div>

<script src="./_shared/js/echarts.min.js"></script>
<script src="assets/charts.js"></script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已生成: {output_path}")

    # 同时生成 charts.js
    generate_charts_js(results, output_path.parent / "assets" / "charts.js")


def generate_charts_js(results: dict, output_path: Path):
    """生成 ECharts 配置"""
    # 速度数据
    vel_data = []
    vel_labels = []
    if "full" in results:
        for name, data in results["full"].items():
            if "velocity" in data:
                vel_labels.append(name)
                vel_data.append(data["velocity"]["linear_velocity_rmse_m_s"])
    if "optical_flow" in results:
        for name, data in results["optical_flow"].items():
            label = "稠密光流" if "farneback" in name else "稀疏光流 LK"
            vel_labels.append(label)
            vel_data.append(data["linear_velocity_rmse_m_s"])
    # 按 RMSE 升序
    sorted_pairs = sorted(zip(vel_labels, vel_data), key=lambda x: x[1])
    vel_labels = [p[0] for p in sorted_pairs]
    vel_data = [p[1] for p in sorted_pairs]

    # 接触数据
    con_data = {"f1": [], "prec": [], "rec": []}
    con_labels = []
    if "full" in results:
        for name, data in results["full"].items():
            if "contact" in data:
                con_labels.append(name)
                con_data["f1"].append(data["contact"]["F1"])
                con_data["prec"].append(data["contact"]["precision"])
                con_data["rec"].append(data["contact"]["recall"])

    # 滑移数据
    slip_data = {"f1": [], "prec": [], "rec": []}
    slip_labels = []
    if "full" in results:
        for name, data in results["full"].items():
            if "slip" in data:
                slip_labels.append(name)
                slip_data["f1"].append(data["slip"]["F1"])
                slip_data["prec"].append(data["slip"]["precision"])
                slip_data["rec"].append(data["slip"]["recall"])

    # 位姿数据
    pose_labels = []
    pose_data = []
    if "pose_rgb" in results:
        pose_labels.append("RGB")
        pose_data.append(results["pose_rgb"]["add_mean_mm"])
    if "pose_rgbd" in results:
        pose_labels.append("RGBD")
        pose_data.append(results["pose_rgbd"]["add_mean_mm"])

    js = f"""(function() {{
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  // ---- 速度估计 ----
  var velChart = echarts.init(document.getElementById('chart-velocity'), null, {{ renderer: 'svg' }});
  velChart.setOption({{
    animation: false,
    tooltip: {{ trigger: 'axis', appendToBody: true, axisPointer: {{ type: 'shadow' }} }},
    grid: {{ left: 120, right: 30, top: 20, bottom: 40 }},
    xAxis: {{ type: 'value', name: 'RMSE (m/s)', axisLabel: {{ color: muted }}, splitLine: {{ lineStyle: {{ color: rule }} }} }},
    yAxis: {{ type: 'category', data: {json.dumps(vel_labels)}, axisLabel: {{ color: ink, fontSize: 12 }}, axisLine: {{ show: false }}, axisTick: {{ show: false }} }},
    series: [{{
      type: 'bar',
      data: {json.dumps(vel_data)},
      barWidth: 22,
      itemStyle: {{
        color: function(params) {{
          if (params.dataIndex === 0) return accent;
          if (params.dataIndex === 1) return accent2;
          return muted;
        }},
        borderRadius: [0, 4, 4, 0]
      }},
      label: {{ show: true, position: 'right', formatter: '{{c}} m/s', color: ink, fontWeight: 600, fontSize: 11 }}
    }}]
  }});
  window.addEventListener('resize', function() {{ velChart.resize(); }});

  // ---- 接触检测 ----
  var conChart = echarts.init(document.getElementById('chart-contact'), null, {{ renderer: 'svg' }});
  conChart.setOption({{
    animation: false,
    tooltip: {{ trigger: 'axis', appendToBody: true, axisPointer: {{ type: 'shadow' }} }},
    legend: {{ data: ['F1', 'Precision', 'Recall'], top: 0, textStyle: {{ color: muted }} }},
    grid: {{ left: 120, right: 30, top: 35, bottom: 30 }},
    xAxis: {{ type: 'value', max: 1, axisLabel: {{ color: muted }}, splitLine: {{ lineStyle: {{ color: rule }} }} }},
    yAxis: {{ type: 'category', data: {json.dumps(con_labels)}, axisLabel: {{ color: ink, fontSize: 12 }}, axisLine: {{ show: false }}, axisTick: {{ show: false }} }},
    series: [
      {{ name: 'F1', type: 'bar', data: {json.dumps(con_data['f1'])}, barWidth: 12, itemStyle: {{ color: accent, borderRadius: [0, 4, 4, 0] }} }},
      {{ name: 'Precision', type: 'bar', data: {json.dumps(con_data['prec'])}, barWidth: 12, itemStyle: {{ color: accent2, borderRadius: [0, 4, 4, 0] }} }},
      {{ name: 'Recall', type: 'bar', data: {json.dumps(con_data['rec'])}, barWidth: 12, itemStyle: {{ color: accent + '99', borderRadius: [0, 4, 4, 0] }} }}
    ]
  }});
  window.addEventListener('resize', function() {{ conChart.resize(); }});

  // ---- 滑移检测 ----
  var slipChart = echarts.init(document.getElementById('chart-slip'), null, {{ renderer: 'svg' }});
  slipChart.setOption({{
    animation: false,
    tooltip: {{ trigger: 'axis', appendToBody: true, axisPointer: {{ type: 'shadow' }} }},
    legend: {{ data: ['F1', 'Precision', 'Recall'], top: 0, textStyle: {{ color: muted }} }},
    grid: {{ left: 120, right: 30, top: 35, bottom: 30 }},
    xAxis: {{ type: 'value', max: 1, axisLabel: {{ color: muted }}, splitLine: {{ lineStyle: {{ color: rule }} }} }},
    yAxis: {{ type: 'category', data: {json.dumps(slip_labels)}, axisLabel: {{ color: ink, fontSize: 12 }}, axisLine: {{ show: false }}, axisTick: {{ show: false }} }},
    series: [
      {{ name: 'F1', type: 'bar', data: {json.dumps(slip_data['f1'])}, barWidth: 12, itemStyle: {{ color: accent, borderRadius: [0, 4, 4, 0] }} }},
      {{ name: 'Precision', type: 'bar', data: {json.dumps(slip_data['prec'])}, barWidth: 12, itemStyle: {{ color: accent2, borderRadius: [0, 4, 4, 0] }} }},
      {{ name: 'Recall', type: 'bar', data: {json.dumps(slip_data['rec'])}, barWidth: 12, itemStyle: {{ color: accent + '99', borderRadius: [0, 4, 4, 0] }} }}
    ]
  }});
  window.addEventListener('resize', function() {{ slipChart.resize(); }});

  // ---- 位姿估计（如有） ----
  var poseEl = document.getElementById('chart-pose');
  if (poseEl) {{
    var poseChart = echarts.init(poseEl, null, {{ renderer: 'svg' }});
    poseChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis', appendToBody: true, axisPointer: {{ type: 'shadow' }} }},
      grid: {{ left: 100, right: 30, top: 20, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'ADD (mm)', axisLabel: {{ color: muted }}, splitLine: {{ lineStyle: {{ color: rule }} }} }},
      yAxis: {{ type: 'category', data: {json.dumps(pose_labels)}, axisLabel: {{ color: ink, fontSize: 12 }}, axisLine: {{ show: false }}, axisTick: {{ show: false }} }},
      series: [{{
        type: 'bar',
        data: {json.dumps(pose_data)},
        barWidth: 28,
        itemStyle: {{ color: accent, borderRadius: [0, 6, 6, 0] }},
        label: {{ show: true, position: 'right', formatter: '{{c}} mm', color: ink, fontWeight: 600 }}
      }}]
    }});
    window.addEventListener('resize', function() {{ poseChart.resize(); }});
  }}
}})();
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"charts.js 已生成: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成最终对比评估报告")
    parser.add_argument("--eval_dir", type=str, default="outputs/evaluation_full")
    parser.add_argument("--output", type=str, default="outputs/final-report/final-report.html")
    args = parser.parse_args()

    results = load_results(Path(args.eval_dir))
    print(f"加载了 {len(results)} 组结果: {list(results.keys())}")

    generate_report(results, Path(args.output))


if __name__ == "__main__":
    main()
