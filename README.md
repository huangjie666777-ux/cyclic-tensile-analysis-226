# cyclic_dic226

无前端的二维散斑图位移/应变测量后端（FastAPI + NumPy + SciPy）。对 8 位灰度 PNG
参考图与变形图逐点进行**整数平移搜索 + 六参数局部仿射亚像素优化**，再由有效位移的
3×3 邻域平面拟合恢复 Green–Lagrange 应变。

## 安装与启动

```bash
.venv/bin/pip install -e .          # 或直接使用已就绪的 .venv
.venv/bin/python -m cyclic_dic226 # 监听 127.0.0.1:8000
# 交互式文档: http://127.0.0.1:8000/docs
```

## 可复现合成示例

```bash
.venv/bin/python examples/generate_sample.py
# 生成 examples/sample/{reference.png,deformed.png,truth.json}
```

curl 测量（JSON）：

```bash
curl -s http://127.0.0.1:8000/analyze \
  -F reference=@examples/sample/reference.png \
  -F deformed=@examples/sample/deformed.png \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 | python -m json.tool
```

下载 ZIP：

```bash
curl -s -o result.zip http://127.0.0.1:8000/download \
  -F reference=@examples/sample/reference.png \
  -F deformed=@examples/sample/deformed.png \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50
unzip -l result.zip
```

## 请求格式（multipart/form-data）

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `reference`, `deformed` | 8 位单通道灰度 PNG | 同尺寸；每边 ≤ 512 px |
| `scale_mm_per_px` | 毫米/像素 | > 0 |
| `roi_x`,`roi_y`,`roi_w`,`roi_h` | 矩形 ROI（像素，左上原点） | 完全落在图内，w,h > 0 |
| `subset_size` | 奇数子区边长（px） | 正奇数 |
| `grid_step` | 网格步距（px） | 正整数；端点自动补点 |
| `search_radius` | 整数搜索半径（px） | 0–16 |
| `max_iterations` | 亚像素最大迭代数 | 1–100 |

网格点至多 400 个，超出或任何图像/参数非法时整请求返回 **422**，不产生部分结果。
请求间不共享状态。

## 坐标系与输出

- 坐标原点为图像左上角，**x 向右、y 向下**；报告的是**参考构型**网格，保留全部请求点。
- 位移 `u_mm`/`v_mm` 为像素位移乘 `scale_mm_per_px`。
- 相关系数 `zncc ∈ [-1,1]`，由零均值归一化 SSD 的最优仿射亮度拟合给出
  （`ZNCC = 1 − ZNSSD/2`），容忍整体亮度偏移与正比例变化。
- 每点先做半径内穷举整数平移，再对 `[u,v,du/dx,du/dy,dv/dx,dv/dy]` 六参数
  局部仿射映射做双三次采样的阻尼 Gauss–Newton 优化；这是**逐点局部配准**，不是整图配准。
- 逐点失效原因：`flat_texture`、`out_of_bounds`、`integer_search_out_of_bounds`、
  `subpixel_out_of_bounds`、`flat_deformed_texture`、`degenerate_jacobian`、
  `solver_failure`、`not_converged`。失效点位移/应变留空，**绝不用零位移补齐**。
- 应变：在每个有效点的 **3×3 网格邻域**内分别对 u、v 做二维线性最小二乘拟合，
  有效点少于 3 个或点共线（秩退化）时该点应变无效（留空，不插补）。由梯度构造
  变形梯度 `F`，输出 `E = (FᵀF − I)/2` 的 `Exx`、`Eyy`、`Exy`；`Exy` 为张量剪切
  （= 工程剪切的一半）。位移单位毫米，应变无量纲。

## ZIP 内容

- `points.csv`：逐点 `grid_x_mm,grid_y_mm,valid,u_mm,v_mm,zncc,converged,
  iterations,exx,eyy,exy,failure_reason`，缺测为空字段。
- `valid_mask.png`：与网格行列对齐的 8 位图，有效点 255、失效点 0。
- `result.json`：回显参数、坐标系、点数/有效点数统计与失败原因计数。

## 适用范围

仅测**面内**小/中等变形的散斑图；假设子区内仿射近似成立、散斑纹理充分且两幅图像
成像条件接近（允许亮度的仿射变化）。不处理离面运动、大旋转、透视畸变、相机标定与
三维重构。边缘点要求整数搜索中子区整体位于画面内。

## 测试

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q cyclic_dic226
```

## 拉伸试验虚拟引伸计（/tensile/*）

在原有逐点 DIC 之上新增两个接口（旧 `/analyze`、`/download` 保持不变）：

- `POST /tensile/analyze`：返回 JSON（逐帧标距/应变/应力、模量拟合、屈服点）。
- `POST /tensile/download`：返回 ZIP（`curve.csv` 全帧曲线、`frames/<id>_points.csv`
  逐帧测量点、`result.json` 参数/标距/拟合/屈服汇总，三者按 frame_id 互相引用）。

### 输入（multipart/form-data）

除原有 DIC 参数外，还需：

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `reference` | 8 位灰度 PNG 参考图 | 同旧接口（16 位 PNG 明确拒绝） |
| `frames` | 2–12 帧变形 PNG 的 ZIP | 文件名（去扩展名）与 CSV 的 `frame_id` 一一对应 |
| `curve` | CSV，列 `frame_id,time_s,force_N` | `frame_id` 唯一；`time_s` 有限且严格递增；`force_N` 非负有限 |
| `area_mm2` | 原始截面积 | 正有限值 |
| `p1_x,p1_y,p2_x,p2_y` | 参考图像素坐标的两个标距端点 | 两点不同且落在网格覆盖内，否则整请求 422 |
| `fit_strain_min,fit_strain_max` | 拟合应变闭区间 | 非负且递增 |

任何非法输入（CSV/ZIP/图像/参数）都**整次拒绝**（422），不产生部分结果。

### 计算约定

- 每帧独立对同一参考图做子区 DIC，**不累积位移**；网格与相关算法与旧接口完全一致。
- 端点位移由所在网格单元四角双线性插值；**四角位移全部有效**才给出该帧标距，
  否则该帧标距无效（曲线保留，应变留空），绝不补缺测。
- 统一毫米坐标与毫米位移：工程应变 = 变形后端点欧氏距离 / 原标距 − 1；
  工程应力 = `force_N / area_mm2`（N/mm² = MPa）。
- 模量：闭区间内全部有效点最小二乘拟合 `σ = Eε + b`，至少 3 个不同应变且 E > 0，
  返回 `E_MPa`、`b_MPa`、`R²`；失败时给出 `reason`，曲线照常交付。
- 屈服：拟合上界之后用 0.2 % 偏置线 `σ = E(ε − 0.002) + b`，仅在**相邻有效且应变
  递增**的帧间寻找「实测 − 偏置线」由正到非正的首次交点并线性插值；不跨缺测、
  不外推，无交点返回 `null` 及原因。

### 示例

```bash
.venv/bin/python examples/generate_tensile_sample.py
# 生成 examples/tensile/{reference.png, frames.zip, curve.csv, truth.json}
# 双线性材料：E=70000 MPa，0.2% 屈服后硬化 8000 MPa

curl -s http://127.0.0.1:8000/tensile/analyze \
  -F reference=@examples/tensile/reference.png \
  -F frames=@examples/tensile/frames.zip \
  -F curve=@examples/tensile/curve.csv \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 \
  -F area_mm2=25 -F p1_x=56 -F p1_y=128 -F p2_x=200 -F p2_y=128 \
  -F fit_strain_min=0.0002 -F fit_strain_max=0.002 | python -m json.tool

curl -s -o tensile_result.zip http://127.0.0.1:8000/tensile/download \
  -F reference=@examples/tensile/reference.png \
  -F frames=@examples/tensile/frames.zip \
  -F curve=@examples/tensile/curve.csv \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 \
  -F area_mm2=25 -F p1_x=56 -F p1_y=128 -F p2_x=200 -F p2_y=128 \

## 循环加载分析（/cyclic/*）

在虚拟引伸计之上新增循环加载分析（旧 `/analyze`、`/download`、`/tensile/*` 保持不变）：

- `POST /cyclic/analyze`：返回 JSON（逐帧应力应变、逐周期峰值应力/永久应变/回线功、卸载模量拟合与 E 保持率）。
- `POST /cyclic/download`：返回 ZIP（`curve.csv` 全帧曲线、`cycles.csv` 逐周期汇总、`frames/<id>_points.csv` 逐帧测量点、`result.json` 测量与参数汇总，按 frame_id 互相引用）。

### 输入（multipart/form-data）

DIC 测量参数与 `area_mm2`、`p1_x,p1_y,p2_x,p2_y` 同 /tensile/*（不需要拟合区间字段）：

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `reference` | 8 位灰度 PNG 参考图 | 同旧接口 |
| `frames` | 6–40 帧变形 PNG 的 ZIP | 文件名（去扩展名）与 CSV 的 `frame_id` 一一对应 |
| `curve` | CSV，列 `frame_id,time_s,force_N,cycle_id,phase` | 见下 |

协议校验（任一违反即整次 422，不产生部分结果）：

- `frame_id` 唯一；`time_s` 有限且严格递增；`force_N` 非负有限；`phase` 为 load/unload。
- 同一 `cycle_id` 的行必须连续；每周期先 load 后 unload，两段各至少 3 帧。
- 加载段力不减、卸载段力不增；每周期首末力为零且峰值为正。

### 计算约定

- 每帧独立对同一参考图测量，复用四角全有效双线性插值与工程应力应变，不重置参考、不累积位移、不补缺测。
- 逐周期输出：峰值应力（MPa）、永久应变（末帧应变 − 首帧应变）、带符号回线功密度——按时间相邻点梯形积分 σdε，MPa·应变 = MJ/m³，不排序、不取绝对值；加载末帧即卸载起点，全程只积分一次，不重复计数。
- 周期内任一帧标距无效（缺测）时，该周期永久应变与回线功为 `null` 并给出 `measure.reason`（`gauge_invalid_frames: ...`），其余周期照常输出。
- 卸载模量：取该周期峰值应力 20 %–80 % 闭区间内全部卸载有效点拟合 `σ = Eε + b`，至少 3 个不同应变且 E > 0，返回 `E_MPa`、`b_MPa`、`R²`；失败给出 `reason`。
- E 保持率：首个拟合成功的周期为基准（`modulus_retention_baseline`），各周期 `modulus_retention = E / E_基准`；无任何成功拟合时基准为 `null` 及原因，不影响其他结果。

### 示例

```bash
.venv/bin/python examples/generate_cyclic_sample.py
# 生成 examples/cyclic/{reference.png, frames.zip, curve.csv, truth.json}
# 两个循环：加载 E=70000 MPa，卸载 80000/78000 MPa，含永久伸长与正回线功

curl -s http://127.0.0.1:8000/cyclic/analyze \
  -F reference=@examples/cyclic/reference.png \
  -F frames=@examples/cyclic/frames.zip \
  -F curve=@examples/cyclic/curve.csv \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 \
  -F area_mm2=25 -F p1_x=56 -F p1_y=128 -F p2_x=200 -F p2_y=128 | python -m json.tool

curl -s -o cyclic_result.zip http://127.0.0.1:8000/cyclic/download \
  -F reference=@examples/cyclic/reference.png \
  -F frames=@examples/cyclic/frames.zip \
  -F curve=@examples/cyclic/curve.csv \
  -F scale_mm_per_px=0.05 -F roi_x=32 -F roi_y=32 -F roi_w=192 -F roi_h=192 \
  -F subset_size=31 -F grid_step=16 -F search_radius=8 -F max_iterations=50 \
  -F area_mm2=25 -F p1_x=56 -F p1_y=128 -F p2_x=200 -F p2_y=128
unzip -l cyclic_result.zip
```
