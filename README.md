# gpumd_toolkit — GPUMD を Python から使いこなすためのクラス群

`mission.md` の要件に沿って、GPUMD (https://gpumd.org) を Python から
「構造の準備 → `run.in` の生成 → 実行 → 解析・可視化 → トラジェクトリ変換」
まで一貫して扱えるようにしたツールキット。

追加依頼 (20260926) により、**GPUMD の `run.in` キーワードを一通り**
Python API にしてある。機能分類ごとにワークフロークラスが 1 つ対応する。

| # | 分類 | クラス |
|---|------|--------|
| 1 | 静的計算・構造計算 | `StaticCalculation` |
| 2 | 通常の MD・アンサンブル | `GPUMDCalculation` (全ワークフローの基底) |
| 3 | 自由エネルギー・熱力学的積分 | `FreeEnergyCalculation` |
| 4 | Monte Carlo・組成自由度 | `MonteCarloCalculation` |
| – | 熱輸送 (GK / HNEMD / NEMD / SHC / モード分解) | `ThermalTransport` |
| 6 | 拡散・イオン伝導・液体物性 | `DiffusionCalculation` |
| 7 | 電子・電気的性質 | `ElectronicCalculation` |
| 8 | PIMD・核量子効果 | `PathIntegralCalculation` |
| 9 | 高圧・衝撃波 | `ShockCalculation` |
| 10 | Two-Temperature Model | `TwoTemperatureCalculation` |
| 11 | 機械的・外場・特殊操作 | `MechanicalCalculation` |
| 12 | NEP の不確かさ・Active Learning | `ActiveLearningCalculation` |

```
03_gpumd_toolkit/
├── pyproject.toml        pip でインストールするための定義 (console script つき)
├── conda/                conda ビルドレシピ (meta.yaml) と environment.yml
├── gpumd_toolkit/
│   ├── config.py         GPUMDEnvironment  … 仮想環境・実行ファイル・GPU の設定
│   ├── fastio.py         read_fast など     … 構造ファイルの高速読み込み / xyz 変換
│   ├── structure.py      StructureHandler  … POSCAR / CIF などの入出力と model.xyz 生成
│   ├── groups.py         GroupingScheme    … grouping method (fix/heat/compute 用) の組み立て
│   ├── profiles.py       TemperatureProfile… 昇温・降温・定温・任意温度プロファイル
│   ├── inputs/           run.in の全キーワード
│   │   ├── builder.py      MDStage / RunInputBuilder … 全体の骨組み
│   │   ├── ensembles.py    すべての ensemble (QTB/NEMD/TTM/PIMD/TI/衝撃波)
│   │   ├── computes.py     すべての compute_*
│   │   ├── dumps.py        すべての dump_* と active
│   │   └── modifiers.py    セットアップ・外場・MC・最小化などその他
│   ├── md.py             GPUMDCalculation  … ★ MD 計算のメインクラス
│   ├── workflows/        機能分類ごとの高レベルクラス (上の表)
│   ├── outputs.py        OutputReader      … GPUMD の全出力ファイルの読み込み
│   ├── postprocess.py    弾性率 / κ / D / η / 自由エネルギー曲線などの後処理
│   ├── trajectory.py     TrajectoryConverter … 拡張 XYZ / XDATCAR などへの変換
│   ├── analysis.py       MDAnalyzer        … thermo.out 解析と matplotlib 出力
│   ├── parallel.py       ParallelRunner    … joblib による並列実行
│   ├── ase_interface.py  ASEMDRunner       … ★ ASE ベースの実行 (CPU/GPU/pyNEP)
│   ├── plotting.py       日本語フォントの設定と英語へのフォールバック
│   ├── cli.py            gpumd-toolkit コマンド
│   ├── validation.py     ValidationReport  … 予実 (期待値 vs 実測値) 管理
│   ├── diagnostics.py    environment_report… 実行環境の確認
│   └── examples/dpmd.py  DPMDExample       … ★ DPMD の example + 予実
├── scripts/              コマンドラインから使うスクリプト
├── tests/                自己テスト (pytest)
└── results/              同梱の実行済み結果 (DPMD example の予実)
```

---

## 1. セットアップ

### 1.1 パッケージとしてインストールする

```bash
source /home/tajimamainpc/.venv/gpumd312/bin/activate

# pip (開発用は -e)
uv pip install -e ".[all]"

# conda パッケージを作る
conda build conda/
conda install --use-local gpumd-toolkit

# conda 環境をまるごと作る
conda env create -f conda/environment.yml
conda activate gpumd-toolkit
pip install -e .
```

インストールすると `gpumd-toolkit` コマンドが使えるようになる (→ 8 章)。

### 1.2 依存だけ入れる場合

```bash
source /home/tajimamainpc/.venv/gpumd312/bin/activate
uv pip install -r requirements.txt

# GitHub からのみ入るもの (任意)
uv pip install setuptools wheel
uv pip install --no-build-isolation "gpyumd @ git+https://github.com/AlexGabourie/gpyumd.git"
uv pip install "pynep @ git+https://github.com/bigd4/PyNEP.git"
```

`gpumd` / `nep` コマンドはログインシェルの `PATH` (ここでは
`/home/tajimamainpc/repos/GPUMD/src`) から解決される。本ツールキットは
`subprocess` で実行するとき **必ず仮想環境を `source` したログインシェル**
(`bash -lc "source .../activate && gpumd"`) を使う。

確認:

```bash
python scripts/check_environment.py
```

```
  OK gpumd       : /home/tajimamainpc/repos/GPUMD/src/gpumd
  OK nep         : /home/tajimamainpc/repos/GPUMD/src/nep
  GPU            : [0]  NVIDIA GeForce RTX 5070 Ti, 16303 MiB
  OK calorine 3.5 / OK pynep 0.0.1 / OK gpyumd
  ASE backends   : cpu OK / gpu OK / pynep OK
```

> **gpyumd について**: mission で挙げられている 3 つ (calorine / gpyumd / pyNEP)
> のうち gpyumd (v0.1) はローダが `pandas.read_csv(delim_whitespace=…)` に依存しており、
> pandas 2.2 以降で削除されたため**この環境では動作しない**。
> したがって入出力は calorine を主に使い、CPU 実行の代替として pyNEP を使う構成にしている。
> (gpyumd 自体はインストール済みなので、pandas を下げれば利用できる。)

---

## 2. MD 計算 — `GPUMDCalculation`

### 2.1 基本

```python
from gpumd_toolkit import GPUMDCalculation

calc = GPUMDCalculation(
    "POSCAR",                       # POSCAR / CIF / xyz / traj / lammps-data ...
    potential="nep89.txt",
    workdir="runs/si_300K",
    time_step=1.0,                  # fs
    min_cell_length=12.0,           # この長さ以上になるよう自動でスーパーセル化
    max_atoms=10_000,               # mission の上限 (超えると例外)
)
calc.set_dump(thermo=100, traj=500) # 任意ステップごとにトラジェクトリ保存
calc.minimize()                     # 任意: MD 前に FIRE で構造最適化
calc.nvt(temperature=300, steps=20_000)

result = calc.run()                 # gpumd を実行
print(result.summary())             # ステージごとの平均・標準偏差
result.plot()                       # runs/si_300K/analysis/*.png
result.to_xdatcar()                 # runs/si_300K/XDATCAR
```

### 2.2 アンサンブル

| メソッド | GPUMD キーワード | 備考 |
|---|---|---|
| `calc.nve(steps=…)` | `ensemble nve` | |
| `calc.nvt(temperature=…, steps=…, thermostat=…)` | `nvt_ber` / `nvt_nhc` (既定) / `nvt_bdp` / `nvt_lan` / `nvt_bao` / `nvt_mttk` | `temperature_end=` で昇温/降温 |
| `calc.npt(temperature=…, pressure=…, steps=…, barostat=…)` | `npt_ber` / `npt_scr` (既定) / `npt_mttk` | 圧力はスカラー / 3 成分 / 6 成分 |
| `calc.nph(pressure=…, steps=…)` | `nph_mttk` | 熱浴なしでセルだけ緩和する |

```python
# 段階的に積み上げる (すべて 1 つの run.in にステージとして並ぶ)
calc.nvt(temperature=300, steps=10_000)                       # 定温
calc.nvt(temperature=300, temperature_end=1200, steps=40_000) # 昇温
calc.npt(temperature=1200, pressure=0.0, steps=20_000)        # NPT
calc.nve(steps=10_000)                                        # NVE
```

### 2.2.1 熱浴・圧浴の時定数

GPUMD の `<T_coup>` / `<p_coup>` / `tperiod` / `pperiod` は
**時間刻みを単位とする無次元量** (τ/Δt) であって時間ではない
(目安: `T_coup ≈ 100`, `p_coup ≈ 1000`, `pperiod ≥ 200`)。
無次元量をそのまま渡すことも、`tau_T` / `tau_p` に **fs** で渡して
自動換算させることもできる。

```python
calc = GPUMDCalculation("POSCAR", NEP, "runs/x", time_step=2.0)

calc.nvt(temperature=300, steps=10_000, T_coup=100)     # tau_T/dt = 100
calc.nvt(temperature=300, steps=10_000, tau_T=200)      # 200 fs / 2 fs = 100 と同じ

calc.npt(temperature=300, steps=10_000,
         tau_T=200,      # 熱浴の時定数 [fs]
         tau_p=4000)     # 圧浴の時定数 [fs] (npt_mttk なら pperiod になる)
```

`tau_*/time_step < 1` は GPUMD が受け付けないので、その場で例外にする。
ステージごとに `time_step=` を変えた場合はそのステージの刻みで換算する。

### 2.2.2 セル形状と変調する軸

GPUMD の `model.xyz` は 9 成分の `lattice` を取るので、**セルの形状に制限は
ない** (三斜晶でもそのまま計算できる)。制限があるのは NPT の圧力指定の方で、

| `cell_mode` | 圧力成分 | セルの動き | セルの条件 |
|---|---|---|---|
| `iso` | 1 (静水圧) | 体積のみ (等方) | 直交セル |
| `ortho` | 3 (xx, yy, zz) | 3 軸独立 | 直交セル |
| `tri` | 6 (xx, yy, zz, yz, xz, xy) | 6 自由度すべて | 任意 (三斜晶可) |

`cell_mode` を省略すると、**構造が三斜晶なら自動的に `tri`** に切り替える
(直交セルなら `pressure` の成分数から決める)。直方晶セルに直したい場合だけ
`GPUMDCalculation(..., orthorhombic=True)` を使う。

変調する軸は `free_axes` / `fixed_axes` で選ぶ。

```python
calc.npt(temperature=300, steps=10_000, free_axes=("z",))     # c 軸だけ動かす
calc.npt(temperature=300, steps=10_000, fixed_axes=("xy",))   # xy 剪断だけ止める
calc.npt(temperature=300, steps=10_000, pressure=(5, 0, 0))   # x に 5 GPa
```

`npt_ber` / `npt_scr` では「固定したい成分の弾性率を 2000 GPa 超にすると
GPUMD がその成分のカップリングを 0 にする」という仕様を使っている
(`gpumd_toolkit.inputs.FROZEN_MODULUS`)。実際に `free_axes=("z",)` で
300 ステップ回すと a, b の標準偏差は 0.00 Å、c だけが 0.019 Å 揺らぐ。

`npt_mttk` は GPUMD 側に方向指定があるのでそれをそのまま使う。

```python
calc.npt(temperature=300, steps=10_000, barostat="npt_mttk",
         mttk_direction="aniso")                       # 3 軸独立
calc.npt(temperature=300, steps=10_000, barostat="npt_mttk",
         mttk_direction="x", pressure=5.0)             # x に 5 GPa、他は固定
calc.npt(temperature=300, steps=10_000, barostat="npt_mttk",
         mttk_direction={"x": 5.0, "y": 0.0, "z": 0.0})  # 成分ごとに圧力
calc.npt(temperature=300, steps=10_000, barostat="npt_mttk",
         pressure=0.0, pressure_end=5.0)               # 圧力ランプ
```

`calc.describe()` は各 NPT ステージがどの軸をどう動かすかを表示する。

```
 1. npt_scr_300K                  npt_scr       10,000 steps
    cell_mode=ortho (xx: 固定, yy: 固定, zz: 0 GPa)
```

セルまわりのユーティリティ:

```python
StructureHandler.cell_thickness(atoms)   # 面間距離 (GPUMD が近接リストに使う量)
StructureHandler.classify_cell(atoms)    # 'cubic' … 'triclinic'
StructureHandler.to_standard_cell(atoms) # 下三角形 (LAMMPS 標準形) へ回転
StructureHandler.reduce_cell(atoms)      # Niggli / Minkowski 簡約
```

`min_cell_length` は **辺の長さではなく面間距離** の下限として扱う。
斜めのセルでは辺が長くても厚みが足りないことがあり、GPUMD 本体も
近接リストの繰り返し数を厚み (`volume / 面積`) から決めているため。
例: fcc Cu の菱面体プリミティブセル (a=2.546 Å, 60°) の厚みは 2.08 Å しかなく、
12 Å を満たすには 5×5×5 ではなく 6×6×6 が必要になる。

### 2.3 任意の温度プロファイル

```python
from gpumd_toolkit import TemperatureProfile

# (a) アニール: 平衡化 → 昇温 → 保持 → 降温
profile = TemperatureProfile.anneal(
    300, 1200, equilibrate_steps=10_000,
    heat_steps=20_000, hold_steps=20_000, cool_steps=40_000)

# (b) 折れ線で任意指定 (時刻 ps, 温度 K)
profile = TemperatureProfile.from_points(
    [(0, 300), (10, 1000), (20, 1000), (40, 300)], time_step_fs=1.0)

print(profile.summary(time_step_fs=1.0))
profile.plot(1.0, filename="profile.png")

calc.apply_profile(profile, ensemble="nvt_nhc")   # 各区間が 1 ステージになる
```

GPUMD の `ensemble nvt_xxx <T1> <T2> <T_coup>` は 1 つの `run` の中で目標温度を
`T1 → T2` へ線形変化させる。任意プロファイルはこれを区分線形に並べて表現する。

### 2.4 出力設定

```python
calc.set_dump(
    thermo=100,                     # thermo.out の間隔
    traj=1000,                      # トラジェクトリの間隔
    traj_file="dump.xyz",
    traj_properties=("velocity", "force", "potential", "unwrapped_position"),
    restart=10_000,                 # restart.xyz
    extra=("compute_msd 10 400", "compute_rdf 100 100 8.0"),
)
```

> この GPUMD では `dump_exyz` / `dump_position` / `dump_force` は **削除済み**で、
> すべて `dump_xyz <interval> <file> [properties]` に統合されている。
> 本ツールキットは新しい書式のみを出力する。

### 2.5 実行前の確認

```python
print(calc.describe())   # ステージ一覧
print(calc.preview())    # 生成される run.in
calc.run(dry_run=True)   # model.xyz と run.in だけ書き出す
```

---

## 2.6 構造の高速読み込み — `gpumd_toolkit.fastio`

`ase.io.read` は汎用だが、原子数が増えると形式によって極端に遅くなる。
そこで読み込みは既定で高速経路を通るようにした
(`StructureHandler.read(..., fast=False)` で従来どおりの ASE 経由に戻せる)。

| 形式 | `ase.io.read` | 本ツールキット | 経路 |
|---|---|---|---|
| CIF | 28.7 s | **1.84 s** / 2 回目以降 **0.002 s** | pymatgen + xyz キャッシュ |
| POSCAR | 13 ms | **3 ms** | numpy だけの専用パーサ |
| extxyz | 4 ms | **2 ms** | numpy だけの専用パーサ |

(Cu 4,000 原子, ase 3.29 / pymatgen 2026.5.4 で実測。
`python scripts/convert_structure.py <file> --benchmark` で再現できる。)

CIF などの遅い形式は、一度読んだ結果を extxyz に変換して使い回す。
キャッシュのキーは元ファイルの絶対パス + mtime + サイズなので、構造を
書き換えれば自動的に作り直される。保存先は
`cache_dir` 引数 → 環境変数 `GPUMD_TOOLKIT_CACHE` → `~/.cache/gpumd_toolkit/xyz`
の順で決まる。

```python
from gpumd_toolkit import cif_to_xyz, poscar_to_xyz, convert_to_xyz, read_fast

cif_to_xyz("big.cif")                 # -> big.xyz (入力の隣に作る)
poscar_to_xyz("POSCAR", "model.xyz")
convert_to_xyz("structure.pdb")

atoms = read_fast("big.xyz")          # 0.002 s
```

コマンドラインからも使える。

```bash
python scripts/convert_structure.py big.cif               # cif2xyz
python scripts/convert_structure.py structures/*.cif --outdir xyz/
python scripts/convert_structure.py big.cif --benchmark   # 予実 (どれだけ速いか)
python scripts/convert_structure.py --clear-cache
```

`GPUMDCalculation` / `ASEMDRunner` も既定でこの経路を使う
(`fast_read=False` / `cache_structure=False` で無効化)。

---

## 2.7 GPUMD の全機能 — `gpumd_toolkit.workflows`

`GPUMDCalculation` は NVE / NVT / NPT / NPH までを扱う。それ以外の GPUMD の機能は
分類ごとのサブクラスにまとめてある。どのクラスも構造の読み込み・スーパーセル化・
`run()`・解析は `GPUMDCalculation` と同じ作法で使える。

```python
from gpumd_toolkit.workflows import (
    StaticCalculation, FreeEnergyCalculation, MonteCarloCalculation,
    ThermalTransport, DiffusionCalculation, ElectronicCalculation,
    PathIntegralCalculation, ShockCalculation, TwoTemperatureCalculation,
    MechanicalCalculation, ActiveLearningCalculation,
)
```

### 2.7.1 静的計算・構造計算 (分類 1)

`minimize` / `compute_cohesive` / `compute_elastic` / `compute_phonon` は
GPUMD が `run.in` を読んだ時点で実行されるので `ensemble` / `run` を必要としない。
`add_action()` がその違いを吸収する。

```python
calc = StaticCalculation("POSCAR", "nep.txt", "runs/static", repeat=(2, 2, 2))
calc.relax(relax_cell=True)          # FIRE でセルごと最適化
calc.elastic(strain=0.01)            # 弾性定数 C_ij
calc.cohesive_curve(0.95, 1.05)      # 状態方程式
calc.single_point()                  # エネルギー・力・応力
calc.run()

print(calc.elastic_matrix())         # 6x6 [GPa]
print(calc.elastic_moduli())         # B = 169.3 GPa, G = 61.1 GPa, E = 163.6 GPa, nu = 0.339
print(calc.energy_per_atom(), calc.forces().shape, calc.stress())
print(calc.equilibrium_from_cohesive())
calc.plot_cohesive()
```

フォノン分散は `kpoints.in` とスーパーセル (`replicate`) が要る:

```python
from gpumd_toolkit.workflows import band_path_kpoints

path = band_path_kpoints([[0,0,0], [0.5,0,0], [0.5,0.5,0], [0,0,0]], n_per_segment=50)
calc.phonon(displacement=0.01, kpoints=path, replicate=(4, 4, 4))
calc.run()
calc.plot_phonon()                   # 虚振動は負の振動数として描かれる
```

### 2.7.2 自由エネルギー・熱力学的積分 (分類 3)

| 経路 | メソッド | 用途 |
|------|----------|------|
| Frenkel-Ladd | `frenkel_ladd()` | 固体の絶対自由エネルギー (参照系 = Einstein 結晶) |
| Uhlenbeck-Ford | `uhlenbeck_ford()` | 液体の絶対自由エネルギー |
| 可逆スケーリング | `reversible_scaling()` | 等圧線に沿った G(T) |
| 断熱スイッチング | `adiabatic_switching()` | 等温線に沿った G(P) |

`run` のステップ数は `2 * (tequil + tswitch)` でなければならない。これは自動計算される。

```python
solid = FreeEnergyCalculation("POSCAR", "nep.txt", "runs/fe_solid", repeat=(3, 3, 3))
solid.equilibrate(temperature=300, steps=20000)
solid.frenkel_ladd(temperature=300, t_equil=5000, t_switch=20000)
solid.run()

print(solid.gibbs_free_energy())
# {'E_Einstein': -0.055, 'E_diff': -4.251, 'F': -4.306, 'T': 300.0, 'G': -4.306, ...}
print(solid.hysteresis())      # 往復のずれ。大きければ t_switch を伸ばす
```

融点は「固相と液相の G(T) が交わる温度」として出す:

```python
from gpumd_toolkit.workflows import melting_point_from_curves

G0 = solid.gibbs_free_energy()["G"]
rs = FreeEnergyCalculation("POSCAR", "nep.txt", "runs/rs")
rs.reversible_scaling(T_min=300, T_max=2000, pressure=0.0, t_switch=50000)
rs.run()
solid_curve = rs.free_energy_curve(T0=300, G0=G0)
# 液相も同様に ti_liquid + ti_rs で出してから
print(melting_point_from_curves(solid_curve, liquid_curve))
```

### 2.7.3 Monte Carlo・組成自由度 (分類 4)

MD の合間に MC 試行を挟む (NEP 専用)。`mc canonical` は組成を変えずに原子を交換、
`sgc` / `vcsgc` は組成そのものを動かす。

```python
mc = MonteCarloCalculation("alloy.xyz", "nep.txt", "runs/sqs", min_cell_length=20.0)
mc.canonical(temperature=1000, temperature_end=300,   # 焼きなましで規則化
             steps=200000, md_steps=100, mc_trials=200)
mc.run()
print(mc.acceptance_ratio())         # 0.2-0.5 程度なら健全
mc.plot_mcmd()

mc2 = MonteCarloCalculation("alloy.xyz", "nep.txt", "runs/vcsgc", min_cell_length=20.0)
mc2.vcsgc(temperature=800, steps=200000, phi={"Cu": -0.5, "Ni": 0.5}, kappa=100)
mc2.run()
print(mc2.concentrations())
```

> GPUMD の MCMD は周期方向のセル厚みが `2.5*(rc+1)` (NEP の radial cutoff が 6 Å なら
> 17.5 Å) より大きいことを要求する。`check_box_size()` が実行前に警告する。

#### GPUMD だけで回す MCMC (原子交換のみ) — `mcmc()`

GPUMD の `mc` は各 MD ステップの位置更新と力計算の間に MC を挟む。`mcmc()` は
`time_step 0` の NVE ステージに `mc canonical 1 <trials_per_call> ...` を付けるので、
座標は一切動かず原子種の交換 MC だけが GPU 上で回る (`MetropolisMC` よりはるかに速い)。

```python
mc = MonteCarloCalculation("alloy.xyz", "nep.txt", "runs/mcmc_gpu", min_cell_length=20.0)
mc.mcmc(temperature=1500, temperature_end=300,       # 1 呼び出しごとに線形に降温
        trials=200000, trials_per_call=2000)         # run 100 (= 100 回の mc 呼び出し)
mc.run()
mc.mcmd()                            # 呼び出しごとの受理率
mc.result.thermo()                   # thermo.out のポテンシャルエネルギー = MCMC のエネルギー推移
```

- 各呼び出しで力計算が 1 回走るので `trials_per_call` は大きめ (1000 以上) が効率的。
- 後続のステージで `time_step` を指定しなければ、既定の時間刻みに自動で戻す。
- 原子変位の MC は無い。変位も必要なら下の `MetropolisMC` を使う。
- CLI: `gpumd-toolkit mc --mode mcmc --steps <総試行数> --mc-trials <1 呼び出しあたり>`

#### 純粋な Metropolis MC (MCMC) — `MetropolisMC`

GPUMD 本体の `mc` には原子変位の MC 試行は無い (原子交換だけなら上の `mcmc()`)。そこで ASE calculator
(NEP の `cpu` / `pynep` / `gpu`、または任意の ASE calculator) でエネルギーを評価し、
Python 側で Metropolis 法を回すクラスを用意している。

| 試行 | 内容 |
|------|------|
| `displace` | ランダムな 1 原子を各成分 ±`max_displacement` Å の一様乱数で動かす |
| `swap` | 元素の異なる 2 原子の位置を入れ替える (組成は保存) |

```python
from gpumd_toolkit import MetropolisMC

mc = MetropolisMC("alloy.xyz", model="nep.txt", backend="cpu",
                  workdir="runs/mcmc", seed=0)
mc.run(steps=20000, temperature=1500, temperature_end=300,   # 焼きなまし
       moves={"displace": 0.5, "swap": 0.5}, max_displacement=0.1,
       log_interval=200, trajectory_interval=1000)            # runs/mcmc/mc.xyz
print(mc.acceptance_ratio("displace"), mc.acceptance_ratio("swap"))
mc.save_log()                        # mc.csv
mc.plot()                            # mc.png (エネルギー・受理率・温度)
best = mc.lowest_energy_structure()  # 訪れた中で最低エネルギーの構造
```

- 1 試行ごとに全エネルギーを 1 回計算するので、数百原子までが目安
  (54 原子の NEP・CPU で 2000 試行 ≈ 4 秒)。
- `gpu` バックエンドは試行ごとに gpumd をファイル経由で起動するため非常に遅い。`cpu` を推奨。
- `adapt_displacement=True` で受理率が `target_acceptance` に近づくよう変位幅を自動調整する
  (詳細釣り合いが崩れるので平衡化の段階だけで使う)。
- `run()` は何度呼んでも続きから回る (平衡化 → 本計算の 2 段に分けられる)。

### 2.7.4 熱輸送

| 手法 | メソッド | 出力 |
|------|----------|------|
| 平衡 MD + Green-Kubo | `green_kubo()` | `hac.out` |
| 均一非平衡 MD (HNEMD) | `hnemd()` | `kappa.out` |
| 多成分 HNEMDEC | `hnemdec()` | `onsager.out` |
| source/sink NEMD | `nemd()` | `compute.out` |
| スペクトル熱流 | `spectral()` | `shc.out` |
| モード分解 (GKMA/HNEMA) | `modal()` | `heatmode.out` / `kappamode.out` |
| フォノン状態密度 | `phonon_dos()` | `dos.out` / `mvac.out` |

```python
# HNEMD (最も効率がよい)
calc = ThermalTransport("POSCAR", "nep.txt", "runs/hnemd", min_cell_length=20.0)
calc.equilibrate(temperature=300, steps=50000)
calc.hnemd(temperature=300, steps=2000000, Fe=1e-5, direction="x")
calc.run()
print(calc.thermal_conductivity())   # {'kappa_x': ..., 'kappa_x_stderr': ...}
calc.plot_kappa()
```

NEMD はグループ分けが要るので専用のコンストラクタを使う:

```python
calc = ThermalTransport.for_nemd("POSCAR", "nep.txt", "runs/nemd",
                                 axis="x", n_blocks=8, repeat=(14, 6, 6))
calc.equilibrate(temperature=300, steps=20000)
calc.nemd(temperature=300, steps=2000000, delta_T=40, method="lan")
calc.run()
print(calc.nemd_result())            # kappa, 熱流, 温度勾配, ΔT
calc.plot_temperature_profile()
```

> GPUMD の `fix` は 1 つの `run` につき 1 グループしか固定できない。
> `nemd_layout()` は両端の固定層を **同じグループ 0** にまとめてこれを回避している。

### 2.7.5 拡散・イオン伝導・液体物性 (分類 6)

```python
calc = DiffusionCalculation("liquid.xyz", "nep.txt", "runs/liquid", min_cell_length=16.0)
calc.group_by_species()                       # 元素別に解析したい場合
calc.equilibrate(temperature=1500, steps=50000, pressure=0.0)
calc.production(temperature=1500, steps=500000,
                msd=True, viscosity=True, rdf=True,
                ic=(0, 1.0),                  # 元素 0、電荷 +1 のイオン伝導度
                orientorder=[4, 6],           # 結晶核生成の判定
                per_species=True)
calc.run()

print(calc.diffusion_coefficient())           # {'D_cm2_per_s': 6e-05, ...}
print(calc.diffusion_per_species())
print(calc.viscosity())                       # せん断 / 縦 / 体積粘性率 [Pa s]
print(calc.ionic_conductivity())              # [mS/cm]
print(calc.coordination_number(r_max=3.3))    # RDF の積分
calc.plot_summary()
```

`compute_rdf` のカットオフはセルに収まる値へ自動で丸められる (警告が出る)。

### 2.7.6 電子・電気的性質 (分類 7)

```python
calc = ElectronicCalculation("water.xyz", "nep.txt", "runs/ir")
calc.infrared(temperature=300, steps=500000, nep_dipole="nep_dipole.txt", interval=10)
calc.run()
calc.plot_spectrum("infrared")                # 双極子自己相関の FT
```

- `raman(...)` … `dump_polarizability` からラマンスペクトル
- `polarization(...)` … Born 有効電荷から `compute_dpdt`
- `lsqt(...)` … 線形スケーリング量子輸送 (現状は炭素の強束縛模型のみ)
- `electric_field(...)` … `add_efield` で一様電場をかける
- `use_ewald()` … 静電相互作用の逆空間項を Ewald にする

### 2.7.7 PIMD・核量子効果 (分類 8)

```python
calc = PathIntegralCalculation("water.xyz", "nep.txt", "runs/pimd")
calc.pimd(num_beads=32, temperature=300, steps=200000)
calc.dump_beads(interval=1000)
calc.run()
print(calc.gyration_radius().head())   # ring polymer の広がり (H は重原子よりずっと大きい)
```

- `pimd(pressure=..., stochastic_cell_rescaling=True)` … NPT 版 (`pimd_scr`)
- `pimd(eco_omega_max=4000)` … 少ないビーズ数で同じ精度を狙う Eco 法
- `rpmd(num_beads=32, thermostatted=True)` … 近似的な量子ダイナミクス
- `qtb(temperature=300, f_max=200)` … 1 レプリカで済む量子熱浴

> QTB では零点エネルギーの分だけ `thermo.out` の運動温度が目標温度より高く出る
> (水 300 K で 1000 K 前後)。これは異常ではない。

### 2.7.8 高圧・衝撃波 (分類 9)

```python
# MSST: 小さいセルで定常衝撃波後方の状態を再現する
calc = ShockCalculation("POSCAR", "nep.txt", "runs/msst", min_cell_length=16.0)
calc.equilibrate(temperature=300, steps=20000)
calc.msst(shock_velocity=8.0, steps=200000, direction="x", qmass=10000, mu=10)
calc.run()
print(calc.hugoniot_state())   # {'temperature': 2898, 'pressure_GPa': 145, ...}
```

- `hugoniostat(pressure=..., ...)` … NPHug で Hugoniot 上の 1 点に収束させる
- `piston(vp=..., bin_size=...)` … 実際に壁を動かして波面を走らせる (x 方向)
- `compression_ramp(...)` … 圧力を連続的に上げる

```python
calc.piston(vp=1.5, steps=200000, kind="piston", bin_size=3.0, dump_interval=500)
calc.run()
print(calc.shock_front_position())                       # 波面位置と反射の判定
print(calc.shock_velocity(output_interval_fs=500.0))     # [km/s]
calc.plot_profiles()
```

### 2.7.9 Two-Temperature Model (分類 10)

```python
calc = TwoTemperatureCalculation("POSCAR", "nep.txt", "runs/ttm", repeat=(4, 4, 14))
calc.limit_displacement(0.05)                 # 高エネルギー衝突の暴走を防ぐ
calc.equilibrate(temperature=300, steps=20000)
calc.ttm(steps=200000,
         Ce=1.0, rho_e=1.0,                   # 電子比熱 [eV/K] と数密度 [1/Å^3]
         kappa_e=0.005, gamma_p=0.01,         # 電子熱伝導率・電子フォノン結合
         grid=(1, 1, 12), T_e_init=2000,
         source=0.05,                         # レーザー吸収 [eV/(ps Å^3)]
         out_interval=200, active={"z": "3:10"})
calc.run()
calc.plot_two_temperature()
```

- `heat_ttm(...)` … 局所 source/sink Langevin 熱浴と電子グリッドを併用
- `electron_stopping("stopping.txt")` … 照射損傷カスケードの電子阻止能
- `write_electron_temperature_file()` / `write_electron_properties_file()` … 入力ファイル生成

### 2.7.10 機械的・外場・特殊操作 (分類 11)

```python
calc = MechanicalCalculation("POSCAR", "nep.txt", "runs/tensile", repeat=(8, 3, 3))
calc.equilibrate(temperature=300, steps=20000, pressure=0.0)
calc.tensile(temperature=300, steps=500000, strain_rate=1e8, axis="x")
calc.run()
print(calc.mechanical_properties())   # ヤング率・最大応力・破断ひずみ
calc.plot_stress_strain()
```

`strain_rate` [1/s] はセル長と時間刻みから `deform` の Å/step に自動換算される。

そのほか:

| 操作 | メソッド |
|------|----------|
| せん断 | `shear(rate=..., component="xy")` |
| 片端固定 + 片端駆動 | `add_grip_groups()` → `pull_grips()` |
| 一定外力 / 電場 | `constant_force()` / `electric_field()` |
| ばね拘束 (steered MD) | `spring(mode="ghost_com", ...)` |
| 成膜・照射 | `deposition(...)` |
| 電子阻止能 | `electron_stopping(...)` |
| PLUMED | `plumed("plumed.dat")` |
| DFT-D3 補正 | `dispersion_correction("pbe")` |
| 運動量のリセット | `reset_momentum(50)` |

### 2.7.11 NEP の不確かさ・Active Learning (分類 12)

committee (複数 NEP のばらつき) を使う場合は `potential` を入れ子で渡す:

```python
calc = ActiveLearningCalculation(
    "POSCAR", [["nep0.txt"], ["nep1.txt"], ["nep2.txt"]], "runs/al")
calc.explore(T_start=300, T_end=2000, steps=500000, threshold=0.05, interval=10)
calc.run(check=False)

print(calc.uncertainty_summary())     # 平均・中央値・95 パーセンタイル・超過割合
calc.plot_uncertainty()
calc.write_candidates()               # DFT に投げる構造をまとめる
```

- `extrapolation(nep_file=..., asi_file=...)` … active set からの外挿グレード gamma
  (1 モデルで済む。`gamma_high` を超えたら MD が止まる)
- `observe(mode="observe")` … `dump_observer` で複数 NEP の予測を並べて記録

---

## 2.8 キーワードを直接使う — `gpumd_toolkit.inputs`

ワークフローに無い組み合わせは、キーワード関数を直接 `run.in` に差し込める。
各関数は引数をその場で検証してから 1 行の文字列を返すので、GPUMD を起動してから
「引数の個数が違う」と怒られることがない。

```python
from gpumd_toolkit import GPUMDCalculation
from gpumd_toolkit.inputs import computes, dumps, modifiers
from gpumd_toolkit.inputs.ensembles import MSST

calc = GPUMDCalculation("POSCAR", "nep.txt", "runs/custom")
calc.add_preamble(modifiers.dftd3("pbe", 12, 6))         # potential の直後
calc.nvt(temperature=300, steps=100000)
calc.add_commands(                                        # 直近のステージの run 直前
    computes.compute_rdf(8.0, 400, 1000),
    computes.compute_msd(5, 500),
    dumps.dump_netcdf(1000, "movie.nc", compression=1),
)
calc.add_ensemble(MSST("x", 15, qmass=10000, mu=10), steps=50000)   # 特殊アンサンブル
print(calc.preview())
```

対応キーワードの一覧は `gpumd-toolkit keywords` で出る。

---

## 2.9 グループ分け — `gpumd_toolkit.groups`

`fix` / `move` / `heat_*` / `ttm` / `compute` / `add_force` / `add_efield` /
`compute_msd all_groups` / `dump_xyz group` / `mc ... group` は
「grouping method 番号 + グループ番号」で原子集合を指定する。
`model.xyz` の `group:I:<n>` 列がその定義にあたる。

```python
from gpumd_toolkit.groups import (
    GroupingScheme, uniform_slabs, by_species, by_region, by_molecule, nemd_layout,
)

scheme = GroupingScheme()
scheme.add(uniform_slabs(atoms, "z", 10), "slabs")   # 空間スラブ
scheme.add(by_species(atoms), "species")             # 元素別
scheme.add(by_molecule(atoms, cutoff=1.4), "mols")   # 分子別 (連結成分)
print(scheme.summary())

calc = GPUMDCalculation("POSCAR", "nep.txt", "runs/x", groupings=scheme)
```

グループを使うキーワードは `model.xyz` に grouping method が 1 つ以上ないと
GPUMD が起動直後に止まる。各ワークフローは `ensure_grouping()` で
自動的に「全原子 1 グループ」を足す。

---

## 2.10 出力の読み込みと後処理 — `outputs` / `postprocess`

GPUMD の出力はほぼ「ヘッダなしの数値テーブル」なので、列の意味を知らないと使えない。
`OutputReader` は正しい列名を付けた `DataFrame` を返す。

```python
from gpumd_toolkit.outputs import OutputReader

reader = OutputReader("runs/liquid")
print(reader.available())
reader.thermo()          # 温度・エネルギー・応力・セル + volume / pressure を派生
reader.msd()             # MSD と SDC (all_groups ならグループ数を自動判定)
reader.rdf()
reader.viscosity()       # せん断 / 縦 / 体積粘性率を派生列で追加
reader.shc()             # (相関関数 K(t), スペクトル J_q(ω)) に分割
reader.elastic()         # 6x6 [GPa]
reader.ttm()             # 電子温度グリッドのスナップショット列
reader.shock_profiles()  # *_hist.txt をまとめて
reader.ti_summary("ti_spring")
```

`postprocess` は「DataFrame → 物理量」を担当する。単位換算とフィット範囲の扱いを
ここに集約してある。

```python
from gpumd_toolkit.postprocess import (
    elastic_moduli,                 # C_ij -> B / G / E / ν (Voigt-Reuss-Hill)
    nemd_thermal_conductivity,      # 温度勾配と熱浴出力 -> κ [W/mK]
    green_kubo_kappa, hnemd_kappa,
    diffusion_from_msd,             # MSD の傾き -> D [cm^2/s]
    shear_viscosity, ionic_conductivity,
    free_energy_reversible_scaling, # ti_rs.csv -> G(T)
    free_energy_adiabatic_switching,# ti_as.csv -> G(P)
    vibrational_spectrum,           # 双極子/分極率 -> 赤外/ラマン
    committee_uncertainty_summary,
)
```

---

## 3. 解析・可視化 — `MDAnalyzer`

`thermo.out` を `run.in` と突き合わせて **時間軸・ステージ番号・目標温度を復元**する。

```python
from gpumd_toolkit import MDAnalyzer

a = MDAnalyzer("runs/si_300K")
a.thermo.frame            # DataFrame: time_ps, step, stage, temperature, pressure, volume, ...
a.thermo.summary()        # ステージごとの平均/標準偏差 (先頭 30% は平衡化として除外)
a.thermo.energy_drift()   # 全エネルギードリフト [meV/atom/ps] (NVE の品質指標)
a.plot_all()              # overview/temperature/energy/pressure/cell(+msd/sdc/rdf).png と CSV
a.diffusion_coefficient(source="msd")   # msd.out から自己拡散係数
```

複数計算の重ね描き:

```python
from gpumd_toolkit.analysis import compare_runs
compare_runs(["runs/T300K", "runs/T600K"], quantity="temperature", filename="cmp.png")
```

---

## 4. トラジェクトリ変換 — `TrajectoryConverter`

**出力は拡張 XYZ (`dump.xyz`) と XDATCAR を基本にしている。**
GPUMD が吐く `dump.xyz` はそのまま拡張 XYZ なので、OVITO / VESTA / VMD / ASE から
変換なしで開ける。ASE の `.traj` は専用ツールがないと中身を見られず扱いにくいので、
明示的に `fmt="traj"` を指定したときだけ書く。


```python
from gpumd_toolkit import TrajectoryConverter

t = TrajectoryConverter("runs/si_300K/dump.xyz")
t.n_frames
t.to_xdatcar("runs/si_300K/XDATCAR", stride=10)   # VASP XDATCAR
t.to_xyz("runs/si_300K/trajectory.xyz", stride=10)  # 拡張 XYZ (間引き)
t.to_poscar("runs/si_300K/CONTCAR")              # 最終フレームを POSCAR で
t.convert("out.lammpstrj", fmt="lammps-dump")     # 他形式
t.msd(time_step_ps=0.5)                           # アンラップ座標から MSD / 拡散係数
t.rdf(rmax=8.0, stride=10)                        # g(r)
```

対応形式: `xyz` (= `extxyz`), `xdatcar`, `lammps-dump`, `pdb`, `cif`, `netcdf`, `vasp`, `traj`。

> GPUMD が吐く `dump.xyz` はそのまま拡張 XYZ なので、OVITO / VESTA / ASE / VMD から
> 追加の変換なしで開ける。`convert_all()` の既定も **xyz と XDATCAR**。
> ASE の `.traj` は専用ツールがないと中身を見られないため、明示指定したときだけ書く。

---

## 5. 並列実行 — `ParallelRunner` (joblib)

```python
from functools import partial
from gpumd_toolkit import GPUMDCalculation, JobSpec, ParallelRunner

def build(temperature, workdir):        # トップレベル関数にすること (pickle 可能)
    calc = GPUMDCalculation("POSCAR", "nep89.txt", workdir, min_cell_length=12.0)
    calc.set_dump(thermo=100, traj=1000)
    calc.nvt(temperature=temperature, steps=20_000)
    return calc

jobs = [JobSpec(partial(build, T, f"runs/T{T}K"), name=f"T{T}K")
        for T in (300, 600, 900, 1200)]
results = ParallelRunner(n_jobs=2).run(jobs)   # GPU が複数あれば自動で振り分け
```

`ParallelRunner(gpu_ids=[0, 1])` でジョブへ `CUDA_VISIBLE_DEVICES` をラウンドロビン割当。

---

## 6. ASE ベースの実行 — `ASEMDRunner`

3 つのバックエンドを同じ API で切り替えられる。

| backend | 実体 | 用途 |
|---|---|---|
| `cpu` | calorine `CPUNEP` (GPUMD 同梱の `nep_cpu`) | CPU のみの環境、軽い 1 点計算 |
| `pynep` | pyNEP (`NEP_CPU`) | 同上。pyNEP を使いたい場合 |
| `gpu` | calorine `GPUNEP` (`gpumd` 実行ファイル経由) | GPU がある環境 |

```python
from gpumd_toolkit import ASEMDRunner, create_nep_calculator, available_backends

available_backends()      # {'cpu': True, 'pynep': True, 'gpu': True}

runner = ASEMDRunner("POSCAR", model="nep89.txt", backend="cpu", workdir="runs/ase")
runner.single_point()     # energy, forces, stress, per-atom energies
runner.relax(fmax=0.01, relax_cell=True)
runner.run_md(ensemble="nvt", temperature=300, temperature_end=600,
              steps=5000, time_step=1.0)
runner.plot(); runner.save_log()
runner.write_trajectory("runs/ase/XDATCAR", fmt="xdatcar")
```

### 6.1 積分器と時定数 (`taut` / `taup` / `pfactor`)

`ensemble` で ASE 側の積分器を選ぶ。時定数はすべて **fs** で指定する。

| `ensemble` | ASE のクラス | 調整できる時定数 |
|---|---|---|
| `nve` | `VelocityVerlet` | — |
| `nvt` (既定) | `Langevin` | `friction` [1/fs] |
| `nvt_berendsen` | `NVTBerendsen` | `taut` |
| `nvt_bussi` | `Bussi` | `taut` |
| `nvt_nose_hoover` | `MelchionnaNPT` (barostat なし) | `ttime` |
| `npt` = `npt_berendsen` | `NPTBerendsen` | `taut`, `taup`, `bulk_modulus_GPa` |
| `npt_inhomogeneous` | `Inhomogeneous_NPTBerendsen` | 同上 + `npt_axes` |
| `npt_parrinello_rahman` | `MelchionnaNPT` | `ttime`, `ptime`, `pfactor`, `npt_axes` |

```python
runner.run_md(ensemble="nvt_berendsen", temperature=300, taut=200, steps=5000)

runner.run_md(ensemble="npt", temperature=300, pressure_GPa=0.0,
              taut=100, taup=1000, bulk_modulus_GPa=140, steps=5000)

# Parrinello-Rahman で c 軸だけ動かす
runner.run_md(ensemble="npt_parrinello_rahman", temperature=300,
              ttime=50, ptime=2000, bulk_modulus_GPa=140,
              npt_axes=("z",), steps=5000)
```

* `pfactor` は `ptime**2 * bulk_modulus` として組み立てる
  (`pfactor=` を直接渡せばそちらが優先)。
* Berendsen の圧縮率は `compressibility_au = 1 / bulk_modulus_GPa`。
  既定値は 100 GPa (固体向け)。
* `npt_axes` は ASE の `mask` に変換される。`('x','y','z')` なら 3 軸独立、
  `('z',)` なら c 軸のみ、`('xy',)` のように剪断成分も指定できる
  (`mask=` を直接渡すことも可能)。
* ASE の `MelchionnaNPT` は三角行列のセルしか扱えないので、そうでない場合は
  自動的に標準形へ回転する (原子の相対配置は不変。警告を出す)。

積分器だけ取り出したいときは `runner.make_dynamics(...)`。
NEP 以外の calculator を使いたいときは
`ASEMDRunner(atoms, calculator=EMT(), ...)` のように注入できる。

calculator 単体で使う場合:

```python
atoms.calc = create_nep_calculator("nep89.txt", backend="cpu", atoms=atoms)
atoms.get_potential_energy()      # 全エネルギー
atoms.get_potential_energies()    # per-atom エネルギー
```

> **NEP_CPU / pyNEP への切り替え**は `backend="cpu"` ↔ `"pynep"` の 1 引数だけ。
> GPU が無い環境でも `backend="cpu"` (calorine 同梱の `nep_cpu`) で動く。
>
> 本モジュールでは calorine 3.5 の以下 2 点を補正している。
> 1. `CPUNEP` が per-atom エネルギーを `results` に入れていない
>    → `get_potential_energies()` が例外になる問題を修正。
> 2. `GPUNEP` の 1 点計算が削除済みキーワード `dump_force` / `dump_position` を
>    使うため新しい GPUMD で失敗する → `dump_xyz … force potential` に置換し、
>    ついでに per-atom エネルギーも取得。
>
> 3 バックエンドのエネルギー・力は相対誤差 1e-6 以内で一致することを
> `tests/test_toolkit.py::test_backends_agree` で確認している。

---

### 6.2 トラジェクトリ

`ASEMDRunner.run_md()` の既定の出力は `md.xyz` (拡張 XYZ)。

```python
runner.run_md(ensemble="nvt", temperature=300, steps=10000, log_interval=10)
print(runner.trajectory_path)                      # <workdir>/md.xyz
frames = runner.trajectory().read(":")             # ASE Atoms のリスト
runner.write_trajectory("XDATCAR", fmt="xdatcar")  # XDATCAR へ変換
```

`trajectory="md.traj"` のように `.traj` で終わる名前を渡したときだけ
ASE の Trajectory 形式になる。`trajectory=None` で出力を止められる。

---

## 7. DPMD の example と予実 — `DPMDExample`

題材は GPUMD 公式 example `14_DP/water_msd`
(液体水 512 H2O = 1536 原子、330 K NVT、`compute_sdc` / `compute_msd` で自己拡散係数)。
gpyumd のドキュメント例 (https://gpyumd.readthedocs.io/en/latest/example.html)
と同じ「平衡化 → 本計算 → 輸送係数」の流れになっている。

```python
from gpumd_toolkit.examples import DPMDExample

example = DPMDExample(
    "runs/dpmd_water",
    dp_setting_file="dp.txt",          # "dp 2 O H"
    dp_model="DNN_seed2.pb",           # deepmd-kit の凍結モデル
    fallback_potential="nep89.txt",    # DP が使えない場合のフォールバック
)
print(example.check_dp_support())
report = example.execute()             # 実行 → 解析 → 予実表の書き出し
print(report.to_markdown())
```

### 予実 (expected vs actual)

公式 example には DP で実行済みの `thermo.out` / `sdc.out` が同梱されている。
これを **予 (期待値)** とし、本クラスの実行結果を **実 (実測値)** として
`analysis/validation.{md,csv,png}` に出力する。

項目は 2 種類に分けてある。

* **ポテンシャル非依存** (原子数・セル長・平均温度) … 合否判定の対象
* **ポテンシャル依存** (エネルギー・圧力・拡散係数) … DP 以外では「参考」扱い

### DP が使えない環境について

`potential dp.txt <model>.pb` を使うには

1. GPUMD を `make -f makefile.dp` (= `-DUSE_TENSORFLOW`) でビルドすること
2. deepmd-kit の凍結モデル `.pb` があること

の両方が必要。**この環境はどちらも満たしていない**
(`/home/tajimamainpc/repos/GPUMD/src/gpumd` は DP 無しビルド、
example にも `.pb` は同梱されていない)。
そのため `DPMDExample` は自動的に NEP へフォールバックし、
同じワークフロー・同じ予実表で実行する。DP が使える環境へ持っていけば
`dp_model=` を渡すだけで DP 実行に切り替わる。

同梱の実行済み結果は `results/dpmd_water/` を参照 (NEP89 でのフォールバック実行)。

---

## 8. コマンドライン

```bash
source /home/tajimamainpc/.venv/gpumd312/bin/activate
NEP=/home/tajimamainpc/08_gpumd/02_benchmark/01_CuMoTaVW/nep89_20250409.txt
```

### 8.0 `gpumd-toolkit` コマンド

`pip install -e .` すると入る統合 CLI。機能分類ごとのサブコマンドがある。

```bash
gpumd-toolkit --help
gpumd-toolkit keywords          # 対応している run.in キーワードの一覧
gpumd-toolkit check             # GPUMD 実行環境の確認

# 2. 通常の MD
gpumd-toolkit md POSCAR -p $NEP -o runs/nvt --temperature 300 --steps 20000

# 1. 静的計算
gpumd-toolkit static POSCAR -p $NEP -o runs/static --relax --relax-cell --elastic

# 3. 自由エネルギー (Frenkel-Ladd)
gpumd-toolkit free-energy POSCAR -p $NEP -o runs/fe --path solid \
    --temperature 300 --t-equil 5000 --t-switch 20000

# 4. Monte Carlo
gpumd-toolkit mc alloy.xyz -p $NEP -o runs/mc --mode vcsgc \
    --species Cu Ni --values -0.5 0.5 --temperature 800

# 熱輸送 (HNEMD / Green-Kubo / NEMD)
gpumd-toolkit transport POSCAR -p $NEP -o runs/kappa --method hnemd \
    --temperature 300 --steps 1000000 --driving-force 1e-5

# 6. 拡散・液体物性
gpumd-toolkit diffusion liquid.xyz -p $NEP -o runs/liquid \
    --temperature 1500 --viscosity --rdf --per-species

# 8. PIMD / 量子熱浴
gpumd-toolkit pimd water.xyz -p $NEP -o runs/pimd --beads 32 --temperature 300

# 9. 衝撃波
gpumd-toolkit shock POSCAR -p $NEP -o runs/shock --method msst --vp 8

# 10. 二温度モデル
gpumd-toolkit ttm POSCAR -p $NEP -o runs/ttm --te 2000 --source 0.05

# 11. 引張試験
gpumd-toolkit mechanical POSCAR -p $NEP -o runs/tensile --strain-rate 1e8

# 12. Active Learning (committee)
gpumd-toolkit active-learning POSCAR -p nep0.txt nep1.txt nep2.txt \
    -o runs/al --temperature 1000 --threshold 0.05

# 解析・変換
gpumd-toolkit analyze runs/nvt
gpumd-toolkit convert runs/nvt/dump.xyz runs/nvt/XDATCAR --to xdatcar --stride 10
```

どのサブコマンドも `--dry-run` で `run.in` の生成だけ行える。

### 8.0.1 全機能のデモ

```bash
# 入力だけ作って全部の run.in を見る
python scripts/run_workflows_demo.py -p $NEP -o runs/demo --dry-run

# 短いステップ数で全部流す
python scripts/run_workflows_demo.py -p $NEP -o runs/demo --scale 800

# 一部だけ
python scripts/run_workflows_demo.py -p $NEP -o runs/demo \
    --only static free_energy transport_nemd
```

### MD 実行

```bash
# 定温 NVT
python scripts/run_md.py POSCAR -p $NEP -o runs/si_300K \
    --ensemble nvt --temperature 300 --steps 20000

# アニール (昇温→保持→降温)
python scripts/run_md.py structure.cif -p $NEP -o runs/anneal \
    --profile anneal --t-low 300 --t-high 1200 \
    --heat-steps 20000 --hold-steps 20000 --cool-steps 40000 \
    --convert xyz xdatcar

# 任意プロファイル ("時刻ps:温度K")
python scripts/run_md.py POSCAR -p $NEP -o runs/custom \
    --profile points --points 0:300,10:1000,20:1000,40:300

# NPT
python scripts/run_md.py POSCAR -p $NEP -o runs/npt --orthorhombic \
    --ensemble npt --temperature 500 --pressure 0 --steps 10000

# 入力だけ作って中身を確認
python scripts/run_md.py POSCAR -p $NEP -o runs/check --dry-run

# 熱浴・圧浴の時定数を fs で指定
python scripts/run_md.py POSCAR -p $NEP -o runs/npt \
    --ensemble npt --temperature 500 --tau-t 100 --tau-p 1000

# c 軸だけ動かす NPT
python scripts/run_md.py POSCAR -p $NEP -o runs/npt_z \
    --ensemble npt --temperature 500 --free-axes z

# 三斜晶セルを 6 成分指定でフルに緩和 (cell-mode を省略しても自動で tri になる)
python scripts/run_md.py structure.cif -p $NEP -o runs/npt_tri \
    --ensemble npt --temperature 500 --cell-mode tri

# npt_mttk で x 方向にだけ 5 GPa
python scripts/run_md.py POSCAR -p $NEP -o runs/uniaxial \
    --ensemble npt --barostat npt_mttk --mttk-direction x --pressure 5

# NPH (熱浴なし)
python scripts/run_md.py POSCAR -p $NEP -o runs/nph --ensemble nph --pressure 0
```

### 構造の変換・ベンチマーク

```bash
python scripts/convert_structure.py big.cif                # cif2xyz
python scripts/convert_structure.py POSCAR -o model.xyz    # POSCAR2xyz
python scripts/convert_structure.py big.cif --benchmark
python scripts/convert_structure.py --clear-cache
```

### 結果解析

```bash
python scripts/analyze_results.py runs/si_300K --msd --rdf --convert xyz xdatcar
python scripts/analyze_results.py runs/T*K --compare temperature --compare-output cmp.png
```

### DPMD example (予実つき)

```bash
python scripts/run_dpmd_example.py -o runs/dpmd_water --fallback-potential $NEP
# DP が使える環境なら:
python scripts/run_dpmd_example.py -o runs/dpmd_water --dp-setting dp.txt --dp-model model.pb
```

### ASE デモ

```bash
python scripts/run_ase_demo.py POSCAR -p $NEP -o runs/ase --backend cpu \
    --relax --relax-cell --md --steps 2000 --temperature 300 --convert xdatcar

# 熱浴の時定数を変える / 変調軸を選ぶ
python scripts/run_ase_demo.py POSCAR -p $NEP -o runs/ase_nvt \
    --md --ensemble nvt_berendsen --taut 200 --steps 2000
python scripts/run_ase_demo.py POSCAR -p $NEP -o runs/ase_npt \
    --md --ensemble npt_parrinello_rahman --pressure 0 \
    --ttime 50 --ptime 2000 --bulk-modulus 140 --npt-axes z
```

### 環境確認

```bash
python scripts/check_environment.py          # 人間向け
python scripts/check_environment.py --json   # 機械可読
```

### 並列温度スキャン

```bash
python scripts/run_parallel_scan.py POSCAR -p $NEP -o runs/scan \
    --temperatures 300 600 900 1200 --steps 20000 --n-jobs 2
```

---

## 9. テスト

```bash
source /home/tajimamainpc/.venv/gpumd312/bin/activate
python -m pytest tests/ -v

# 基本機能だけ / 追加機能だけ
python -m pytest tests/test_toolkit.py -v
python -m pytest tests/test_features.py -v
```

| ファイル | 対象 |
|----------|------|
| `tests/test_toolkit.py` | 構造入出力・温度プロファイル・`run.in` の基本・NPT の軸制御・ASE 連携・GPUMD 実行 |
| `tests/test_features.py` | 全 `ensemble` / `compute_*` / `dump_*` / 修飾キーワードの行生成、グループ分け、出力の読み込み、後処理の数式、各ワークフロー、CLI |

`gpumd` が見つからない環境では GPUMD 実行を伴うテストは自動 skip される。
`test_features.py` は GPUMD 実行を必要としない (生成した `run.in` の中身と
解析の数式を検証する)。

---

## 10. 設計メモ

* **構造の読み込み**は `fastio` の高速経路が既定。POSCAR / extxyz は numpy
  だけの専用パーサ、CIF は pymatgen、それ以外は ASE。遅い形式は extxyz に
  変換してキャッシュする。`fast=False` でいつでも ASE 経由に戻せる。
* **入力生成** は calorine (`calorine.gpumd.write_xyz`) を使用。ASE の extxyz
  ライタは速度の単位換算 (ASE 単位 ↔ Å/fs) をしないため。
  (`fastio.write_xyz_fast` は単位換算込みの軽量ライタで、
  `StructureHandler.write_model(..., fast=True)` から使える。)
* **時定数**は GPUMD が τ/Δt という無次元量を取るので、`tau_T` / `tau_p`
  (fs) は `RunInputBuilder` / `MDStage` の `time_step` を見て換算する。
  ステージごとに刻みを変えた場合はそのステージの刻みを使う。
* **NPT の軸固定**は `npt_ber` / `npt_scr` では弾性率 > 2000 GPa、
  `npt_mttk` では `direction` の指定で実現する。どちらも GPUMD 本体の仕様
  (`src/integrate/integrate.cu`, `doc/gpumd/input_parameters/ensemble_*.rst`)
  に沿ったもので、ツールキット側で積分器に手を入れているわけではない。
* **セルの厚み**は `volume / 面積` で評価する (GPUMD の `Box::thickness_*`
  と同じ定義)。スーパーセルの判定も辺長ではなくこちらで行う。
* **実行** は `subprocess` + `bash -lc "source <venv>/bin/activate && gpumd"`。
  GPU 指定は `CUDA_VISIBLE_DEVICES`。標準出力は `gpumd.out` に保存する。
* **原子数**は既定で 10,000 を上限とし、超えると例外 (mission の指示)。
  `min_cell_length` を指定すると、NEP のカットオフを満たす最小のスーパーセルを
  上限内で自動決定する。
* **`run.in` のキーワードは非 propagating** (次の `run` に引き継がれない) ため、
  `MDStage` ごとに `dump_thermo` / `dump_xyz` を書き直している。
  `MDAnalyzer` はこの構造を逆にたどって時間軸とステージを復元する。
* **特殊なアンサンブル** (QTB / NEMD 熱浴 / TTM / PIMD / 熱力学的積分 / 衝撃波) は
  引数の構造が標準アンサンブルとまったく違うので、`MDStage(ensemble=...)` に
  `inputs.ensembles` のクラスを渡す形にした。文字列指定 (NVE/NVT/NPT/NPH) は
  従来どおり動く。
* **`minimize` / `compute_cohesive` / `compute_elastic` / `compute_phonon`** は
  GPUMD が構文解析した時点で実行され、`ensemble` と `run` を必要としない。
  `RunInputBuilder.add_action()` がこの系統を別扱いにしている。
* **GPUMD 側の制約を実行前に検出する**ようにした。どれも実機で踏んだもの:
  - `fix` は 1 つの `run` につき 1 グループしか効かない
    (`Integrate::parse_fix` が単一の `fixed_group` を持つだけ)。
    `nemd_layout()` は両端の固定層を同じグループ 0 にまとめて回避する。
  - グループを取るキーワード (`fix` / `move` / `add_force` / `add_efield` /
    `add_spring` / `compute` / `ttm`) は `model.xyz` に grouping method が
    1 つ以上ないと起動直後に止まる → `ensure_grouping()` が自動で足す。
  - MCMD と `compute_rdf` は「周期方向のセル厚み > 2.5 × カットオフ」を要求する
    → `check_box_size()` が警告し、RDF のカットオフは自動で丸める。
  - NEP は「薄い周期方向 (≤ 2.5(rc+1)) と厚い方向 (> 10 rc) の同居」を拒否する
    → `check_cell()` が `write_inputs()` で警告する。
  - NEMD ピストンはセルを走り切ると密度が発散して CUDA エラーになる
    → `max_piston_steps()` が上限を計算して警告する。
* **出力の読み込み**は列名を持たせた `DataFrame` に統一した (`outputs.py`)。
  `shc.out` のように 1 ファイルに性質の違う 2 ブロックが入るものは
  GPUMD が書くヘッダ (`# num_correlation_rows`) を見て分割する。
* **トラジェクトリは拡張 XYZ と XDATCAR が既定**。GPUMD の `dump.xyz` は
  そのまま拡張 XYZ なので追加の変換なしで OVITO / VESTA / VMD / ASE から開ける。
  ASE の `.traj` は専用ツールが要るので、明示指定したときだけ書く
  (`ASEMDRunner` も既定で `md.xyz` を書く)。
* **作図の日本語**は `plotting.setup_matplotlib()` が CJK フォントを探して
  `font.family` のフォールバックに入れる。無い環境ではラベルを英語に差し替える
  (豆腐や警告を出さない)。
