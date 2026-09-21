# gpumd_toolkit — GPUMD を Python から使いこなすためのクラス群

`mission.md` の要件に沿って、GPUMD (https://gpumd.org) を Python から
「構造の準備 → `run.in` の生成 → 実行 → 解析・可視化 → トラジェクトリ変換」
まで一貫して扱えるようにしたツールキット。

クラスは mission の指示どおり **MD 計算** / **DPMD の example** / **ASE ベースの実行**
の 3 系統に分けてある。

```
03_gpumd_python_script/
├── gpumd_toolkit/
│   ├── config.py         GPUMDEnvironment  … 仮想環境・実行ファイル・GPU の設定
│   ├── fastio.py         read_fast など     … 構造ファイルの高速読み込み / xyz 変換
│   ├── structure.py      StructureHandler  … POSCAR / CIF などの入出力と model.xyz 生成
│   ├── profiles.py       TemperatureProfile… 昇温・降温・定温・任意温度プロファイル
│   ├── inputs.py         RunInputBuilder / MDStage / DumpSettings … run.in の組み立て
│   ├── md.py             GPUMDCalculation  … ★ MD 計算のメインクラス
│   ├── trajectory.py     TrajectoryConverter … XDATCAR / traj などへの変換
│   ├── analysis.py       MDAnalyzer        … thermo.out 解析と matplotlib 出力
│   ├── parallel.py       ParallelRunner    … joblib による並列実行
│   ├── ase_interface.py  ASEMDRunner       … ★ ASE ベースの実行 (CPU/GPU/pyNEP)
│   ├── validation.py     ValidationReport  … 予実 (期待値 vs 実測値) 管理
│   ├── diagnostics.py    environment_report… 実行環境の確認
│   └── examples/dpmd.py  DPMDExample       … ★ DPMD の example + 予実
├── scripts/              コマンドラインから使うスクリプト
├── tests/                自己テスト (pytest)
└── results/              同梱の実行済み結果 (DPMD example の予実)
```

---

## 1. セットアップ

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

```python
from gpumd_toolkit import TrajectoryConverter

t = TrajectoryConverter("runs/si_300K/dump.xyz")
t.n_frames
t.to_xdatcar("runs/si_300K/XDATCAR", stride=10)   # VASP XDATCAR
t.to_ase_traj("runs/si_300K/md.traj")             # ASE .traj
t.convert("out.lammpstrj", fmt="lammps-dump")     # 他形式
t.msd(time_step_ps=0.5)                           # アンラップ座標から MSD / 拡散係数
t.rdf(rmax=8.0, stride=10)                        # g(r)
```

対応形式: `xdatcar`, `traj`, `extxyz`, `lammps-dump`, `pdb`, `cif`, `netcdf`, `vasp`。

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

## 8. コマンドラインスクリプト

```bash
source /home/tajimamainpc/.venv/gpumd312/bin/activate
NEP=/home/tajimamainpc/08_gpumd/02_benchmark/01_CuMoTaVW/nep89_20250409.txt
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
    --convert xdatcar traj

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
python scripts/analyze_results.py runs/si_300K --msd --rdf --convert xdatcar traj
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
```

`gpumd` が見つからない環境では GPUMD 実行を伴うテストは自動 skip される。

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
