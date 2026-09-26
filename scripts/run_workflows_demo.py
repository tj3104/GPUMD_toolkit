#!/usr/bin/env python
"""GPUMD の全機能分類を一通り動かすデモ。

各ワークフローの `run.in` を短いステップ数で作り、GPUMD があれば実行して
主要な物理量まで出す。新しい環境で「どこまで動くか」を確かめるのに使う。

使用例
------
# 入力だけ作って中身を確認する (GPUMD 不要)
python scripts/run_workflows_demo.py -p nep.txt -o runs/demo --dry-run

# 実際に流す (短いので数分で終わる)
python scripts/run_workflows_demo.py -p nep.txt -o runs/demo

# 一部だけ
python scripts/run_workflows_demo.py -p nep.txt -o runs/demo \
    --only static free_energy transport
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

from gpumd_toolkit.workflows import (
    ActiveLearningCalculation,
    DiffusionCalculation,
    FreeEnergyCalculation,
    MechanicalCalculation,
    MonteCarloCalculation,
    PathIntegralCalculation,
    ShockCalculation,
    StaticCalculation,
    ThermalTransport,
    TwoTemperatureCalculation,
)

#: デモの一覧 (名前 -> 構築関数)。すべて短いステップ数にしてある。
DEMOS: dict[str, str] = {
    "static": "1. 最適化・弾性定数・凝集エネルギー・一点計算",
    "free_energy": "3. Frenkel-Ladd による固体の自由エネルギー",
    "monte_carlo": "4. カノニカル MC/MD",
    "transport_hnemd": "熱輸送: HNEMD + スペクトル熱流",
    "transport_nemd": "熱輸送: source/sink NEMD",
    "diffusion": "6. 液体の拡散係数・粘性率・RDF",
    "pimd": "8. 経路積分 MD",
    "qtb": "8. 量子熱浴",
    "shock_msst": "9. MSST による衝撃圧縮",
    "shock_piston": "9. NEMD ピストン",
    "ttm": "10. 二温度モデル",
    "mechanical": "11. 一軸引張",
    "active_learning": "12. committee による不確かさ評価",
}


def build(name: str, structure, potential: str, root: Path, scale: int):
    """デモ名からワークフローを組み立てる。"""
    common = dict(structure=structure, potential=potential, overwrite=True)
    workdir = root / name

    if name == "static":
        calc = StaticCalculation(**common, workdir=workdir, repeat=(2, 2, 2))
        calc.relax(relax_cell=True, max_steps=200)
        calc.elastic(0.01)
        calc.cohesive_curve(0.97, 1.03)
        calc.single_point()
        return calc

    if name == "free_energy":
        calc = FreeEnergyCalculation(**common, workdir=workdir, repeat=(3, 3, 3))
        calc.equilibrate(temperature=300, steps=2 * scale)
        calc.frenkel_ladd(temperature=300, t_equil=1 * scale, t_switch=4 * scale)
        return calc

    if name == "monte_carlo":
        calc = MonteCarloCalculation(**common, workdir=workdir, min_cell_length=20.0)
        symbols = calc.atoms.get_chemical_symbols()
        if len(set(symbols)) == 1:   # 単元素なら擬似的に 2 元素合金にする
            for i in range(0, len(symbols), 2):
                symbols[i] = "Ni" if symbols[i] != "Ni" else "Cu"
            calc.atoms.set_chemical_symbols(symbols)
        calc.canonical(temperature=800, steps=2 * scale, md_steps=100, mc_trials=50)
        return calc

    if name == "transport_hnemd":
        calc = ThermalTransport(**common, workdir=workdir, min_cell_length=16.0)
        calc.equilibrate(temperature=300, steps=2 * scale)
        calc.hnemd(temperature=300, steps=20 * scale, Fe=1e-4, direction="x")
        calc.spectral(sample_interval=2, Nc=100, direction="x", num_omega=200, max_omega=100)
        return calc

    if name == "transport_nemd":
        calc = ThermalTransport.for_nemd(
            **common, workdir=workdir, axis="x", n_blocks=8, repeat=(14, 6, 6)
        )
        calc.equilibrate(temperature=300, steps=2 * scale)
        calc.nemd(temperature=300, steps=30 * scale, delta_T=40, method="lan",
                  sample_interval=10, output_interval=10)
        return calc

    if name == "diffusion":
        calc = DiffusionCalculation(**common, workdir=workdir, min_cell_length=16.0)
        calc.equilibrate(temperature=2500, steps=5 * scale)
        calc.production(temperature=2500, steps=30 * scale, msd=True,
                        viscosity=True, rdf=True, sample_interval=5, Nc=200)
        return calc

    if name == "pimd":
        calc = PathIntegralCalculation(**common, workdir=workdir, repeat=(2, 2, 2))
        calc.pimd(num_beads=4, temperature=300, steps=1 * scale)
        calc.dump_beads(interval=scale // 2 or 1)
        return calc

    if name == "qtb":
        calc = PathIntegralCalculation(**common, workdir=workdir, repeat=(2, 2, 2))
        calc.qtb(temperature=300, steps=1 * scale, f_max=100)
        return calc

    if name == "shock_msst":
        calc = ShockCalculation(**common, workdir=workdir, min_cell_length=14.0)
        calc.equilibrate(temperature=300, steps=2 * scale)
        calc.msst(shock_velocity=8.0, steps=5 * scale, direction="x")
        return calc

    if name == "shock_piston":
        calc = ShockCalculation(**common, workdir=workdir, repeat=(40, 6, 6))
        calc.equilibrate(temperature=300, steps=2 * scale)
        # セルを走り切る手前で止める
        steps = min(8 * scale, calc.max_piston_steps(1.5, thickness=10) - 500)
        calc.piston(vp=1.5, steps=max(steps, 1000), thickness=10,
                    bin_size=3.0, dump_interval=250)
        return calc

    if name == "ttm":
        calc = TwoTemperatureCalculation(**common, workdir=workdir, repeat=(4, 4, 14))
        calc.equilibrate(temperature=300, steps=2 * scale)
        calc.ttm(steps=10 * scale, Ce=1e-4, rho_e=1.0, kappa_e=0.005,
                 gamma_p=0.01, grid=(1, 1, 12), T_e_init=2000, out_interval=200)
        return calc

    if name == "mechanical":
        calc = MechanicalCalculation(**common, workdir=workdir, repeat=(8, 3, 3))
        calc.equilibrate(temperature=300, steps=2 * scale, pressure=0.0)
        calc.tensile(temperature=300, steps=20 * scale, strain_rate=5e9, axis="x")
        return calc

    if name == "active_learning":
        calc = ActiveLearningCalculation(
            structure=structure, potential=[[potential], [potential]],
            workdir=workdir, min_cell_length=12.0, overwrite=True,
        )
        calc.committee(temperature=800, steps=2 * scale, threshold=0.05, interval=10)
        return calc

    raise ValueError(f"未知のデモ: {name}")


def report(name: str, calc) -> None:
    """実行後に主要な物理量を出す。"""
    if name == "static":
        print("   弾性率:", calc.elastic_moduli())
        print("   E/atom:", round(calc.energy_per_atom(), 6), "eV")
        print("   平衡点:", calc.equilibrium_from_cohesive())
    elif name == "free_energy":
        print("   自由エネルギー:", calc.gibbs_free_energy())
        print("   ヒステリシス:", calc.hysteresis())
    elif name == "monte_carlo":
        print("   MC 受理率:", round(calc.acceptance_ratio(), 3))
    elif name == "transport_hnemd":
        print("   kappa:", calc.hnemd_result())
    elif name == "transport_nemd":
        print("   kappa:", calc.nemd_result())
        print(calc.temperature_profile().to_string(index=False))
    elif name == "diffusion":
        print("   D:", calc.diffusion_coefficient())
        print("   eta:", calc.viscosity())
    elif name == "pimd":
        print("   ビーズ:", [p.name for p in calc.bead_files()])
    elif name in ("shock_msst",):
        print("   Hugoniot:", calc.hugoniot_state())
    elif name == "shock_piston":
        print(calc.shock_front_position().head().to_string(index=False))
    elif name == "ttm":
        snapshots = calc.electron_temperature()
        print("   電子温度スナップショット:", len(snapshots.steps))
    elif name == "mechanical":
        print("   力学特性:", calc.mechanical_properties())
    elif name == "active_learning":
        print("   不確かさ:", calc.uncertainty_summary())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GPUMD の全機能分類のデモ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--potential", required=True, help="NEP ポテンシャル")
    parser.add_argument("-o", "--workdir", required=True, help="出力先ディレクトリ")
    parser.add_argument("-s", "--structure", default=None,
                        help="入力構造 (省略すると fcc Cu を使う)")
    parser.add_argument("--only", nargs="*", default=None, choices=sorted(DEMOS),
                        help="実行するデモ (既定はすべて)")
    parser.add_argument("--scale", type=int, default=1000,
                        help="ステップ数の基準 (大きいほど長く回す)")
    parser.add_argument("--dry-run", action="store_true", help="run.in の生成だけ")
    args = parser.parse_args(argv)

    if args.structure:
        structure = args.structure
    else:
        from ase.build import bulk

        structure = bulk("Cu", "fcc", 3.615, cubic=True)

    root = Path(args.workdir).expanduser().resolve()
    names = args.only or list(DEMOS)
    results: list[tuple[str, bool]] = []

    for name in names:
        print("=" * 72)
        print(f"{name}: {DEMOS[name]}")
        print("-" * 72)
        try:
            calc = build(name, structure, args.potential, root, args.scale)
        except Exception:
            traceback.print_exc(limit=2)
            results.append((name, False))
            continue

        if args.dry_run:
            calc.write_inputs()
            print(calc.preview())
            results.append((name, True))
            continue

        result = calc.run(check=False)
        ok = result.succeeded
        print(f"  {'OK' if ok else '失敗'}  ({result.elapsed:.1f} s)")
        if not ok:
            tail = (result.command.stderr or result.command.stdout or "").splitlines()[-6:]
            print("   " + "\n   ".join(tail))
        else:
            try:
                report(name, calc)
            except Exception as error:
                print(f"   後処理でエラー: {type(error).__name__}: {error}")
        results.append((name, ok))

    print("=" * 72)
    for name, ok in results:
        print(f"  {'OK  ' if ok else '失敗 '} {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
