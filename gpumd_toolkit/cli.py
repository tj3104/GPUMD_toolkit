"""``gpumd-toolkit`` コマンド。

機能分類ごとのサブコマンドから GPUMD を回す。
細かい設定が要る場合は Python API を直接使うほうが早いが、
一通りの計算はここから起動できる。

    gpumd-toolkit --help
    gpumd-toolkit md POSCAR -p nep.txt -o runs/nvt --temperature 300 --steps 20000
    gpumd-toolkit static POSCAR -p nep.txt -o runs/static --elastic --relax
    gpumd-toolkit transport POSCAR -p nep.txt -o runs/kappa --method hnemd
    gpumd-toolkit keywords            # 対応している run.in キーワード一覧
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

__all__ = ["main", "build_parser"]


# --------------------------------------------------------------- 共通オプション
def _add_common(parser: argparse.ArgumentParser, *, need_potential: bool = True) -> None:
    parser.add_argument("structure", help="入力構造 (POSCAR / CIF / xyz / traj など)")
    parser.add_argument(
        "-p", "--potential", required=need_potential, nargs="+",
        help="NEP ポテンシャル。複数指定すると committee になる",
    )
    parser.add_argument("-o", "--workdir", required=True, help="計算ディレクトリ")
    parser.add_argument("--repeat", type=int, nargs=3, default=None, help="スーパーセルの繰り返し数")
    parser.add_argument(
        "--min-cell-length", type=float, default=None,
        help="この面間距離 [Å] 以上になるよう自動でスーパーセル化",
    )
    parser.add_argument("--max-atoms", type=int, default=10000, help="原子数の上限")
    parser.add_argument("--time-step", type=float, default=1.0, help="時間刻み [fs]")
    parser.add_argument("--overwrite", action="store_true", help="既存の計算ディレクトリを消す")
    parser.add_argument("--dry-run", action="store_true", help="入力だけ作って実行しない")
    parser.add_argument("--gpu", type=int, default=None, help="使用する GPU の index")
    parser.add_argument("--venv", default=None, help="GPUMD を起動する仮想環境")


def _environment(args):
    from gpumd_toolkit import GPUMDEnvironment

    kwargs = {}
    if args.venv:
        kwargs["venv"] = Path(args.venv)
    if args.gpu is not None:
        kwargs["gpu_id"] = args.gpu
    return GPUMDEnvironment(**kwargs) if kwargs else None


def _common_kwargs(args) -> dict:
    potential = args.potential
    if potential is not None and len(potential) > 1:
        potential = [[item] for item in potential]   # committee = 複数の potential 行
    elif potential is not None:
        potential = potential[0]
    return {
        "structure": args.structure,
        "potential": potential,
        "workdir": args.workdir,
        "repeat": tuple(args.repeat) if args.repeat else None,
        "min_cell_length": args.min_cell_length,
        "max_atoms": args.max_atoms,
        "time_step": args.time_step,
        "overwrite": args.overwrite,
        "environment": _environment(args),
    }


def _finish(calculation, args, *, analyze: bool = True) -> int:
    print(calculation.describe())
    if args.dry_run:
        calculation.write_inputs()
        print(f"\n入力ファイルを生成しました: {calculation.workdir}\n")
        print(calculation.preview())
        return 0
    result = calculation.run()
    print(f"\n完了: {result}")
    if analyze:
        try:
            print(result.summary().to_string(index=False))
        except Exception as error:  # pragma: no cover - thermo.out が無い計算
            print(f"(thermo の要約なし: {error})")
    return 0


# ------------------------------------------------------------------ 各サブコマンド
def cmd_md(args) -> int:
    from gpumd_toolkit import GPUMDCalculation, TemperatureProfile

    calc = GPUMDCalculation(**_common_kwargs(args))
    calc.set_dump(thermo=args.thermo_interval, traj=args.traj_interval)
    if args.minimize:
        calc.minimize()
    if args.profile == "anneal":
        profile = TemperatureProfile.anneal(
            T_low=args.temperature, T_high=args.temperature_high,
            equilibrate_steps=args.steps, heat_steps=args.steps,
            hold_steps=args.steps, cool_steps=args.steps,
        )
        calc.apply_profile(profile, ensemble=args.thermostat)
    elif args.ensemble == "nve":
        calc.nve(steps=args.steps)
    elif args.ensemble == "npt":
        calc.npt(
            temperature=args.temperature, temperature_end=args.temperature_end,
            steps=args.steps, pressure=args.pressure[0] if len(args.pressure) == 1
            else tuple(args.pressure),
            barostat=args.barostat, tau_T=args.tau_t, tau_p=args.tau_p,
            free_axes=tuple(args.free_axes) if args.free_axes else None,
        )
    else:
        calc.nvt(
            temperature=args.temperature, temperature_end=args.temperature_end,
            steps=args.steps, thermostat=args.thermostat, tau_T=args.tau_t,
        )
    return _finish(calc, args)


def cmd_static(args) -> int:
    from gpumd_toolkit.workflows import StaticCalculation

    calc = StaticCalculation(**_common_kwargs(args))
    if args.relax:
        calc.relax(relax_cell=args.relax_cell, max_steps=args.minimize_steps)
    if args.cohesive:
        calc.cohesive_curve(args.cohesive[0], args.cohesive[1])
    if args.elastic:
        calc.elastic(args.strain)
    if args.single_point or not (args.cohesive or args.elastic):
        calc.single_point()
    status = _finish(calc, args, analyze=False)
    if status or args.dry_run:
        return status
    if args.elastic:
        print("\n弾性定数 C_ij [GPa]:")
        print(calc.elastic_matrix())
        print(calc.elastic_moduli())
    if args.cohesive:
        print("\n凝集エネルギー曲線の最小点:", calc.equilibrium_from_cohesive())
        print("  図:", calc.plot_cohesive())
    if args.single_point or not (args.cohesive or args.elastic):
        print(f"\nE/atom = {calc.energy_per_atom():.6f} eV")
        print(f"|F|max = {abs(calc.forces()).max():.6f} eV/Å")
    return 0


def cmd_free_energy(args) -> int:
    from gpumd_toolkit.workflows import FreeEnergyCalculation

    calc = FreeEnergyCalculation(**_common_kwargs(args))
    if args.equilibrate:
        calc.equilibrate(temperature=args.temperature, steps=args.equilibrate)
    if args.path == "solid":
        calc.frenkel_ladd(
            temperature=args.temperature, t_equil=args.t_equil, t_switch=args.t_switch,
            pressure=args.pressure,
        )
    elif args.path == "liquid":
        calc.uhlenbeck_ford(
            temperature=args.temperature, t_equil=args.t_equil, t_switch=args.t_switch,
            pressure=args.pressure,
        )
    elif args.path == "rs":
        calc.reversible_scaling(
            T_min=args.temperature, T_max=args.temperature_high,
            pressure=args.pressure, t_equil=args.t_equil, t_switch=args.t_switch,
        )
    else:
        calc.adiabatic_switching(
            temperature=args.temperature, p_min=args.pressure, p_max=args.pressure_high,
            t_equil=args.t_equil, t_switch=args.t_switch,
        )
    status = _finish(calc, args, analyze=False)
    if status or args.dry_run:
        return status
    if args.path in ("solid", "liquid"):
        print("\n自由エネルギー [eV/atom]:")
        for key, value in calc.gibbs_free_energy().items():
            print(f"  {key:12s} {value}")
    print("\nヒステリシス (小さいほど良い):", calc.hysteresis())
    return 0


def cmd_mc(args) -> int:
    from gpumd_toolkit.workflows import MonteCarloCalculation

    calc = MonteCarloCalculation(**_common_kwargs(args))
    species = dict(zip(args.species, args.values)) if args.species else {}
    if args.mode == "canonical":
        calc.canonical(
            temperature=args.temperature, temperature_end=args.temperature_end,
            steps=args.steps, md_steps=args.md_steps, mc_trials=args.mc_trials,
        )
    elif args.mode == "sgc":
        calc.semi_grand_canonical(
            temperature=args.temperature, steps=args.steps,
            chemical_potentials=species, md_steps=args.md_steps, mc_trials=args.mc_trials,
        )
    else:
        calc.vcsgc(
            temperature=args.temperature, steps=args.steps, phi=species,
            kappa=args.kappa, md_steps=args.md_steps, mc_trials=args.mc_trials,
        )
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    print(f"\nMC 受理率: {calc.acceptance_ratio():.3f}")
    concentrations = calc.concentrations()
    if len(concentrations):
        print("濃度:", concentrations.to_dict())
    print("図:", calc.plot_mcmd())
    return 0


def cmd_transport(args) -> int:
    from gpumd_toolkit.workflows import ThermalTransport

    if args.method == "nemd":
        calc = ThermalTransport.for_nemd(
            axis=args.direction, n_blocks=args.n_blocks, **_common_kwargs(args)
        )
    else:
        calc = ThermalTransport(**_common_kwargs(args))
    if args.equilibrate:
        calc.equilibrate(temperature=args.temperature, steps=args.equilibrate)
    if args.method == "gk":
        calc.green_kubo(
            temperature=args.temperature, steps=args.steps,
            correlation_steps=args.correlation_steps,
        )
    elif args.method == "hnemd":
        calc.hnemd(
            temperature=args.temperature, steps=args.steps,
            Fe=args.driving_force, direction=args.direction,
        )
    else:
        calc.nemd(temperature=args.temperature, steps=args.steps, delta_T=args.delta_t)
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    print("\n熱伝導率 [W/mK]:")
    for key, value in calc.thermal_conductivity().items():
        print(f"  {key:26s} {value}")
    print("図:", calc.plot_kappa())
    if args.method == "nemd":
        print("図:", calc.plot_temperature_profile())
    return 0


def cmd_diffusion(args) -> int:
    from gpumd_toolkit.workflows import DiffusionCalculation

    calc = DiffusionCalculation(**_common_kwargs(args))
    if args.per_species:
        calc.group_by_species()
    if args.equilibrate:
        calc.equilibrate(
            temperature=args.temperature, steps=args.equilibrate, pressure=args.pressure
        )
    calc.production(
        temperature=args.temperature, steps=args.steps,
        msd=True, sdc=args.sdc, viscosity=args.viscosity, rdf=args.rdf,
        ic=(args.ic_type, args.ic_charge) if args.ic_type is not None else None,
        per_species=args.per_species,
    )
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    print("\n自己拡散係数:", calc.diffusion_coefficient())
    if args.viscosity:
        print("粘性率 [Pa s]:", calc.viscosity())
    if args.ic_type is not None:
        print("イオン伝導度 [mS/cm]:", calc.ionic_conductivity())
    print("図:", calc.plot_summary())
    return 0


def cmd_pimd(args) -> int:
    from gpumd_toolkit.workflows import PathIntegralCalculation

    calc = PathIntegralCalculation(**_common_kwargs(args))
    if args.method == "qtb":
        calc.qtb(temperature=args.temperature, steps=args.steps, f_max=args.f_max)
    else:
        calc.pimd(
            num_beads=args.beads, temperature=args.temperature, steps=args.steps,
            eco_omega_max=args.eco,
        )
        if args.dump_beads:
            calc.dump_beads(interval=args.dump_beads)
    return _finish(calc, args)


def cmd_shock(args) -> int:
    from gpumd_toolkit.workflows import ShockCalculation

    calc = ShockCalculation(**_common_kwargs(args))
    if args.equilibrate:
        calc.equilibrate(temperature=args.temperature, steps=args.equilibrate)
    if args.method == "msst":
        calc.msst(shock_velocity=args.vp, steps=args.steps, direction=args.direction)
    elif args.method == "nphug":
        calc.hugoniostat(pressure=args.pressure, steps=args.steps)
    else:
        calc.piston(vp=args.vp, steps=args.steps, kind=args.wall, bin_size=args.bin_size)
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    if args.method == "piston":
        print("図:", calc.plot_profiles())
    else:
        print("\nHugoniot 状態:", calc.hugoniot_state())
    return 0


def cmd_ttm(args) -> int:
    from gpumd_toolkit.workflows import TwoTemperatureCalculation

    calc = TwoTemperatureCalculation(**_common_kwargs(args))
    if args.equilibrate:
        calc.equilibrate(temperature=args.temperature, steps=args.equilibrate)
    calc.ttm(
        steps=args.steps, Ce=args.ce, rho_e=args.rho_e, kappa_e=args.kappa_e,
        gamma_p=args.gamma_p, grid=tuple(args.grid), T_e_init=args.te,
        source=args.source, out_interval=args.out_interval,
    )
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    print("図:", calc.plot_two_temperature())
    return 0


def cmd_mechanical(args) -> int:
    from gpumd_toolkit.workflows import MechanicalCalculation

    calc = MechanicalCalculation(**_common_kwargs(args))
    if args.equilibrate:
        calc.equilibrate(temperature=args.temperature, steps=args.equilibrate)
    if args.mode == "tensile":
        calc.tensile(
            temperature=args.temperature, steps=args.steps,
            strain_rate=args.strain_rate, axis=args.axis,
        )
    else:
        calc.shear(
            temperature=args.temperature, steps=args.steps,
            rate=args.rate, component=args.component,
        )
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    print("\n力学特性:", calc.mechanical_properties())
    print("図:", calc.plot_stress_strain())
    return 0


def cmd_active_learning(args) -> int:
    from gpumd_toolkit.workflows import ActiveLearningCalculation

    calc = ActiveLearningCalculation(**_common_kwargs(args))
    if args.asi_file:
        calc.extrapolation(
            temperature=args.temperature, steps=args.steps,
            nep_file=args.potential[0], asi_file=args.asi_file,
            gamma_low=args.gamma_low, gamma_high=args.gamma_high,
        )
    else:
        calc.explore(
            T_start=args.temperature, T_end=args.temperature_end or args.temperature,
            steps=args.steps, threshold=args.threshold, interval=args.interval,
        )
    status = _finish(calc, args)
    if status or args.dry_run:
        return status
    if not args.asi_file:
        print("\n不確かさ:", calc.uncertainty_summary())
        print("図:", calc.plot_uncertainty())
        try:
            structures = calc.selected_structures()
            print(f"選ばれた構造: {len(structures)} 個 -> {calc.write_candidates()}")
        except FileNotFoundError:
            print("しきい値を超える構造はありませんでした。")
    return 0


def cmd_analyze(args) -> int:
    from gpumd_toolkit.analysis import MDAnalyzer
    from gpumd_toolkit.outputs import OutputReader

    reader = OutputReader(args.workdir)
    print(f"{reader.directory} にある出力:")
    for name in reader.available():
        print(f"  {name}")
    if reader.has("thermo.out"):
        analyzer = MDAnalyzer(reader.directory)
        print()
        print(analyzer.thermo.summary().to_string(index=False))
        if not args.no_plot:
            for path in analyzer.plot_all():
                print(f"  図: {path}")
    return 0


def cmd_convert(args) -> int:
    from gpumd_toolkit.fastio import convert_to_xyz
    from gpumd_toolkit.trajectory import TrajectoryConverter

    source = Path(args.source)
    if args.to == "xyz":
        print(convert_to_xyz(source, args.output))
        return 0
    converter = TrajectoryConverter(source)
    print(converter.convert(args.output, fmt=args.to, stride=args.stride))
    return 0


def cmd_check(args) -> int:
    from gpumd_toolkit.diagnostics import environment_report, format_report

    environment = None
    if args.venv:
        from gpumd_toolkit import GPUMDEnvironment

        environment = GPUMDEnvironment(venv=Path(args.venv))
    report = environment_report(environment)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else format_report(report))
    return 0 if report["executables"].get("gpumd") else 1


def cmd_keywords(args) -> int:
    """対応している ``run.in`` キーワードを一覧する。"""
    from gpumd_toolkit.inputs import computes, dumps, ensembles, modifiers

    sections = [
        ("ensemble (文字列指定)", list(ensembles.STANDARD_ENSEMBLES)),
        ("ensemble (クラス指定)", [
            f"{name} -> {cls}" for name, cls in (
                ("nvt_qtb / npt_qtb", "QTB"),
                ("heat_nhc / heat_bdp / heat_lan", "HeatBath"),
                ("ttm / heat_ttm", "TTM / HeatTTM"),
                ("pimd / pimd_scr / rpmd / trpmd", "PIMD / RPMD / TRPMD"),
                ("ti / ti_spring / ti_liquid / ti_as / ti_rs",
                 "TIFixedLambda / TISpring / TILiquid / TIAdiabaticSwitching / TIReversibleScaling"),
                ("msst / nphug", "MSST / NPHug"),
                ("wall_piston / wall_mirror / wall_harmonic", "Wall"),
            )
        ]),
        ("compute_*", sorted(n for n in computes.__all__ if n.startswith("compute"))),
        ("dump_* / active", sorted(n for n in dumps.__all__ if n.startswith(("dump", "active")))),
        ("setup / 外場 / MC / 最適化", sorted(
            n for n in modifiers.__all__ if n.islower() and not n.isupper()
        )),
    ]
    for title, items in sections:
        print(f"\n[{title}]")
        for item in items:
            print(f"  {item}")
    return 0


# ------------------------------------------------------------------ パーサ
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpumd-toolkit",
        description="GPUMD を Python から回すツールキットの CLI",
    )
    from gpumd_toolkit import __version__

    parser.add_argument("--version", action="version", version=f"gpumd-toolkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- md ---
    p = sub.add_parser("md", help="2. 通常の MD (NVE / NVT / NPT)")
    _add_common(p)
    p.add_argument("--ensemble", default="nvt", choices=["nve", "nvt", "npt"])
    p.add_argument("--thermostat", default="nvt_nhc")
    p.add_argument("--barostat", default="npt_scr")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--temperature-end", type=float, default=None)
    p.add_argument("--temperature-high", type=float, default=1000.0, help="アニールの上限温度")
    p.add_argument("--pressure", type=float, nargs="+", default=[0.0])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--tau-t", type=float, default=None, help="熱浴の時定数 [fs]")
    p.add_argument("--tau-p", type=float, default=None, help="圧浴の時定数 [fs]")
    p.add_argument("--free-axes", nargs="*", default=None, help="NPT で動かす軸")
    p.add_argument("--profile", default=None, choices=[None, "anneal"])
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--thermo-interval", type=int, default=100)
    p.add_argument("--traj-interval", type=int, default=1000)
    p.set_defaults(func=cmd_md)

    # --- static ---
    p = sub.add_parser("static", help="1. 静的計算 (最適化・弾性定数・凝集エネルギー)")
    _add_common(p)
    p.add_argument("--relax", action="store_true", help="構造最適化する")
    p.add_argument("--relax-cell", action="store_true", help="セルも最適化する")
    p.add_argument("--minimize-steps", type=int, default=1000)
    p.add_argument("--elastic", action="store_true", help="弾性定数を計算する")
    p.add_argument("--strain", type=float, default=0.01)
    p.add_argument("--cohesive", type=float, nargs=2, default=None, metavar=("E1", "E2"))
    p.add_argument("--single-point", action="store_true")
    p.set_defaults(func=cmd_static)

    # --- free-energy ---
    p = sub.add_parser("free-energy", help="3. 自由エネルギー (熱力学的積分)")
    _add_common(p)
    p.add_argument("--path", default="solid", choices=["solid", "liquid", "rs", "as"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--temperature-high", type=float, default=2000.0, help="rs の上限温度")
    p.add_argument("--pressure", type=float, default=0.0)
    p.add_argument("--pressure-high", type=float, default=10.0, help="as の上限圧力")
    p.add_argument("--t-equil", type=int, default=5000)
    p.add_argument("--t-switch", type=int, default=20000)
    p.add_argument("--equilibrate", type=int, default=10000, help="TI 前の平衡化ステップ数")
    p.set_defaults(func=cmd_free_energy)

    # --- mc ---
    p = sub.add_parser("mc", help="4. Monte Carlo (組成・配置サンプリング)")
    _add_common(p)
    p.add_argument("--mode", default="canonical", choices=["canonical", "sgc", "vcsgc"])
    p.add_argument("--temperature", type=float, default=1000.0)
    p.add_argument("--temperature-end", type=float, default=None)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--md-steps", type=int, default=100)
    p.add_argument("--mc-trials", type=int, default=100)
    p.add_argument("--species", nargs="*", default=None, help="MC で扱う元素")
    p.add_argument("--values", type=float, nargs="*", default=None, help="mu [eV] または phi")
    p.add_argument("--kappa", type=float, default=100.0)
    p.set_defaults(func=cmd_mc)

    # --- transport ---
    p = sub.add_parser("transport", help="熱輸送 (Green-Kubo / HNEMD / NEMD)")
    _add_common(p)
    p.add_argument("--method", default="hnemd", choices=["gk", "hnemd", "nemd"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=1000000)
    p.add_argument("--equilibrate", type=int, default=50000)
    p.add_argument("--driving-force", type=float, default=1e-5, help="HNEMD の Fe [1/Å]")
    p.add_argument("--direction", default="x", choices=["x", "y", "z"])
    p.add_argument("--delta-t", type=float, default=30.0, help="NEMD の温度差の半分 [K]")
    p.add_argument("--n-blocks", type=int, default=10, help="NEMD のブロック数")
    p.add_argument("--correlation-steps", type=int, default=50000)
    p.set_defaults(func=cmd_transport)

    # --- diffusion ---
    p = sub.add_parser("diffusion", help="6. 拡散・イオン伝導・液体物性")
    _add_common(p)
    p.add_argument("--temperature", type=float, default=1000.0)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--equilibrate", type=int, default=50000)
    p.add_argument("--pressure", type=float, default=None)
    p.add_argument("--sdc", action="store_true")
    p.add_argument("--viscosity", action="store_true")
    p.add_argument("--rdf", action="store_true")
    p.add_argument("--per-species", action="store_true")
    p.add_argument("--ic-type", type=int, default=None, help="イオン伝導度を出す元素の型番号")
    p.add_argument("--ic-charge", type=float, default=1.0)
    p.set_defaults(func=cmd_diffusion)

    # --- pimd ---
    p = sub.add_parser("pimd", help="8. PIMD・量子熱浴")
    _add_common(p)
    p.add_argument("--method", default="pimd", choices=["pimd", "qtb"])
    p.add_argument("--beads", type=int, default=32)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--eco", type=float, default=None, help="Eco の最大波数 [cm^-1]")
    p.add_argument("--f-max", type=float, default=None, help="QTB のフィルタ上限 [1/ps]")
    p.add_argument("--dump-beads", type=int, default=None)
    p.set_defaults(func=cmd_pimd)

    # --- shock ---
    p = sub.add_parser("shock", help="9. 高圧・衝撃波")
    _add_common(p)
    p.add_argument("--method", default="msst", choices=["msst", "nphug", "piston"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--equilibrate", type=int, default=20000)
    p.add_argument("--vp", type=float, default=5.0, help="衝撃波速度 / ピストン速度 [km/s]")
    p.add_argument("--pressure", type=float, default=100.0, help="NPHug の目標圧力 [GPa]")
    p.add_argument("--direction", default="x", choices=["x", "y", "z"])
    p.add_argument("--wall", default="piston", choices=["piston", "mirror", "harmonic"])
    p.add_argument("--bin-size", type=float, default=10.0)
    p.set_defaults(func=cmd_shock)

    # --- ttm ---
    p = sub.add_parser("ttm", help="10. Two-Temperature Model")
    _add_common(p)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--equilibrate", type=int, default=20000)
    p.add_argument("--ce", type=float, default=1.0, help="電子比熱 [eV/K]")
    p.add_argument("--rho-e", type=float, default=1.0, help="電子数密度 [1/Å^3]")
    p.add_argument("--kappa-e", type=float, default=0.005, help="電子熱伝導率 [eV/(ps K Å)]")
    p.add_argument("--gamma-p", type=float, default=0.01, help="電子-フォノン結合 [amu/ps]")
    p.add_argument("--grid", type=int, nargs=3, default=[1, 1, 12])
    p.add_argument("--te", type=float, default=300.0, help="初期電子温度 [K]")
    p.add_argument("--source", type=float, default=None, help="体積熱源 [eV/(ps Å^3)]")
    p.add_argument("--out-interval", type=int, default=100)
    p.set_defaults(func=cmd_ttm)

    # --- mechanical ---
    p = sub.add_parser("mechanical", help="11. 引張・せん断などの変形")
    _add_common(p)
    p.add_argument("--mode", default="tensile", choices=["tensile", "shear"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--equilibrate", type=int, default=20000)
    p.add_argument("--strain-rate", type=float, default=1e8, help="ひずみ速度 [1/s]")
    p.add_argument("--axis", default="x", choices=["x", "y", "z"])
    p.add_argument("--rate", type=float, default=1e-5, help="せん断の変化率 [Å/step]")
    p.add_argument("--component", default="xy", choices=["xy", "xz", "yz"])
    p.set_defaults(func=cmd_mechanical)

    # --- active-learning ---
    p = sub.add_parser("active-learning", help="12. NEP の不確かさ・Active Learning")
    _add_common(p)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--temperature-end", type=float, default=None)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--threshold", type=float, default=0.05, help="committee のしきい値 [eV/Å]")
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--asi-file", default=None, help="外挿グレードを使う場合の ASI ファイル")
    p.add_argument("--gamma-low", type=float, default=5.0)
    p.add_argument("--gamma-high", type=float, default=10.0)
    p.set_defaults(func=cmd_active_learning)

    # --- analyze / convert / check / keywords ---
    p = sub.add_parser("analyze", help="計算ディレクトリの出力を解析・作図する")
    p.add_argument("workdir")
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("convert", help="構造・トラジェクトリを変換する")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument(
        "--to", default="xyz",
        help="出力形式。xyz (拡張 XYZ) と xdatcar が扱いやすい。"
             " 他に extxyz / lammps-dump / pdb / cif / netcdf / traj / vasp",
    )
    p.add_argument("--stride", type=int, default=1)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("check", help="GPUMD 実行環境を確認する")
    p.add_argument("--json", action="store_true")
    p.add_argument("--venv", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("keywords", help="対応している run.in キーワードを一覧する")
    p.set_defaults(func=cmd_keywords)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
