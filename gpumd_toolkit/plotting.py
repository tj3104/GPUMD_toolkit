"""作図の共通設定。

本ツールキットの図はラベルに日本語を使う。環境に日本語フォントが無いと
matplotlib が豆腐 (□) を描いて大量の警告を出すので、
:func:`setup_matplotlib` で

* 日本語が出せるフォントがあればそれを使う
* 無ければラベルを英語に切り替える (:func:`label`)

という切り分けをする。
"""

from __future__ import annotations

import functools

__all__ = ["setup_matplotlib", "japanese_available", "label", "new_figure"]

#: 日本語が描けるフォントの候補 (上から順に探す)
_JP_FONTS = (
    "Noto Sans CJK JP",
    "Noto Serif CJK JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
    "Source Han Sans JP",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Hiragino Sans",
    "Droid Sans Fallback",
    "WenQuanYi Zen Hei",
)

#: 日本語が使えないときの差し替え (日本語 -> 英語)
_FALLBACK = {
    "温度": "Temperature",
    "時間": "Time",
    "ステップ": "Step",
    "位置": "Position",
    "強度 (任意単位)": "Intensity (arb. u.)",
    "波数": "Wavenumber",
    "振動数": "Frequency",
    "距離": "Distance",
    "体積": "Volume",
    "エネルギー": "Energy",
    "応力": "Stress",
    "ひずみ": "Strain",
    "濃度": "Concentration",
    "出現回数": "Count",
    "相関時間": "Correlation time",
    "出力回数": "Output index",
    "各ブロック": "blocks",
    "勾配": "gradient",
    "電子": "electron",
    "格子": "lattice",
    "せん断": "shear",
    "体積粘性": "bulk",
    "累積平均": "cumulative mean",
    "しきい値": "threshold",
    "フォノン分散": "Phonon dispersion",
    "凝集エネルギー曲線": "Cohesive energy curve",
    "高対称線に沿った距離": "Distance along high-symmetry path",
    "熱伝導率の収束": "Thermal conductivity convergence",
    "NEMD 温度プロファイル": "NEMD temperature profile",
    "衝撃波の空間分布": "Shock wave spatial profiles",
    "二温度モデル": "Two-temperature model",
    "拡散・液体物性": "Diffusion / liquid properties",
    "応力-ひずみ": "Stress-strain",
    "赤外スペクトル": "Infrared spectrum",
    "ラマンスペクトル": "Raman spectrum",
    "committee 不確かさ": "Committee uncertainty",
    "方向のグリッド番号": " grid index",
    "受理率": "acceptance ratio",
    "セル固定": "fixed cell",
}


@functools.lru_cache(maxsize=1)
def japanese_available() -> str | None:
    """日本語が描けるフォント名を返す (無ければ ``None``)。"""
    try:
        import matplotlib.font_manager as fm
    except ImportError:  # pragma: no cover
        return None
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _JP_FONTS:
        if name in installed:
            return name
    return None


@functools.lru_cache(maxsize=1)
def setup_matplotlib() -> str | None:
    """matplotlib を Agg + 日本語フォントで初期化し、使うフォント名を返す。

    何度呼んでも 1 回しか効かない。
    """
    import matplotlib

    matplotlib.use("Agg")
    font = japanese_available()
    if font:
        # matplotlib 3.6+ は font.family をリストにするとグリフ単位で
        # 後ろのフォントにフォールバックする。ラテン文字と数式は DejaVu Sans に
        # 任せ、日本語だけ CJK フォントに落とすのが一番きれいに出る。
        matplotlib.rcParams["font.family"] = ["DejaVu Sans", font]
        matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
        matplotlib.rcParams["axes.unicode_minus"] = False
    return font


def label(text: str) -> str:
    """日本語フォントが無い環境では英語に差し替える。"""
    if japanese_available():
        return text
    for japanese, english in _FALLBACK.items():
        text = text.replace(japanese, english)
    return text


def new_figure(*args, **kwargs):
    """:func:`setup_matplotlib` を済ませてから ``plt.subplots`` を呼ぶ。"""
    setup_matplotlib()
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)
