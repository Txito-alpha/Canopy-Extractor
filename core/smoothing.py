"""CHM の平滑化 (QGIS 非依存).

itcSegment の原著手順では, 局所最大値を探す前にローパスフィルタで CHM を平滑化して
偽の極大を減らす。lidR のドキュメント例でも 3x3 平均フィルタをかけている。

NoData の扱いが要点になる。素朴に畳み込むと NoData が周囲のセルに滲み出すので,
正規化畳み込み (normalized convolution) を使う:

    有効セルだけを 1, NoData を 0 とした重み w を用意し,
    smooth(値 * w) / smooth(w) を計算する。

これでラスタの端も NoData の縁も, 「実際に存在するセルだけの平均」になる。
ゼロ詰めした分母で割るので, 端のセルが暗くなるような偏りも生じない。

scipy があれば scipy.ndimage を使い, 無ければ numpy のみで同じ結果を出す。
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - 環境依存
    _ndimage = None

try:
    from numpy.lib.stride_tricks import sliding_window_view as _swv
except Exception:  # pragma: no cover - numpy < 1.20
    _swv = None


METHOD_NONE = "none"
METHOD_MEAN = "mean"
METHOD_GAUSSIAN = "gaussian"

# scipy.ndimage.gaussian_filter の既定値に合わせる
GAUSSIAN_TRUNCATE = 4.0


# ---------------------------------------------------------------------------
# カーネル半径
# ---------------------------------------------------------------------------

def gaussian_radius(sigma_cells, truncate=GAUSSIAN_TRUNCATE):
    """scipy.ndimage.gaussian_filter と同じ規則でカーネル半径を求める."""
    return int(truncate * float(sigma_cells) + 0.5)


def smoothing_radius(method, size=3, sigma_cells=1.0):
    """平滑化に必要なカーネル半径 (セル数) を返す.

    タイル処理の halo を決めるのに使う。
    """
    if method == METHOD_MEAN:
        return int(size) // 2
    if method == METHOD_GAUSSIAN:
        return gaussian_radius(sigma_cells)
    return 0


def gaussian_kernel1d(sigma_cells, truncate=GAUSSIAN_TRUNCATE):
    """1 次元ガウシアンカーネル. scipy の _gaussian_kernel1d と同じ定義."""
    radius = gaussian_radius(sigma_cells, truncate)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma_cells)) ** 2)
    kernel /= kernel.sum()
    return kernel


# ---------------------------------------------------------------------------
# 分離型フィルタ (numpy のみ)
# ---------------------------------------------------------------------------

def _box_sum_axis(array, size, axis):
    """ゼロ詰めした窓和. 端は「はみ出した分を 0 として数えない」挙動になる."""
    radius = size // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="constant", constant_values=0.0)

    if _swv is not None:
        windows = _swv(padded, size, axis=axis)
        return windows.sum(axis=-1)

    out = np.zeros_like(array)
    for offset in range(size):
        if axis == 0:
            out += padded[offset:offset + array.shape[0], :]
        else:
            out += padded[:, offset:offset + array.shape[1]]
    return out


def _convolve1d_axis(array, kernel, axis):
    """ゼロ詰めの 1 次元畳み込み."""
    radius = kernel.size // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="constant", constant_values=0.0)

    out = np.zeros_like(array)
    for offset, weight in enumerate(kernel):
        if weight == 0.0:
            continue
        if axis == 0:
            out += weight * padded[offset:offset + array.shape[0], :]
        else:
            out += weight * padded[:, offset:offset + array.shape[1]]
    return out


def _box_filter(array, size):
    if _ndimage is not None:
        # 平均ではなく和がほしいが, 正規化畳み込みでは分子分母で係数が相殺するため
        # uniform_filter (平均) のままでよい。
        return _ndimage.uniform_filter(
            array, size=size, mode="constant", cval=0.0)
    total = _box_sum_axis(_box_sum_axis(array, size, 0), size, 1)
    return total / float(size * size)


def _gaussian_filter(array, sigma_cells):
    if _ndimage is not None:
        return _ndimage.gaussian_filter(
            array, sigma=sigma_cells, mode="constant", cval=0.0,
            truncate=GAUSSIAN_TRUNCATE)
    kernel = gaussian_kernel1d(sigma_cells)
    return _convolve1d_axis(_convolve1d_axis(array, kernel, 0), kernel, 1)


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def smooth_chm(chm, valid, method, size=3, sigma_cells=1.0):
    """NoData を考慮して CHM を平滑化する.

    Args:
        chm: 2 次元配列
        valid: 有効セルの真偽配列 (None なら全て有効)
        method: METHOD_NONE / METHOD_MEAN / METHOD_GAUSSIAN
        size: 平均フィルタの窓サイズ (奇数)
        sigma_cells: ガウシアンの標準偏差 (セル数)

    Returns:
        平滑化した float32 配列。無効セルは NaN になる。
    """
    if method == METHOD_NONE:
        return np.asarray(chm, dtype=np.float32)

    if method == METHOD_MEAN and int(size) % 2 == 0:
        raise ValueError("平均フィルタの窓サイズは奇数で指定してください: %r" % (size,))
    if method == METHOD_GAUSSIAN and float(sigma_cells) <= 0.0:
        raise ValueError("ガウシアンの sigma は正の値で指定してください: %r"
                         % (sigma_cells,))

    values = np.asarray(chm, dtype=np.float64)
    if valid is None:
        weights = np.ones(values.shape, dtype=np.float64)
    else:
        weights = valid.astype(np.float64)

    masked = np.where(weights > 0.0, values, 0.0)

    if method == METHOD_MEAN:
        numerator = _box_filter(masked, int(size))
        denominator = _box_filter(weights, int(size))
    elif method == METHOD_GAUSSIAN:
        numerator = _gaussian_filter(masked, float(sigma_cells))
        denominator = _gaussian_filter(weights, float(sigma_cells))
    else:
        raise ValueError("未知の平滑化方式です: %r" % (method,))

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator > 1e-12, numerator / denominator, np.nan)
    return result.astype(np.float32)


def backend_description():
    """使用中の平滑化バックエンドを説明する文字列 (ログ出力用)."""
    return "scipy.ndimage" if _ndimage is not None else "numpy (分離型畳み込み)"
