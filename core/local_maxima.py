"""CHM の局所最大値による樹頂点抽出コア.

QGIS に依存しない純粋な numpy 実装。単体テスト・ベンチマークがそのまま走る。

元の Processing モデル (樹頂点抽出0_5_0.model3) の

    r.neighbors(max, size=7)
      -> gdal:rastercalculator (A == B)
      -> gdal:polygonize
      -> native:centroids
      -> native:rastersampling
      -> native:extractbyattribute (height >= 6)

を 1 パスの配列演算に置き換えたもの。中間ファイルは一切生成しない。

scipy があれば使い、無ければ numpy のみで同じ結果を出す。
"""

from __future__ import annotations

import numpy as np

# scipy は必須ではない。ImportError だけでなく, numpy との ABI 不一致による
# 実行時エラーもありうるので Exception 全体を拾って numpy 経路に落とす。
try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - 環境依存
    _ndimage = None

try:
    from scipy.sparse import coo_matrix as _coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
except Exception:  # pragma: no cover - 環境依存
    _coo_matrix = None
    _cc = None

try:
    from numpy.lib.stride_tricks import sliding_window_view as _swv
except Exception:  # pragma: no cover - numpy < 1.20
    _swv = None


HAS_SCIPY = _ndimage is not None


# ---------------------------------------------------------------------------
# 最大値フィルタ
# ---------------------------------------------------------------------------

def maximum_filter(array, size):
    """size x size の最大値フィルタ. 端は 'nearest' (端値の複製) で処理する.

    scipy.ndimage.maximum_filter と同じ結果を返す。
    """
    if size <= 1:
        return array.copy()
    if _ndimage is not None:
        return _ndimage.maximum_filter(array, size=size, mode="nearest")
    return _maximum_filter_numpy(array, size)


def _maximum_filter_numpy(array, size):
    """分離可能な最大値フィルタ (行方向 -> 列方向).

    2 次元 size x size の最大値は 1 次元 size の最大値を 2 回かければ得られる。
    O(size^2) ではなく O(2 * size) パスで済む。
    """
    tmp = _max_along_axis(array, size, axis=0)
    return _max_along_axis(tmp, size, axis=1)


def _max_along_axis(array, size, axis):
    radius = size // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="edge")

    if _swv is not None:
        windows = _swv(padded, size, axis=axis)
        return windows.max(axis=-1)

    # numpy < 1.20 用フォールバック: シフトしながら累積 max
    out = None
    for offset in range(size):
        if axis == 0:
            chunk = padded[offset:offset + array.shape[0], :]
        else:
            chunk = padded[:, offset:offset + array.shape[1]]
        out = chunk.copy() if out is None else np.maximum(out, chunk)
    return out


# ---------------------------------------------------------------------------
# 候補セルの抽出
# ---------------------------------------------------------------------------

def local_maxima_mask(chm, window_size=7, min_height=None, nodata=None,
                      valid=None):
    """局所最大セルの真偽マスクを返す.

    Args:
        chm: 2 次元配列 (極大を探す対象の面。平滑化後の CHM でもよい)
        window_size: 近傍窓のセル数 (奇数)
        min_height: この値未満のセルは候補にしない (None で無効)
        nodata: NoData 値 (None で無効). NaN は常に NoData 扱い
        valid: 有効セルの真偽配列。平滑化した面を渡すときに, 元 CHM 由来の
            有効セル判定を明示的に渡すために使う (None なら chm から求める)

    Returns:
        bool の 2 次元配列
    """
    if window_size % 2 == 0:
        raise ValueError("window_size は奇数で指定してください: %r" % (window_size,))

    values = np.asarray(chm, dtype=np.float32)

    computed_valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        computed_valid &= values != np.float32(nodata)
    if valid is not None:
        computed_valid &= valid
    valid = computed_valid

    # NoData を -inf に落としてから最大値フィルタをかける。
    # これで NoData が近傍最大値を汚さない。
    work = np.where(valid, values, np.float32(-np.inf))
    neighborhood_max = maximum_filter(work, window_size)

    mask = valid & (work >= neighborhood_max)
    if min_height is not None:
        mask &= values >= np.float32(min_height)
    return mask


def valid_mask(chm, nodata=None):
    """有効セル (NaN でも NoData でもない) の真偽配列を返す."""
    values = np.asarray(chm, dtype=np.float32)
    result = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        result &= values != np.float32(nodata)
    return result


# ---------------------------------------------------------------------------
# 平坦部 (プラトー) の統合
# ---------------------------------------------------------------------------

def cluster_candidates(rows, cols):
    """8 近傍で連結する候補セルをグループ化し, 各セルのラベルを返す.

    窓が 3 以上あれば, 隣接する候補セル同士は必ず同値 (プラトー) になる。
    値が異なれば小さいほうが相手の窓に入って候補から落ちるため。
    したがって値を見ずに隣接判定だけでよい。

    元モデルの polygonize -> centroids に対応する処理。
    """
    n = rows.size
    if n == 0:
        return np.empty(0, dtype=np.int64), 0

    keys = rows.astype(np.int64) * (2 ** 32) + cols.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    # 4 方向だけ見れば 8 近傍の連結は網羅できる (逆向きは対称)
    edges_a = []
    edges_b = []
    for d_row, d_col in ((0, 1), (1, -1), (1, 0), (1, 1)):
        neighbor_keys = (rows.astype(np.int64) + d_row) * (2 ** 32) \
            + (cols.astype(np.int64) + d_col)
        pos = np.searchsorted(sorted_keys, neighbor_keys)
        pos_clipped = np.clip(pos, 0, n - 1)
        hit = sorted_keys[pos_clipped] == neighbor_keys
        if not np.any(hit):
            continue
        edges_a.append(np.nonzero(hit)[0])
        edges_b.append(order[pos_clipped[hit]])

    if not edges_a:
        return np.arange(n, dtype=np.int64), n

    edge_a = np.concatenate(edges_a)
    edge_b = np.concatenate(edges_b)

    if _cc is not None and _coo_matrix is not None:
        graph = _coo_matrix(
            (np.ones(edge_a.size, dtype=np.int8), (edge_a, edge_b)),
            shape=(n, n),
        )
        count, labels = _cc(graph, directed=False)
        return labels.astype(np.int64), count

    return _label_propagation(edge_a, edge_b, n)


def _label_propagation(edge_a, edge_b, n):
    """scipy 無し用の連結成分ラベリング.

    プラトーは数セル程度なのでクラスタ径が小さく, 数回で収束する。
    """
    labels = np.arange(n, dtype=np.int64)
    for _ in range(64):
        previous = labels.copy()
        np.minimum.at(labels, edge_a, previous[edge_b])
        np.minimum.at(labels, edge_b, previous[edge_a])
        labels = np.minimum(labels, labels[labels])
        if np.array_equal(labels, previous):
            break
    unique, compacted = np.unique(labels, return_inverse=True)
    return compacted.astype(np.int64), unique.size


# ---------------------------------------------------------------------------
# メイン API
# ---------------------------------------------------------------------------

def extract_tree_tops(chm, window_size=7, min_height=6.0, nodata=None,
                      merge_plateaus=True):
    """CHM 配列から樹頂点を抽出する.

    Returns:
        (rows, cols, heights, cell_counts) のタプル。
        rows / cols はセル座標 (プラトー統合時は重心なので実数)。
        heights はその樹頂点の CHM 値。
        cell_counts はプラトーを構成したセル数。
    """
    values = np.asarray(chm, dtype=np.float32)
    mask = local_maxima_mask(values, window_size, min_height, nodata)

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return empty_f, empty_f, empty_f, np.empty(0, dtype=np.int32)

    heights = values[rows, cols].astype(np.float64)

    if not merge_plateaus:
        return (rows.astype(np.float64), cols.astype(np.float64), heights,
                np.ones(rows.size, dtype=np.int32))

    labels, count = cluster_candidates(rows, cols)
    cell_counts = np.bincount(labels, minlength=count).astype(np.int32)
    row_centroid = np.bincount(labels, weights=rows, minlength=count) / cell_counts
    col_centroid = np.bincount(labels, weights=cols, minlength=count) / cell_counts
    # プラトー内は同値なので max でも mean でも同じ。max のほうが頑健。
    height_out = np.zeros(count, dtype=np.float64)
    np.maximum.at(height_out, labels, heights)

    return row_centroid, col_centroid, height_out, cell_counts


def cell_to_map(rows, cols, geotransform):
    """セル座標 (行, 列) を地図座標 (x, y) に変換する.

    セル中心を返すので +0.5 を足す。geotransform は GDAL 形式の 6 要素。
    """
    gt = geotransform
    px = cols + 0.5
    py = rows + 0.5
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return x, y


def backend_description():
    """使用中の計算バックエンドを説明する文字列 (ログ出力用)."""
    filter_backend = "scipy.ndimage" if _ndimage is not None else (
        "numpy (sliding_window_view)" if _swv is not None else "numpy (shift)")
    cluster_backend = ("scipy.sparse.csgraph" if _cc is not None
                       else "numpy (label propagation)")
    return "最大値フィルタ: %s / 連結成分: %s" % (filter_backend, cluster_backend)
