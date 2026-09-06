"""ラベル配列をポリゴン (WKB) に変換する (QGIS 非依存).

主経路は gdal.Polygonize。MEM ドライバのラスタに WriteRaster() でバイト列を
渡すので, osgeo.gdal_array を一切経由しない。numpy の ABI 不一致で
ReadAsArray が落ちる環境でも動く。

フォールバックはセル境界の辺を追跡する自前実装。GDAL が使えない環境向けの
保険で, Python ループなので大量の樹冠では遅い。

どちらの経路も {ラベル: WKB バイト列} を返す。座標は geotransform を適用した
地図座標。複数の連結成分に分かれたラベルは MultiPolygon にまとめる。
"""

from __future__ import annotations

import struct

import numpy as np

WKB_NDR = 1  # リトルエンディアン
WKB_POLYGON = 3
WKB_MULTIPOLYGON = 6


# ---------------------------------------------------------------------------
# WKB 組み立て
# ---------------------------------------------------------------------------

def polygon_wkb(rings):
    """外環 + 内環のリストから Polygon の WKB を組み立てる."""
    parts = [struct.pack("<BII", WKB_NDR, WKB_POLYGON, len(rings))]
    for ring in rings:
        parts.append(struct.pack("<I", len(ring)))
        parts.append(np.asarray(ring, dtype="<f8").tobytes())
    return b"".join(parts)


def multipolygon_wkb(polygons):
    """Polygon の WKB 列から MultiPolygon の WKB を組み立てる."""
    if len(polygons) == 1:
        return polygons[0]
    parts = [struct.pack("<BII", WKB_NDR, WKB_MULTIPOLYGON, len(polygons))]
    parts.extend(polygons)
    return b"".join(parts)


# ---------------------------------------------------------------------------
# GDAL 経路
# ---------------------------------------------------------------------------

def _polygonize_gdal_features(labels, geotransform):
    """gdal.Polygonize を実行し (ラベル, OGR ジオメトリ) のリストを返す.

    黙って 0 件を返すのが一番困るので, ラベルがあるのに何も出てこなければ
    例外にして呼び出し側で自前実装に切り替えられるようにする。
    マスクバンドは Byte で作る (GDAL のマスクは Byte が前提)。
    """
    from osgeo import gdal, ogr

    height, width = labels.shape
    array = np.ascontiguousarray(labels, dtype=np.int32)
    expected = int((array > 0).sum())

    driver = gdal.GetDriverByName("MEM")
    if driver is None:
        raise IOError("MEM ドライバが使えません。")

    label_ds = driver.Create("", width, height, 1, gdal.GDT_Int32)
    mask_ds = driver.Create("", width, height, 1, gdal.GDT_Byte)
    if label_ds is None or mask_ds is None:
        raise IOError("MEM ラスタを作成できません。")
    label_ds.SetGeoTransform(geotransform)
    mask_ds.SetGeoTransform(geotransform)

    label_ds.GetRasterBand(1).WriteRaster(
        0, 0, width, height, array.tobytes(), width, height, gdal.GDT_Int32)
    mask = np.ascontiguousarray((array > 0).astype(np.uint8))
    mask_ds.GetRasterBand(1).WriteRaster(
        0, 0, width, height, mask.tobytes(), width, height, gdal.GDT_Byte)

    ogr_driver = ogr.GetDriverByName("Memory")
    if ogr_driver is None:
        raise IOError("OGR Memory ドライバが使えません。")

    features = _run_polygonize(
        ogr_driver, label_ds.GetRasterBand(1), mask_ds.GetRasterBand(1))

    if not features and expected > 0:
        # マスクバンドが効いていない可能性があるので, マスク無しで再試行する。
        # 背景も出てくるが, ラベル 0 として捨てる。
        features = _run_polygonize(
            ogr_driver, label_ds.GetRasterBand(1), None)

    label_ds = None
    mask_ds = None

    if not features and expected > 0:
        raise IOError(
            "gdal.Polygonize が 1 件も返しませんでした "
            "(割当セル %d)" % expected)
    return features


def _run_polygonize(ogr_driver, label_band, mask_band):
    from osgeo import gdal, ogr

    source = ogr_driver.CreateDataSource("polygonize")
    layer = source.CreateLayer("crowns", None, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("label", ogr.OFTInteger))

    status = gdal.Polygonize(label_band, mask_band, layer, 0, [])
    if status != 0:
        raise IOError("gdal.Polygonize が失敗しました (code %s)" % status)

    features = []
    layer.ResetReading()
    for feature in layer:
        label = feature.GetFieldAsInteger(0)
        if label <= 0:
            continue
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        features.append((label, geometry.Clone()))

    layer = None
    source = None
    return features


def polygonize_gdal(labels, geotransform):
    """gdal.Polygonize でラベル配列をポリゴン化する (WKB を返す)."""
    result = {}
    for label, geometry in _polygonize_gdal_features(labels, geotransform):
        # 明示的にリトルエンディアンで出力する (既定は big endian)
        result.setdefault(label, []).append(
            bytes(geometry.ExportToWkb(byte_order=1)))
    return {label: multipolygon_wkb(parts)
            for label, parts in result.items()}


# ---------------------------------------------------------------------------
# 自前の境界追跡 (フォールバック)
# ---------------------------------------------------------------------------

def _trace_rings(cells, height, width):
    """同一ラベルのセル集合から, セル境界の環を取り出す.

    各セルの 4 辺のうち, 隣が同じラベルでない辺だけを残し, 有向辺として繋ぐ。
    辺の向きは「セルの内側を左に見る」向きで統一するので, 外環と内環が
    符号付き面積で区別できる。
    """
    cell_set = set(cells)
    edges = {}
    for row, col in cells:
        if (row - 1, col) not in cell_set:
            edges.setdefault((col, row), []).append((col + 1, row))
        if (row, col + 1) not in cell_set:
            edges.setdefault((col + 1, row), []).append((col + 1, row + 1))
        if (row + 1, col) not in cell_set:
            edges.setdefault((col + 1, row + 1), []).append((col, row + 1))
        if (row, col - 1) not in cell_set:
            edges.setdefault((col, row + 1), []).append((col, row))

    rings = []
    while edges:
        start = next(iter(edges))
        ring = [start]
        point = start
        while True:
            targets = edges.get(point)
            if not targets:
                break
            nxt = targets.pop()
            if not targets:
                del edges[point]
            ring.append(nxt)
            point = nxt
            if point == start:
                break
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def _signed_area(ring):
    total = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def _to_map(ring, geotransform):
    gt = geotransform
    return [(gt[0] + x * gt[1] + y * gt[2], gt[3] + x * gt[4] + y * gt[5])
            for x, y in ring]


def polygonize_numpy_rings(labels, geotransform):
    """自前実装. {ラベル: [ポリゴン, ...]} を返す.

    ポリゴンは環のリストで, 先頭が外環, 以降が内環 (穴)。
    環は (x, y) タプルのリスト (地図座標)。
    """
    height, width = labels.shape
    rows, cols = np.nonzero(labels)
    if rows.size == 0:
        return {}

    values = labels[rows, cols]
    order = np.argsort(values, kind="stable")
    rows, cols, values = rows[order], cols[order], values[order]

    boundaries = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [values.size]])

    result = {}
    for start, end in zip(starts, ends):
        label = int(values[start])
        cells = list(zip(rows[start:end].tolist(), cols[start:end].tolist()))
        rings = _trace_rings(cells, height, width)
        if not rings:
            continue

        outer = []
        inner = []
        for ring in rings:
            (outer if _signed_area(ring) > 0 else inner).append(ring)
        if not outer:
            # 向きの想定が崩れた場合は全部を外環として扱う
            outer, inner = rings, []

        if len(outer) == 1:
            polygons = [[_to_map(r, geotransform) for r in outer + inner]]
        else:
            # 連結成分が複数。穴は落とす (近似)
            polygons = [[_to_map(r, geotransform)] for r in outer]
        result[label] = polygons
    return result


def polygonize_numpy(labels, geotransform):
    """自前実装によるポリゴン化 (WKB を返す)."""
    rings = polygonize_numpy_rings(labels, geotransform)
    return {label: multipolygon_wkb([polygon_wkb(p) for p in polygons])
            for label, polygons in rings.items()}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def polygonize_gdal_rings(labels, geotransform):
    """gdal.Polygonize の結果を環の座標列として返す (WKB を経由しない)."""
    from osgeo import ogr

    result = {}
    for label, geometry in _polygonize_gdal_features(labels, geotransform):
        polygons = result.setdefault(label, [])
        if geometry.GetGeometryType() in (ogr.wkbMultiPolygon,
                                          ogr.wkbMultiPolygon25D):
            parts = [geometry.GetGeometryRef(i)
                     for i in range(geometry.GetGeometryCount())]
        else:
            parts = [geometry]
        for part in parts:
            rings = []
            for i in range(part.GetGeometryCount()):
                ring = part.GetGeometryRef(i)
                rings.append([(p[0], p[1]) for p in ring.GetPoints()])
            if rings:
                polygons.append(rings)
    return result


def polygonize_rings(labels, geotransform, prefer_gdal=True):
    """ラベル配列を環の座標列に変換する. (結果, 使用経路名) を返す.

    WKB を経由しないので, WKB の解釈で問題が起きる環境でも使える。
    """
    if prefer_gdal:
        try:
            return (polygonize_gdal_rings(labels, geotransform),
                    "gdal.Polygonize (環)")
        except Exception as error:  # noqa: BLE001 - 意図的に拾って自前実装へ
            reason = "%s: %s" % (type(error).__name__, error)
            return (polygonize_numpy_rings(labels, geotransform),
                    "numpy 境界追跡 (環, GDAL 失敗: %s)" % reason)
    return polygonize_numpy_rings(labels, geotransform), "numpy 境界追跡 (環)"


def polygonize(labels, geotransform, prefer_gdal=True):
    """ラベル配列をポリゴン化する. (結果, 使用経路名) を返す."""
    if prefer_gdal:
        try:
            return polygonize_gdal(labels, geotransform), "gdal.Polygonize"
        except Exception as error:  # noqa: BLE001 - 意図的に拾って自前実装へ
            reason = "%s: %s" % (type(error).__name__, error)
            return (polygonize_numpy(labels, geotransform),
                    "numpy (境界追跡, GDAL 失敗: %s)" % reason)
    return polygonize_numpy(labels, geotransform), "numpy (境界追跡)"
