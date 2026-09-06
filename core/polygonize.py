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

def polygonize_gdal(labels, geotransform):
    """gdal.Polygonize でラベル配列をポリゴン化する."""
    from osgeo import gdal, ogr

    height, width = labels.shape
    array = np.ascontiguousarray(labels, dtype=np.int32)

    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", width, height, 2, gdal.GDT_Int32)
    if dataset is None:
        raise IOError("MEM ラスタを作成できません。")
    dataset.SetGeoTransform(geotransform)

    # バンド 1 = ラベル, バンド 2 = マスク (0 の背景を除外する)
    dataset.GetRasterBand(1).WriteRaster(
        0, 0, width, height, array.tobytes(), width, height, gdal.GDT_Int32)
    mask = np.ascontiguousarray((array > 0).astype(np.int32))
    dataset.GetRasterBand(2).WriteRaster(
        0, 0, width, height, mask.tobytes(), width, height, gdal.GDT_Int32)

    ogr_driver = ogr.GetDriverByName("Memory")
    source = ogr_driver.CreateDataSource("polygonize")
    layer = source.CreateLayer("crowns", None, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("label", ogr.OFTInteger))

    gdal.Polygonize(dataset.GetRasterBand(1), dataset.GetRasterBand(2),
                    layer, 0, [])

    result = {}
    layer.ResetReading()
    for feature in layer:
        label = feature.GetFieldAsInteger(0)
        if label <= 0:
            continue
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        result.setdefault(label, []).append(bytes(geometry.ExportToWkb()))

    layer = None
    source = None
    dataset = None

    return {label: multipolygon_wkb(parts) for label, parts in result.items()}


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


def polygonize_numpy(labels, geotransform):
    """自前実装によるポリゴン化 (GDAL が使えない環境向けの保険)."""
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
            polygons = [polygon_wkb(
                [_to_map(r, geotransform) for r in outer + inner])]
        else:
            # 連結成分が複数。穴の帰属は面積最大の外環にまとめる (近似)
            polygons = [polygon_wkb([_to_map(r, geotransform)])
                        for r in outer]
        result[label] = multipolygon_wkb(polygons)
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

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
