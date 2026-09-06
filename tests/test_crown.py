"""樹冠切り出しの検証.

    python -m unittest canopy_extractor.tests.test_crown -v
"""

from __future__ import annotations

import struct
import sys
import types
import unittest

import numpy as np

from ..core import crown, polygonize, raster_reader

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _ring_area(ring):
    points = np.asarray(ring, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0)


def reference_grow(chm, seed_rows, seed_cols, th_tree=2.0, th_seed=0.45,
                   th_cr=0.55, max_cr=10):
    """PyCrown の Cython 実装と同じ規則を素直に書いた参照版 (逐次, 遅い)."""
    height, width = chm.shape
    n_seed = seed_rows.size
    labels = np.zeros(chm.shape, dtype=np.int32)
    for i in range(n_seed):
        labels[seed_rows[i], seed_cols[i]] = i + 1

    seed_h = np.zeros(n_seed + 1)
    seed_r = np.zeros(n_seed + 1, dtype=int)
    seed_c = np.zeros(n_seed + 1, dtype=int)
    sum_h = np.zeros(n_seed + 1)
    n_px = np.ones(n_seed + 1)
    for i in range(n_seed):
        seed_h[i + 1] = chm[seed_rows[i], seed_cols[i]]
        seed_r[i + 1] = seed_rows[i]
        seed_c[i + 1] = seed_cols[i]
        sum_h[i + 1] = seed_h[i + 1]

    for _ in range(max_cr * 2):
        temp = labels.copy()
        grown = False
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                tid = labels[row, col]
                if tid == 0:
                    continue
                mean_h = sum_h[tid] / n_px[tid]
                for d_row, d_col in NEIGHBOURS:
                    near_row, near_col = row + d_row, col + d_col
                    near_h = chm[near_row, near_col]
                    if (near_h > th_tree and temp[near_row, near_col] == 0
                            and near_h > seed_h[tid] * th_seed
                            and near_h > mean_h * th_cr
                            and near_h <= seed_h[tid] * 1.05
                            and abs(seed_c[tid] - near_col) < max_cr
                            and abs(seed_r[tid] - near_row) < max_cr):
                        temp[near_row, near_col] = tid
                        n_px[tid] += 1
                        sum_h[tid] += near_h
                        grown = True
        labels = temp
        if not grown:
            break
    return labels


def synthetic_stand(size=200, spacing=14, crown_radius=6.0, seed=0):
    """円錐樹冠を格子状に並べた合成 CHM."""
    rng = np.random.default_rng(seed)
    grid_r, grid_c = np.ogrid[:size, :size]
    chm = np.zeros((size, size), dtype=np.float32)
    rows, cols = [], []
    for row in range(spacing, size - spacing, spacing):
        for col in range(spacing, size - spacing, spacing):
            r0 = row + int(rng.integers(-2, 3))
            c0 = col + int(rng.integers(-2, 3))
            height = 18.0 + float(rng.random()) * 8.0
            dist = np.sqrt((grid_r - r0) ** 2 + (grid_c - c0) ** 2)
            cone = np.clip(height * (1.0 - dist / crown_radius), 0.0, None)
            chm = np.maximum(chm, cone.astype(np.float32))
            rows.append(r0)
            cols.append(c0)
    return chm, np.array(rows), np.array(cols)


class TestRegionGrowing(unittest.TestCase):

    def test_matches_sequential_reference(self):
        """逐次参照実装とのラベル一致率が 99% を超えること."""
        chm, rows, cols = synthetic_stand(150, 14)
        fast = crown.grow_region(
            chm, rows, cols, max_cr=10, shape=crown.SHAPE_SQUARE)
        slow = reference_grow(chm, rows, cols, max_cr=10)

        both = (fast > 0) & (slow > 0)
        agreement = float((fast[both] == slow[both]).mean())
        self.assertGreater(agreement, 0.99)

        area_ratio = float((fast > 0).sum()) / float((slow > 0).sum())
        self.assertAlmostEqual(area_ratio, 1.0, places=2)

    def test_single_tree_is_bounded_by_max_cr(self):
        """樹冠が種から max_cr を超えて広がらないこと (タイル処理の前提)."""
        chm = np.full((60, 60), 20.0, dtype=np.float32)
        rows = np.array([30])
        cols = np.array([30])
        max_cr = 8
        labels = crown.grow_region(chm, rows, cols, max_cr=max_cr,
                                   shape=crown.SHAPE_SQUARE)
        assigned_r, assigned_c = np.nonzero(labels)
        self.assertLess(int(np.abs(assigned_r - 30).max()), max_cr)
        self.assertLess(int(np.abs(assigned_c - 30).max()), max_cr)

    def test_circle_is_subset_of_square(self):
        chm, rows, cols = synthetic_stand(120, 14)
        circle = crown.grow_region(chm, rows, cols, max_cr=10,
                                   shape=crown.SHAPE_CIRCLE)
        square = crown.grow_region(chm, rows, cols, max_cr=10,
                                   shape=crown.SHAPE_SQUARE)
        self.assertLessEqual(int((circle > 0).sum()), int((square > 0).sum()))

    def test_low_neighbour_is_excluded(self):
        """th_seed を下回るセルが樹冠に入らないこと."""
        chm = np.full((21, 21), 20.0, dtype=np.float32)
        chm[10, 15:] = 5.0  # 種高の 25% しかない帯
        labels = crown.grow_region(
            np.ascontiguousarray(chm), np.array([10]), np.array([10]),
            th_seed=0.45, max_cr=10, shape=crown.SHAPE_SQUARE)
        self.assertEqual(int(labels[10, 16]), 0)

    def test_no_seeds_returns_empty(self):
        chm = np.full((10, 10), 20.0, dtype=np.float32)
        labels = crown.grow_region(
            chm, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        self.assertEqual(int(labels.sum()), 0)


class TestVoronoi(unittest.TestCase):

    def test_assigns_to_nearest_seed(self):
        chm = np.full((41, 41), 20.0, dtype=np.float32)
        rows = np.array([20, 20])
        cols = np.array([10, 30])
        labels = crown.voronoi_crowns(chm, rows, cols, max_cr=15,
                                      exclusion=0.0, th_tree=0.0)
        self.assertEqual(int(labels[20, 12]), 1)
        self.assertEqual(int(labels[20, 28]), 2)

    def test_exclusion_removes_low_cells(self):
        chm = np.full((31, 31), 20.0, dtype=np.float32)
        chm[15, 20:] = 4.0  # 種高の 20%
        labels = crown.voronoi_crowns(
            np.ascontiguousarray(chm), np.array([15]), np.array([15]),
            max_cr=12, exclusion=0.3, th_tree=0.0)
        self.assertEqual(int(labels[15, 21]), 0)

    def test_bounded_by_max_cr(self):
        chm = np.full((61, 61), 20.0, dtype=np.float32)
        labels = crown.voronoi_crowns(
            chm, np.array([30]), np.array([30]), max_cr=8,
            exclusion=0.0, th_tree=0.0)
        rows, cols = np.nonzero(labels)
        self.assertLess(int(np.abs(rows - 30).max()), 8)
        self.assertLess(int(np.abs(cols - 30).max()), 8)


class TestTiling(unittest.TestCase):
    """タイル分割が全域処理と一致すること.

    halo は 2 * max_cr + 1 が必要。あるセルを奪いうる競合の種は種から
    最大 2 * max_cr 離れているため (セルは種から max_cr 以内, 競合は
    そのセルから max_cr 以内)。halo = max_cr では密な林分でずれる。

    種の並び順も結果に影響する (同距離の競合はラベル番号の小さいほうが勝つ)
    ので, 全域処理と同じ行順に揃えて比較する。
    """

    @staticmethod
    def _tiled(chm, seed_rows, seed_cols, max_cr, halo, block_size, method):
        ids = np.arange(1, seed_rows.size + 1, dtype=np.int64)
        result = np.zeros(chm.shape, dtype=np.int32)

        for window in raster_reader.iter_blocks(
                chm.shape[1], chm.shape[0], block_size, halo):
            lo = np.searchsorted(seed_rows, window.read_row, side="left")
            hi = np.searchsorted(
                seed_rows, window.read_row + window.read_height, side="left")
            if lo == hi:
                continue
            inside = ((seed_cols[lo:hi] >= window.read_col)
                      & (seed_cols[lo:hi]
                         < window.read_col + window.read_width))
            local_row = seed_rows[lo:hi][inside] - window.read_row
            local_col = seed_cols[lo:hi][inside] - window.read_col
            local_id = ids[lo:hi][inside]
            if local_row.size == 0:
                continue

            block = chm[
                window.read_row:window.read_row + window.read_height,
                window.read_col:window.read_col + window.read_width]
            labels = method(block, local_row, local_col, max_cr)

            core_row0 = window.core_row - window.read_row
            core_col0 = window.core_col - window.read_col
            keep = ((local_row >= core_row0)
                    & (local_row < core_row0 + window.core_height)
                    & (local_col >= core_col0)
                    & (local_col < core_col0 + window.core_width))
            lookup = np.zeros(local_row.size + 1, dtype=np.int64)
            lookup[np.nonzero(keep)[0] + 1] = local_id[keep]
            mapped = lookup[labels]

            target = result[
                window.read_row:window.read_row + window.read_height,
                window.read_col:window.read_col + window.read_width]
            np.copyto(target, mapped.astype(np.int32), where=mapped > 0)
        return result

    @staticmethod
    def _region(block, rows, cols, max_cr):
        return crown.grow_region(block, rows, cols, max_cr=max_cr)

    @staticmethod
    def _voronoi(block, rows, cols, max_cr):
        return crown.voronoi_crowns(block, rows, cols, max_cr=max_cr)

    def _check(self, method, spacing, seed):
        chm, rows, cols = synthetic_stand(
            240, spacing, crown_radius=9.0, seed=seed)
        order = np.argsort(rows, kind="stable")
        seed_rows, seed_cols = rows[order], cols[order]

        max_cr = 8
        whole = method(chm, seed_rows, seed_cols, max_cr)
        tiled = self._tiled(
            chm, seed_rows, seed_cols, max_cr, 2 * max_cr + 1, 64, method)
        np.testing.assert_array_equal(tiled, whole)

    def test_region_growing_sparse(self):
        for seed in (0, 1, 2):
            self._check(self._region, 14, seed)

    def test_region_growing_dense(self):
        for seed in (0, 1, 2, 3, 4):
            self._check(self._region, 10, seed)

    def test_voronoi_sparse(self):
        for seed in (0, 1, 2):
            self._check(self._voronoi, 14, seed)

    def test_voronoi_dense(self):
        for seed in (0, 1, 2, 3, 4):
            self._check(self._voronoi, 10, seed)

    def test_insufficient_halo_is_detected(self):
        """halo = max_cr では一致しないこと (定数の根拠を残す回帰テスト)."""
        chm, rows, cols = synthetic_stand(240, 10, crown_radius=9.0, seed=1)
        order = np.argsort(rows, kind="stable")
        seed_rows, seed_cols = rows[order], cols[order]

        max_cr = 8
        whole = self._voronoi(chm, seed_rows, seed_cols, max_cr)
        too_small = self._tiled(
            chm, seed_rows, seed_cols, max_cr, max_cr + 1, 64, self._voronoi)
        self.assertGreater(int((too_small != whole).sum()), 0)


class TestFillHoles(unittest.TestCase):

    def test_fills_enclosed_hole(self):
        if crown._ndimage is None:
            self.skipTest("scipy 無し")
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[3:8, 3:8] = 1
        labels[5, 5] = 0  # 内側に穴
        filled = crown.fill_holes(labels)
        self.assertEqual(int(filled[5, 5]), 1)

    def test_keeps_background(self):
        if crown._ndimage is None:
            self.skipTest("scipy 無し")
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[3:8, 3:8] = 1
        filled = crown.fill_holes(labels)
        self.assertEqual(int(filled[0, 0]), 0)


class TestCrownStatistics(unittest.TestCase):

    def test_counts_and_heights(self):
        chm = np.array([[10.0, 12.0], [8.0, 0.0]], dtype=np.float32)
        labels = np.array([[1, 1], [1, 0]], dtype=np.int32)
        counts, maxima, means = crown.crown_statistics(chm, labels, 1)
        self.assertEqual(int(counts[1]), 3)
        self.assertAlmostEqual(float(maxima[1]), 12.0)
        self.assertAlmostEqual(float(means[1]), 10.0)


class TestPolygonize(unittest.TestCase):
    """自前の境界追跡の検証 (GDAL が無い環境向けの保険経路)."""

    GT = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)

    @staticmethod
    def _rings_from_wkb(wkb):
        """Polygon / MultiPolygon の WKB から環の座標列を取り出す."""
        order, kind = struct.unpack_from("<BI", wkb, 0)
        if order != 1:
            raise ValueError("リトルエンディアンの WKB ではありません")
        offset = 5
        if kind == polygonize.WKB_MULTIPOLYGON:
            (count,) = struct.unpack_from("<I", wkb, offset)
            offset += 4
            rings = []
            for _ in range(count):
                sub_rings, offset = TestPolygonize._read_polygon(wkb, offset)
                rings.extend(sub_rings)
            return rings
        rings, _ = TestPolygonize._read_polygon(wkb, 0)
        return rings

    @staticmethod
    def _read_polygon(wkb, offset):
        order, kind = struct.unpack_from("<BI", wkb, offset)
        if order != 1 or kind != polygonize.WKB_POLYGON:
            raise ValueError("Polygon の WKB ではありません: %r" % (kind,))
        offset += 5
        (n_rings,) = struct.unpack_from("<I", wkb, offset)
        offset += 4
        rings = []
        for _ in range(n_rings):
            (n_points,) = struct.unpack_from("<I", wkb, offset)
            offset += 4
            points = np.frombuffer(
                wkb, dtype="<f8", count=n_points * 2, offset=offset)
            offset += n_points * 16
            rings.append(points.reshape(n_points, 2))
        return rings, offset

    @staticmethod
    def _ring_area(ring):
        x, y = ring[:, 0], ring[:, 1]
        return abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0)

    def test_single_square(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[2:5, 3:7] = 1  # 3 行 x 4 列 = 12 セル
        shapes = polygonize.polygonize_numpy(labels, self.GT)
        self.assertEqual(set(shapes), {1})
        rings = self._rings_from_wkb(shapes[1])
        self.assertEqual(len(rings), 1)
        self.assertAlmostEqual(self._ring_area(rings[0]), 12.0, places=6)

    def test_ring_is_closed(self):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:5, 2:5] = 1
        rings = self._rings_from_wkb(
            polygonize.polygonize_numpy(labels, self.GT)[1])
        for ring in rings:
            np.testing.assert_allclose(ring[0], ring[-1])

    def test_hole_becomes_interior_ring(self):
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[2:9, 2:9] = 1
        labels[5, 5] = 0
        rings = self._rings_from_wkb(
            polygonize.polygonize_numpy(labels, self.GT)[1])
        self.assertEqual(len(rings), 2)
        areas = sorted(self._ring_area(r) for r in rings)
        self.assertAlmostEqual(areas[0], 1.0, places=6)   # 穴
        self.assertAlmostEqual(areas[1], 49.0, places=6)  # 外周

    def test_multiple_labels(self):
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[7:10, 7:10] = 2
        shapes = polygonize.polygonize_numpy(labels, self.GT)
        self.assertEqual(set(shapes), {1, 2})

    def test_coordinates_are_georeferenced(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[0, 0] = 1
        rings = self._rings_from_wkb(
            polygonize.polygonize_numpy(labels, self.GT)[1])
        xs = rings[0][:, 0]
        ys = rings[0][:, 1]
        self.assertAlmostEqual(float(xs.min()), 100.0)
        self.assertAlmostEqual(float(xs.max()), 101.0)
        self.assertAlmostEqual(float(ys.max()), 200.0)
        self.assertAlmostEqual(float(ys.min()), 199.0)

    def test_area_matches_cell_count(self):
        """ポリゴン面積がセル数 x セル面積と一致すること."""
        chm, rows, cols = synthetic_stand(120, 14)
        labels = crown.grow_region(chm, rows, cols, max_cr=6)
        shapes = polygonize.polygonize_numpy(labels, self.GT)
        counts, _, _ = crown.crown_statistics(chm, labels, rows.size)
        for label, wkb in list(shapes.items())[:20]:
            rings = self._rings_from_wkb(wkb)
            outer = sum(self._ring_area(r) for r in rings
                        if self._ring_area(r) > 1.5)
            inner = sum(self._ring_area(r) for r in rings
                        if self._ring_area(r) <= 1.5)
            self.assertAlmostEqual(outer - inner, float(counts[label]),
                                   delta=0.5)


if __name__ == "__main__":
    unittest.main()


class TestPolygonizeRings(unittest.TestCase):
    """環経路 (WKB を経由しない出力) の検証."""

    GT = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)

    def test_rings_and_wkb_agree(self):
        chm, rows, cols = synthetic_stand(120, 14)
        order = np.argsort(rows, kind="stable")
        labels = crown.grow_region(
            chm, rows[order], cols[order], max_cr=6)

        rings = polygonize.polygonize_numpy_rings(labels, self.GT)
        wkbs = polygonize.polygonize_numpy(labels, self.GT)
        self.assertEqual(set(rings), set(wkbs))
        self.assertGreater(len(rings), 0)

    def test_ring_area_matches_cell_count(self):
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[2:9, 2:9] = 1
        labels[5, 5] = 0
        rings = polygonize.polygonize_numpy_rings(labels, self.GT)
        polygons = rings[1]
        self.assertEqual(len(polygons), 1)
        outer, inner = polygons[0][0], polygons[0][1]
        self.assertAlmostEqual(_ring_area(outer), 49.0, places=6)
        self.assertAlmostEqual(_ring_area(inner), 1.0, places=6)

    def test_rings_are_closed(self):
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:5, 2:5] = 1
        for polygon in polygonize.polygonize_numpy_rings(labels, self.GT)[1]:
            for ring in polygon:
                self.assertEqual(ring[0], ring[-1])
                self.assertGreaterEqual(len(ring), 4)


class TestGdalPolygonizeFailure(unittest.TestCase):
    """gdal.Polygonize が黙って 0 件を返す状況を検知できること.

    0.3.0 ではこれが握り潰されて出力が空になっていた。
    """

    def setUp(self):
        self.labels = np.zeros((10, 10), dtype=np.int32)
        self.labels[3:7, 3:7] = 1
        self.gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    def tearDown(self):
        for name in ("osgeo", "osgeo.gdal", "osgeo.ogr"):
            sys.modules.pop(name, None)

    def _install_fake_osgeo(self, feature_count):
        gdal = types.ModuleType("osgeo.gdal")
        gdal.GDT_Int32 = 5
        gdal.GDT_Byte = 1

        class FakeBand(object):
            def WriteRaster(self, *args, **kwargs):
                return 0

        class FakeDataset(object):
            def SetGeoTransform(self, gt):
                return 0

            def GetRasterBand(self, index):
                return FakeBand()

        class FakeDriver(object):
            def Create(self, name, width, height, bands, dtype):
                return FakeDataset()

        gdal.GetDriverByName = lambda name: FakeDriver()
        gdal.Polygonize = lambda band, mask, layer, field, options: 0

        ogr = types.ModuleType("osgeo.ogr")
        ogr.wkbPolygon = 3
        ogr.wkbMultiPolygon = 6
        ogr.wkbMultiPolygon25D = 0x80000006
        ogr.OFTInteger = 0
        ogr.FieldDefn = lambda name, kind: object()

        class FakeLayer(object):
            def CreateField(self, defn):
                return 0

            def ResetReading(self):
                return None

            def __iter__(self):
                return iter([])

        class FakeSource(object):
            def CreateLayer(self, name, srs, kind):
                return FakeLayer()

        class FakeOgrDriver(object):
            def CreateDataSource(self, name):
                return FakeSource()

        ogr.GetDriverByName = lambda name: FakeOgrDriver()

        osgeo = types.ModuleType("osgeo")
        osgeo.gdal = gdal
        osgeo.ogr = ogr
        sys.modules["osgeo"] = osgeo
        sys.modules["osgeo.gdal"] = gdal
        sys.modules["osgeo.ogr"] = ogr

    def test_empty_result_raises(self):
        self._install_fake_osgeo(0)
        with self.assertRaises(IOError):
            polygonize._polygonize_gdal_features(self.labels, self.gt)

    def test_wrapper_falls_back_to_numpy(self):
        self._install_fake_osgeo(0)
        shapes, route = polygonize.polygonize(self.labels, self.gt)
        self.assertEqual(set(shapes), {1})
        self.assertIn("numpy", route)
        self.assertIn("GDAL 失敗", route)

    def test_rings_wrapper_falls_back_to_numpy(self):
        self._install_fake_osgeo(0)
        shapes, route = polygonize.polygonize_rings(self.labels, self.gt)
        self.assertEqual(set(shapes), {1})
        self.assertIn("numpy", route)
