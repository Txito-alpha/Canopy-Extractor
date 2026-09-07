"""処理範囲の絞り込み (extent / region) の検証.

    python -m unittest canopy_extractor.tests.test_region -v
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from ..core import local_maxima, raster_reader


class TestExtentToRegion(unittest.TestCase):

    GT = (100.0, 0.5, 0.0, 500.0, 0.0, -0.5)  # 0.5m CHM, 原点 (100, 500)

    def test_full_raster_extent(self):
        # ラスタ全体を覆う範囲を渡すと, 範囲もラスタ全体になること
        region = raster_reader.extent_to_region(
            self.GT, 400, 300, 100.0, 350.0, 300.0, 500.0)
        self.assertEqual(region, (0, 0, 400, 300))

    def test_partial_extent(self):
        region = raster_reader.extent_to_region(
            self.GT, 400, 300, 110.0, 480.0, 130.0, 495.0)
        # x: (110-100)/0.5=20 -> (130-100)/0.5=60
        # y: (500-495)/0.5=10 -> (500-480)/0.5=40
        self.assertEqual(region, (20, 10, 40, 30))

    def test_extent_outside_raster_returns_none(self):
        region = raster_reader.extent_to_region(
            self.GT, 400, 300, 1000.0, 1000.0, 1100.0, 1100.0)
        self.assertIsNone(region)

    def test_extent_clipped_to_raster_bounds(self):
        # ラスタの外にはみ出す範囲は, ラスタ境界にクリップされること
        region = raster_reader.extent_to_region(
            self.GT, 400, 300, 50.0, 400.0, 150.0, 600.0)
        col0, row0, width, height = region
        self.assertEqual(col0, 0)  # x=50 は範囲外 -> 0 にクリップ
        self.assertLessEqual(col0 + width, 400)
        self.assertLessEqual(row0 + height, 300)

    def test_fractional_bounds_are_expanded_not_truncated(self):
        # セル境界に一致しない範囲は, 内側に切り詰めず外側に広げること
        # (中途半端な範囲指定でセルが 1 つも入らない事態を避ける)
        region = raster_reader.extent_to_region(
            self.GT, 400, 300, 100.1, 499.1, 100.4, 499.4)
        col0, row0, width, height = region
        # x: floor((100.1-100)/0.5)=0, ceil((100.4-100)/0.5)=1
        self.assertEqual((col0, width), (0, 1))
        # y: floor((499.4-500)/-0.5)=1, ceil((499.1-500)/-0.5)=2
        self.assertEqual((row0, height), (1, 1))


class TestIterBlocksRegion(unittest.TestCase):
    """region 引数を渡しても, 対応する範囲だけを走査すること."""

    def test_none_region_matches_full_raster_default(self):
        """region=None は従来どおり全域を走査すること (後方互換)."""
        blocks_a = list(raster_reader.iter_blocks(100, 80, 32, 2))
        blocks_b = list(raster_reader.iter_blocks(
            100, 80, 32, 2, region=None))
        self.assertEqual(len(blocks_a), len(blocks_b))
        for a, b in zip(blocks_a, blocks_b):
            self.assertEqual(
                (a.core_col, a.core_row, a.core_width, a.core_height),
                (b.core_col, b.core_row, b.core_width, b.core_height))

    def test_region_restricts_core_area(self):
        region = (20, 10, 40, 30)
        blocks = list(raster_reader.iter_blocks(100, 80, 16, 2, region=region))
        for block in blocks:
            self.assertGreaterEqual(block.core_col, 20)
            self.assertGreaterEqual(block.core_row, 10)
            self.assertLessEqual(block.core_col + block.core_width, 60)
            self.assertLessEqual(block.core_row + block.core_height, 40)

    def test_halo_reads_beyond_region_but_within_raster(self):
        """halo はラスタ全体の範囲までなら region の外を読んでよいこと."""
        region = (20, 10, 40, 30)
        halo = 5
        blocks = list(raster_reader.iter_blocks(
            100, 80, 16, halo, region=region))
        min_read_col = min(b.read_col for b in blocks)
        min_read_row = min(b.read_row for b in blocks)
        max_read_right = max(b.read_col + b.read_width for b in blocks)
        max_read_bottom = max(b.read_row + b.read_height for b in blocks)
        # halo のぶん region より外側まで読んでいること
        self.assertLess(min_read_col, 20)
        self.assertLess(min_read_row, 10)
        # ただしラスタ境界は超えないこと
        self.assertLessEqual(max_read_right, 100)
        self.assertLessEqual(max_read_bottom, 80)

    def test_region_covers_whole_area_without_gaps_or_overlap(self):
        region = (5, 5, 50, 40)
        blocks = list(raster_reader.iter_blocks(100, 80, 12, 3, region=region))
        covered = np.zeros((80, 100), dtype=bool)
        for block in blocks:
            patch = covered[
                block.core_row:block.core_row + block.core_height,
                block.core_col:block.core_col + block.core_width]
            self.assertFalse(patch.any())  # 重複が無いこと
            patch[:] = True
        expected = np.zeros((80, 100), dtype=bool)
        expected[5:45, 5:55] = True
        np.testing.assert_array_equal(covered, expected)

    def test_local_maxima_matches_whole_raster_within_region(self):
        """region 制限つきタイル処理の結果が, 全域処理の対応範囲と一致すること."""
        rng = np.random.default_rng(0)
        chm = (rng.random((120, 140)) * 30).astype(np.float32)
        window_size = 7
        halo = window_size // 2
        region = (30, 20, 60, 50)

        whole = local_maxima.local_maxima_mask(chm, window_size, 6.0)

        tiled = np.zeros(chm.shape, dtype=bool)
        for block in raster_reader.iter_blocks(
                140, 120, 32, halo, region=region):
            patch = chm[
                block.read_row:block.read_row + block.read_height,
                block.read_col:block.read_col + block.read_width]
            mask = local_maxima.local_maxima_mask(patch, window_size, 6.0)
            tiled[
                block.core_row:block.core_row + block.core_height,
                block.core_col:block.core_col + block.core_width,
            ] = mask[block.core_slice]

        col0, row0, width, height = region
        region_slice = (slice(row0, row0 + height), slice(col0, col0 + width))
        np.testing.assert_array_equal(tiled[region_slice], whole[region_slice])
        # region の外は触れていないこと
        self.assertFalse(tiled[:row0, :].any())
        self.assertFalse(tiled[row0 + height:, :].any())


class TestCountBlocksRegion(unittest.TestCase):

    def test_matches_iter_blocks_count(self):
        region = (20, 10, 40, 30)
        n = raster_reader.count_blocks(100, 80, 16, region=region)
        actual = len(list(raster_reader.iter_blocks(
            100, 80, 16, 2, region=region)))
        self.assertEqual(n, actual)

    def test_none_region_matches_full_raster(self):
        self.assertEqual(
            raster_reader.count_blocks(100, 80, 16),
            raster_reader.count_blocks(100, 80, 16, region=None))


class TestAreaRasterizer(unittest.TestCase):
    """gdal.RasterizeLayer 経路のモック検証.

    実際の QGIS / GDAL 無しでは動かせないので, 呼び出し方が正しいこと
    (ReadRaster を使い gdal_array を経由しないこと, 空間フィルタを設定して
    いること) を偽の osgeo モジュールで確認する。
    """

    def tearDown(self):
        for name in ("osgeo", "osgeo.gdal", "osgeo.ogr", "qgis", "qgis.core"):
            sys.modules.pop(name, None)

    def _install_fake_qgis(self):
        qgis_core = types.ModuleType("qgis.core")

        class FakeCoordinateTransform(object):
            def __init__(self, source_crs, target_crs, context):
                pass

        qgis_core.QgsCoordinateTransform = FakeCoordinateTransform
        qgis = types.ModuleType("qgis")
        qgis.core = qgis_core
        sys.modules["qgis"] = qgis
        sys.modules["qgis.core"] = qgis_core

    def _install_fake_osgeo(self, burned):
        """burned: RasterizeLayer が塗ったことにする (row0,col0,row1,col1)."""
        gdal = types.ModuleType("osgeo.gdal")
        gdal.GDT_Byte = 1
        calls = {"rasterize": 0, "spatial_filter_rect": None}

        class FakeBand(object):
            def ReadRaster(self, xoff, yoff, xsize, ysize):
                array = np.zeros((ysize, xsize), dtype=np.uint8)
                r0, c0, r1, c1 = burned
                array[max(0, r0):min(ysize, r1),
                      max(0, c0):min(xsize, c1)] = 1
                return array.tobytes()

        class FakeDataset(object):
            def SetGeoTransform(self, gt):
                return 0

            def GetRasterBand(self, index):
                return FakeBand()

        class FakeDriver(object):
            def Create(self, name, width, height, bands, dtype):
                return FakeDataset()

        gdal.GetDriverByName = lambda name: FakeDriver()

        def fake_rasterize(dataset, bands, layer, burn_values, options):
            calls["rasterize"] += 1
            return 0

        gdal.RasterizeLayer = fake_rasterize

        ogr = types.ModuleType("osgeo.ogr")
        ogr.wkbMultiPolygon = 6
        ogr.CreateGeometryFromWkt = lambda wkt: object()

        class FakeFeature(object):
            def __init__(self, defn):
                pass

            def SetGeometry(self, geometry):
                return None

        ogr.Feature = FakeFeature

        class FakeLayer(object):
            def GetLayerDefn(self):
                return object()

            def CreateFeature(self, feature):
                return 0

            def SetSpatialFilterRect(self, x0, y0, x1, y1):
                calls["spatial_filter_rect"] = (x0, y0, x1, y1)

            def SetSpatialFilter(self, geometry):
                return None

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
        return calls

    def _fake_source(self, n_features=2):
        class FakeGeometry(object):
            def isNull(self):
                return False

            def isEmpty(self):
                return False

            def transform(self, tr):
                return 0

            def asWkt(self):
                return "POLYGON((0 0,1 0,1 1,0 1,0 0))"

        class FakeFeature(object):
            def geometry(self):
                return FakeGeometry()

        class FakeCrs(object):
            def isValid(self):
                return True

            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        class FakeSource(object):
            def getFeatures(self):
                return [FakeFeature() for _ in range(n_features)]

            def sourceCrs(self):
                return FakeCrs()

        return FakeSource(), FakeCrs()

    def test_rasterize_is_called_and_result_thresholded(self):
        self._install_fake_qgis()
        calls = self._install_fake_osgeo(burned=(2, 2, 8, 8))
        source, crs = self._fake_source(n_features=3)

        rasterizer = raster_reader.AreaRasterizer(source, crs, None)
        self.assertEqual(rasterizer.feature_count, 3)

        window = raster_reader.BlockWindow(0, 0, 10, 10, 0, 0, 10, 10)
        mask = rasterizer.valid_mask(window, (0.0, 1.0, 0.0, 0.0, 0.0, -1.0))

        self.assertEqual(calls["rasterize"], 1)
        self.assertIsNotNone(calls["spatial_filter_rect"])
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[0, 0])

    def test_no_features_returns_all_false_without_calling_gdal(self):
        self._install_fake_qgis()
        calls = self._install_fake_osgeo(burned=(0, 0, 0, 0))
        source, crs = self._fake_source(n_features=0)
        rasterizer = raster_reader.AreaRasterizer(source, crs, None)
        self.assertEqual(rasterizer.feature_count, 0)

        window = raster_reader.BlockWindow(0, 0, 5, 5, 0, 0, 5, 5)
        mask = rasterizer.valid_mask(window, (0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        self.assertFalse(mask.any())
        self.assertEqual(calls["rasterize"], 0)


if __name__ == "__main__":
    unittest.main()
