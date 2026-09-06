"""GdalReader の ReadRaster 経路を GDAL 無しで検証する.

osgeo をモックに差し替えて, 生バイト列 -> numpy 配列の解釈が正しいことを確かめる。
gdal_array に依存していないこと (import されないこと) も確認する。
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _install_fake_osgeo(source_array, gdal_dtype_name="GDT_Float32"):
    """osgeo.gdal のごく一部だけを持つ偽モジュールを sys.modules に入れる."""
    gdal = types.ModuleType("osgeo.gdal")
    gdal.GA_ReadOnly = 0
    gdal.GDT_Byte = 1
    gdal.GDT_UInt16 = 2
    gdal.GDT_Int16 = 3
    gdal.GDT_UInt32 = 4
    gdal.GDT_Int32 = 5
    gdal.GDT_Float32 = 6
    gdal.GDT_Float64 = 7
    gdal.GetDataTypeName = lambda code: "code%d" % code

    dtype_code = getattr(gdal, gdal_dtype_name)

    class FakeBand(object):
        DataType = dtype_code

        def ReadRaster(self, xoff, yoff, xsize, ysize,
                       buf_xsize=None, buf_ysize=None, buf_type=None):
            chunk = source_array[yoff:yoff + ysize, xoff:xoff + xsize]
            # GDAL はホストのバイトオーダで生バイト列を返す
            return chunk.tobytes()

        def GetNoDataValue(self):
            return -9999.0

    class FakeDataset(object):
        RasterXSize = source_array.shape[1]
        RasterYSize = source_array.shape[0]

        def GetRasterBand(self, index):
            return FakeBand()

        def GetGeoTransform(self):
            return (100.0, 0.4, 0.0, 500.0, 0.0, -0.4)

    gdal.Open = lambda source, mode=0: FakeDataset()

    osgeo = types.ModuleType("osgeo")
    osgeo.gdal = gdal

    sys.modules["osgeo"] = osgeo
    sys.modules["osgeo.gdal"] = gdal
    # gdal_array が読まれたら即座に分かるように壊れたモジュールを置く
    broken = types.ModuleType("osgeo.gdal_array")

    def _explode(*args, **kwargs):
        raise ImportError("numpy.core.multiarray failed to import")

    broken.__getattr__ = _explode
    sys.modules["osgeo.gdal_array"] = broken


def _remove_fake_osgeo():
    for name in ("osgeo", "osgeo.gdal", "osgeo.gdal_array"):
        sys.modules.pop(name, None)


class TestGdalReaderReadRaster(unittest.TestCase):

    def setUp(self):
        self.source = (np.arange(40 * 30, dtype=np.float32)
                       .reshape(40, 30) / 7.0)

    def tearDown(self):
        _remove_fake_osgeo()

    def _read(self, window, dtype_name="GDT_Float32", array=None):
        from ..core import raster_reader

        data = self.source if array is None else array
        _install_fake_osgeo(data, dtype_name)
        reader = raster_reader.GdalReader("dummy.tif", 1)
        try:
            return reader.read(window)
        finally:
            reader.close()

    def test_full_read_matches_source(self):
        from ..core import raster_reader

        window = raster_reader.BlockWindow(0, 0, 30, 40, 0, 0, 30, 40)
        result = self._read(window)
        np.testing.assert_array_equal(result, self.source)

    def test_offset_read_matches_source(self):
        from ..core import raster_reader

        window = raster_reader.BlockWindow(5, 7, 10, 12, 5, 7, 10, 12)
        result = self._read(window)
        np.testing.assert_array_equal(result, self.source[7:19, 5:15])

    def test_integer_raster_is_converted_to_float32(self):
        from ..core import raster_reader

        source = np.arange(20 * 20, dtype=np.int16).reshape(20, 20)
        window = raster_reader.BlockWindow(0, 0, 20, 20, 0, 0, 20, 20)
        result = self._read(window, "GDT_Int16", source)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, source.astype(np.float32))

    def test_byte_raster(self):
        from ..core import raster_reader

        source = (np.arange(16 * 16) % 256).astype(np.uint8).reshape(16, 16)
        window = raster_reader.BlockWindow(0, 0, 16, 16, 0, 0, 16, 16)
        result = self._read(window, "GDT_Byte", source)
        np.testing.assert_array_equal(result, source.astype(np.float32))

    def test_result_is_writable(self):
        from ..core import raster_reader

        window = raster_reader.BlockWindow(0, 0, 30, 40, 0, 0, 30, 40)
        result = self._read(window)
        result[0, 0] = 1.0  # frombuffer の読み取り専用ビューなら例外になる
        self.assertEqual(result[0, 0], 1.0)

    def test_gdal_array_is_never_imported(self):
        from ..core import raster_reader

        window = raster_reader.BlockWindow(0, 0, 30, 40, 0, 0, 30, 40)
        self._read(window)
        # 壊れたモジュールを置いてあるので, 触っていれば ImportError で落ちている
        self.assertIn("osgeo.gdal_array", sys.modules)

    def test_metadata_is_read(self):
        from ..core import raster_reader

        _install_fake_osgeo(self.source)
        reader = raster_reader.GdalReader("dummy.tif", 1)
        try:
            self.assertEqual(reader.width, 30)
            self.assertEqual(reader.height, 40)
            self.assertEqual(reader.nodata, -9999.0)
            self.assertAlmostEqual(reader.geotransform[1], 0.4)
        finally:
            reader.close()


class TestCellToMap(unittest.TestCase):
    """セル座標 -> 地図座標の変換 (0.4m CHM を想定)."""

    def test_cell_centre(self):
        from ..core import local_maxima

        geotransform = (100.0, 0.4, 0.0, 500.0, 0.0, -0.4)
        rows = np.array([0.0, 10.0])
        cols = np.array([0.0, 5.0])
        xs, ys = local_maxima.cell_to_map(rows, cols, geotransform)
        np.testing.assert_allclose(xs, [100.2, 102.2])
        np.testing.assert_allclose(ys, [499.8, 495.8])


if __name__ == "__main__":
    unittest.main()
