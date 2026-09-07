"""タイル (ブロック) 単位でラスタを numpy 配列として読むためのラッパ.

GDAL で開ければ GDAL を使い, 開けなければ QGIS のラスタプロバイダ経由で読む。
オーバーラップ (halo) 付きでブロックを切り出すので, 近傍窓を使う処理を
ブロック境界の欠けなしに実行できる。
"""

from __future__ import annotations

import numpy as np


class BlockWindow(object):
    """1 ブロック分の読み出し範囲.

    core_* が結果として採用する範囲, read_* が halo を含めた読み出し範囲。
    """

    __slots__ = ("core_col", "core_row", "core_width", "core_height",
                 "read_col", "read_row", "read_width", "read_height")

    def __init__(self, core_col, core_row, core_width, core_height,
                 read_col, read_row, read_width, read_height):
        self.core_col = core_col
        self.core_row = core_row
        self.core_width = core_width
        self.core_height = core_height
        self.read_col = read_col
        self.read_row = read_row
        self.read_width = read_width
        self.read_height = read_height

    @property
    def core_slice(self):
        """読み出し配列の中から core 部分を取り出すスライス."""
        row0 = self.core_row - self.read_row
        col0 = self.core_col - self.read_col
        return (slice(row0, row0 + self.core_height),
                slice(col0, col0 + self.core_width))


def iter_blocks(width, height, block_size, halo):
    """ラスタ全体を halo 付きブロックに分割して yield する."""
    for row in range(0, height, block_size):
        core_height = min(block_size, height - row)
        read_row = max(0, row - halo)
        read_bottom = min(height, row + core_height + halo)
        for col in range(0, width, block_size):
            core_width = min(block_size, width - col)
            read_col = max(0, col - halo)
            read_right = min(width, col + core_width + halo)
            yield BlockWindow(
                col, row, core_width, core_height,
                read_col, read_row,
                read_right - read_col, read_bottom - read_row,
            )


def count_blocks(width, height, block_size):
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    return rows * cols


def _gdal_dtype_map():
    """GDAL データ型 -> numpy データ型の対応表を作る.

    GDT_Int8 (GDAL 3.7 以降) のようにバージョン依存の型があるので getattr で拾う。
    """
    from osgeo import gdal

    names = (
        ("GDT_Byte", np.uint8),
        ("GDT_Int8", np.int8),
        ("GDT_UInt16", np.uint16),
        ("GDT_Int16", np.int16),
        ("GDT_UInt32", np.uint32),
        ("GDT_Int32", np.int32),
        ("GDT_UInt64", np.uint64),
        ("GDT_Int64", np.int64),
        ("GDT_Float32", np.float32),
        ("GDT_Float64", np.float64),
    )
    mapping = {}
    for name, dtype in names:
        code = getattr(gdal, name, None)
        if code is not None:
            mapping[code] = dtype
    return mapping


class GdalReader(object):
    """GDAL データセットからの読み出し.

    ReadAsArray() は内部で osgeo.gdal_array (C 拡張) を import するが、
    これは numpy の ABI に結び付いてビルドされているため、GDAL と numpy の
    ビルド組み合わせによっては ImportError になる環境がある。
    そこで ReadRaster() で生バイト列を受け取り、numpy 側で解釈する。
    gdal_array に一切依存しない。
    """

    def __init__(self, source, band=1):
        from osgeo import gdal

        self.dataset = gdal.Open(source, gdal.GA_ReadOnly)
        if self.dataset is None:
            raise IOError("GDAL でラスタを開けません: %s" % source)

        self.band = self.dataset.GetRasterBand(band)
        if self.band is None:
            raise IOError("バンド %d が存在しません: %s" % (band, source))

        self.gdal_dtype = self.band.DataType
        mapping = _gdal_dtype_map()
        if self.gdal_dtype not in mapping:
            raise IOError(
                "未対応のラスタデータ型です: %s"
                % gdal.GetDataTypeName(self.gdal_dtype))
        self.numpy_dtype = np.dtype(mapping[self.gdal_dtype])

        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.geotransform = self.dataset.GetGeoTransform()
        self.nodata = self.band.GetNoDataValue()

    def read(self, window):
        raw = self.band.ReadRaster(
            xoff=window.read_col, yoff=window.read_row,
            xsize=window.read_width, ysize=window.read_height,
            buf_xsize=window.read_width, buf_ysize=window.read_height,
            buf_type=self.gdal_dtype,
        )
        if raw is None:
            raise IOError("ラスタの読み出しに失敗しました。")
        array = np.frombuffer(raw, dtype=self.numpy_dtype)
        array = array.reshape(window.read_height, window.read_width)
        # frombuffer は読み取り専用ビューなので astype で書き込み可能な複製にする
        return array.astype(np.float32)

    def close(self):
        self.band = None
        self.dataset = None


class QgisProviderReader(object):
    """QGIS ラスタプロバイダからの読み出し (GDAL で開けない場合の保険)."""

    def __init__(self, layer, band=1):
        from qgis.core import QgsRectangle

        self._rectangle_class = QgsRectangle
        self.layer = layer
        self.provider = layer.dataProvider()
        self.band = band
        self.width = layer.width()
        self.height = layer.height()

        extent = layer.extent()
        self.x_size = extent.width() / self.width
        self.y_size = extent.height() / self.height
        self.geotransform = (
            extent.xMinimum(), self.x_size, 0.0,
            extent.yMaximum(), 0.0, -self.y_size,
        )
        self.nodata = None
        if self.provider.sourceHasNoDataValue(band):
            self.nodata = self.provider.sourceNoDataValue(band)

    def read(self, window):
        origin_x = self.geotransform[0]
        origin_y = self.geotransform[3]
        rectangle = self._rectangle_class(
            origin_x + window.read_col * self.x_size,
            origin_y - (window.read_row + window.read_height) * self.y_size,
            origin_x + (window.read_col + window.read_width) * self.x_size,
            origin_y - window.read_row * self.y_size,
        )
        block = self.provider.block(
            self.band, rectangle, window.read_width, window.read_height)
        if block is None or not block.isValid():
            raise IOError("ラスタブロックの読み出しに失敗しました。")
        array = np.frombuffer(
            bytes(block.data()), dtype=_numpy_dtype(block.dataType()))
        array = array.reshape(window.read_height, window.read_width)
        return array.astype(np.float32)

    def close(self):
        self.provider = None


def _numpy_dtype(qgis_data_type):
    from qgis.core import Qgis

    mapping = {
        Qgis.DataType.Byte: np.uint8,
        Qgis.DataType.Int8: np.int8,
        Qgis.DataType.UInt16: np.uint16,
        Qgis.DataType.Int16: np.int16,
        Qgis.DataType.UInt32: np.uint32,
        Qgis.DataType.Int32: np.int32,
        Qgis.DataType.Float32: np.float32,
        Qgis.DataType.Float64: np.float64,
    }
    if qgis_data_type not in mapping:
        raise ValueError("未対応のラスタデータ型です: %s" % qgis_data_type)
    return mapping[qgis_data_type]


class MaskSampler(object):
    """CHM のブロックと同じ格子でマスクラスタを読むためのラッパ.

    QGIS のラスタプロバイダは範囲とサイズを指定して読めるので, マスクの
    セルサイズが CHM と違っていても最近傍で合わせてくれる。
    CRS が違う場合は再投影されないので, 呼び出し側で警告する。
    """

    def __init__(self, layer, band=1):
        self.layer = layer
        self.provider = layer.dataProvider()
        self.band = band
        self.nodata = None
        if self.provider.sourceHasNoDataValue(band):
            self.nodata = self.provider.sourceNoDataValue(band)

    def read(self, window, geotransform):
        """CHM の読み出し窓に対応するマスク値を float32 配列で返す."""
        from qgis.core import QgsRectangle

        x_size = geotransform[1]
        y_size = geotransform[5]
        x_min = geotransform[0] + window.read_col * x_size
        x_max = geotransform[0] + (window.read_col + window.read_width) * x_size
        y_top = geotransform[3] + window.read_row * y_size
        y_bottom = geotransform[3] + (
            window.read_row + window.read_height) * y_size

        rectangle = QgsRectangle(
            min(x_min, x_max), min(y_top, y_bottom),
            max(x_min, x_max), max(y_top, y_bottom))

        block = self.provider.block(
            self.band, rectangle, window.read_width, window.read_height)
        if block is None or not block.isValid():
            raise IOError("マスクラスタのブロックを読めません。")

        array = np.frombuffer(
            bytes(block.data()), dtype=_numpy_dtype(block.dataType()))
        return array.reshape(
            window.read_height, window.read_width).astype(np.float32)

    def valid_mask(self, window, geotransform, invert=False):
        """有効セル (マスクが 0 でも NoData でもない) の真偽配列を返す."""
        values = self.read(window, geotransform)
        valid = np.isfinite(values) & (values != 0)
        if self.nodata is not None and np.isfinite(self.nodata):
            valid &= values != np.float32(self.nodata)
        return ~valid if invert else valid


def open_reader(layer, band=1, feedback=None):
    """QgsRasterLayer からリーダを作る. GDAL 優先, 失敗したらプロバイダ経由.

    生成できただけでは不十分なので, 1 セルだけ試し読みして実際に読めることを
    確認してから返す。読めなければ QGIS プロバイダ経由に切り替える。
    """
    source = layer.source()
    try:
        reader = GdalReader(source, band)
        reader.read(BlockWindow(0, 0, 1, 1, 0, 0, 1, 1))
        return reader
    except Exception as error:  # noqa: BLE001 - 意図的に全部拾ってフォールバック
        if feedback is not None:
            feedback.pushInfo(
                "GDAL 経由で読めなかったため QGIS プロバイダ経由に切り替えます"
                " (%s: %s)" % (type(error).__name__, error))
        return QgisProviderReader(layer, band)
