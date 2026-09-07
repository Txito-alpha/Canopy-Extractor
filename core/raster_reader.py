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


def iter_blocks(width, height, block_size, halo, region=None):
    """ラスタを halo 付きブロックに分割して yield する.

    Args:
        width, height: ラスタ全体のセル数 (halo の読み出しはこの範囲に
            クリップされる)
        block_size: コアブロックの一辺のセル数
        halo: コアの周囲に追加で読む幅 (セル数)
        region: 処理対象を絞る場合の (col0, row0, region_width, region_height)。
            None ならラスタ全体を対象にする。ブロックはこの範囲内だけを
            走査するが, halo による読み出しは region の外, ラスタ全体の
            範囲までは及んでよい (境界付近の計算精度を落とさないため)。
    """
    if region is None:
        origin_col, origin_row = 0, 0
        region_width, region_height = width, height
    else:
        origin_col, origin_row, region_width, region_height = region

    for row in range(origin_row, origin_row + region_height, block_size):
        core_height = min(block_size, origin_row + region_height - row)
        read_row = max(0, row - halo)
        read_bottom = min(height, row + core_height + halo)
        for col in range(origin_col, origin_col + region_width, block_size):
            core_width = min(block_size, origin_col + region_width - col)
            read_col = max(0, col - halo)
            read_right = min(width, col + core_width + halo)
            yield BlockWindow(
                col, row, core_width, core_height,
                read_col, read_row,
                read_right - read_col, read_bottom - read_row,
            )


def count_blocks(width, height, block_size, region=None):
    if region is not None:
        _, _, width, height = region
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


def extent_to_region(geotransform, raster_width, raster_height,
                     x_min, y_min, x_max, y_max):
    """地図座標の範囲をピクセル範囲 (col0, row0, width, height) に変換する.

    ラスタが北が上 (回転無し, geotransform[2] == geotransform[4] == 0) で
    あることを前提にする。本プラグインが扱う CHM はすべてこの形式。

    範囲がラスタと重ならない場合は None を返す。
    """
    gt = geotransform
    x_size = gt[1]
    y_size = gt[5]  # 通常は負値

    col0 = int(np.floor((x_min - gt[0]) / x_size))
    col1 = int(np.ceil((x_max - gt[0]) / x_size))
    # y_size が負値なので, y の大小関係を反転させて row に変換する
    row0 = int(np.floor((y_max - gt[3]) / y_size))
    row1 = int(np.ceil((y_min - gt[3]) / y_size))

    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(raster_width, col1)
    row1 = min(raster_height, row1)

    if col1 <= col0 or row1 <= row0:
        return None
    return (col0, row0, col1 - col0, row1 - row0)


class AreaRasterizer(object):
    """ポリゴンレイヤをブロックごとにラスタ化し, 有効セルのマスクを作る.

    gdal.RasterizeLayer を使う。gdal.Polygonize と同じく MEM ドライバ +
    ReadRaster 経由なので osgeo.gdal_array を通らない。

    ポリゴンの CRS が CHM と違う場合は, 構築時に一括で座標変換してから
    OGR メモリレイヤに積む (ブロックごとの変換は行わない)。
    """

    def __init__(self, source, target_crs, transform_context,
                 all_touched=False):
        from osgeo import ogr
        from qgis.core import QgsCoordinateTransform

        transform = None
        source_crs = source.sourceCrs()
        if (source_crs.isValid() and target_crs.isValid()
                and source_crs != target_crs):
            transform = QgsCoordinateTransform(
                source_crs, target_crs, transform_context)

        driver = ogr.GetDriverByName("Memory")
        self._datasource = driver.CreateDataSource("area")
        self._layer = self._datasource.CreateLayer(
            "area", None, ogr.wkbMultiPolygon)
        self._all_touched = all_touched
        self.feature_count = 0

        for feature in source.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isNull() or geometry.isEmpty():
                continue
            if transform is not None:
                if geometry.transform(transform) != 0:
                    continue
            ogr_geometry = ogr.CreateGeometryFromWkt(geometry.asWkt())
            if ogr_geometry is None:
                continue
            ogr_feature = ogr.Feature(self._layer.GetLayerDefn())
            ogr_feature.SetGeometry(ogr_geometry)
            self._layer.CreateFeature(ogr_feature)
            self.feature_count += 1

    def valid_mask(self, window, geotransform):
        """window の読み出し範囲について, ポリゴン内側の真偽配列を返す."""
        from osgeo import gdal

        if self.feature_count == 0:
            return np.zeros((window.read_height, window.read_width),
                            dtype=bool)

        block_gt = (
            geotransform[0] + window.read_col * geotransform[1],
            geotransform[1], geotransform[2],
            geotransform[3] + window.read_row * geotransform[5],
            geotransform[4], geotransform[5],
        )

        driver = gdal.GetDriverByName("MEM")
        dataset = driver.Create(
            "", window.read_width, window.read_height, 1, gdal.GDT_Byte)
        if dataset is None:
            raise IOError("MEM ラスタを作成できません。")
        dataset.SetGeoTransform(block_gt)

        x_min = block_gt[0]
        x_max = block_gt[0] + window.read_width * block_gt[1]
        y_max = block_gt[3]
        y_min = block_gt[3] + window.read_height * block_gt[5]
        self._layer.SetSpatialFilterRect(
            min(x_min, x_max), min(y_min, y_max),
            max(x_min, x_max), max(y_min, y_max))

        options = ["ALL_TOUCHED=%s" % ("TRUE" if self._all_touched
                                       else "FALSE")]
        status = gdal.RasterizeLayer(
            dataset, [1], self._layer, burn_values=[1], options=options)
        self._layer.SetSpatialFilter(None)
        if status != 0:
            raise IOError("gdal.RasterizeLayer が失敗しました (code %s)"
                          % status)

        raw = dataset.GetRasterBand(1).ReadRaster(
            0, 0, window.read_width, window.read_height)
        array = np.frombuffer(raw, dtype=np.uint8).reshape(
            window.read_height, window.read_width)
        return array > 0


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
