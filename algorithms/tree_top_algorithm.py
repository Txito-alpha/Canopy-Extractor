"""樹頂点抽出 Processing アルゴリズム."""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication, QVariant

from ..core import local_maxima, raster_reader, smoothing


class TreeTopAlgorithm(QgsProcessingAlgorithm):
    """CHM から局所最大値法で樹頂点を抽出する."""

    INPUT = "INPUT"
    BAND = "BAND"
    WINDOW_SIZE = "WINDOW_SIZE"
    MIN_HEIGHT = "MIN_HEIGHT"
    MERGE_PLATEAUS = "MERGE_PLATEAUS"
    SMOOTH_METHOD = "SMOOTH_METHOD"
    SMOOTH_SIZE = "SMOOTH_SIZE"
    SMOOTH_SIGMA = "SMOOTH_SIGMA"
    HEIGHT_SOURCE = "HEIGHT_SOURCE"
    BLOCK_SIZE = "BLOCK_SIZE"
    OUTPUT = "OUTPUT"

    def tr(self, string):
        return QCoreApplication.translate("TreeTopAlgorithm", string)

    def createInstance(self):
        return TreeTopAlgorithm()

    def name(self):
        return "extracttreetops"

    def displayName(self):
        return self.tr("樹頂点抽出")

    def group(self):
        return self.tr("森林解析")

    def groupId(self):
        return "forestanalysis"

    def shortHelpString(self):
        return self.tr(
            "CHM (樹冠高モデル) から局所最大値法で樹頂点をポイントとして抽出します。\n\n"
            "近傍窓のなかで最大となるセルを樹頂点候補とし, 指定した最低樹高以上の"
            "ものだけを出力します。同じ高さのセルが連続する平坦な樹冠は, 既定では"
            "1 点に統合します。\n\n"
            "【平滑化】\n"
            "CHM の細かい凹凸は偽の極大を生み, 1 本の木から複数の樹頂点が出る"
            "原因になります。平滑化はこれを抑えるための前処理です。平均フィルタは"
            "手軽で, ガウシアンは平滑の強さを sigma (m) で連続的に調整できます。"
            "NoData は計算から除外されるので, 無立木地の縁の値が引きずられることは"
            "ありません。\n\n"
            "【樹高の取得元】\n"
            "平滑化は樹冠の頂点をわずかに削るため, 平滑化後の CHM から樹高を取ると"
            "系統的な過小評価になります。既定では極大の探索だけを平滑化後の面で行い, "
            "樹高は元の CHM から取得します。\n\n"
            "処理はすべてメモリ上の配列演算で行うため, 中間ファイルは生成しません。"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT, self.tr("CHM ラスタ")))

        self.addParameter(QgsProcessingParameterBand(
            self.BAND, self.tr("バンド"), parentLayerParameterName=self.INPUT,
            defaultValue=1))

        self.addParameter(QgsProcessingParameterNumber(
            self.WINDOW_SIZE, self.tr("近傍窓のサイズ (セル数, 奇数)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=7, minValue=3, maxValue=101))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_HEIGHT, self.tr("最低樹高 (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=6.0))

        self.addParameter(QgsProcessingParameterBoolean(
            self.MERGE_PLATEAUS, self.tr("平坦な樹頂部を 1 点に統合する"),
            defaultValue=True))

        self.addParameter(QgsProcessingParameterEnum(
            self.SMOOTH_METHOD, self.tr("平滑化"),
            options=[self.tr("なし"), self.tr("平均フィルタ"),
                     self.tr("ガウシアン")],
            defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.SMOOTH_SIZE, self.tr("平均フィルタの窓サイズ (セル数, 奇数)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=3, maxValue=51))

        self.addParameter(QgsProcessingParameterNumber(
            self.SMOOTH_SIGMA, self.tr("ガウシアンの sigma (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.5, minValue=0.01))

        self.addParameter(QgsProcessingParameterEnum(
            self.HEIGHT_SOURCE, self.tr("樹高の取得元"),
            options=[self.tr("元の CHM"), self.tr("平滑化後の CHM")],
            defaultValue=0))

        block_size = QgsProcessingParameterNumber(
            self.BLOCK_SIZE, self.tr("処理ブロックサイズ (セル数)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=2048, minValue=256, maxValue=32768)
        block_size.setFlags(
            block_size.flags() | QgsProcessingParameterNumber.FlagAdvanced)
        self.addParameter(block_size)

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("樹頂点"), QgsProcessing.TypeVectorPoint))

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.INPUT))

        band = self.parameterAsInt(parameters, self.BAND, context)
        window_size = self.parameterAsInt(parameters, self.WINDOW_SIZE, context)
        min_height = self.parameterAsDouble(parameters, self.MIN_HEIGHT, context)
        merge_plateaus = self.parameterAsBool(
            parameters, self.MERGE_PLATEAUS, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)

        method_index = self.parameterAsEnum(
            parameters, self.SMOOTH_METHOD, context)
        smooth_method = (smoothing.METHOD_NONE, smoothing.METHOD_MEAN,
                         smoothing.METHOD_GAUSSIAN)[method_index]
        smooth_size = self.parameterAsInt(parameters, self.SMOOTH_SIZE, context)
        smooth_sigma_m = self.parameterAsDouble(
            parameters, self.SMOOTH_SIGMA, context)
        height_from_smoothed = self.parameterAsEnum(
            parameters, self.HEIGHT_SOURCE, context) == 1

        if window_size % 2 == 0:
            raise QgsProcessingException(
                self.tr("近傍窓のサイズは奇数で指定してください。"))
        if smooth_method == smoothing.METHOD_MEAN and smooth_size % 2 == 0:
            raise QgsProcessingException(
                self.tr("平均フィルタの窓サイズは奇数で指定してください。"))

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("height", QVariant.Double))
        fields.append(QgsField("n_cells", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.Point, layer.crs())
        if sink is None:
            raise QgsProcessingException(
                self.invalidSinkError(parameters, self.OUTPUT))

        feedback.pushInfo(local_maxima.backend_description())

        reader = raster_reader.open_reader(layer, band, feedback)
        geotransform = reader.geotransform
        cell_size = abs(geotransform[1])
        feedback.pushInfo(
            self.tr("ラスタ %d x %d セル, セルサイズ %.3f, NoData=%s")
            % (reader.width, reader.height, cell_size, reader.nodata))

        sigma_cells = 1.0
        if smooth_method == smoothing.METHOD_GAUSSIAN:
            if cell_size <= 0.0:
                raise QgsProcessingException(
                    self.tr("セルサイズを取得できないため sigma を換算できません。"))
            sigma_cells = smooth_sigma_m / cell_size
            if sigma_cells < 0.3:
                feedback.pushWarning(self.tr(
                    "sigma がセルサイズに対して小さすぎます (%.2f セル)。"
                    "平滑化がほとんど効きません。") % sigma_cells)

        if smooth_method != smoothing.METHOD_NONE:
            radius = smoothing.smoothing_radius(
                smooth_method, smooth_size, sigma_cells)
            feedback.pushInfo(self.tr(
                "平滑化: %s (カーネル半径 %d セル, バックエンド %s)")
                % (smooth_method, radius, smoothing.backend_description()))

        try:
            rows, cols, heights, counts = self._scan(
                reader, window_size, min_height, merge_plateaus,
                block_size, smooth_method, smooth_size, sigma_cells,
                height_from_smoothed, feedback)
        finally:
            reader.close()

        if feedback.isCanceled():
            return {self.OUTPUT: dest_id}

        feedback.pushInfo(
            self.tr("樹頂点 %d 点を抽出しました。") % rows.size)

        xs, ys = local_maxima.cell_to_map(rows, cols, geotransform)
        self._write(sink, xs, ys, heights, counts, feedback)

        return {self.OUTPUT: dest_id}

    # ------------------------------------------------------------------
    def _scan(self, reader, window_size, min_height, merge_plateaus,
              block_size, smooth_method, smooth_size, sigma_cells,
              height_from_smoothed, feedback):
        """ブロックを走査して候補セルを集める."""
        # 平滑化もカーネル半径のぶん周囲を必要とするので halo に足す。
        # これを忘れるとブロック境界付近の平滑値が全域処理と一致しなくなる。
        halo = window_size // 2 + smoothing.smoothing_radius(
            smooth_method, smooth_size, sigma_cells)
        total = raster_reader.count_blocks(
            reader.width, reader.height, block_size)

        row_parts = []
        col_parts = []
        height_parts = []
        processed = 0

        for window in raster_reader.iter_blocks(
                reader.width, reader.height, block_size, halo):
            if feedback.isCanceled():
                break

            array = reader.read(window)
            valid = local_maxima.valid_mask(array, reader.nodata)

            if smooth_method == smoothing.METHOD_NONE:
                surface = array
            else:
                surface = smoothing.smooth_chm(
                    array, valid, smooth_method, smooth_size, sigma_cells)

            # 極大は平滑化後の面で探すが, 有効セルの判定は元 CHM に従う。
            mask = local_maxima.local_maxima_mask(
                surface, window_size, min_height=None, nodata=None,
                valid=valid)

            # 最低樹高は「出力する樹高」に対して掛ける。
            height_surface = surface if height_from_smoothed else array
            if min_height is not None:
                mask &= height_surface >= np.float32(min_height)

            core_mask = mask[window.core_slice]

            local_rows, local_cols = np.nonzero(core_mask)
            if local_rows.size:
                row_parts.append(local_rows.astype(np.int64) + window.core_row)
                col_parts.append(local_cols.astype(np.int64) + window.core_col)
                # 高さは読み込み済みの配列からそのまま取る (再読込しない)
                core_heights = height_surface[window.core_slice]
                height_parts.append(
                    core_heights[local_rows, local_cols].astype(np.float64))

            processed += 1
            feedback.setProgress(int(90.0 * processed / total))

        if not row_parts:
            empty_f = np.empty(0, dtype=np.float64)
            return empty_f, empty_f, empty_f, np.empty(0, dtype=np.int32)

        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        heights = np.concatenate(height_parts)

        return self._finalize(rows, cols, heights, merge_plateaus, feedback)

    def _finalize(self, rows, cols, heights, merge_plateaus, feedback):
        if not merge_plateaus:
            return (rows.astype(np.float64), cols.astype(np.float64), heights,
                    np.ones(rows.size, dtype=np.int32))

        feedback.setProgressText(
            self.tr("平坦な樹頂部を統合しています..."))
        labels, count = local_maxima.cluster_candidates(rows, cols)
        cell_counts = np.bincount(labels, minlength=count).astype(np.int32)
        row_centroid = np.bincount(
            labels, weights=rows, minlength=count) / cell_counts
        col_centroid = np.bincount(
            labels, weights=cols, minlength=count) / cell_counts
        height_out = np.zeros(count, dtype=np.float64)
        np.maximum.at(height_out, labels, heights)
        return row_centroid, col_centroid, height_out, cell_counts

    def _write(self, sink, xs, ys, heights, counts, feedback):
        total = xs.size
        for i in range(total):
            if feedback.isCanceled():
                return
            feature = QgsFeature()
            feature.setGeometry(
                QgsGeometry.fromPointXY(QgsPointXY(float(xs[i]), float(ys[i]))))
            feature.setAttributes(
                [i + 1, float(heights[i]), int(counts[i])])
            sink.addFeature(feature, QgsFeatureSink.FastInsert)
            if total and i % 5000 == 0:
                feedback.setProgress(90 + int(10.0 * i / total))
