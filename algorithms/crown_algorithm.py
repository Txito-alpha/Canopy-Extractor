"""樹冠ポリゴン生成 Processing アルゴリズム."""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication, QVariant

from ..core import crown, polygonize, raster_reader


class CrownAlgorithm(QgsProcessingAlgorithm):
    """樹頂点と CHM から単木の樹冠ポリゴンを作る."""

    CHM = "CHM"
    BAND = "BAND"
    TREE_TOPS = "TREE_TOPS"
    ID_FIELD = "ID_FIELD"
    METHOD = "METHOD"
    MIN_HEIGHT = "MIN_HEIGHT"
    TH_SEED = "TH_SEED"
    TH_CROWN = "TH_CROWN"
    EXCLUSION = "EXCLUSION"
    MAX_RADIUS = "MAX_RADIUS"
    SHAPE = "SHAPE"
    FILL_HOLES = "FILL_HOLES"
    BLOCK_SIZE = "BLOCK_SIZE"
    OUTPUT = "OUTPUT"

    def tr(self, string):
        return QCoreApplication.translate("CrownAlgorithm", string)

    def createInstance(self):
        return CrownAlgorithm()

    def name(self):
        return "extractcrowns"

    def displayName(self):
        return self.tr("樹冠ポリゴン生成")

    def group(self):
        return self.tr("森林解析")

    def groupId(self):
        return "forestanalysis"

    def shortHelpString(self):
        return self.tr(
            "樹頂点ポイントと CHM から, 単木ごとの樹冠ポリゴンを作成します。\n\n"
            "【領域拡張】Dalponte & Coomes (2016) の方式。樹頂点を種として"
            "4 近傍へ条件付きで広げます。樹冠の形が CHM の起伏に沿うため, "
            "針葉樹人工林では概ね良好な結果になります。\n\n"
            "【ボロノイ】Silva et al. (2016) の方式。最大半径を切った最近傍割当で, "
            "低いセルを除去します。単純で速いぶん, 樹冠の形は幾何的になります。\n\n"
            "樹冠の最大半径はメートルで指定します (内部でセル数に換算)。"
            "どちらの方式も樹冠が種から一定距離内に収まるため, タイル分割しても"
            "ポリゴンが境界で分断されることはありません。\n\n"
            "出力の crown_area は樹冠投影面積 (m2) で, セル数から直接求めています。"
            "胸高直径推定式に使う値です。\n\n"
            "注意: 衛星画像から推論した CHM は実効解像度がグリッド間隔より粗いため, "
            "樹冠の形状の細部は信頼できません。面積を使う分には問題ありませんが, "
            "形状から樹種を判別するような用途には使わないでください。"
        )

    # ------------------------------------------------------------------
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.CHM, self.tr("CHM ラスタ")))

        self.addParameter(QgsProcessingParameterBand(
            self.BAND, self.tr("バンド"), parentLayerParameterName=self.CHM,
            defaultValue=1))

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.TREE_TOPS, self.tr("樹頂点"),
            [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterField(
            self.ID_FIELD, self.tr("樹木 ID フィールド (任意)"),
            parentLayerParameterName=self.TREE_TOPS, optional=True))

        self.addParameter(QgsProcessingParameterEnum(
            self.METHOD, self.tr("分割方式"),
            options=[self.tr("領域拡張 (Dalponte)"),
                     self.tr("ボロノイ (Silva)")],
            defaultValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_RADIUS, self.tr("樹冠の最大半径 (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=5.0, minValue=0.1))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_HEIGHT, self.tr("樹木とみなす最低高 (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=2.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.TH_SEED, self.tr("種の高さに対する割合 (領域拡張)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.45, minValue=0.0, maxValue=1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.TH_CROWN, self.tr("領域平均高に対する割合 (領域拡張)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.55, minValue=0.0, maxValue=1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.EXCLUSION, self.tr("除去する高さの割合 (ボロノイ)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.3, minValue=0.0, maxValue=1.0))

        shape = QgsProcessingParameterEnum(
            self.SHAPE, self.tr("樹冠の広がり方"),
            options=[self.tr("円形"), self.tr("正方形 (原著と同じ)")],
            defaultValue=0)
        shape.setFlags(shape.flags() | QgsProcessingParameterEnum.FlagAdvanced)
        self.addParameter(shape)

        fill = QgsProcessingParameterBoolean(
            self.FILL_HOLES, self.tr("樹冠内の穴を埋める"), defaultValue=True)
        fill.setFlags(fill.flags() | QgsProcessingParameterBoolean.FlagAdvanced)
        self.addParameter(fill)

        block = QgsProcessingParameterNumber(
            self.BLOCK_SIZE, self.tr("処理ブロックサイズ (セル数)"),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=2048, minValue=256, maxValue=32768)
        block.setFlags(block.flags()
                       | QgsProcessingParameterNumber.FlagAdvanced)
        self.addParameter(block)

        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, self.tr("樹冠ポリゴン"),
            QgsProcessing.TypeVectorPolygon))

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsRasterLayer(parameters, self.CHM, context)
        if layer is None:
            raise QgsProcessingException(
                self.invalidRasterError(parameters, self.CHM))

        source = self.parameterAsSource(parameters, self.TREE_TOPS, context)
        if source is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.TREE_TOPS))

        band = self.parameterAsInt(parameters, self.BAND, context)
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        method = (crown.METHOD_REGION_GROWING, crown.METHOD_VORONOI)[
            self.parameterAsEnum(parameters, self.METHOD, context)]
        max_radius_m = self.parameterAsDouble(
            parameters, self.MAX_RADIUS, context)
        min_height = self.parameterAsDouble(
            parameters, self.MIN_HEIGHT, context)
        th_seed = self.parameterAsDouble(parameters, self.TH_SEED, context)
        th_crown = self.parameterAsDouble(parameters, self.TH_CROWN, context)
        exclusion = self.parameterAsDouble(parameters, self.EXCLUSION, context)
        shape = (crown.SHAPE_CIRCLE, crown.SHAPE_SQUARE)[
            self.parameterAsEnum(parameters, self.SHAPE, context)]
        do_fill = self.parameterAsBool(parameters, self.FILL_HOLES, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("height", QVariant.Double))
        fields.append(QgsField("crown_area", QVariant.Double))
        fields.append(QgsField("crown_dia", QVariant.Double))
        fields.append(QgsField("crown_max", QVariant.Double))
        fields.append(QgsField("crown_mean", QVariant.Double))
        fields.append(QgsField("n_cells", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.MultiPolygon, layer.crs())
        if sink is None:
            raise QgsProcessingException(
                self.invalidSinkError(parameters, self.OUTPUT))

        reader = raster_reader.open_reader(layer, band, feedback)
        try:
            geotransform = reader.geotransform
            cell_size = abs(geotransform[1])
            if cell_size <= 0.0:
                raise QgsProcessingException(
                    self.tr("セルサイズを取得できません。"))

            max_cr = max(1, int(round(max_radius_m / cell_size)))
            cell_area = abs(geotransform[1] * geotransform[5])

            feedback.pushInfo(self.tr(
                "セルサイズ %.3f m, 樹冠最大半径 %.1f m = %d セル")
                % (cell_size, max_radius_m, max_cr))
            feedback.pushInfo(crown.backend_description())
            if do_fill and crown._ndimage is None:
                feedback.pushWarning(self.tr(
                    "scipy が無いため穴埋めをスキップします。"))
                do_fill = False

            seeds = self._read_seeds(
                source, geotransform, reader, id_field, feedback)
            if seeds is None:
                return {self.OUTPUT: dest_id}
            seed_row, seed_col, seed_id = seeds
            feedback.pushInfo(
                self.tr("樹頂点 %d 点を読み込みました。") % seed_row.size)

            self._process_blocks(
                reader, sink, seed_row, seed_col, seed_id, geotransform,
                cell_area, method, max_cr, min_height, th_seed, th_crown,
                exclusion, shape, do_fill, block_size, feedback)
        finally:
            reader.close()

        return {self.OUTPUT: dest_id}

    # ------------------------------------------------------------------
    def _read_seeds(self, source, geotransform, reader, id_field, feedback):
        """樹頂点をセル座標に変換し, 行順に並べ替えて返す."""
        origin_x, x_size = geotransform[0], geotransform[1]
        origin_y, y_size = geotransform[3], geotransform[5]

        rows, cols, ids = [], [], []
        for index, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                return None
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            point = geometry.asPoint()
            col = int((point.x() - origin_x) / x_size)
            row = int((point.y() - origin_y) / y_size)
            if not (0 <= row < reader.height and 0 <= col < reader.width):
                continue
            rows.append(row)
            cols.append(col)
            if id_field:
                value = feature[id_field]
                ids.append(int(value) if value is not None else index + 1)
            else:
                ids.append(index + 1)

        if not rows:
            feedback.pushWarning(self.tr("有効な樹頂点がありません。"))
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64))

        seed_row = np.array(rows, dtype=np.int64)
        seed_col = np.array(cols, dtype=np.int64)
        seed_id = np.array(ids, dtype=np.int64)

        order = np.argsort(seed_row, kind="stable")
        return seed_row[order], seed_col[order], seed_id[order]

    # ------------------------------------------------------------------
    def _process_blocks(self, reader, sink, seed_row, seed_col, seed_id,
                        geotransform, cell_area, method, max_cr, min_height,
                        th_seed, th_crown, exclusion, shape, do_fill,
                        block_size, feedback):
        # ある樹冠のセルを奪いうる競合の種は, 種から最大 2 * max_cr 離れている。
        # (セルは種から max_cr 以内, 競合はそのセルから max_cr 以内)
        # halo を max_cr にすると密な林分でタイル境界付近の帰属がずれる。
        halo = 2 * max_cr + 1
        total = raster_reader.count_blocks(
            reader.width, reader.height, block_size)
        processed = 0
        written = 0
        route_logged = False

        for window in raster_reader.iter_blocks(
                reader.width, reader.height, block_size, halo):
            if feedback.isCanceled():
                return

            # 読み出し窓に入る種を選ぶ。種は行でソート済み。
            lo = np.searchsorted(seed_row, window.read_row, side="left")
            hi = np.searchsorted(
                seed_row, window.read_row + window.read_height, side="left")
            if lo == hi:
                processed += 1
                feedback.setProgress(int(100.0 * processed / total))
                continue

            in_block = ((seed_col[lo:hi] >= window.read_col)
                        & (seed_col[lo:hi]
                           < window.read_col + window.read_width))
            local_row = seed_row[lo:hi][in_block] - window.read_row
            local_col = seed_col[lo:hi][in_block] - window.read_col
            local_id = seed_id[lo:hi][in_block]
            if local_row.size == 0:
                processed += 1
                feedback.setProgress(int(100.0 * processed / total))
                continue

            array = reader.read(window)
            valid = np.isfinite(array)
            if reader.nodata is not None and np.isfinite(reader.nodata):
                valid &= array != np.float32(reader.nodata)

            if method == crown.METHOD_REGION_GROWING:
                labels = crown.grow_region(
                    array, local_row, local_col, th_tree=min_height,
                    th_seed=th_seed, th_cr=th_crown, max_cr=max_cr,
                    shape=shape, valid=valid)
            else:
                labels = crown.voronoi_crowns(
                    array, local_row, local_col, max_cr=max_cr,
                    exclusion=exclusion, th_tree=min_height, shape=shape,
                    valid=valid)

            if do_fill:
                labels = crown.fill_holes(labels)

            # 種がコア領域内にある樹冠だけを残す。
            # 樹冠は種から max_cr 以内に収まるので, 残した樹冠は
            # 読み出し窓の内側で完結しており全域処理と一致する。
            core_row0 = window.core_row - window.read_row
            core_col0 = window.core_col - window.read_col
            keep = ((local_row >= core_row0)
                    & (local_row < core_row0 + window.core_height)
                    & (local_col >= core_col0)
                    & (local_col < core_col0 + window.core_width))
            if not keep.any():
                processed += 1
                feedback.setProgress(int(100.0 * processed / total))
                continue

            keep_labels = np.nonzero(keep)[0] + 1
            drop = np.ones(local_row.size + 1, dtype=bool)
            drop[keep_labels] = False
            labels[drop[labels]] = 0

            counts, maxima, means = crown.crown_statistics(
                array, labels, local_row.size)

            block_gt = (
                geotransform[0] + window.read_col * geotransform[1],
                geotransform[1], geotransform[2],
                geotransform[3] + window.read_row * geotransform[5],
                geotransform[4], geotransform[5],
            )
            shapes, route = polygonize.polygonize(labels, block_gt)
            if not route_logged:
                feedback.pushInfo(self.tr("ポリゴン化: %s") % route)
                route_logged = True

            seed_heights = array[local_row, local_col]
            for label, wkb in shapes.items():
                geometry = QgsGeometry()
                geometry.fromWkb(wkb)
                if geometry.isEmpty():
                    continue
                if not geometry.isMultipart():
                    geometry.convertToMultiType()

                index = label - 1
                n_cells = int(counts[label])
                area = n_cells * cell_area
                feature = QgsFeature()
                feature.setGeometry(geometry)
                feature.setAttributes([
                    int(local_id[index]),
                    float(seed_heights[index]),
                    float(area),
                    float(2.0 * np.sqrt(area / np.pi)) if area > 0 else 0.0,
                    float(maxima[label]),
                    float(means[label]),
                    n_cells,
                ])
                sink.addFeature(feature, QgsFeatureSink.FastInsert)
                written += 1

            processed += 1
            feedback.setProgress(int(100.0 * processed / total))

        feedback.pushInfo(
            self.tr("樹冠 %d 個を出力しました。") % written)
