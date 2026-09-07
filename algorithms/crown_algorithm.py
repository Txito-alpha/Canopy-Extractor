"""樹冠ポリゴン生成 Processing アルゴリズム."""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsFillSymbol,
    QgsField,
    QgsFields,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QCoreApplication, QVariant

from ..core import crown, polygonize, raster_reader


class CrownStyler(QgsProcessingLayerPostProcessorInterface):
    """出力レイヤに既定のスタイル (不透明度 50%) を適用する.

    ポストプロセッサは QGIS 側から参照されるだけなので, Python 側で
    参照を保持しておかないとガベージコレクトされてしまう。クラス変数に
    持たせるのが Processing での定石。
    """

    instance = None

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        if not isinstance(layer, QgsVectorLayer):
            return
        symbol = QgsFillSymbol.createSimple({
            "color": "111,168,86",
            "outline_color": "45,84,32",
            "outline_width": "0.2",
            "outline_width_unit": "MM",
            "style": "solid",
        })
        symbol.setOpacity(0.5)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()

    @staticmethod
    def create():
        CrownStyler.instance = CrownStyler()
        return CrownStyler.instance


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
    REQUIRE_CONNECTED = "REQUIRE_CONNECTED"
    MASK = "MASK"
    MASK_BAND = "MASK_BAND"
    MASK_INVERT = "MASK_INVERT"
    EXTENT = "EXTENT"
    AREA = "AREA"
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
            "【ボロノイ】Silva et al. (2016) の方式 (既定)。最大半径を切った"
            "最近傍割当で, 低いセルを除去します。単純で速く, 樹冠の大きさが"
            "揃うため樹冠投影面積の集計には扱いやすい方式です。\n\n"
            "【領域拡張】Dalponte & Coomes (2016) の方式。樹頂点を種として"
            "4 近傍へ条件付きで広げます。樹冠の形が CHM の起伏に沿います。\n\n"
            "【無立木地の除外】3 段階で効かせられます。\n"
            "1. 樹木とみなす最低高: この高さに満たないセルは樹冠に入りません。\n"
            "2. 樹頂点と連結した部分に限る: 林道や無立木地のギャップを飛び越えて"
            "向こう側のセルを取り込むのを防ぎます。ボロノイは距離だけで割り当てる"
            "ため, この指定が効きます。\n"
            "3. 立木地マスク: 立木地を 0 以外, 無立木地を 0 か NoData とした"
            "ラスタを指定すると, そのセルを樹冠から除外します。小班界などの"
            "ポリゴンをラスタ化したものも使えます。\n\n"
            "【処理範囲】\n"
            "「処理範囲」でキャンバスの表示範囲や座標を指定すると, その範囲だけを"
            "処理します。「処理範囲ポリゴン」でポリゴンレイヤを指定すると, "
            "ポリゴンの外側にあるセルをすべて無立木地扱いにして樹冠から除外します"
            "(立木地マスクと同じ扱いで, 内部でラスタ化して使います)。\n\n"
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
            defaultValue=1))

        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_RADIUS, self.tr("樹冠の最大半径 (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0, minValue=0.1))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_HEIGHT, self.tr("樹木とみなす最低高 (m)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=2.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.TH_SEED, self.tr("種の高さに対する割合 (領域拡張)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.55, minValue=0.0, maxValue=1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.TH_CROWN, self.tr("領域平均高に対する割合 (領域拡張)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.55, minValue=0.0, maxValue=1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.EXCLUSION, self.tr("除去する高さの割合 (ボロノイ)"),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.30, minValue=0.0, maxValue=1.0))

        self.addParameter(QgsProcessingParameterBoolean(
            self.REQUIRE_CONNECTED,
            self.tr("樹冠を樹頂点と連結した部分に限る"), defaultValue=True))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.MASK, self.tr("立木地マスク (任意)"), optional=True))

        self.addParameter(QgsProcessingParameterBand(
            self.MASK_BAND, self.tr("マスクのバンド"),
            parentLayerParameterName=self.MASK, defaultValue=1,
            optional=True))

        self.addParameter(QgsProcessingParameterBoolean(
            self.MASK_INVERT, self.tr("マスクの意味を反転する"),
            defaultValue=False))

        extent = QgsProcessingParameterExtent(
            self.EXTENT, self.tr("処理範囲 (任意)"), optional=True)
        self.addParameter(extent)

        area = QgsProcessingParameterFeatureSource(
            self.AREA, self.tr("処理範囲ポリゴン (任意)"),
            [QgsProcessing.TypeVectorPolygon], optional=True)
        self.addParameter(area)

        shape = QgsProcessingParameterEnum(
            self.SHAPE, self.tr("樹冠の広がり方"),
            options=[self.tr("円形"), self.tr("正方形 (原著と同じ)")],
            defaultValue=0)
        shape.setFlags(shape.flags() | QgsProcessingParameterEnum.FlagAdvanced)
        self.addParameter(shape)

        fill = QgsProcessingParameterBoolean(
            self.FILL_HOLES, self.tr("樹冠内の穴を埋める"), defaultValue=False)
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
        require_connected = self.parameterAsBool(
            parameters, self.REQUIRE_CONNECTED, context)
        mask_layer = self.parameterAsRasterLayer(
            parameters, self.MASK, context)
        mask_band = self.parameterAsInt(parameters, self.MASK_BAND, context)
        mask_invert = self.parameterAsBool(
            parameters, self.MASK_INVERT, context)
        area_source = self.parameterAsSource(parameters, self.AREA, context)
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

        if context.willLoadLayerOnCompletion(dest_id):
            context.layerToLoadOnCompletionDetails(dest_id).setPostProcessor(
                CrownStyler.create())

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

            mask_sampler = None
            if mask_layer is not None:
                if mask_layer.crs() != layer.crs():
                    feedback.pushWarning(self.tr(
                        "マスクの CRS (%s) が CHM (%s) と異なります。"
                        "再投影は行いません。")
                        % (mask_layer.crs().authid(), layer.crs().authid()))
                mask_sampler = raster_reader.MaskSampler(
                    mask_layer, mask_band)
                feedback.pushInfo(self.tr(
                    "立木地マスクを使用します (%s%s)")
                    % (mask_layer.name(),
                       self.tr(", 反転") if mask_invert else ""))

            region = None
            extent = self.parameterAsExtent(
                parameters, self.EXTENT, context, layer.crs())
            if not extent.isNull():
                region = raster_reader.extent_to_region(
                    geotransform, reader.width, reader.height,
                    extent.xMinimum(), extent.yMinimum(),
                    extent.xMaximum(), extent.yMaximum())
                if region is None:
                    raise QgsProcessingException(
                        self.tr("処理範囲が CHM の範囲と重なりません。"))
                feedback.pushInfo(self.tr(
                    "処理範囲: 列 %d-%d, 行 %d-%d (%d x %d セル)")
                    % (region[0], region[0] + region[2],
                       region[1], region[1] + region[3],
                       region[2], region[3]))

            area_rasterizer = None
            if area_source is not None:
                area_rasterizer = raster_reader.AreaRasterizer(
                    area_source, layer.crs(), context.transformContext())
                if area_rasterizer.feature_count == 0:
                    feedback.pushWarning(self.tr(
                        "処理範囲ポリゴンに有効なフィーチャがありません。"
                        "無視します。"))
                    area_rasterizer = None
                else:
                    feedback.pushInfo(self.tr(
                        "処理範囲ポリゴンを使用します (%d フィーチャ)")
                        % area_rasterizer.feature_count)

            seeds = self._read_seeds(
                source, layer, geotransform, reader, id_field, context,
                feedback)
            if seeds is None:
                return {self.OUTPUT: dest_id}
            seed_row, seed_col, seed_id = seeds
            if seed_row.size == 0:
                return {self.OUTPUT: dest_id}

            self._process_blocks(
                reader, sink, seed_row, seed_col, seed_id, geotransform,
                cell_area, method, max_cr, min_height, th_seed, th_crown,
                exclusion, shape, do_fill, require_connected, mask_sampler,
                mask_invert, area_rasterizer, region, block_size, feedback)
        finally:
            reader.close()

        return {self.OUTPUT: dest_id}

    # ------------------------------------------------------------------
    def _read_seeds(self, source, layer, geotransform, reader, id_field,
                    context, feedback):
        """樹頂点をセル座標に変換し, 行順に並べ替えて返す.

        樹頂点レイヤの CRS が CHM と違う場合は変換する。変換しないと
        全ての点がラスタ範囲外と判定され, 黙って 0 件になる。
        """
        origin_x, x_size = geotransform[0], geotransform[1]
        origin_y, y_size = geotransform[3], geotransform[5]

        transform = None
        source_crs = source.sourceCrs()
        raster_crs = layer.crs()
        if (source_crs.isValid() and raster_crs.isValid()
                and source_crs != raster_crs):
            transform = QgsCoordinateTransform(
                source_crs, raster_crs, context.transformContext())
            feedback.pushInfo(self.tr("樹頂点を %s から %s に変換します。")
                              % (source_crs.authid(), raster_crs.authid()))

        rows, cols, ids = [], [], []
        total = 0
        no_geometry = 0
        outside = 0

        for index, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                return None
            total += 1
            geometry = feature.geometry()
            if geometry is None or geometry.isNull() or geometry.isEmpty():
                no_geometry += 1
                continue
            if transform is not None:
                if geometry.transform(transform) != 0:
                    no_geometry += 1
                    continue

            if geometry.isMultipart():
                points = geometry.asMultiPoint()
            else:
                points = [geometry.asPoint()]

            for point in points:
                col = int(np.floor((point.x() - origin_x) / x_size))
                row = int(np.floor((point.y() - origin_y) / y_size))
                if not (0 <= row < reader.height and 0 <= col < reader.width):
                    outside += 1
                    continue
                rows.append(row)
                cols.append(col)
                ids.append(self._seed_id(feature, id_field, index))

        feedback.pushInfo(self.tr(
            "樹頂点: 入力 %d 件 / 採用 %d 点 / ジオメトリ無効 %d / 範囲外 %d")
            % (total, len(rows), no_geometry, outside))

        if not rows:
            self._report_extent_mismatch(source, layer, geotransform, reader,
                                         feedback)
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64))

        seed_row = np.array(rows, dtype=np.int64)
        seed_col = np.array(cols, dtype=np.int64)
        seed_id = np.array(ids, dtype=np.int64)

        order = np.argsort(seed_row, kind="stable")
        return seed_row[order], seed_col[order], seed_id[order]

    @staticmethod
    def _seed_id(feature, id_field, index):
        if not id_field:
            return index + 1
        try:
            value = feature[id_field]
            return index + 1 if value is None else int(value)
        except (KeyError, TypeError, ValueError):
            return index + 1

    def _report_extent_mismatch(self, source, layer, geotransform, reader,
                                feedback):
        """1 点も採用されなかったとき, 範囲の食い違いをログに出す."""
        feedback.pushWarning(self.tr(
            "有効な樹頂点がありません。樹頂点と CHM の範囲を確認してください。"))
        extent = source.sourceExtent()
        feedback.pushInfo(self.tr("樹頂点の範囲: %s (%s)")
                          % (extent.toString(2), source.sourceCrs().authid()))
        right = geotransform[0] + reader.width * geotransform[1]
        bottom = geotransform[3] + reader.height * geotransform[5]
        feedback.pushInfo(self.tr("CHM の範囲: %.2f, %.2f - %.2f, %.2f (%s)")
                          % (geotransform[0], bottom, right, geotransform[3],
                             layer.crs().authid()))

    # ------------------------------------------------------------------
    @staticmethod
    def _wkb_roundtrip_works():
        """自前で組んだ WKB を QgsGeometry が読めるか試す.

        1 セル四方のポリゴンを作って面積が 1 になることを確認する。
        """
        try:
            probe = polygonize.polygon_wkb(
                [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),
                  (0.0, 0.0)]])
            geometry = QgsGeometry()
            geometry.fromWkb(probe)
            return (not geometry.isEmpty()
                    and abs(geometry.area() - 1.0) < 1e-6)
        except Exception:  # noqa: BLE001 - 判定できなければ使わない
            return False

    @staticmethod
    def _geometry_from_wkb(wkb):
        geometry = QgsGeometry()
        geometry.fromWkb(wkb)
        return geometry

    @staticmethod
    def _geometry_from_rings(polygons):
        """環の座標列から MultiPolygon を組み立てる (WKB を経由しない)."""
        parts = []
        for rings in polygons:
            converted = [[QgsPointXY(float(x), float(y)) for x, y in ring]
                         for ring in rings if len(ring) >= 4]
            if converted:
                parts.append(converted)
        if not parts:
            return None
        return QgsGeometry.fromMultiPolygonXY(parts)

    # ------------------------------------------------------------------
    def _process_blocks(self, reader, sink, seed_row, seed_col, seed_id,
                        geotransform, cell_area, method, max_cr, min_height,
                        th_seed, th_crown, exclusion, shape, do_fill,
                        require_connected, mask_sampler, mask_invert,
                        area_rasterizer, region, block_size, feedback):
        # ある樹冠のセルを奪いうる競合の種は, 種から最大 2 * max_cr 離れている。
        # (セルは種から max_cr 以内, 競合はそのセルから max_cr 以内)
        # halo を max_cr にすると密な林分でタイル境界付近の帰属がずれる。
        halo = 2 * max_cr + 1
        total = raster_reader.count_blocks(
            reader.width, reader.height, block_size, region=region)
        processed = 0
        written = 0
        assigned_cells = 0
        produced_shapes = 0
        geometry_failures = 0
        sink_failures = 0
        excluded_seeds = 0
        route_logged = False

        # WKB 経路が使えるか先に確かめる。QgsGeometry.fromWkb は戻り値が無く,
        # 失敗しても null ジオメトリになるだけなので, 黙って全件消える。
        use_wkb = self._wkb_roundtrip_works()
        if use_wkb:
            feedback.pushInfo(self.tr("ジオメトリ生成: WKB 経路"))
            build_geometry = self._geometry_from_wkb
            run_polygonize = polygonize.polygonize
        else:
            feedback.pushWarning(self.tr(
                "WKB の解釈に失敗したため, 座標列からジオメトリを組み立てます。"))
            build_geometry = self._geometry_from_rings
            run_polygonize = polygonize.polygonize_rings

        for window in raster_reader.iter_blocks(
                reader.width, reader.height, block_size, halo,
                region=region):
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

            if mask_sampler is not None:
                try:
                    valid &= mask_sampler.valid_mask(
                        window, geotransform, mask_invert)
                except Exception as error:  # noqa: BLE001
                    raise QgsProcessingException(self.tr(
                        "立木地マスクを読めません: %s") % error)
                masked_seeds = valid[local_row, local_col]
                if not masked_seeds.any():
                    processed += 1
                    feedback.setProgress(int(100.0 * processed / total))
                    continue
                excluded_seeds += int((~masked_seeds).sum())

            if area_rasterizer is not None:
                try:
                    valid &= area_rasterizer.valid_mask(window, geotransform)
                except Exception as error:  # noqa: BLE001
                    raise QgsProcessingException(self.tr(
                        "処理範囲ポリゴンをラスタ化できません: %s") % error)
                masked_seeds = valid[local_row, local_col]
                if not masked_seeds.any():
                    processed += 1
                    feedback.setProgress(int(100.0 * processed / total))
                    continue
                excluded_seeds += int((~masked_seeds).sum())

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

            if require_connected:
                labels = crown.enforce_connectivity(
                    labels, local_row, local_col, max_cr)

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
            assigned_cells += int((labels > 0).sum())
            shapes, route = run_polygonize(labels, block_gt)
            produced_shapes += len(shapes)
            if not route_logged:
                feedback.pushInfo(self.tr("ポリゴン化: %s") % route)
                route_logged = True

            seed_heights = array[local_row, local_col]
            for label, source_shape in shapes.items():
                geometry = build_geometry(source_shape)
                if geometry is None or geometry.isEmpty():
                    geometry_failures += 1
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
                if not sink.addFeature(feature, QgsFeatureSink.FastInsert):
                    sink_failures += 1
                else:
                    written += 1

            processed += 1
            feedback.setProgress(int(100.0 * processed / total))

        feedback.pushInfo(self.tr(
            "割当セル %d / ポリゴン %d / 出力 %d 件")
            % (assigned_cells, produced_shapes, written))
        if excluded_seeds:
            feedback.pushInfo(self.tr(
                "マスクにより除外された樹頂点: 延べ %d 点") % excluded_seeds)
        if geometry_failures:
            feedback.pushWarning(self.tr(
                "ジオメトリを組み立てられなかった樹冠が %d 件あります。")
                % geometry_failures)
        if sink_failures:
            feedback.pushWarning(self.tr(
                "出力レイヤへの書き込みに失敗した樹冠が %d 件あります。")
                % sink_failures)
        if written == 0 and assigned_cells > 0:
            feedback.pushWarning(self.tr(
                "セルは樹冠に割り当てられましたが 1 件も出力されませんでした。"
                "ポリゴン化の経路を確認してください。"))
