"""Canopy Extractor - CHM から樹頂点と樹冠ポリゴンを抽出する QGIS プラグイン."""


def classFactory(iface):  # noqa: N802 (QGIS API)
    from .plugin import CanopyExtractorPlugin

    return CanopyExtractorPlugin(iface)
