"""プラグイン本体 (Processing プロバイダの登録のみ)."""

from __future__ import annotations

from qgis.core import QgsApplication

from .provider import CanopyExtractorProvider


class CanopyExtractorPlugin(object):

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = CanopyExtractorProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
