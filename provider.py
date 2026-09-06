"""Processing プロバイダ."""

from __future__ import annotations

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtCore import QCoreApplication

from .algorithms.crown_algorithm import CrownAlgorithm
from .algorithms.tree_top_algorithm import TreeTopAlgorithm


class CanopyExtractorProvider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(TreeTopAlgorithm())
        self.addAlgorithm(CrownAlgorithm())

    def id(self):
        return "canopyextractor"

    def name(self):
        return QCoreApplication.translate(
            "CanopyExtractorProvider", "Canopy Extractor")

    def longName(self):
        return self.name()
