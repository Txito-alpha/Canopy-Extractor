"""コアの検証: 総当たり実装との一致確認とタイル分割の整合性確認.

QGIS なしで実行できる:
    python -m canopy_extractor.tests.test_local_maxima
"""

from __future__ import annotations

import sys
import time
import unittest

import numpy as np

from ..core import local_maxima, raster_reader


def reference_mask(chm, window_size, min_height, nodata=None):
    """モデルと同じことを愚直に書いた参照実装 (遅いが確実)."""
    height, width = chm.shape
    radius = window_size // 2
    result = np.zeros(chm.shape, dtype=bool)
    for row in range(height):
        for col in range(width):
            value = chm[row, col]
            if not np.isfinite(value):
                continue
            if nodata is not None and value == nodata:
                continue
            if min_height is not None and value < min_height:
                continue
            r0, r1 = max(0, row - radius), min(height, row + radius + 1)
            c0, c1 = max(0, col - radius), min(width, col + radius + 1)
            window = chm[r0:r1, c0:c1]
            valid = np.isfinite(window)
            if nodata is not None:
                valid &= window != nodata
            if not valid.any():
                continue
            if value >= window[valid].max():
                result[row, col] = True
    return result


class TestMaximumFilter(unittest.TestCase):

    def test_matches_scipy(self):
        if not local_maxima.HAS_SCIPY:
            self.skipTest("scipy 無し")
        from scipy import ndimage

        rng = np.random.default_rng(0)
        array = rng.random((97, 131)).astype(np.float32)
        for size in (3, 5, 7, 15):
            expected = ndimage.maximum_filter(array, size=size, mode="nearest")
            actual = local_maxima._maximum_filter_numpy(array, size)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


class TestLocalMaximaMask(unittest.TestCase):

    def test_matches_reference(self):
        rng = np.random.default_rng(1)
        array = (rng.random((80, 90)) * 30).astype(np.float32)
        for size in (3, 7, 11):
            expected = reference_mask(array, size, 6.0)
            actual = local_maxima.local_maxima_mask(array, size, 6.0)
            np.testing.assert_array_equal(actual, expected)

    def test_nodata_is_excluded(self):
        array = np.full((20, 20), 10.0, dtype=np.float32)
        array[5, 5] = -9999.0
        array[10, 10] = 25.0
        mask = local_maxima.local_maxima_mask(
            array, 7, min_height=6.0, nodata=-9999.0)
        self.assertFalse(mask[5, 5])
        self.assertTrue(mask[10, 10])

    def test_min_height_filters(self):
        array = np.zeros((30, 30), dtype=np.float32)
        array[10, 10] = 4.0
        array[20, 20] = 12.0
        mask = local_maxima.local_maxima_mask(array, 7, min_height=6.0)
        self.assertFalse(mask[10, 10])
        self.assertTrue(mask[20, 20])


class TestPlateauMerging(unittest.TestCase):

    def test_flat_top_becomes_single_point(self):
        array = np.zeros((40, 40), dtype=np.float32)
        array[18:21, 18:21] = 20.0  # 3x3 の平坦な樹冠
        rows, cols, heights, counts = local_maxima.extract_tree_tops(
            array, window_size=7, min_height=6.0)
        self.assertEqual(rows.size, 1)
        self.assertAlmostEqual(float(rows[0]), 19.0)
        self.assertAlmostEqual(float(cols[0]), 19.0)
        self.assertAlmostEqual(float(heights[0]), 20.0)
        self.assertEqual(int(counts[0]), 9)

    def test_two_separate_trees(self):
        array = np.zeros((60, 60), dtype=np.float32)
        array[15, 15] = 18.0
        array[45, 45] = 22.0
        rows, cols, heights, _ = local_maxima.extract_tree_tops(
            array, window_size=7, min_height=6.0)
        self.assertEqual(rows.size, 2)
        np.testing.assert_allclose(sorted(heights), [18.0, 22.0])

    def test_label_propagation_matches_scipy(self):
        rng = np.random.default_rng(2)
        array = (rng.random((120, 120)) * 30).astype(np.float32)
        array = np.round(array)  # 同値を大量に作ってプラトーを誘発
        mask = local_maxima.local_maxima_mask(array, 7, 6.0)
        rows, cols = np.nonzero(mask)
        labels_a, count_a = local_maxima.cluster_candidates(rows, cols)

        saved_cc = local_maxima._cc
        local_maxima._cc = None  # scipy 無し経路を強制
        try:
            labels_b, count_b = local_maxima.cluster_candidates(rows, cols)
        finally:
            local_maxima._cc = saved_cc

        self.assertEqual(count_a, count_b)
        self.assertEqual(
            _partition(labels_a), _partition(labels_b))


def _partition(labels):
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return sorted(tuple(v) for v in groups.values())


class TestTiling(unittest.TestCase):
    """ブロック分割しても全域処理と同じ結果になることを確認する."""

    def test_blocks_match_whole_raster(self):
        rng = np.random.default_rng(3)
        array = (rng.random((300, 260)) * 30).astype(np.float32)
        window_size = 7
        halo = window_size // 2

        whole = local_maxima.local_maxima_mask(array, window_size, 6.0)

        tiled = np.zeros(array.shape, dtype=bool)
        for window in raster_reader.iter_blocks(
                array.shape[1], array.shape[0], 64, halo):
            block = array[
                window.read_row:window.read_row + window.read_height,
                window.read_col:window.read_col + window.read_width]
            mask = local_maxima.local_maxima_mask(block, window_size, 6.0)
            tiled[
                window.core_row:window.core_row + window.core_height,
                window.core_col:window.core_col + window.core_width,
            ] = mask[window.core_slice]

        np.testing.assert_array_equal(tiled, whole)


def benchmark(size=4000, window_size=7):
    """参考: 大きめの合成 CHM での所要時間."""
    rng = np.random.default_rng(0)
    base = rng.random((size // 20, size // 20)) * 25
    array = np.kron(base, np.ones((20, 20)))[:size, :size]
    array = array + rng.random((size, size)) * 2
    array = array.astype(np.float32)

    start = time.perf_counter()
    rows, _, _, _ = local_maxima.extract_tree_tops(array, window_size, 6.0)
    elapsed = time.perf_counter() - start
    print("%dx%d セル / 窓 %d -> 樹頂点 %d 点, %.2f 秒 (scipy=%s)"
          % (size, size, window_size, rows.size, elapsed,
             local_maxima.HAS_SCIPY))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        sys.argv.remove("--bench")
        benchmark()
    unittest.main()
