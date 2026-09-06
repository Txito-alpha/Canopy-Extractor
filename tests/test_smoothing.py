"""平滑化の検証: 総当たり参照実装との一致, NoData の非汚染, タイル整合性.

    python -m unittest canopy_extractor.tests.test_smoothing -v
"""

from __future__ import annotations

import unittest

import numpy as np

from ..core import local_maxima, raster_reader, smoothing


def reference_mean(chm, valid, size):
    """窓内の有効セルだけの平均を愚直に計算した参照実装."""
    height, width = chm.shape
    radius = size // 2
    out = np.full(chm.shape, np.nan, dtype=np.float64)
    for row in range(height):
        for col in range(width):
            r0, r1 = max(0, row - radius), min(height, row + radius + 1)
            c0, c1 = max(0, col - radius), min(width, col + radius + 1)
            window = chm[r0:r1, c0:c1]
            window_valid = valid[r0:r1, c0:c1]
            if not window_valid.any():
                continue
            out[row, col] = window[window_valid].mean()
    return out


def reference_gaussian(chm, valid, sigma):
    """ゼロ詰め正規化畳み込みの参照実装 (2 次元カーネルを直接適用)."""
    kernel_1d = smoothing.gaussian_kernel1d(sigma)
    kernel = np.outer(kernel_1d, kernel_1d)
    radius = kernel_1d.size // 2

    height, width = chm.shape
    values = np.where(valid, chm, 0.0).astype(np.float64)
    weights = valid.astype(np.float64)
    out = np.full(chm.shape, np.nan, dtype=np.float64)

    padded_v = np.pad(values, radius, mode="constant", constant_values=0.0)
    padded_w = np.pad(weights, radius, mode="constant", constant_values=0.0)

    for row in range(height):
        for col in range(width):
            v = (padded_v[row:row + kernel_1d.size,
                          col:col + kernel_1d.size] * kernel).sum()
            w = (padded_w[row:row + kernel_1d.size,
                          col:col + kernel_1d.size] * kernel).sum()
            if w > 1e-12:
                out[row, col] = v / w
    return out


class TestMeanFilter(unittest.TestCase):

    def test_matches_reference_all_valid(self):
        rng = np.random.default_rng(0)
        chm = (rng.random((30, 34)) * 25).astype(np.float32)
        valid = np.ones(chm.shape, dtype=bool)
        for size in (3, 5, 7):
            actual = smoothing.smooth_chm(
                chm, valid, smoothing.METHOD_MEAN, size=size)
            expected = reference_mean(chm, valid, size)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_matches_reference_with_nodata(self):
        rng = np.random.default_rng(1)
        chm = (rng.random((25, 25)) * 25).astype(np.float32)
        valid = rng.random(chm.shape) > 0.25
        actual = smoothing.smooth_chm(
            chm, valid, smoothing.METHOD_MEAN, size=5)
        expected = reference_mean(chm, valid, 5)
        both = np.isfinite(actual) & np.isfinite(expected)
        np.testing.assert_allclose(
            actual[both], expected[both], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(
            np.isfinite(actual), np.isfinite(expected))

    def test_nodata_does_not_bleed(self):
        """NoData の穴が周囲のセルの値を引き下げないこと."""
        chm = np.full((11, 11), 10.0, dtype=np.float32)
        valid = np.ones(chm.shape, dtype=bool)
        valid[5, 5] = False
        chm[5, 5] = -9999.0
        result = smoothing.smooth_chm(
            chm, valid, smoothing.METHOD_MEAN, size=3)
        # 穴の隣は「10.0 のセルだけの平均」なので 10.0 のまま
        self.assertAlmostEqual(float(result[4, 5]), 10.0, places=4)
        self.assertAlmostEqual(float(result[5, 4]), 10.0, places=4)
        # 素朴な畳み込みなら (8*10 + 0) / 9 = 8.89 に下がってしまう
        self.assertGreater(float(result[4, 5]), 9.9)

    def test_rejects_even_size(self):
        chm = np.zeros((5, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            smoothing.smooth_chm(chm, None, smoothing.METHOD_MEAN, size=4)


class TestGaussianFilter(unittest.TestCase):

    def test_matches_reference(self):
        rng = np.random.default_rng(2)
        chm = (rng.random((22, 20)) * 25).astype(np.float32)
        valid = rng.random(chm.shape) > 0.2
        for sigma in (0.8, 1.5):
            actual = smoothing.smooth_chm(
                chm, valid, smoothing.METHOD_GAUSSIAN, sigma_cells=sigma)
            expected = reference_gaussian(chm, valid, sigma)
            both = np.isfinite(actual) & np.isfinite(expected)
            np.testing.assert_allclose(
                actual[both], expected[both], rtol=1e-4, atol=1e-4)

    def test_numpy_path_matches_scipy_path(self):
        if smoothing._ndimage is None:
            self.skipTest("scipy 無し")
        rng = np.random.default_rng(3)
        chm = (rng.random((40, 45)) * 25).astype(np.float32)
        valid = rng.random(chm.shape) > 0.15

        with_scipy = smoothing.smooth_chm(
            chm, valid, smoothing.METHOD_GAUSSIAN, sigma_cells=1.2)

        saved = smoothing._ndimage
        smoothing._ndimage = None
        try:
            without_scipy = smoothing.smooth_chm(
                chm, valid, smoothing.METHOD_GAUSSIAN, sigma_cells=1.2)
        finally:
            smoothing._ndimage = saved

        both = np.isfinite(with_scipy) & np.isfinite(without_scipy)
        np.testing.assert_allclose(
            with_scipy[both], without_scipy[both], rtol=1e-4, atol=1e-4)

    def test_mean_numpy_path_matches_scipy_path(self):
        if smoothing._ndimage is None:
            self.skipTest("scipy 無し")
        rng = np.random.default_rng(4)
        chm = (rng.random((40, 45)) * 25).astype(np.float32)
        valid = rng.random(chm.shape) > 0.15

        with_scipy = smoothing.smooth_chm(
            chm, valid, smoothing.METHOD_MEAN, size=5)
        saved = smoothing._ndimage
        smoothing._ndimage = None
        try:
            without_scipy = smoothing.smooth_chm(
                chm, valid, smoothing.METHOD_MEAN, size=5)
        finally:
            smoothing._ndimage = saved

        both = np.isfinite(with_scipy) & np.isfinite(without_scipy)
        np.testing.assert_allclose(
            with_scipy[both], without_scipy[both], rtol=1e-4, atol=1e-4)

    def test_radius_matches_scipy_rule(self):
        # scipy.ndimage.gaussian_filter の truncate=4.0 と同じ半径になること
        self.assertEqual(smoothing.gaussian_radius(1.0), 4)
        self.assertEqual(smoothing.gaussian_radius(0.5), 2)
        self.assertEqual(smoothing.gaussian_radius(2.0), 8)


class TestSmoothingEffect(unittest.TestCase):
    """平滑化が偽の極大を減らすこと (導入の目的そのもの)."""

    @staticmethod
    def _noisy_stand(noise_amplitude, seed=5):
        rng = np.random.default_rng(seed)
        size = 200
        spacing = 14
        rows = range(spacing, size - spacing, spacing)
        n_trees = len(list(rows)) ** 2

        grid_r, grid_c = np.ogrid[:size, :size]
        base = np.zeros((size, size), dtype=np.float32)
        for row in rows:
            for col in rows:
                dist = np.sqrt((grid_r - row) ** 2 + (grid_c - col) ** 2)
                crown = 20.0 * np.clip(1.0 - (dist / 10.0) ** 2, 0.0, None)
                base = np.maximum(base, crown.astype(np.float32))

        noisy = base + (rng.random(base.shape) * noise_amplitude
                        ).astype(np.float32)
        return noisy, n_trees

    def test_recovers_true_tree_count(self):
        """細かいノイズで過剰抽出になる条件で, 平滑化が正解本数に戻すこと."""
        noisy, n_trees = self._noisy_stand(3.0)

        raw = int(local_maxima.local_maxima_mask(noisy, 5, 6.0).sum())
        smoothed = smoothing.smooth_chm(
            noisy, None, smoothing.METHOD_GAUSSIAN, sigma_cells=1.5)
        after = int(local_maxima.local_maxima_mask(smoothed, 5, 6.0).sum())

        self.assertGreater(raw, n_trees)      # 平滑化なしでは過剰抽出
        self.assertEqual(after, n_trees)      # 平滑化後は正解本数

    def test_small_window_benefits_most(self):
        """窓が小さいほど平滑化の効果が大きいこと."""
        noisy, n_trees = self._noisy_stand(3.0)
        smoothed = smoothing.smooth_chm(
            noisy, None, smoothing.METHOD_GAUSSIAN, sigma_cells=1.5)

        for window_size in (3, 5, 7):
            raw = int(local_maxima.local_maxima_mask(
                noisy, window_size, 6.0).sum())
            after = int(local_maxima.local_maxima_mask(
                smoothed, window_size, 6.0).sum())
            self.assertLessEqual(after, raw)
            self.assertEqual(after, n_trees)

    def test_mean_filter_also_reduces_maxima(self):
        noisy, n_trees = self._noisy_stand(3.0)
        raw = int(local_maxima.local_maxima_mask(noisy, 5, 6.0).sum())
        smoothed = smoothing.smooth_chm(
            noisy, None, smoothing.METHOD_MEAN, size=5)
        after = int(local_maxima.local_maxima_mask(smoothed, 5, 6.0).sum())
        self.assertLess(after, raw)
        self.assertGreaterEqual(after, n_trees * 0.9)


class TestTilingWithSmoothing(unittest.TestCase):
    """平滑化を入れてもタイル分割が全域処理と一致すること.

    halo に平滑化のカーネル半径を足し忘れると, ここで落ちる。
    """

    def _run(self, method, size=3, sigma=1.0):
        rng = np.random.default_rng(6)
        chm = (rng.random((260, 230)) * 30).astype(np.float32)
        window_size = 7
        halo = window_size // 2 + smoothing.smoothing_radius(
            method, size, sigma)

        valid_all = local_maxima.valid_mask(chm)
        surface_all = smoothing.smooth_chm(chm, valid_all, method, size, sigma)
        whole = local_maxima.local_maxima_mask(
            surface_all, window_size, min_height=6.0, valid=valid_all)

        tiled = np.zeros(chm.shape, dtype=bool)
        for window in raster_reader.iter_blocks(
                chm.shape[1], chm.shape[0], 64, halo):
            block = chm[
                window.read_row:window.read_row + window.read_height,
                window.read_col:window.read_col + window.read_width]
            valid = local_maxima.valid_mask(block)
            surface = smoothing.smooth_chm(block, valid, method, size, sigma)
            mask = local_maxima.local_maxima_mask(
                surface, window_size, min_height=6.0, valid=valid)
            tiled[
                window.core_row:window.core_row + window.core_height,
                window.core_col:window.core_col + window.core_width,
            ] = mask[window.core_slice]

        np.testing.assert_array_equal(tiled, whole)

    def test_mean(self):
        self._run(smoothing.METHOD_MEAN, size=5)

    def test_gaussian(self):
        self._run(smoothing.METHOD_GAUSSIAN, sigma=1.0)

    def test_none(self):
        self._run(smoothing.METHOD_NONE)


if __name__ == "__main__":
    unittest.main()
