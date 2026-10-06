import unittest

import numpy as np

from finitewave.numerics.fdm.asymmetric_diffusion import AsymmetricDiffusion


class TestHarmonicDiffusion(unittest.TestCase):
    def test_component_values_and_zero_coefficients(self):
        average = AsymmetricDiffusion('harmonic').diffusion_averaging_method
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                left = np.array([0, 0, 2, 1, -2], dtype=dtype)
                right = np.array([0, 2, 0, 3, -4], dtype=dtype)
                with np.errstate(all='raise'):
                    result = average(left, right)
                np.testing.assert_allclose(result, [0, 0, 0, 1.5, -8 / 3])
                self.assertEqual(result.dtype, dtype)
                np.testing.assert_array_equal(left, [0, 0, 2, 1, -2])
                np.testing.assert_array_equal(right, [0, 2, 0, 3, -4])

    def test_cancelling_nonzero_components_raise(self):
        average = AsymmetricDiffusion('harmonic').diffusion_averaging_method
        with self.assertRaisesRegex(ValueError, 'cancelling nonzero'):
            average(np.array([1.]), np.array([-1.]))

    def test_constant_diagonal_tensor_matches_arithmetic(self):
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                mesh = np.ones((3,) * ndim, dtype=np.int8)
                tensor = np.broadcast_to(np.diag(np.arange(1., ndim + 1)),
                                         (mesh.size, ndim, ndim)).copy()
                with np.errstate(all='raise'):
                    harmonic = AsymmetricDiffusion('harmonic').build_diffusion_operator(
                        mesh, dr=0.5, diffusion=tensor)
                arithmetic = AsymmetricDiffusion().build_diffusion_operator(
                    mesh, dr=0.5, diffusion=tensor)
                np.testing.assert_allclose(harmonic.toarray(), arithmetic.toarray())
                np.testing.assert_allclose(harmonic.sum(axis=1), 0)


if __name__ == '__main__':
    unittest.main()
