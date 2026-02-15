"""Unit tests for action head loss functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.heads.losses import continuous_loss, masked_mean, sample_tau


class TestMaskedMean:
    """Tests for masked_mean utility."""

    def test_masked_mean_all_ones(self):
        """Test masked mean with all ones mask."""
        x = jnp.ones((2, 3, 4))
        mask = jnp.ones((2, 3, 4))
        result = masked_mean(x, mask)
        assert result.shape == ()
        assert jnp.allclose(result, 1.0)

    def test_masked_mean_all_zeros(self):
        """Test masked mean with all zeros mask (should not divide by zero)."""
        x = jnp.ones((2, 3, 4))
        mask = jnp.zeros((2, 3, 4))
        result = masked_mean(x, mask)
        assert result.shape == ()
        # Result should be 0 due to zero mask
        assert jnp.allclose(result, 0.0)

    def test_masked_mean_partial_mask(self):
        """Test masked mean with partial mask."""
        x = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
        mask = jnp.array([[[1.0, 0.0], [1.0, 1.0]]])  # mask out one value
        result = masked_mean(x, mask)
        assert result.shape == ()
        # Masked values: 1, 3, 4 (3 values), mean = 8/3
        assert jnp.allclose(result, 8.0 / 3.0)

    def test_masked_mean_broadcasted_mask(self):
        """Test masked mean with broadcastable mask."""
        x = jnp.ones((2, 3, 4))
        mask = jnp.ones((2, 3, 1))  # Will broadcast to (2, 3, 4)
        result = masked_mean(x, mask)
        assert result.shape == ()
        assert jnp.allclose(result, 1.0)


class TestContinuousLoss:
    """Tests for continuous_loss function."""

    def test_continuous_loss_mse_perfect_prediction(self):
        """Test MSE loss with perfect prediction."""
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[1.0, 2.0, 3.0]])
        mask = jnp.ones_like(pred)

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")

        assert loss.shape == ()
        assert jnp.allclose(loss, 0.0)
        assert "loss" in metrics
        assert "mse" in metrics
        assert "lsign" in metrics
        assert jnp.allclose(metrics["loss"], 0.0)
        assert jnp.allclose(metrics["mse"], 0.0)

    def test_continuous_loss_l1_perfect_prediction(self):
        """Test L1 loss with perfect prediction."""
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[1.0, 2.0, 3.0]])
        mask = jnp.ones_like(pred)

        loss, metrics = continuous_loss(pred, target, mask, loss_type="l1")

        assert loss.shape == ()
        assert jnp.allclose(loss, 0.0)
        assert jnp.allclose(metrics["mse"], 0.0)

    def test_continuous_loss_mse_with_error(self):
        """Test MSE loss with non-zero error."""
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 3.0, 4.0]])
        mask = jnp.ones_like(pred)

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")

        assert loss.shape == ()
        assert loss > 0.0
        # Each error is 1.0, so squared error is 1.0, mean is 1.0
        assert jnp.allclose(loss, 1.0)
        assert jnp.allclose(metrics["mse"], 1.0)

    def test_continuous_loss_l1_with_error(self):
        """Test L1 loss with non-zero error."""
        pred = jnp.array([[0.0, 1.0, 2.0]])
        target = jnp.array([[1.0, 2.0, 3.0]])
        mask = jnp.ones_like(pred)

        loss, _metrics = continuous_loss(pred, target, mask, loss_type="l1")

        assert loss.shape == ()
        assert loss > 0.0
        # Each error is 1.0, so L1 loss is 1.0
        assert jnp.allclose(loss, 1.0)

    def test_continuous_loss_partial_mask(self):
        """Test loss with partial mask."""
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[1.0, 5.0, 3.0]])  # Middle value differs
        mask = jnp.array([[1.0, 0.0, 1.0]])  # Mask out middle value

        loss, _metrics = continuous_loss(pred, target, mask, loss_type="mse")

        assert loss.shape == ()
        # Only unmasked values: pred[0]=1, target[0]=1 (error=0); pred[2]=3, target[2]=3 (error=0)
        assert jnp.allclose(loss, 0.0)

    def test_continuous_loss_sign_error(self):
        """Test sign error detection."""
        pred = jnp.array([[1.0, -1.0]])  # Pred positive, negative
        target = jnp.array([[-1.0, 1.0]])  # Target negative, positive (opposite signs)
        mask = jnp.ones_like(pred)

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")

        assert loss.shape == ()
        # Both values have sign mismatches, so lsign should be 1.0
        assert jnp.allclose(metrics["lsign"], 1.0)

    def test_continuous_loss_batch_shape(self):
        """Test loss with batched inputs."""
        batch_size = 4
        action_dim = 7
        pred = jax.random.normal(jax.random.PRNGKey(0), (batch_size, action_dim))
        target = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
        mask = jnp.ones_like(pred)

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")

        assert loss.shape == ()
        assert loss > 0.0
        assert jnp.isfinite(loss)
        for key in ["loss", "mse", "lsign"]:
            assert key in metrics
            assert jnp.isfinite(metrics[key])

    def test_continuous_loss_invalid_type(self):
        """Test that invalid loss type raises error."""
        pred = jnp.ones((2, 3))
        target = jnp.ones((2, 3))
        mask = jnp.ones_like(pred)

        with pytest.raises(ValueError, match="Invalid loss type"):
            continuous_loss(pred, target, mask, loss_type="invalid")


class TestSampleTau:
    """Tests for sample_tau (beta distribution sampling)."""

    def test_sample_tau_shape(self):
        # @claude TODO wouldnt this be better done with static type checking / analysis?
        """Test that sample_tau returns correct shape."""
        key = jax.random.PRNGKey(0)
        shape = (10, 20)
        result = sample_tau(key, shape=shape)

        assert result.shape == shape

    def test_sample_tau_scalar_shape(self):
        """Test sample_tau with scalar shape."""
        key = jax.random.PRNGKey(0)
        shape = (5,)
        result = sample_tau(key, shape=shape)

        assert result.shape == shape

    def test_sample_tau_range(self):
        """Test that samples are in valid range [0, s]."""
        key = jax.random.PRNGKey(0)
        s = 0.99
        shape = (1000,)
        result = sample_tau(key, shape=shape, s=s)

        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= s)

    def test_sample_tau_different_s_values(self):
        """Test sample_tau with different s parameters."""
        key = jax.random.PRNGKey(0)
        shape = (100,)

        # Test with s=0.99 (default)
        result_099 = sample_tau(key, shape=shape, s=0.99)
        assert jnp.all(result_099 <= 0.99)

        # Test with s=0.5
        result_05 = sample_tau(key, shape=shape, s=0.5)
        assert jnp.all(result_05 <= 0.5)

        # Test with s=1.0
        result_10 = sample_tau(key, shape=shape, s=1.0)
        assert jnp.all(result_10 <= 1.0)

    def test_sample_tau_randomness(self):
        """Test that different keys produce different samples."""
        shape = (100,)
        result1 = sample_tau(jax.random.PRNGKey(0), shape=shape)
        result2 = sample_tau(jax.random.PRNGKey(1), shape=shape)

        # Results should be different (with extremely high probability)
        assert not jnp.allclose(result1, result2)

    def test_sample_tau_distribution_properties(self):
        """Test that samples have reasonable distribution properties."""
        key = jax.random.PRNGKey(42)
        shape = (10000,)
        s = 0.99

        samples = sample_tau(key, shape=shape, s=s)

        # Check mean is roughly in middle of range (but closer to 0 due to beta distribution)
        mean = jnp.mean(samples)
        assert 0.0 < mean < s / 2  # Beta(1.5, 1.0) should be skewed toward 0

        # Check std is positive
        std = jnp.std(samples)
        assert std > 0.0

    def test_sample_tau_multidim_shape(self):
        """Test sample_tau with multi-dimensional shapes."""
        key = jax.random.PRNGKey(0)
        shape = (4, 8, 16)  # 3D shape
        result = sample_tau(key, shape=shape)

        assert result.shape == shape
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 0.999)
