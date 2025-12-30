import jax.numpy as jnp
import flax.linen as nn
from crossformer.model.act_vae_model import ACTVAEModel

class ACTVAEHead(nn.Module):
    act_model: ACTVAEModel

    @nn.compact
    def __call__(self, transformer_outputs, train=False):
        # obs tokens: [B,H,K,D]
        obs_tokens = transformer_outputs["obs"].tokens

        # spatial mean pool: [B,H,D]
        context = jnp.mean(obs_tokens, axis=2)

        # Actions_chunk training sırasında train loop tarafından verilir.
        actions_pred, kl = self.act_model(
            context,
            actions_chunk=None,
            train=train
        )

        return {"actions": actions_pred, "kl": kl}
