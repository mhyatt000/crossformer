from __future__ import annotations

from flax import linen as nn


class VGGBlock(nn.Module):
    ch: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.depth):
            x = nn.Conv(self.ch, (3, 3), padding="SAME", name=f"conv{i + 1}")(x)
            x = nn.relu(x)
        return x


class DecoderBlock(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(
            self.ch,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            name="up",
        )(x)
        x = nn.Conv(self.ch, (3, 3), padding="SAME", name="refine")(x)
        x = nn.relu(x)
        return x


class DreamVGG(nn.Module):
    num_keypoints: int
    variant: str = "full"

    def setup(self):
        if self.variant not in {"quarter", "half", "full"}:
            raise ValueError(f"unknown variant: {self.variant}")

    @nn.compact
    def __call__(self, x):
        stages = []

        x = VGGBlock(64, 2, name="enc1")(x)
        stages.append(("enc1", x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        stages.append(("pool1", x))

        x = VGGBlock(128, 2, name="enc2")(x)
        stages.append(("enc2", x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        stages.append(("pool2", x))

        x = VGGBlock(256, 4, name="enc3")(x)
        stages.append(("enc3", x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        stages.append(("pool3", x))

        x = VGGBlock(512, 4, name="enc4")(x)
        stages.append(("enc4", x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
        stages.append(("pool4", x))

        for i, ch in enumerate((256, 128), start=1):
            x = DecoderBlock(ch, name=f"dec_q{i}")(x)
            stages.append((f"dec_q{i}", x))

        if self.variant in {"half", "full"}:
            x = DecoderBlock(64, name="dec_h")(x)
            stages.append(("dec_h", x))

        if self.variant == "full":
            x = DecoderBlock(32, name="dec_f")(x)
            stages.append(("dec_f", x))

        x = nn.Conv(64, (3, 3), padding="SAME", name="head1")(x)
        x = nn.relu(x)
        stages.append(("head1", x))

        x = nn.Conv(32, (3, 3), padding="SAME", name="head2")(x)
        x = nn.relu(x)
        stages.append(("head2", x))

        heatmaps = nn.Conv(self.num_keypoints, (3, 3), padding="SAME", name="head_out")(x)
        stages.append(("heatmaps", heatmaps))
        return heatmaps, tuple((name, arr.shape) for name, arr in stages)
