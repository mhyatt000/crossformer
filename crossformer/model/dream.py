from __future__ import annotations

from flax import linen as nn

VGG19_BLOCKS = (
    (64, 2),
    (128, 2),
    (256, 4),
    (512, 4),
    (512, 4),
)


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

        for i, (ch, depth) in enumerate(VGG19_BLOCKS, start=1):
            x = VGGBlock(ch, depth, name=f"enc{i}")(x)
            stages.append((f"enc{i}", x))
            if i < len(VGG19_BLOCKS):
                x = nn.max_pool(x, (2, 2), strides=(2, 2), padding="SAME")
                stages.append((f"pool{i}", x))

        dec_ch = {
            "quarter": (256, 128),
            "half": (256, 128, 64),
            "full": (256, 128, 64, 32),
        }[self.variant]
        dec_names = ("dec_q1", "dec_q2", "dec_h", "dec_f")
        for name, ch in zip(dec_names, dec_ch, strict=True):
            x = DecoderBlock(ch, name=name)(x)
            stages.append((name, x))

        x = nn.Conv(64, (3, 3), padding="SAME", name="head1")(x)
        x = nn.relu(x)
        stages.append(("head1", x))

        x = nn.Conv(32, (3, 3), padding="SAME", name="head2")(x)
        x = nn.relu(x)
        stages.append(("head2", x))

        heatmaps = nn.Conv(self.num_keypoints, (3, 3), padding="SAME", name="head_out")(x)
        stages.append(("heatmaps", heatmaps))
        return heatmaps, tuple((name, arr.shape) for name, arr in stages)
