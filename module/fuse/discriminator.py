from torch import nn, Tensor


class Discriminator(nn.Module):
    """
    Use to discriminate fused images and source images.
    """

    def __init__(self, dim: int = 32, size: tuple[int, int] = (224, 224)):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, dim, (3, 3), (2, 2), 1),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.Conv2d(dim, dim * 2, (3, 3), (2, 2), 1),
                nn.LeakyReLU(0.2, True),
            ),
            nn.Sequential(
                nn.Conv2d(dim * 2, dim * 4, (3, 3), (2, 2), 1),
                nn.LeakyReLU(0.2, True),
            ),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((size[0] // 8) * (size[1] // 8) * 4 * dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
