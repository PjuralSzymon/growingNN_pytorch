import torch.nn as nn


class ModelFactory:
    @staticmethod
    def simple_chain_2() -> nn.Module:
        class ModelSimpleTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.l2 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.l1(x)
                x = self.l2(x)
                return x

        return ModelSimpleTest()

    @staticmethod
    def simple_conv_chain_2() -> nn.Module:
        class ModelSimpleConvTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
                self.c2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.l1 = nn.Linear(4, 4)
                self.l2 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.c1(x)
                x = self.c2(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.l1(x)
                x = self.l2(x)
                return x

        return ModelSimpleConvTest()

    def simple_chain_2_diffrent_input_output_features() -> nn.Module:
        class ModelSimpleTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 3)
                self.l2 = nn.Linear(3, 2)

            def forward(self, x):
                x = self.l1(x)
                x = self.l2(x)
                return x

        return ModelSimpleTest()

    @staticmethod
    def simple_chain_3() -> nn.Module:
        class ModelSimpleTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.l2 = nn.Linear(4, 4)
                self.l3 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.l1(x)
                x = self.l2(x)
                x = self.l3(x)
                return x

        return ModelSimpleTest()

    @staticmethod
    def complex_residual_many_widths() -> nn.Module:
        """Deep residual MLP: several additive skip blocks, each block uses distinct hidden sizes."""

        class ComplexResidualManyWidths(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()
                self.stem = nn.Linear(4, 12)
                self.r1_up = nn.Linear(12, 20)
                self.r1_down = nn.Linear(20, 12)
                self.expand = nn.Linear(12, 18)
                self.r2_a = nn.Linear(18, 18)
                self.r2_b = nn.Linear(18, 18)
                self.contract = nn.Linear(18, 7)
                self.r3_a = nn.Linear(7, 15)
                self.r3_b = nn.Linear(15, 7)
                self.merge = nn.Linear(7, 11)
                self.r4_a = nn.Linear(11, 24)
                self.r4_b = nn.Linear(24, 11)
                self.head = nn.Linear(11, 4)

            def forward(self, x):
                h = self.act(self.stem(x))
                h = h + self.r1_down(self.act(self.r1_up(h)))
                h = self.act(self.expand(h))
                h = h + self.r2_b(self.act(self.r2_a(h)))
                h = self.act(self.contract(h))
                h = h + self.r3_b(self.act(self.r3_a(h)))
                h = self.act(self.merge(h))
                h = h + self.r4_b(self.act(self.r4_a(h)))
                return self.head(h)

        return ComplexResidualManyWidths()

    @staticmethod
    def complex_residual_conv_many_widths() -> nn.Module:
        """Deep irregular CNN: conv residual trunk, then a small ``Linear`` MLP on pooled features."""

        class ComplexResidualConvManyWidths(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()
                self.stem = nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1)
                self.r1_up = nn.Conv2d(12, 20, kernel_size=5, stride=1, padding=2)
                self.r1_down = nn.Conv2d(20, 12, kernel_size=3, stride=1, padding=1)
                self.expand = nn.Conv2d(12, 18, kernel_size=3, stride=1, padding=1)
                self.r2_a = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
                self.r2_b = nn.Conv2d(18, 18, kernel_size=1, stride=1, padding=0)
                self.contract = nn.Conv2d(18, 7, kernel_size=3, stride=1, padding=1)
                self.r3_a = nn.Conv2d(7, 15, kernel_size=3, stride=1, padding=1)
                self.r3_b = nn.Conv2d(15, 7, kernel_size=3, stride=1, padding=1)
                self.merge = nn.Conv2d(7, 11, kernel_size=3, stride=1, padding=1)
                self.r4_a = nn.Conv2d(11, 24, kernel_size=3, stride=1, padding=1)
                self.r4_b = nn.Conv2d(24, 11, kernel_size=3, stride=1, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.post_lin1 = nn.Linear(11, 22)
                self.post_lin2 = nn.Linear(22, 14)
                self.post_lin3 = nn.Linear(14, 9)
                self.head = nn.Linear(9, 4)

            def forward(self, x):
                h = self.act(self.stem(x))
                h = h + self.r1_down(self.act(self.r1_up(h)))
                h = self.act(self.expand(h))
                h = h + self.r2_b(self.act(self.r2_a(h)))
                h = self.act(self.contract(h))
                h = h + self.r3_b(self.act(self.r3_a(h)))
                h = self.act(self.merge(h))
                h = h + self.r4_b(self.act(self.r4_a(h)))
                h = self.pool(h).flatten(1)
                h = self.act(self.post_lin1(h))
                h = self.act(self.post_lin2(h))
                h = self.act(self.post_lin3(h))
                return self.head(h)

        return ComplexResidualConvManyWidths()

    @staticmethod
    def residual_skip() -> nn.Module:
        class ModelWithResidualSkip(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.l2 = nn.Linear(4, 4)
                self.l3 = nn.Linear(4, 4)
                self.l4 = nn.Linear(4, 4)

            def forward(self, x):
                a = self.l1(x)
                b = self.l2(a)
                c = self.l3(b)
                d = self.l4(a)
                return c + d

        return ModelWithResidualSkip()
