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
