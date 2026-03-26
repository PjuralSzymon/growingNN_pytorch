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
