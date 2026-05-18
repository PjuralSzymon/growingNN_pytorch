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
    def simple_chain_3_with_activation() -> nn.Module:
        class ModelSimpleTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.act = nn.ReLU()
                self.dropout = nn.Dropout(0.5)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.l2 = nn.Linear(4, 4)
                self.batch_norm = nn.BatchNorm1d(4)
                self.l3 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.l1(x)
                x = self.act(x)
                x = self.dropout(x)
                x = self.pool(x)
                x = self.l2(x)
                x = self.batch_norm(x)
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
    def complex_residual_many_widths_with_activation() -> nn.Module:
        """Identical residual structure to ``complex_residual_many_widths``,
        but every Linear is wrapped with non-editable modules:
        Norm (LayerNorm or BatchNorm1d), an activation, and Dropout.
        The forward statements stay in one-to-one correspondence with the original."""

        class ComplexResidualManyWidths(nn.Module):
            def __init__(self):
                super().__init__()
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

                self.norm_stem = nn.LayerNorm(12)
                self.norm_r1_up = nn.BatchNorm1d(20)
                self.norm_r1_down = nn.LayerNorm(12)
                self.norm_expand = nn.LayerNorm(18)
                self.norm_r2_a = nn.BatchNorm1d(18)
                self.norm_r2_b = nn.LayerNorm(18)
                self.norm_contract = nn.LayerNorm(7)
                self.norm_r3_a = nn.BatchNorm1d(15)
                self.norm_r3_b = nn.LayerNorm(7)
                self.norm_merge = nn.LayerNorm(11)
                self.norm_r4_a = nn.BatchNorm1d(24)
                self.norm_r4_b = nn.LayerNorm(11)

                self.act_stem = nn.ReLU()
                self.act_r1_up = nn.GELU()
                self.act_r1_down = nn.Tanh()
                self.act_expand = nn.SiLU()
                self.act_r2_a = nn.ReLU()
                self.act_r2_b = nn.GELU()
                self.act_contract = nn.LeakyReLU(0.01)
                self.act_r3_a = nn.ELU()
                self.act_r3_b = nn.SiLU()
                self.act_merge = nn.GELU()
                self.act_r4_a = nn.SiLU()
                self.act_r4_b = nn.Tanh()

                self.drop_stem = nn.Dropout(0.10)
                self.drop_r1_up = nn.Dropout(0.10)
                self.drop_r1_down = nn.Dropout(0.10)
                self.drop_expand = nn.Dropout(0.15)
                self.drop_r2_a = nn.Dropout(0.15)
                self.drop_r2_b = nn.Dropout(0.15)
                self.drop_contract = nn.Dropout(0.20)
                self.drop_r3_a = nn.Dropout(0.20)
                self.drop_r3_b = nn.Dropout(0.20)
                self.drop_merge = nn.Dropout(0.25)
                self.drop_r4_a = nn.Dropout(0.25)
                self.drop_r4_b = nn.Dropout(0.25)

            def _wrap_stem(self, y):
                return self.drop_stem(self.act_stem(self.norm_stem(y)))

            def _wrap_r1_up(self, y):
                return self.drop_r1_up(self.act_r1_up(self.norm_r1_up(y)))

            def _wrap_r1_down(self, y):
                return self.drop_r1_down(self.act_r1_down(self.norm_r1_down(y)))

            def _wrap_expand(self, y):
                return self.drop_expand(self.act_expand(self.norm_expand(y)))

            def _wrap_r2_a(self, y):
                return self.drop_r2_a(self.act_r2_a(self.norm_r2_a(y)))

            def _wrap_r2_b(self, y):
                return self.drop_r2_b(self.act_r2_b(self.norm_r2_b(y)))

            def _wrap_contract(self, y):
                return self.drop_contract(self.act_contract(self.norm_contract(y)))

            def _wrap_r3_a(self, y):
                return self.drop_r3_a(self.act_r3_a(self.norm_r3_a(y)))

            def _wrap_r3_b(self, y):
                return self.drop_r3_b(self.act_r3_b(self.norm_r3_b(y)))

            def _wrap_merge(self, y):
                return self.drop_merge(self.act_merge(self.norm_merge(y)))

            def _wrap_r4_a(self, y):
                return self.drop_r4_a(self.act_r4_a(self.norm_r4_a(y)))

            def _wrap_r4_b(self, y):
                return self.drop_r4_b(self.act_r4_b(self.norm_r4_b(y)))

            def forward(self, x):
                h = self._wrap_stem(self.stem(x))
                h = h + self._wrap_r1_down(self.r1_down(self._wrap_r1_up(self.r1_up(h))))
                h = self._wrap_expand(self.expand(h))
                h = h + self._wrap_r2_b(self.r2_b(self._wrap_r2_a(self.r2_a(h))))
                h = self._wrap_contract(self.contract(h))
                h = h + self._wrap_r3_b(self.r3_b(self._wrap_r3_a(self.r3_a(h))))
                h = self._wrap_merge(self.merge(h))
                h = h + self._wrap_r4_b(self.r4_b(self._wrap_r4_a(self.r4_a(h))))
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

    @staticmethod
    def deeply_nested_submodules() -> nn.Module:
        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.act = nn.ReLU()
                self.l2 = nn.Linear(4, 4)

            def forward(self, x):
                x = self.l1(x)
                x = self.act(x)
                x = self.l2(x)
                return x

        class MiddleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerBlock()
                self.l1 = nn.Linear(4, 4)
                self.act = nn.ReLU()

            def forward(self, x):
                x = self.inner(x)
                x = self.l1(x)
                x = self.act(x)
                return x

        class OuterBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.middle = MiddleBlock()
                self.l1 = nn.Linear(4, 4)
                self.act = nn.ReLU()

            def forward(self, x):
                x = self.middle(x)
                x = self.l1(x)
                x = self.act(x)
                return x

        class ModelDeeplyNested(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Linear(4, 4)
                self.outer = OuterBlock()
                self.head = nn.Linear(4, 4)

            def forward(self, x):
                x = self.stem(x)
                x = self.outer(x)
                x = self.head(x)
                return x
        return ModelDeeplyNested()
