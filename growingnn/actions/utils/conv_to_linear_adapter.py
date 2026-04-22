def can_insert_conv_before_linear(conv_out_channels: int, linear_in_features: int) -> bool:
    return (
        conv_out_channels > 0
        and linear_in_features > 0
        and linear_in_features % conv_out_channels == 0
    )