from . import (
    I_VT,
    I_2MC,
    I_DiT,
)

IMPLEMENTATIONS = {
    "I_VT": (I_VT.process_impl, I_VT.default_config_impl),
    "I_2MC": (I_2MC.process_impl, I_2MC.default_config_impl),
    "I_DiT": (I_DiT.process_impl, I_DiT.default_config_impl),
}
