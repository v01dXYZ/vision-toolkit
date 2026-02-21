from . import (
    I_VT,
    I_2MC,
    I_DiT,
    I_DeT,
    I_HMM,
    I_KF,
    I_MST,
)

IMPLEMENTATIONS = {
    "I_VT": (I_VT.process_impl, I_VT.default_config_impl),
    "I_2MC": (I_2MC.process_impl, I_2MC.default_config_impl),
    "I_DiT": (I_DiT.process_impl, I_DiT.default_config_impl),
    "I_DeT": (I_DeT.process_impl, I_DeT.default_config_impl),
    "I_HMM": (I_HMM.process_impl, I_HMM.default_config_impl),
    "I_KF": (I_KF.process_impl, I_KF.default_config_impl),
    "I_MST": (I_MST.process_impl, I_MST.default_config_impl),
}
