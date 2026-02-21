from . import (
    I_VDT,
    I_VMP,
    I_VVT,
    I_BDT,
)

IMPLEMENTATIONS = {
    "I_VDT": (I_VDT.process_impl, I_VDT.default_config_impl),
    "I_VMP": (I_VMP.process_impl, I_VMP.default_config_impl),
    "I_VVT": (I_VVT.process_impl, I_VVT.default_config_impl),
    "I_BDT": (I_BDT.process_impl, I_BDT.default_config_impl),
}
