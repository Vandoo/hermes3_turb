"""Test energy-channel classification without loading simulation data.

Radiative losses, prescribed inputs, and internal transfers are distinct.
The R/E prefix distinguishes recombination radiation from species exchange;
reflection and recycling distinguish outgoing and incoming boundary energy.
"""

from hermes3_turb.balance import classify_energy_channel

# Real dump variable list -> expected class.
CHANNELS = {
    # --- genuine losses: energy leaves the box -----------------------
    "Rd+_ex": "loss",              # radiation AND ionisation potential (one channel)
    "Rd+_rec": "loss",             # recombination radiation
    "Ed_target_refl": "loss",      # neutral energy onto the plate
    "Ed_wall_refl": "loss",        # neutral energy onto the wall
    "Ed_pump": "loss",             # removed with pumped neutrals
    "Ee_sheath": "loss",           # only with [sheath_boundary_simple] diagnose
    "Ed+_sheath": "loss",
    # --- external inputs ---------------------------------------------
    "Pe_src": "input",             # prescribed heating, [Pa/s], needs the 3/2
    "Pd+_src": "input",
    "Pd_src": "input",
    "Ed_target_recycle": "input",  # prescribed eV per returning atom (f_fast = 0)
    "Ed_wall_recycle": "input",
    # --- internal transfers: must NOT enter the budget ---------------
    "Edd+_cx": "transfer",         # the double-counting trap
    "Ed+_iz": "transfer",
    "Ed+_rec": "transfer",
    "Ed+e_coll": "transfer",
    "Ed+e_coll_friction": "transfer",
    "Eed+_coll": "transfer",
    "Eed+_coll_friction": "transfer",
    "Ed+_VgradP": "transfer",
    "Ee_VgradP": "transfer",
    # --- not energy channels at all ----------------------------------
    "Fd+_iz": None,                # momentum
    "Fdd+_cx": None,
    "Sd+_iz": None,                # particles
    "Sd_pump": None,
    "Se_src": None,                # particle source, not pressure
    "Pe": None,                    # a state variable, not a source
    "Pd+": None,
    "Ne": None,
    "phi": None,
    "J": None,
    "": None,
    None: None,
}


def test_every_real_channel_classified():
    wrong = {v: (classify_energy_channel(v), exp)
             for v, exp in CHANNELS.items()
             if classify_energy_channel(v) != exp}
    assert not wrong, "misclassified: %r" % wrong


def test_R_and_E_rec_differ():
    """Same suffix, opposite class — the leading letter decides."""
    assert classify_energy_channel("Rd+_rec") == "loss"
    assert classify_energy_channel("Ed+_rec") == "transfer"


def test_refl_is_a_loss_but_recycle_is_an_input():
    assert classify_energy_channel("Ed_target_refl") == "loss"
    assert classify_energy_channel("Ed_target_recycle") == "input"
    assert classify_energy_channel("Ed_wall_refl") == "loss"
    assert classify_energy_channel("Ed_wall_recycle") == "input"


def test_cx_is_never_a_loss():
    """Counting CX as a sink double-counts: the loss is the _refl channel."""
    for v in ("Edd+_cx", "Ed+d_cx", "Edd_cx"):
        assert classify_energy_channel(v) == "transfer"


def test_unknown_E_channel_defaults_to_transfer():
    """Treat an unrecognised E* diagnostic as an internal transfer.
    """
    assert classify_energy_channel("Ed+_somethingnew") == "transfer"
