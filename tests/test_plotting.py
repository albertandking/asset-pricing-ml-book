import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from apsim import plotting


def test_setup_configures_fonts():
    chosen = plotting.setup()
    assert plt.rcParams["axes.unicode_minus"] is False
    sans = plt.rcParams["font.sans-serif"]
    assert sans[0] == "Arial"
    if chosen is not None:
        assert chosen in plotting.CJK_FONT_CANDIDATES
        assert chosen in sans


def test_setup_is_idempotent():
    first = plotting.setup()
    second = plotting.setup()
    assert first == second
    assert plt.rcParams["font.sans-serif"].count("Arial") == 1
