import numpy as np
import pytest

from apsim._rng import substreams


def test_int_seed_matches_seedsequence_spawn():
    gens = substreams(7, 3)
    ref = [np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(3)]
    for g, r in zip(gens, ref):
        assert g.standard_normal() == r.standard_normal()


def test_appending_stream_keeps_existing_ones():
    a = [g.standard_normal(5) for g in substreams(11, 2)]
    b = [g.standard_normal(5) for g in substreams(11, 3)]
    for x, y in zip(a, b[:2]):
        np.testing.assert_array_equal(x, y)


def test_generator_seed_consumes_one_key_per_call():
    rng = np.random.default_rng(5)
    substreams(rng, 4)
    assert rng.bit_generator.seed_seq.n_children_spawned == 1


def test_generator_seed_is_append_safe():
    a = [g.standard_normal(5) for g in substreams(np.random.default_rng(9), 2)]
    b = [g.standard_normal(5) for g in substreams(np.random.default_rng(9), 3)]
    for x, y in zip(a, b[:2]):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("seed", [True, 1.5])
def test_bool_and_float_seeds_raise_typeerror(seed):
    with pytest.raises(TypeError):
        substreams(seed, 2)
