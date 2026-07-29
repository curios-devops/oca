"""The rules the live architecture is held to, enforced before there is any code.

These exist now, while `corvus/` is nearly empty, because that is when they are free. Written
after the fact they would each be a refactor, and the specific failure they guard against --
adapting a frozen model piece by piece until no behaviour can be attributed to any hypothesis
-- happens one convenient import at a time.

Three architectures are now frozen in `legacy/`: Wren, Swift and Heron. All three satisfied
every contract they were given. The tests here include the two requirements that would have
caught what those contracts missed.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIVE = ROOT / "corvus"
FROZEN = {"legacy"}


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_the_live_architecture_directory_exists():
    """A guard that silently passes because its target is missing is worse than no guard.

    `rglob` on a directory that does not exist yields nothing, so every import check below
    would vacuously pass if `corvus/` were renamed or moved. This assertion stops that -- and
    it is not hypothetical: moving `dcn/` into `legacy/` left exactly that hole behind.
    """
    assert LIVE.is_dir(), f"{LIVE} is missing; the import guards below would pass vacuously"


def test_live_architecture_never_imports_from_a_frozen_one():
    """Nothing enters by compatibility. This is that rule, in code.

    `legacy/` holds three frozen architectures, available as benchmark opponents. None of
    them is a base class. Carry the *idea* across and re-justify it in the specs; never the
    code.
    """
    offenders = []
    for p in LIVE.rglob("*.py"):
        for mod in _imports(p):
            if mod.split(".")[0] in FROZEN:
                offenders.append(f"{p.name}: {mod}")
    assert not offenders, (
        "corvus/ must not import from legacy/. Three architectures are frozen there and "
        "each failed for a reason that is now written down; re-justify the idea in "
        f"docs/CORVUS/, not the import. Found: {offenders}")


def test_the_live_architecture_may_use_the_shared_substrate():
    """Worlds, sensors, probes, metrics and baselines are common ground, not any version's."""
    import core.baselines  # noqa: F401
    import core.data  # noqa: F401
    import core.metrics  # noqa: F401
    import core.probes  # noqa: F401
    import core.world  # noqa: F401


def test_core_never_imports_an_architecture():
    """The substrate has to stay architecture-agnostic or the benchmark means nothing."""
    offenders = []
    for p in (ROOT / "core").rglob("*.py"):
        for mod in _imports(p):
            if mod.split(".")[0] in FROZEN | {"corvus"}:
                offenders.append(f"{p.name}: {mod}")
    assert not offenders, f"core/ must not depend on any architecture: {offenders}"


# ------------------------------------------------------- the layer contract itself


def test_a_layer_must_declare_its_horizon():
    """A slow layer scored on a one-tick prediction is competing with persistence.

    Measured: no function of a legacy mesh state beat "assume no change" at one tick, while
    at 64 ticks a 44% gain was available. The horizon is not a free parameter.
    """
    from corvus import Floor, Layer

    f = Floor(beats="raw_input", why="test")
    with pytest.raises(ValueError):
        Layer(name="bad", horizon=0, inputs_from=None, floor=f,
              build=lambda: None, step=lambda s, u: {}, readout=lambda s: s)

    ok = Layer(name="good", horizon=64, inputs_from=None, floor=f,
               build=lambda: None, step=lambda s, u: {}, readout=lambda s: s)
    assert ok.horizon == 64


def test_a_layer_must_name_what_it_beats():
    """The requirement the previous two contracts did not have.

    Wren, Swift and Heron each satisfied their contract completely and were all beaten by
    their own input or by a two-number memory. None of those contracts asked what a layer
    was supposed to outperform, so nothing failed when it did not.
    """
    from corvus import Floor, Layer

    with pytest.raises(ValueError):
        Floor(beats="")                       # a layer with no floor cannot fail
    with pytest.raises(ValueError):
        Floor(beats="raw_input", margin=0.0)  # a floor you tie with is not a floor

    with pytest.raises(TypeError):
        Layer(name="floorless", horizon=1, inputs_from=None, floor="raw pixels",
              build=lambda: None, step=lambda s, u: {}, readout=lambda s: s)


def _chain():
    from corvus import Floor, Layer

    def mk(n, h, i, b):
        return Layer(n, h, i, Floor(beats=b, why="test"),
                     lambda: None, lambda s, u: {}, lambda s: s)

    return [mk("neuron", 1, None, "raw_input"),
            mk("tower", 16, "neuron", "trivial_memory"),
            mk("cluster", 64, "tower", "mean_pool")]


def test_layers_address_only_the_layer_below():
    """Dependencies flow downward, abstractions flow upward, and nothing skips a layer."""
    from corvus import Floor, Layer, check_stack

    check_stack(_chain())

    broken = _chain()
    broken[2] = Layer("cluster", 64, "neuron", Floor(beats="mean_pool", why="t"),
                      lambda: None, lambda s, u: {}, lambda s: s)
    with pytest.raises(ValueError, match="skipping a layer"):
        check_stack(broken)


def test_horizons_must_strictly_increase_up_the_stack():
    """A layer that is not slower than the layer below it is a relabelling.

    Heron's node layer decorrelated faster than its own neurons -- 0.88 against 0.91 at the
    declared horizon -- and not one gate in the battery noticed. This is the check.
    """
    from corvus import Floor, Layer, check_stack

    flat = _chain()
    flat[1] = Layer("tower", 1, "neuron", Floor(beats="trivial_memory", why="t"),
                    lambda: None, lambda s, u: {}, lambda s: s)
    with pytest.raises(ValueError, match="strictly increase"):
        check_stack(flat)


# --------------------------------------------------------------- the frozen line


def test_all_three_frozen_architectures_are_still_reachable():
    """Frozen means unchanged and still runnable, not deleted.

    They are the opponents. A benchmark whose baselines have rotted cannot tell you whether
    a new architecture is an improvement on anything.
    """
    from bench.registry import available, get

    for key in ("v1", "v2", "dcn", "raw"):
        assert key in available(), f"{key} is no longer registered"
        state = get(key).new(seed=0, side=12)
        assert hasattr(state, "n_params")
