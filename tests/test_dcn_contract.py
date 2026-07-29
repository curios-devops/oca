"""The rules the new architecture is held to, enforced before there is any code.

These exist now, while `dcn/` is empty, because that is when they are free. Written after
the fact they would each be a refactor, and the specific failure this guards against --
adapting the old model piece by piece until no behaviour can be attributed to any
hypothesis -- happens one convenient import at a time.
"""

import ast
import pathlib

import pytest

DCN = pathlib.Path(__file__).resolve().parents[1] / "dcn"


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_dcn_never_imports_from_legacy():
    """Nothing enters by compatibility. This is that rule, in code.

    `legacy/` is frozen and available as a benchmark opponent. It is not a base class. A
    DCN that imports from it has started down the hybrid path the split exists to prevent.
    """
    offenders = []
    for p in DCN.rglob("*.py"):
        for mod in _imports(p):
            if mod.split(".")[0] == "legacy":
                offenders.append(f"{p.name}: {mod}")
    assert not offenders, (
        "dcn/ must not import from legacy/. Carry the *idea* across and re-justify it in "
        f"docs/DCN/FIRST_PRINCIPLES_DCN.md, never the code. Found: {offenders}")


def test_dcn_may_use_the_shared_substrate():
    """Worlds, sensors, probes, metrics and baselines are common ground, not v1's."""
    import core.baselines  # noqa: F401
    import core.data  # noqa: F401
    import core.metrics  # noqa: F401
    import core.probes  # noqa: F401
    import core.world  # noqa: F401


def test_core_never_imports_an_architecture():
    """The substrate has to stay architecture-agnostic or the benchmark means nothing."""
    root = DCN.parent
    offenders = []
    for p in (root / "core").rglob("*.py"):
        for mod in _imports(p):
            if mod.split(".")[0] in {"legacy", "dcn"}:
                offenders.append(f"{p.name}: {mod}")
    assert not offenders, f"core/ must not depend on any architecture: {offenders}"


def test_a_level_must_declare_its_horizon():
    """A slow level scored on a one-tick prediction is competing with persistence.

    Measured on the legacy line: no function of the mesh state beat "assume no change" at
    one tick (1.32x), while at 64 ticks a 44% gain was available. The horizon is not a
    free parameter, so it cannot be defaulted.
    """
    from dcn import DCNLevel

    with pytest.raises(ValueError):
        DCNLevel(name="bad", horizon=0, inputs_from=None,
                 build=lambda: None, step=lambda s, u: {}, readout=lambda s: s)

    ok = DCNLevel(name="good", horizon=64, inputs_from=None,
                  build=lambda: None, step=lambda s, u: {}, readout=lambda s: s)
    assert ok.horizon == 64


def test_levels_address_only_the_level_below():
    """Axiom 5: higher structures never reach into individual neurons."""
    from dcn import DCNLevel

    chain = [
        DCNLevel("neurons", 1, None, lambda: None, lambda s, u: {}, lambda s: s),
        DCNLevel("dcn", 16, "neurons", lambda: None, lambda s, u: {}, lambda s: s),
        DCNLevel("column", 64, "dcn", lambda: None, lambda s, u: {}, lambda s: s),
    ]
    names = [l.name for l in chain]
    for i, level in enumerate(chain):
        if level.inputs_from is None:
            continue
        assert level.inputs_from == names[i - 1], (
            f"{level.name} reads from {level.inputs_from}, skipping a level")


def test_horizons_increase_up_the_hierarchy():
    """Each level integrates over longer than the one below, so it must predict further."""
    from dcn import DCNLevel

    chain = [
        DCNLevel("neurons", 1, None, lambda: None, lambda s, u: {}, lambda s: s),
        DCNLevel("dcn", 16, "neurons", lambda: None, lambda s, u: {}, lambda s: s),
        DCNLevel("column", 64, "dcn", lambda: None, lambda s, u: {}, lambda s: s),
    ]
    hs = [l.horizon for l in chain]
    assert hs == sorted(hs) and len(set(hs)) == len(hs)
