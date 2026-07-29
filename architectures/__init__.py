"""Every architecture OCA has built, one package each.

    wren/     frozen  gradient flow on a learned energy landscape
    swift/    frozen  Stuart-Landau limit-cycle oscillators, phase-gated coupling
              swift/pa/  the Predictive Assembly -- a read-only layer over the Swift mesh
    heron/    frozen  event-driven neurons into a reservoir with a resonance spectrum
    corvus/   live    OCA v4, undecided

`mirror` -- no state at all, the current frame and nothing else -- has no package. It is the
floor, it needs no code beyond a few lines in `cge/registry.py`, and it currently beats two of
the three frozen architectures.

Frozen means unchanged and still runnable, never deleted: they are the opponents, and a
benchmark whose baselines have rotted cannot tell you whether anything is an improvement. Each
is rebuilt from a seed rather than restored from a checkpoint, so a scorecard can never
silently describe code that no longer exists.

`corvus/` must not import from any of the other three -- enforced by
`tests/test_corvus_contract.py`, which also asserts its own target directory exists, because a
guard whose target has moved passes vacuously and that has now happened twice.

See docs/ARCHITECTURES.md for what each one established and how each one failed.
"""
