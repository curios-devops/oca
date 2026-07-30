# logs

Every number this project has published comes from a file in here, and nothing in here is a
source. It is output: regenerate it by re-running the experiment that wrote it.

**Deliberately flat.** `ls logs/` is the index of everything that has ever been run, and every
experiment hard-codes its output path -- reorganising outputs into subdirectories would mean
editing every experiment for no measurement benefit. The risk with logs is staleness, not
location.

| pattern | what it is |
|---|---|
| `scorecard*.json` | benchmark cells: `<architecture>\|<gate>\|<side>` with mean and std |
| `*_summary.json` | one experiment's headline numbers |
| `*.jsonl` | per-tick or per-trial traces, one JSON object per line |
| `*_run.txt` | the console output of a run, kept when the table matters more than the JSON |
| `maze_race.json`, `maze_trace.json` | traces the demo pages are generated from |
| `fig_*.png` | figures from `experiments/make_figures.py` |

**Which files carry a published claim**

| file | claim |
|---|---|
| `corvus_l0.json` | `CGE-A-02` on Corvus's own Layer 0, +0.917 -- and that removing Heron's rotor cost -0.006 |
| `scorecard_pathint.json` | `CGE-A-09`, the first floor gate this project passed |
| `corvus_l2.json`, `corvus_l2_coordination.json` | both of Corvus L2's jobs measured at zero, which retired the layer |
| `corvus_assembly_vote.json` | aggregate-then-vote refuted; the loss is at Layer 1 |
| `pose_world_validity.json` | the pose world is measurable at all -- raw control 0.158 on held-out poses against a chance of 0.250 |
| `jay_l1.json` | Jay L1, and the invariance diagnostic that says its readout is not pose-invariant |
| `scorecard_gateb.json` | Gate B across all five entrants |

A number quoted in `docs/` without a file here is a number that cannot be checked.
