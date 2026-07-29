PY := .venv/bin/python
TRAIN ?= 8000
TEST ?= 1500
SIDE ?= 24

.PHONY: help venv test guards repro gates e0 e0b e1 e2 e3 e3b e4 e5 e6 e7 e8 e9 e9t e10 e11 e12 dcn-l1 dcn-l2 race g2 bench demo serve figures clean

help:
	@echo "make venv     create .venv and install numpy/matplotlib/pytest"
	@echo "make test     run the test suite"
	@echo "make guards   legacy stays frozen, dcn/ stays free of legacy imports"
	@echo "make repro    run E0-E4 end to end (~1 hour)"
	@echo "make e0..e4   run one v1 experiment"
	@echo "make e5 e6 g2 world-v2 gates: headroom, representation probe, ablations"
	@echo "make e7      architecture v2 gate: v1 vs v2, coalitions and binding"
	@echo "make e8      identity world: are objects built when they are necessary?"
	@echo "make e9      maze: does an active sensorimotor loop change what is stored?"
	@echo "make e9t     tunnel maze: path integration while blind"
	@echo "make e10     Predictive Assembly gates PA1-PA4"
	@echo "make e11     aggregation operators: where the predictive signal survives"
	@echo "make e12     decoder vs representation: which one is the limit?"
	@echo "make dcn-l1  DCN level 1: neuron gates, and vs the legacy line"
	@echo "make dcn-l2  DCN level 2: node gates, and vs the legacy line"
	@echo "make race    v1 vs v2 vs DCN in one maze, then build the demo page"
	@echo "make bench   scorecard across all mesh versions (3 seeds)"
	@echo "make serve   rebuild the maze demo and serve it at localhost:8080"
	@echo "make figures  regenerate plots from logs/"

venv:
	python3.14 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet numpy matplotlib pytest

test:
	$(PY) -m pytest tests/ -q

# Legacy must keep reproducing, and the new architecture must stay a blank page
guards:
	$(PY) -m pytest tests/test_legacy_frozen.py tests/test_dcn_contract.py -q

# E0 must run first: without baseline numbers none of the later results mean anything.
repro: e0 e0b e1 e2 e3 e3b e4 figures

# World v2 validation gates (see the appendix in RESULTS.md)
gates: e5 e6 g2

e0:
	$(PY) experiments/exp00_baselines.py --train $(TRAIN) --test $(TEST) --gru-epochs 300

e0b:
	$(PY) experiments/exp00b_gru_matched.py --train $(TRAIN) --test $(TEST) --hidden 288

e1:
	$(PY) experiments/exp01_learning.py --train $(TRAIN) --test $(TEST) --side $(SIDE)

e2:
	$(PY) experiments/exp02_occlusion.py --train $(TRAIN) --ticks 14000 --side $(SIDE)

e3:
	$(PY) experiments/exp03_ablation.py --train $(TRAIN) --test $(TEST) --side $(SIDE) --seeds 0 1

e3b:
	$(PY) experiments/exp03b_uncertainty_control.py --train $(TRAIN) --test $(TEST) --side $(SIDE)

e5:
	$(PY) experiments/exp05_world_validation.py --train $(TRAIN) --test 2000

e6:
	$(PY) experiments/exp06_probe.py --ticks 9000

# G2: the v1 mesh's ablations on world v2. eta 0.01, not v1's 0.08 -- see RESULTS.md
g2:
	$(PY) experiments/exp03_ablation.py --world physics --eta-head 0.01 \
		--train $(TRAIN) --test $(TEST) --side $(SIDE) --seeds 0 1 \
		--out logs/exp03_physics.jsonl

e4:
	$(PY) experiments/exp04_silence.py --train $(TRAIN) --driven 2000 --silent 4000 --side $(SIDE)

# Stage 2: architecture v2 increment-1 gate (v1 vs v2 on the same stream)
e7:
	$(PY) experiments/exp07_stage2.py --ticks 12000 --n-objects 3 --side 12

# Increment 3: does the mesh build objects when they are computationally necessary?
e8:
	$(PY) experiments/exp08_identity.py --ticks 24000

# Increment 4: the sensorimotor maze, plus the mesh-guided navigation demo
e9:
	$(PY) experiments/exp09_maze.py --ticks 14000 --demo 900

# Rebuild the maze demo page from the last exp09 trace, then serve it locally
demo:
	$(PY) experiments/make_maze_demo.py

serve: demo
	@echo "open http://127.0.0.1:8080"
	$(PY) -m http.server 8080 --bind 127.0.0.1 --directory demo

e9t:
	$(PY) experiments/exp09_maze.py --tunnels --ticks 14000

e10:
	$(PY) experiments/exp10_assembly.py --ticks 14000

# Which aggregation operator preserves the predictive information?
e11:
	$(PY) experiments/exp11_aggregation.py --ticks 16000

# Decoder x representation: which of the two is the remaining limit?
e12:
	$(PY) experiments/exp12_decoder.py --ticks 16000

# DCN level 1: component gates (phase 1) and comparison vs legacy (phase 2)
dcn-l1:
	$(PY) experiments/dcn_l1_neuron.py --ticks 8000

# DCN level 2: node gates (phase 1) and v1 vs v2 vs DCN on one target (phase 2)
dcn-l2:
	$(PY) experiments/dcn_l2_node.py --ticks 16000

# The race: every architecture, one maze, one planner, one trophy
race:
	$(PY) experiments/maze_race.py --ticks 14000 --steps 900
	$(PY) experiments/make_maze_demo.py

# Scorecard across every registered mesh version, with seed and scale stress
bench:
	$(PY) -m bench.run --seeds 0 1 2

figures:
	$(PY) experiments/make_figures.py

clean:
	rm -rf logs/*.jsonl logs/*.json logs/*.png logs/*.txt
