# IAMU Dynamic Commitment Prototype v0.1

**Event-driven berth–quay-crane scheduling with ETA updates and a limited budget of published-plan changes. No ERL-DDE retraining is required.**

This prototype is deliberately a **new** planning/realization and commitment layer. It reads the **original ERL-DDE instance format** and can optionally use a **frozen original ERL-DDE `ES` checkpoint** to derive proposal order and berth-position hints. It **does not claim** to be the original authors' dynamic simulator, and it currently **does not optimize or simulate shore power**.

## 1. What is implemented

- The original `case_code/*.txt` **tab-separated** columns: ETA, workload, ship length, ship maximum cranes, and total port cranes (fifth column of **the first row**).
- A hidden actual arrival time and **causal** ETA update event stream for each ship; randomized scenarios: `small`, `shock`, and `repeated`, controlled by seed. Only the environment sees actual arrivals.
- An **initially published plan** (initial publication is free), persistent announced assignments, a global modification budget `B`, and per-ship cumulative revision counts. A plan edit is charged **once for each ship whose published start time, berth coordinate or assigned crane count actually changes**; repeated changes to the same ship are charged again.
- **Immutability after real service starts**: ships already in service cannot be revised.
- A separate physical **realization engine**, with actual arrival releases, 3,000-m continuous quay, safety gaps, and total crane capacity. If actual arrival is late or physical capacity is busy, the ship's operational start slips **without silently counting this as an announced revision**.
- Event-driven heuristic controller choices: `no_replan`, `always`, `threshold`, `periodic` (event-triggered periodic **gate**), `budget_aware` (forecast improvement threshold). These are **untrained**, interpretable baselines, **not** learned revision policies.
- Each event can revise at most `--max-edits-per-event` vessels and never exceeds the global budget. Initial/publication, forecasts, revisions and realized operations are saved separately; all methods receive exactly the same scenario and hidden actual arrivals.
- Optional original, **pretrained** ERL-DDE network replay through the **original C++ source compiled in a separate probe library**. This uses the published `ES` policy's action selection order and suggested berth position **as hints to the new commitment planner**; it is not a faithful reuse of the original C++ dynamics as the execution layer.

### Deliberate simplified assumptions (important for research validity)

- The berth and crane **planning/realization engine in this package is simplified**. Continuous berth positions and non-overlap use the original quay-length/safety-ratio constants, and operation duration is `workload / assigned_cranes` at unit crane efficiency. It does not model the original fine-grained crane movement, reassignment, or original reward.
- The candidate scheduler is a **greedy earliest-feasible insertion heuristic**, not the original ERL-DDE static optimizer, not optimal MILP/CP-SAT, and not a trained dynamic model. In ERL-DDE backend, frozen policy **hints** influence order and berth location; all constraints remain enforced by the new planner.
- Future ETA predictions are treated as point forecasts when scoring revisions. Realized arrivals may differ, and the simulated future arrival trajectory remains hidden from the controller.
- `periodic` currently means **period-gated replanning on ETA notification events**, *not* an independent periodic timer. The full periodic-clock policy can be added next.
- `always` means **consider a new plan at every ETA update**, not forcibly change a published assignment when its estimated objective is worse.
- No shore-power compatibility, electricity capacity, grid emissions, connection/disconnection time, or monetary/CO2 metric is modeled. Those require additional data and explicitly validated resource constraints.
- Do not compare numerical rewards from this simulator with ERL-DDE's negative `reward`. The objectives and physics differ.

## 2. Quick start: zero installs beyond Python standard library

Start in a **new directory**, never in the directory of a running ERL-DDE training process:

```bash
cd ~/code/iamu
mkdir -p IAMU_dynamic
cd IAMU_dynamic
# Extract the ZIP contents here.
python run_experiment.py --demo --scenario shock --seed 42 --budgets 0,2,5,unlimited --out results/demo
python -m unittest discover -s tests -v
```

Run **your own original paper instance** (no checkpoint required; new heuristic backend):

```bash
python run_experiment.py \
  --case ../2024LiERLDDE/case_code/0.txt \
  --scenario repeated --seed 123 --updates 2 \
  --budgets 0,2,5 \
  --policies no_replan,threshold,budget_aware \
  --out results/case0_seed123
```

An output directory contains:

```
comparison.csv
manifest.json
budget_aware_B5/
    scenario.json              # includes hidden actual arrivals: for evaluation only
    metrics.json
    result.json
    eta_updates.csv            # decisions and available forecasts
    revisions.csv              # each charged, ship-level announced revision
    initial_published.csv
    final_published.csv
    realized.csv               # actual physical service times
```

### Load existing (frozen) ERL-DDE weights, NO training

1. **Do not** modify the authors' `main.py`, `train.py`, `BACAP_env.cpp`, or `.so`.
2. Copy a **completed, stable** checkpoint to a new location; do not read a file while it is being overwritten by running training.
3. Install PyTorch in the environment if it is not already installed; have `g++` and `python3` available.
4. Compile a **separate** read-only probe library and run pure inference:

```bash
mkdir -p checkpoints
cp ../2024LiERLDDE/BACAP_checkpoints/2/synced_model_0_55906.pth \
   checkpoints/erldde_frozen_0.pth

python run_experiment.py \
  --case ../2024LiERLDDE/case_code/0.txt \
  --backend erldde \
  --erldde-root ../2024LiERLDDE \
  --checkpoint checkpoints/erldde_frozen_0.pth \
  --build-probe \
  --updates 1 --scenario shock --seed 123 \
  --budgets 0,2,5 \
  --policies no_replan,budget_aware \
  --out results/case0_frozen_erldde
```

If the specified checkpoint directory does not exist, substitute the actual stable checkpoint path. The choice of seed and model checkpoint **must be recorded** when reporting experiments. On-demand frozen ERL-DDE replay on multiple ETA snapshots can take time; start with a single case and `--updates 1`.

This mode builds `results/case0_frozen_erldde/probe_build/libBACAP_env_probe.so` from an include of the **local original** `BACAP_env.cpp`. It does not overwrite the published `.so` or Python wrapper. A tiny generated C++ accessor maps a waiting-list action index to a source ship ID so the ranking hints are attached to the right ship. Frozen `model.ES` is executed under `torch.no_grad()` on CPU; **no optimizer, `train_loop`, or parameter updates** occur.

**Integration test scope:** the optional C++ probe build and checkpoint load have been tested against a small compiled **stub** that reproduces the C API; we have *not* run the original published full simulator with your trained checkpoint in this environment. Run this command on your server to complete that validation.

## 3. Protocol and budget definition

At time 0, initial ETA announcements generate a fully published feasible schedule. For an event at time `t`:

1. The physical simulator executes already-published commitments up to `t` using **hidden actual arrivals**, and freezes every ship that already started.
2. The new ETA notice is revealed for its ship; all other future actual arrivals remain hidden.
3. A trigger rule chooses whether to consider edits.
4. For each eligible, not-yet-started candidate, a new assignment is computed **around every unchanged published reservation**, enforcing quay non-overlap and crane capacity. A forecast dispatcher estimates downstream waiting-time improvement.
5. The best positive-gain single-vessel change is announced if permitted by the chosen policy and remaining `B`. Repeat for up to `max-edits-per-event` and count each changed published assignment against the **cumulative** global budget.
6. Save the decision and continue to next event. After the last event, run the hidden physical arrivals to completion and measure realized metrics.

The **candidate score never uses the hidden actual arrival vector**. It uses observed history and current ETA forecasts for ships yet to arrive. A past ship may physically start later than initially published without a publication change; the metric `total_operational_slip` makes this distinction visible.

## 4. Basic tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q iamu_dynamic run_experiment.py
```

The unit tests cover instance parser, seed reproducibility, announcement causality, planned/realized spatial + crane feasibility, budget accounting, started-ship freeze, `B=0` equivalence and non-anticipativity of future arrivals. A separate **stub-based** test verifies that the optional C++ accessor, ctypes ABI and frozen checkpoint adapter can be compiled and invoked.

## 5. Scientific interpretation and next research steps

- Treat this as a **V0.1 protocol prototype / benchmark harness**, not a validated industrial simulator or a reproduction of published ERL-DDE dynamic scheduling results.
- V0.2 should add an actual policy-gated **immutable future commitment ledger** with a calibrated revision-notice interval, multi-ship feasibility-repair moves and a proper independent periodic trigger.
- V0.3 should add ship-level shore-power eligibility, plug count, kW capacity, connection/disconnection time, grid time-of-use prices, auxiliary-engine emissions and emissions factors. Each extension requires documented assumptions and constrained validation.
- A subsequent **learning** stage can replace the `budget_aware` gate with a trained accept/defer/ship-subset policy while **keeping ERL-DDE frozen**. Compare on matched instances and seeds and use held-out cases.

For research code reuse and paper comparison, see `REUSE_LITERATURE.md`.
