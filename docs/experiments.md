# Experiment map

Everything added after the paper submission was originally named by the reviewer question it
answered (`q1_`, `q4x_`, `Q6-`, `q5_chinese/`). Two reviewers each had a "Q4" and a "Q5", so
`q4_` and `q4x_` were unrelated studies, and so were `q5x_` (hyperparameter sensitivity) and
`q5_chinese/` (Chinese evaluation). Everything is now named by topic. This file is the map.

| Topic | Question it answers | Code | Configs | Runner |
|---|---|---|---|---|
| **OOD / gate** | Does the out-of-distribution advantage come from the abstention gate or from the steering rule? | `scripts/analysis/ood_gate/` | `confs/steer/OOD-*` | `experiments/ood_gate_ablation.sh`, `experiments/ood_paper_config.sh` |
| **Gate sweep** | Earlier exploratory pass over gate signals and step modes | — | `confs/steer/GateSweep-*` | `experiments/gate_sweep.sh` |
| **Multilingual** | Does the method work beyond English? | `scripts/multilingual/` | reuses the base configs | `experiments/multilingual.sh` |
| **Attribution** | What does the formulation expose about a query? | `scripts/analysis/attribution/` | — | — |
| **Geometry** | How sensitive is the method to the spherical geometry assumption? | `scripts/analysis/geometry/` | `confs/steer/Ablation-NoSphere.yaml` | — |
| **Cost** | What does it cost at scale, and which approximations hold up? | `scripts/analysis/cost/` | — | — |
| **Ablation** | What does each component contribute? | `scripts/analysis/ablation/` | `confs/steer/Ablation-*` | `experiments/component_ablation.sh` |
| **Sensitivity** | How much do the never-tuned hyperparameters matter? | `scripts/analysis/sensitivity/` | — | — |
| **Derivation** | Is the implemented update the exact Riemannian gradient the appendix derives? | `scripts/analysis/derivation/` | `confs/steer/Deriv-*` | `experiments/derivation_variants.sh` |

`experiments/run_all_fullsize.sh` dispatches the full-size re-runs;
`experiments/bundle_for_remote.sh` packs what they need for another machine.

## The COBRAS classes

| Class | File | What it is |
|---|---|---|
| `COBRAS` | `src/cobras/steer/_cobras.py` | The shipped method, and nothing else. |
| `AblationCOBRAS` | `src/cobras/steer/_cobras_ablation.py` | `COBRAS` plus the switches that turn one part of it off. With default arguments it **is** `COBRAS` — `tests/test_cobras_variants.py` enforces that. |
| `EuclideanCOBRAS` | `src/cobras/steer/_euclidean_cobras.py` | The same bridge with the spherical geometry removed, as a control. |
| `GatedSteer` | `src/cobras/steer/_gated_steer.py` | Wraps *any* steering method in COBRAS's abstention gate, so the gate's contribution can be measured separately from the steering rule's. |

A config that reaches for an ablation has to say so: `type: AblationCOBRAS`. If you add a knob,
its default must be the shipped behaviour or the equivalence test fails.

## The abstention gate and query activations

The contrastive pairs are *answer* activations; the gate is applied to *prompt* activations,
which sit at a different position and so at a systematically different radius. Calibrating on
one and thresholding the other shifts the whole reference. `abstain_on_queries: true` fixes
this by calibrating on real query activations, which
`scripts/prepare/extract_query_activations.py` writes to `data/query_activations/`.

Two things used to fail silently here and now do not:

* `COBRAS.fit` raises if `abstain_on_queries` is set and no `ref_X` is passed, instead of
  falling back to the contrastive negatives under the same run name.
* `load_query_activations` warns when the file is missing, and `fit_steer_model` warns when it
  drops a keyword argument the steer model does not accept.

## A note on run names

Config **file** names were renamed; the `name:` template *inside* each config was not. Those
strings (`q1-cobras-knn-ratio-...`) are the keys the cached files in `results/` are stored
under, and renaming them would orphan the whole results tree. The same goes for the
`results/analysis/q1|q3|q4|q4x|q5x|q6/` directories. They are historical identifiers, not
descriptions.

## Recovering pruned work

The pre-refactor tree is committed whole on the `snapshot/pre-refactor` branch:

```bash
git show snapshot/pre-refactor:scripts/analysis/check_frozen.py
git checkout snapshot/pre-refactor -- scripts/analysis/transport_pairs.py
```
