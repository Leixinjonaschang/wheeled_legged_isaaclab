# Changes vs upstream

This document records how the vendored `rsl_rl/` in this repository
differs from its upstream, per the licensing requirements of the
BSD-3-Clause license inherited from
[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl).

## Upstream reference

| Field | Value |
|-------|-------|
| Upstream project | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) |
| Upstream license | BSD-3-Clause |
| Fork base tag | **`v5.3.0`** |
| Fork base commit | ⚠ TO CONFIRM — see "Provenance of the 5.3.0 re-sync" below |
| Vendored version string | `5.3.0+limx.1` (in `pyproject.toml`) |
| Vendored license | BSD-3-Clause (unchanged from upstream) |

Copyright notices (`Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION`,
`SPDX-License-Identifier: BSD-3-Clause`) in individual files are preserved
as-is; no upstream copyright headers were removed. The files added by LimX
(listed below) carry the same upstream header because they were authored as
derivative works inside the upstream tree.

---

## 2026-07 — re-sync from `v2.0.1` to `v5.3.0`

This is a **replacement, not a rebase**. The previously vendored `v2.0.1`
fork and the `v5.3.0` fork share no meaningful common structure: the six
same-named files (`rsl_rl/__init__.py`, `env/__init__.py`, `env/vec_env.py`,
`modules/__init__.py`, `storage/__init__.py`, `storage/rollout_storage.py`)
were each rewritten in full upstream between the two tags. A three-way merge
was therefore not attempted.

### Why

The WF-TRON1B tasks require the teacher-student representation-learning
stack (`RepresentationActorCritic` and friends), which exists only in the
`v5.3.0`-based fork maintained alongside the mjlab task package. The
`v2.0.1` fork's only representation-learning feature was a hardcoded
three-dimensional base-linear-velocity regression inside `algorithm/ppo.py`.

### Provenance of the 5.3.0 re-sync

The vendored tree was taken from the sibling mjlab task repository, where it
had been committed as a single squashed import. That repository's history
therefore does **not** record the upstream commit SHA. The base *tag* is
attested by two files inside the tree itself:

- `pyproject.toml` → `version = "5.3.0"` (before the local suffix was applied)
- `CITATION.cff` → `version: 5.3.0`, `repository-code: https://github.com/leggedrobotics/rsl_rl`

The exact upstream SHA must be resolved against the published `v5.3.0` tag
before this repository cuts a tagged release; see the reproduce steps below.

### Directory renames — reverted

The `v2.0.1` vendored copy had renamed two upstream directories to the
singular form. The `v5.3.0` tree restores the upstream plural names:

| Previously vendored (`v2.0.1`) | Now (`v5.3.0`) |
|--------------------------------|----------------|
| `rsl_rl/algorithm/` | `rsl_rl/algorithms/` |
| `rsl_rl/runner/` | `rsl_rl/runners/` |

Downstream imports in this repository were updated accordingly
(`from rsl_rl.runners import OnPolicyRunner`).

### Packages present upstream at `v5.3.0` and absent at `v2.0.1`

`rsl_rl/models/`, `rsl_rl/extensions/`, `rsl_rl/utils/`,
`rsl_rl/modules/{cnn,distribution,mlp,normalization,rnn}.py`,
`rsl_rl/runners/distillation_runner.py`,
`rsl_rl/algorithms/distillation.py`, and the in-tree `tests/` and `docs/`
directories. These are upstream code, not LimX additions.

### Files added by LimX on top of `v5.3.0`

| Vendored path | Purpose |
|---------------|---------|
| `rsl_rl/models/representation_actor_critic.py` | Teacher-student actor-critic; a privileged encoder and a proprioceptive-history encoder are aligned by an MSE representation loss. |
| `rsl_rl/models/representation_velocity_actor_critic.py` | As above, plus a base-linear-velocity regression head feeding the actor. |
| `rsl_rl/models/depth_representation_velocity_actor_critic.py` | As above, plus a depth CNN and a GRU over the depth features. |
| `rsl_rl/models/depth_representation_velocity_predictor_actor_critic.py` | As above, plus multi-horizon latent-dynamics prediction heads. |
| `rsl_rl/algorithms/representation_teacher_student_ppo.py` | PPO on the teacher path with a separate proprioceptive-encoder optimiser. |
| `rsl_rl/algorithms/representation_velocity_teacher_student_ppo.py` | As above, with combined representation + linear-velocity student losses. |
| `rsl_rl/algorithms/representation_velocity_predictor_teacher_student_ppo.py` | As above, with latent-dynamics and latent-rollout losses. Requires `extras["applied_actions"]` from the environment. |

Note that `DepthRepresentationVelocityPredictorActorCritic` and
`RepresentationVelocityPredictorTeacherStudentPPO` are deliberately **not**
re-exported from their package `__init__.py`. Configurations reference them
by the fully qualified `"module.path:ClassName"` form so that the plain depth
task does not import the predictor modules.

### Files modified by LimX relative to `v5.3.0`

| Vendored path | Change |
|---------------|--------|
| `rsl_rl/models/__init__.py` | Re-exports the three non-predictor representation models. |
| `rsl_rl/algorithms/__init__.py` | Re-exports the two non-predictor representation algorithms. |
| `rsl_rl/storage/rollout_storage.py` | Optional `applied_actions` buffer (lazily allocated when a transition supplies it) plus the `latent_dynamics_*` mini-batch generators used by the predictor algorithm. |

### Files added by this repository (not upstream, not mjlab)

| Vendored path | Purpose |
|---------------|---------|
| `licenses/dependencies/README.md` | Per-file license classification and update procedure. Referenced by `../THIRD_PARTY_NOTICES.md` §3 and by the CI GPL-scan failure message. Carried over from the `v2.0.1` vendored copy and rewritten for the `v5.3.0` dependency set. |
| `CHANGES_VS_UPSTREAM.md` | This file. |

⚠ The added/modified split above is asserted from the vendored tree's own
structure and export lists. It has **not** been diffed against a fetched
upstream `v5.3.0` checkout, because no network access was available when the
re-sync was performed. Run the reproduce steps below and correct this section
if the diff disagrees.

### License inventory

`licenses/dependencies/` changed from 12 files to 15, reflecting the new
upstream dependency set (`tensordict`, `onnxscript`, `torchvision`,
`gitpython`, `setuptools`, `sphinx`, `furo`, `ruff` added;
`black`, `flake8`, `isort`, `pyupgrade` removed; `numpy_license.txt` and
`torch_license.txt` renamed to hyphenated form). `THIRD_PARTY_NOTICES.md` §3
was updated to match.

`codespell-license.txt` remains the **only** file in the tree containing
GPL license text, so the CI GPL-scan whitelist in
`.github/workflows/ci.yml` is unchanged and still correct.

---

## How to reproduce this diff

```bash
# 1. Fetch the upstream fork base.
git clone --branch v5.3.0 --depth 1 \
    https://github.com/leggedrobotics/rsl_rl.git /tmp/rsl_rl_v5.3.0

# 2. Record the SHA that the tag resolves to (fills the "Fork base commit" row).
git -C /tmp/rsl_rl_v5.3.0 rev-parse HEAD

# 3. Structural summary.
diff -rq rsl_rl/rsl_rl /tmp/rsl_rl_v5.3.0/rsl_rl

# 4. Full per-file diff.
diff -ru --exclude=__pycache__ \
    /tmp/rsl_rl_v5.3.0/rsl_rl rsl_rl/rsl_rl \
    > /tmp/rsl_rl-vs-v5.3.0.patch

# 5. License inventory check.
diff -rq rsl_rl/licenses /tmp/rsl_rl_v5.3.0/licenses
```

## Update policy

Do **not** silently rebase the vendored copy onto a newer upstream tag.
Any upstream re-sync must:

1. Re-run the reproduce steps above at the new upstream tag and record
   the resulting patch in this file (append a new section).
2. Update the *Fork base tag* and *Fork base commit* rows above.
3. Bump the version in `pyproject.toml` to a value that clearly encodes
   both the upstream base and the local suffix (e.g. `5.3.0+limx.1`).
4. Preserve every upstream copyright header. Do not remove BSD-3-Clause
   SPDX identifiers.
5. Verify `rsl_rl/licenses/` is still consistent with the new upstream
   tree (in particular the `codespell-license.txt` classification —
   see `../THIRD_PARTY_NOTICES.md`).

CI does **not** rebuild this diff automatically; a rebase without a
paired update to this file will be rejected in review (see
`.github/PULL_REQUEST_TEMPLATE.md`).
