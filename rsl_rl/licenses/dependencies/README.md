# Dependency license notices

The `.txt` files in this directory are **verbatim license notices** for the
tools and libraries that upstream `rsl_rl` lists as dependencies. They exist
so that anyone auditing the repository can see each license's terms without
fetching the upstream projects.

This file is LimX-authored, not upstream; it is the only addition to
`rsl_rl/licenses/` beyond the upstream notice texts.

## Classification

Unlike the previous `v2.0.1` inventory, the `v5.3.0` set is **not** purely
dev-time. `rsl_rl` at `v5.3.0` imports `torch`, `tensordict`, `numpy`,
`GitPython` and (for policy export) `onnx` / `onnxscript` at runtime.
The dev-time-only entries are the linting, typing and documentation tools.

| File | Tool | License | Role |
|------|------|---------|------|
| `codespell-license.txt`        | [`codespell`](https://pypi.org/project/codespell/)     | **GPL-2.0** | Spelling linter — dev-time. See the note below. |
| `furo-license.txt`             | [`furo`](https://pypi.org/project/furo/)               | MIT          | Sphinx theme — docs-time |
| `gitpython-license.txt`        | [`GitPython`](https://pypi.org/project/GitPython/)     | BSD-3-Clause | **Runtime** — repository state logging |
| `numpy-license.txt`            | [`numpy`](https://pypi.org/project/numpy/)             | BSD-3-Clause | **Runtime** |
| `onnx-license.txt`             | [`onnx`](https://pypi.org/project/onnx/)               | Apache-2.0   | **Runtime** — policy export |
| `onnxscript-license.txt`       | [`onnxscript`](https://pypi.org/project/onnxscript/)   | MIT          | **Runtime** — policy export |
| `pre-commit-hooks-license.txt` | [`pre-commit-hooks`](https://pypi.org/project/pre-commit-hooks/) | MIT | Dev-time |
| `pre-commit-license.txt`       | [`pre-commit`](https://pypi.org/project/pre-commit/)   | MIT          | Dev-time |
| `pyright-license.txt`          | [`pyright`](https://pypi.org/project/pyright/)         | MIT          | Type checker — dev-time |
| `ruff-license.txt`             | [`ruff`](https://pypi.org/project/ruff/)               | MIT          | Linter / formatter — dev-time |
| `setuptools-license.txt`       | [`setuptools`](https://pypi.org/project/setuptools/)   | MIT          | Build-time |
| `sphinx-license.txt`           | [`sphinx`](https://pypi.org/project/Sphinx/)           | BSD-2-Clause | Docs-time |
| `tensordict-license.txt`       | [`tensordict`](https://pypi.org/project/tensordict/)   | MIT          | **Runtime** — observation containers |
| `torch-license.txt`            | [`torch`](https://pytorch.org/)                        | BSD-style    | **Runtime** |
| `torchvision-license.txt`      | [`torchvision`](https://pypi.org/project/torchvision/) | BSD-3-Clause | **Runtime** |

All licenses above are permissive and compatible with the BSD-3-Clause
`rsl_rl` sources and the Apache-2.0 outer repository, with the single
exception handled below.

## Note on `codespell-license.txt` (GPL-2.0)

`codespell` is a **development-time spell checker**. Its GPL-2.0 terms:

- Do **not** apply to any code in this repository.
- Do **not** apply to any release artifact built from this repository.
- Do **not** propagate to end users of `rsl_rl`, because `rsl_rl` neither
  imports nor bundles `codespell`.

The GPL-2.0 text is retained here **only** as an audit-time courtesy, so a
reviewer can confirm the source of the notice without fetching upstream
`codespell`. This is a NOTICE, not a license grant, and its presence does
**not** subject the surrounding BSD-3-Clause `rsl_rl` sources or the
Apache-2.0 outer repository to GPL terms.

If `codespell` is ever added as a runtime dependency of `rsl_rl` (it is not,
today), this classification must be revisited before the next release.

## Repository-level policy

The repository's CI includes a GPL-text scan (see
`.github/workflows/ci.yml`). That scan **allow-lists exactly this one file**
(`rsl_rl/licenses/dependencies/codespell-license.txt`) as a dev-tool notice.
Any additional GPL-licensed file anywhere else in the tree will fail CI and
must either be removed or given its own `⚠ TO CONFIRM` classification in
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md).

## Update procedure

Whenever a dependency is added, removed, or its license changes:

1. Add / update / remove the corresponding `*-license.txt` file here.
2. Add / update / remove the corresponding row in the table above **and** in
   [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) §3.
3. If the new dependency is copyleft (GPL / AGPL / LGPL / SSPL / …), file a
   `⚠ TO CONFIRM` in `THIRD_PARTY_NOTICES.md` and update the CI allow-list;
   a copyleft dependency is acceptable only if it is genuinely dev-only and
   never enters the runtime import graph.
4. Re-run the license kinds check:

   ```bash
   cd rsl_rl/licenses/dependencies
   grep -l 'Neither the name' *.txt                      # BSD-3-Clause candidates
   grep -l 'Permission is hereby granted, free of charge' *.txt   # MIT candidates
   grep -lE 'GNU (GENERAL|LESSER|AFFERO)' *.txt          # copyleft — must be allow-listed
   ```
