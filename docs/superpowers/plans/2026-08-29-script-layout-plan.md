# Script Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move standalone research scripts into categorized `scripts/` subdirectories without changing scientific behavior or results.

**Architecture:** Keep `data/` and `models/` as reusable core modules. Place executable research workflows under `scripts/{training,evaluation,analysis,figures,data_prep}` and run them as modules from the repository root. Update documentation and CI so old root paths cannot silently return.

**Tech Stack:** Python 3.10, GitHub Actions, existing PyTorch/scikit-learn research code.

**Spec:** `docs/superpowers/specs/2026-08-29-script-layout-design.md`

## Global Constraints

- Do not change model logic, scientific methods, checkpoints, split data, metrics, or reported numerical results.
- Do not change dependency versions.
- Preserve script filenames.
- Final public tree must not retain temporary `docs/superpowers/` planning files.
- Verification must run online in GitHub Actions before merge.

---

### Task 1: Create package directories and move scripts

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/training/__init__.py`
- Create: `scripts/evaluation/__init__.py`
- Create: `scripts/analysis/__init__.py`
- Create: `scripts/figures/__init__.py`
- Create: `scripts/data_prep/__init__.py`
- Move: the 17 approved root scripts into the mapped category directories.

**Interfaces:**
- Consumes: existing root-level executable scripts.
- Produces: module entry points such as `python -m scripts.training.train`.

- [ ] Create package marker files.
- [ ] Move each script without changing its scientific code.
- [ ] Verify every expected new path exists and no approved root script remains.

### Task 2: Update documentation and script usage text

**Files:**
- Modify: `README.md`
- Modify: `REPRODUCE.md`
- Modify: moved scripts only where usage examples contain old root commands.

**Interfaces:**
- Consumes: new module paths from Task 1.
- Produces: reproducible commands based on `python -m scripts.<category>.<module>`.

- [ ] Replace root-level execution examples in README and REPRODUCE.
- [ ] Update the repository tree shown in README.
- [ ] Update script usage headers that show old root-level invocation.
- [ ] Verify no documented command invokes a moved script from the repository root.

### Task 3: Strengthen CI for the new structure

**Files:**
- Modify: `.github/workflows/quality.yml`

**Interfaces:**
- Consumes: new categorized script layout.
- Produces: automated structural and syntax checks.

- [ ] Keep Python compile validation.
- [ ] Add assertions that the expected 17 new script paths exist.
- [ ] Add assertions that the old 17 root paths do not exist.
- [ ] Add documentation scan rejecting stale root-level script commands.

### Task 4: Remove temporary planning artifacts and verify PR

**Files:**
- Delete: `docs/superpowers/specs/2026-08-29-script-layout-design.md`
- Delete: `docs/superpowers/plans/2026-08-29-script-layout-plan.md`

**Interfaces:**
- Consumes: completed migration.
- Produces: clean public repository with no internal process files.

- [ ] Remove temporary planning files.
- [ ] Compare branch against `main` and confirm only structural/docs/CI changes.
- [ ] Open PR.
- [ ] Confirm GitHub Actions succeeds.
- [ ] Merge only after successful verification.
