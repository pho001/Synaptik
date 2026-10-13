# Engine 0019 — Restore lifecycle and mixed-owner regression coverage

## Status

Complete

## Goal and class

Class C boundary validation and supported-workflow documentation; production lifecycle and
capabilities must not change. Repair source-confirmed audit findings B2/B3/B4.

## Scope and required reading

- `EngineTypedLifecycleTest.sameResultCopySerializesAndResultAndEngineCloseWait`: submitted
  second copy has not necessarily passed Engine admission when close starts. Make synchronization
  deterministic or split independently proved serialization/close cases; no sleeps or production
  lifecycle relaxation. Preserve rejection when close wins and bounded timeout/finally cleanup.
- `EngineExplicitCompositionMetalIntegrationTest`: rank1/rank16, multi-hop, shared-input and mixed
  publication fixtures now select only Metal after EXP/SIGMOID support grew. Choose genuinely
  heterogeneous supported expressions and assert owner sequences AND prepared transfer facts.
  Keep actual scalar-transfer and single-owner tests distinct. Update focused test-access helper
  only if needed; no production reflection surface.
- `Engine.prepareTuned` Javadoc: describe eligible single-partition CPU and bounded singleton-NEG
  Metal composition, Metal SESSION reuse, same-owner fallback and heterogeneous rejection.

Read AGENTS; root scope index/Core invariants/Module-ownership routing/Dependency rules/Testing;
`runtime-prepare-engine.md` headings `modules/engine`, `Construction failure and closure`,
`Compile ownership and mixed-owner Prepare routing`, `Public reusable inference session` and
`Profile-free Engine and Prepare lifecycle`; relevant backend transfer and tuning contracts in
`backend-execution.md`. Inspect current implementations and existing fixture helpers.

## Acceptance and validation

- New assertions fail on the baseline's missing admission ordering or singleton-owner fixtures
  as appropriate. Tests do not claim a race was observed merely from source inspection.
- Cover both transfer directions, multi-hop, rank0/1/16, input fan-out and mixed publications
  without arithmetic conversion or fallback. Exact owner/transfer assertions are structural and
  must remain meaningful when device tests are unavailable.
- Focused lifecycle tests, then Engine test/Javadoc once; selected Metal integration class with
  source-matched native library. Try the authorized native-enabled execution environment if the
  sandbox reports NO_DEVICE; report real device skips/blockers truthfully. No benchmark or broad
  suite in worker. Existing library may be used only after matching native source identity.
- Independent Class C review checks test determinism, actual transfer coverage, and Javadoc.
- Commit owned tests/Javadoc/explanation only, no shared plans or push. Report exact validation,
  files, limitations and Complete/Incomplete status. An unavailable device is not a device pass.

## Dependencies and integration

- Depends on: Engine 0016–0018 and Metal 0077 Complete; project-review B2/B3/B4.
- Conflicts with: Engine lifecycle/test-access and Metal mixed-owner integration fixture writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea` plus planning-only repair briefs.
- Integration order: no semantic dependency on other repairs; disjoint integration test file from
  Compiler 0006D. Coordinator resolves common final validation.
- Integration validation: focused evidence/review, affected real-device integration where
  available, one combined repository checkpoint and diff check.
- Shared-document integration owner: coordinator; worker owns affected Engine Javadoc/explanation.

Launch verified 2026-10-13: user-authorized repair, source-confirmed test/documentation defects,
dependencies complete; no production lifecycle change or new contract required. Use clean
implementation context and isolated worktree when a worker slot becomes available.

## Integration record

Implementation `a73822ca` integrated as `241ce92b` after independent Class C APPROVE.
Engine module: 122 tests passed; final cleanup delta: three focused tests passed.
Device-free structural regression and counterfactual old-SIGMOID fixture check passed.
Actual-device mixed-owner integration class: 33/33 passed without skips using source-matched
identity 34. Final Engine Javadoc and diff checks passed; production lifecycle unchanged.
Coordinator's glossary and preparing-execution/backend-selection guide corrections received
independent APPROVE. The combined source-matched native checkpoint passed.
A new scalar provider/preflight defect discovered while testing is separately tracked as D5/0079;
using a valid transfer fixture does not claim that defect fixed.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's scope.

Status: Complete
