# [Engine](../../../../glossary.md#term-engine) 0019 — Restore [lifecycle](../../../../glossary.md#lifecycle) and mixed-[owner](../../../../glossary.md#term-owner) [regression](../../../../glossary.md#term-regression) coverage

> **Editorial historical reading key.**
>
> This is a historical regression-test and documentation repair, explicitly not a production lifecycle
> change. One boundary concerns callers racing materialization with result or Engine closure:
> submitting a second copy did not establish that Engine had admitted it. The other concerns
> Compiler-selected CPU/Metal owners: expanded Metal support made formerly heterogeneous fixtures
> single-owner, so tests needed genuine owner sequences and prepared transfer facts. Engine
> orchestrates captured adapters and planned transfers; backends perform physical work, while Runtime
> owns run-state validity and leases. Neither test submission nor operation names alone establish
> those boundaries. Read the recorded device results as dated evidence, not a renewed run. [Current
> composition](../../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition)
> explains owner/transfer routing; the [current Engine
> boundary](../../../../reference/runtime/engine-boundary.md#current-ordinary-engine-boundary)
> explains result materialization and cleanup ownership.

## Status

Complete

## Goal and class

Class C boundary [validation](../../../../glossary.md#term-validation) and supported-workflow documentation; production [lifecycle](../../../../glossary.md#lifecycle) and
capabilities must not change. Repair source-confirmed audit findings B2/B3/B4.

## [Scope](../../../../glossary.md#term-scope) and required reading

- `EngineTypedLifecycleTest.sameResultCopySerializesAndResultAndEngineCloseWait`: submitted
  second copy has not necessarily passed Engine admission when close starts. Make synchronization
  deterministic or split independently proved serialization/close cases; no sleeps or production
  lifecycle relaxation. Preserve rejection when close wins and bounded timeout/finally cleanup.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [admission glossary](../../../../glossary.md#term-admission); [synchronization glossary](../../../../glossary.md#term-synchronization); [serialization glossary](../../../../glossary.md#term-serialization); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `EngineExplicitCompositionMetalIntegrationTest`: rank1/rank16, multi-hop, shared-input and mixed
  publication fixtures now select only Metal after EXP/SIGMOID support grew. Choose genuinely
  heterogeneous supported expressions and assert owner sequences AND prepared transfer facts.
  Keep actual scalar-transfer and single-owner tests distinct. Update focused test-access helper
  only if needed; no production reflection surface.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [Metal glossary](../../../../glossary.md#term-metal); [EXP glossary](../../../../glossary.md#term-expx); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [heterogeneous glossary](../../../../glossary.md#term-heterogeneous); [owner glossary](../../../../glossary.md#term-owner); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- `Engine.prepareTuned` Javadoc: describe eligible single-partition CPU and bounded singleton-NEG
  Metal composition, Metal SESSION reuse, same-owner fallback and heterogeneous rejection.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu); [singleton glossary](../../../../glossary.md#term-singleton))</span>

Read AGENTS; root scope index/Core invariants/[Module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings) routing/[Dependency](../../../../glossary.md#term-dependency) rules/Testing;
`runtime-prepare-engine.md` headings `modules/engine`, `Construction failure and closure`,
`Compile ownership and mixed-owner Prepare routing`, `Public reusable inference session` and
`Profile-free Engine and Prepare lifecycle`; relevant [backend](../../../../glossary.md#backend) transfer and tuning contracts in
`backend-execution.md`. Inspect current implementations and existing [fixture](../../../../glossary.md#term-fixture) helpers.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- New assertions fail on the baseline's missing admission ordering or singleton-owner fixtures
  as appropriate. Tests do not claim a race was observed merely from source inspection.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [singleton glossary](../../../../glossary.md#term-singleton); [owner glossary](../../../../glossary.md#term-owner); [race glossary](../../../../glossary.md#term-race))</span>
- Cover both transfer directions, multi-hop, rank0/1/16, input fan-out and mixed publications
  without arithmetic conversion or fallback. Exact owner/transfer assertions are structural and
  must remain meaningful when device tests are unavailable.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [device glossary](../../../../glossary.md#term-device))</span>
- Focused lifecycle tests, then Engine test/Javadoc once; selected Metal integration class with
  source-matched native library. Try the authorized native-enabled execution environment if the
  sandbox reports NO_DEVICE; report real device skips/blockers truthfully. No benchmark or broad
  suite in worker. Existing library may be used only after matching native source identity.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [native library glossary](../../../../glossary.md#term-native-library); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Independent Class C review checks test determinism, actual transfer coverage, and [Javadoc](../../../../glossary.md#term-javadoc).
- Commit owned tests/Javadoc/explanation only, no shared plans or push. Report exact validation,
  files, limitations and Complete/Incomplete status. An unavailable device is not a device pass.<span data-glossary-navigation="true"> ([Commit glossary](../../../../glossary.md#term-commit))</span>

## Dependencies and integration

- Depends on: [Engine](../../../../glossary.md#term-engine) 0016–0018 and [Metal](../../../../glossary.md#term-metal) 0077 Complete; project-review B2/B3/B4.
- Conflicts with: Engine [lifecycle](../../../../glossary.md#lifecycle)/test-access and Metal mixed-[owner](../../../../glossary.md#term-owner) integration [fixture](../../../../glossary.md#term-fixture) writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea` plus [planning](../../../../glossary.md#term-planning)-only repair briefs.
- Integration order: no semantic dependency on other repairs; disjoint integration test file from
  Compiler 0006D. Coordinator resolves common final validation.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [integration test glossary](../../../../glossary.md#term-integration-test); [Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation))</span>
- Integration validation: focused evidence/review, affected real-device integration where
  available, one combined repository checkpoint and diff check.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: coordinator; worker owns affected Engine [Javadoc](../../../../glossary.md#term-javadoc)/explanation.

Launch verified 2026-10-13: user-authorized repair, source-confirmed test/documentation defects,
dependencies complete; no production lifecycle change or new contract required. Use clean
implementation context and isolated worktree when a worker slot becomes available.

## Integration record

Implementation `a73822ca` integrated as `241ce92b` after independent Class C APPROVE.
[Engine](../../../../glossary.md#term-engine) [module](../../../../glossary.md#term-module): 122 tests passed; final cleanup delta: three focused tests passed.
[Device](../../../../glossary.md#term-device)-free structural [regression](../../../../glossary.md#term-regression) and counterfactual old-[SIGMOID](../../../../glossary.md#term-sigmoid) [fixture](../../../../glossary.md#term-fixture) check passed.
Actual-device mixed-[owner](../../../../glossary.md#term-owner) integration class: 33/33 passed without skips using source-matched
[identity](../../../../glossary.md#term-identity-meanings) 34. Final Engine [Javadoc](../../../../glossary.md#term-javadoc) and diff checks passed; production [lifecycle](../../../../glossary.md#lifecycle) unchanged.
Coordinator's glossary and preparing-execution/[backend](../../../../glossary.md#backend)-selection guide corrections received
independent APPROVE. The combined source-matched native [checkpoint](../../../../glossary.md#term-checkpoint) passed.
A new [scalar](../../../../glossary.md#term-scalar) provider/[preflight](../../../../glossary.md#term-preflight) defect discovered while testing is separately tracked as D5/0079;
using a valid transfer fixture does not claim that defect fixed.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
