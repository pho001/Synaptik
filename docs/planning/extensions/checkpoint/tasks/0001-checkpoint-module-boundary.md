# Task 0001: Model Checkpoint module boundary

## Status

Complete. Implementation, focused validation, independent Class C documentation/architecture
review, and the post-review repository gate completed from base `81d50901`.
The Ready frontier was verified against `44a77c95`: no Checkpoint module or competing Ready
Checkpoint task exists; Training 0003A is Complete; this architecture decision has no
implementation predecessor. The user selected continuation along the Checkpoint path. The
proposed two-artifact split is adopted for this boundary task only; persistence behavior remains
for later tasks.

## Change class

Class C — new extension ownership and dependency direction, Gradle project, and architecture
contract.

## Goal

Authorize a model-only Checkpoint extension downstream of NN and the public Engine facade,
without placing storage access in NN or making Training/Text dependencies mandatory. Reserve a
separate optional Training Checkpoint adapter for a later, explicitly gated task.

## Scope

- Add `extensions/checkpoint` as an empty Gradle project with exactly NN and public Engine project
  dependencies; no persistence, codec, or public Java API in this task.
- Update the root architecture index/dependency rules and the owning extensions scoped contract.
- Record why model persistence and exact training continuation have separate owners in an ADR.
- Update focused architecture explanations and add an automated dependency-direction test.
- State explicitly that `extensions/training-checkpoint` is not included in settings or implemented
  until its later adapter gate.

## Non-goals

- Artifact schema, file I/O, atomic publication, materialization adapter, state installation, or
  exact-resume API.
- New Engine, NN, Training, Text, Runtime, or backend behavior.
- Changing the six-contract authority model or adding a seventh normative contract.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): explicitly update the root to authorize the new edge.
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope), [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining), [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle): extend the owning scoped contract; no reverse NN/Training edge.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): consume only its public facade; do not change Engine ownership or this scoped contract.

## Dependencies and integration

- Depends on: None; architecture decision is the deliverable, not an assumed existing permission.
- Conflicts with: Checkpoint 0002–0006 and any simultaneous edit to extension/dependency contracts.
- Parallel group: None.
- Common base revision: N/A (serial work from `44a77c95`).
- Integration order: 0001 before Checkpoint 0002 and Training 0004/Checkpoint 0005–0006.
- Integration validation: `./gradlew test javadoc` after the focused architecture gate.
- Shared-document integration owner: 0001 implementation owner; no concurrent writers.

## Files and symbols

- `ARCHITECTURE.md` — layout, scope index, ownership, dependency direction, testing rules.
- `docs/architecture/contracts/extensions-training.md` — Checkpoint ownership and separation.
- `docs/architecture/{module-boundaries,dependency-rules,lifecycle}.md` — affected explanation.
- `docs/design/decisions/0031-model-checkpoint-extension-boundary.md` — significant decision.
- `settings.gradle.kts`, `extensions/checkpoint/build.gradle.kts` — project and direct edges.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java` — exact edges and forbidden reverse dependencies.
- This brief, Checkpoint master plan, and roadmap — synchronized status and frontier.

## Acceptance criteria

- The authoritative root and one extensions scoped contract agree on a model-only Checkpoint
  owner, allowed NN/public Engine edges, forbidden lower-layer and reverse edges, and future
  optional Training adapter ownership. No seventh contract or public persistence API appears.
- Gradle includes only `extensions/checkpoint` now; its direct project dependencies are exactly
  `:extensions:nn` and `:modules:engine`. Training and NN do not depend on Checkpoint.
- Focused automated architecture tests enforce the included module and direction; tests and
  Gradle project build pass.
- ADR and explanatory docs distinguish current module boundary from future artifact behavior.

## Validation

Worker validation:

```bash
./gradlew :testing:architecture-tests:test :extensions:checkpoint:check
```

Integration/repository validation: `./gradlew test javadoc` once after review, because the change
adds a project and modifies architecture/dependency boundaries.

## Documentation and review impact

- Planning brief uses the planning profile; the contract, explanations and ADR use their own
  architecture/ADR profiles and targeted glossary search. No Java production type or Javadoc is
  introduced; any Java test helper receives contract-compliant Javadoc.
- A clean Class C implementation context and an independent targeted documentation/architecture
  review context are mandatory. The review reuses successful executable-test evidence unless it
  changes behavior.

## Result

Implemented the empty model-only Checkpoint Gradle project with exactly NN and public Engine
direct dependencies; updated the architecture root and owning extensions contract, focused
explanations and glossary, ADR 0031, and an architecture dependency-direction test. No persistence
API or Training Checkpoint project was added. The current exact two-edge build does not preclude
a later justified direct Model edge when a public API names Model types.

Focused validation: `./gradlew :testing:architecture-tests:test :extensions:checkpoint:check`
passed after the final test edit (`BUILD SUCCESSFUL in 1s`; 37 actionable tasks: 2 executed,
35 up-to-date; `:extensions:checkpoint:test` reported `NO-SOURCE`). The project remains
source-empty by acceptance inspection rather than a durable architecture-test assertion.
Documentation validation: 136 local Markdown links in nine changed boundary documents resolved;
`git diff --check` passed and new files had no trailing whitespace. The glossary's existing
checkpoint term was updated for the empty project. No production Java type was added; the new
architecture-test class and methods have contract-compliant Javadoc.
Independent Class C architecture/documentation review found no blocking issue and made no edit;
it reused the focused test result. Initial unconfigured `./gradlew test javadoc` was stopped after
Metal tests reported the missing required `SYNAPTIK_METAL_TEST_LIBRARY`. After rebuilding the
source-matched native library with `./native/metal-macos-arm64/build.sh`, the repository gate
`SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
passed (`BUILD SUCCESSFUL in 5m 11s`; 101 actionable tasks, 3 executed, 98 up-to-date). The
five-fork CPU evidence test completed; the new Checkpoint project reported `NO-SOURCE` for Java
and Javadoc. No unresolved issue or follow-up is required within 0001. `Status: Complete`.
