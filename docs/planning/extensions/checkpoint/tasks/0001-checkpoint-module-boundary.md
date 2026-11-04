# Task 0001: [Model](../../../../glossary.md#term-model) [Checkpoint](../../../../glossary.md#term-checkpoint) [module](../../../../glossary.md#term-module) boundary

> **Editorial reading context — not a status update.** This task authorized a dependency
> boundary, not a save operation. “Downstream” means the new project could call public NN
> and Engine APIs; it did not acquire their semantic or resource responsibilities. NN describes
> model state, Engine supplies evaluated values, and file transport is a separate concern.
> Later relocation must not be read back into this original dependency decision. Today's
> [artifact/model separation](../../../../reference/checkpoint/artifacts.md#purpose-and-mental-model)
> explains the distinction, while [recovery](../../../../reference/training/recovery.md#caller-coordinated-in-memory-recovery)
> shows why model values alone are not a resumed training program.

## Status

Complete. Implementation, focused [validation](../../../../glossary.md#term-validation), independent Class C documentation/architecture
review, and the post-review repository gate completed from base `81d50901`.
The [Ready](../../../../glossary.md#term-ready) frontier was verified against `44a77c95`: no [Checkpoint](../../../../glossary.md#term-checkpoint) [module](../../../../glossary.md#term-module) or competing Ready
Checkpoint task exists; Training 0003A is Complete; this architecture decision has no
implementation predecessor. The user selected continuation along the Checkpoint path. The
proposed two-[artifact](../../../../glossary.md#term-artifact) split is adopted for this boundary task only; [persistence](../../../../glossary.md#term-persistence) behavior remains
for later tasks.

## Change class

Class C — new extension [ownership](../../../../glossary.md#term-ownership-meanings) and [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [Gradle](../../../../glossary.md#term-gradle) project, and [architecture
contract](../../../../glossary.md#architecture-contract).

## Goal

Authorize a [model](../../../../glossary.md#term-model)-only [Checkpoint](../../../../glossary.md#term-checkpoint) extension downstream of [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and the public [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade),
without placing storage access in NN or making Training/Text dependencies mandatory. Reserve a
separate optional Training Checkpoint adapter for a later, explicitly gated task.

## [Scope](../../../../glossary.md#term-scope)

- Add `extensions/checkpoint` as an empty Gradle project with exactly NN and public Engine project
  dependencies; no persistence, codec, or public Java API in this task.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [codec glossary](../../../../glossary.md#term-codec); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Update the root architecture index/[dependency](../../../../glossary.md#term-dependency) rules and the owning extensions scoped contract.
- Record why [model](../../../../glossary.md#term-model) [persistence](../../../../glossary.md#term-persistence) and [exact training continuation](../../../../glossary.md#exact-training-continuation) have separate [owners](../../../../glossary.md#term-owner) in an [ADR](../../../../glossary.md#term-architecture-decision-record-adr).
- Update focused architecture explanations and add an automated dependency-direction test.
- State explicitly that `extensions/training-checkpoint` is not included in settings or implemented
  until its later adapter gate.

## Non-goals

- Artifact schema, file I/O, atomic publication, materialization adapter, state installation, or
  exact-resume API.<span data-glossary-navigation="true"> ([Artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [materialization glossary](../../../../glossary.md#materialization); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- New [Engine](../../../../glossary.md#term-engine), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training, Text, [Runtime](../../../../glossary.md#term-runtime), or [backend](../../../../glossary.md#backend) behavior.
- Changing the six-contract authority [model](../../../../glossary.md#term-model) or adding a seventh [normative](../../../../glossary.md#term-normative) contract.

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence), [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>, [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>, [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements): explicitly update the root to authorize the new edge.
- [`extensions-training.md` — Scope](../../../../architecture/contracts/extensions-training.md#scope), [`extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn), [`extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining), [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>: extend the owning scoped contract; no reverse [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training edge.
- [`runtime-prepare-engine.md` — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine): consume only its public [facade](../../../../glossary.md#term-facade); do not change [Engine](../../../../glossary.md#term-engine) [ownership](../../../../glossary.md#term-ownership-meanings) or this scoped contract.

## Dependencies and integration

- Depends on: None; architecture decision is the deliverable, not an assumed existing permission.
- Conflicts with: [Checkpoint](../../../../glossary.md#term-checkpoint) 0002–0006 and any simultaneous edit to extension/[dependency](../../../../glossary.md#term-dependency) contracts.
- Parallel group: None.
- Common base revision: N/A (serial work from `44a77c95`).
- Integration order: 0001 before Checkpoint 0002 and Training 0004/Checkpoint 0005–0006.
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew test javadoc` after the focused architecture gate.
- Shared-document integration [owner](../../../../glossary.md#term-owner): 0001 implementation owner; no concurrent writers.

## Files and symbols

- `ARCHITECTURE.md` — [layout](../../../../glossary.md#layout), [scope](../../../../glossary.md#term-scope) index, [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), testing rules.
- `docs/architecture/contracts/extensions-training.md` — [Checkpoint](../../../../glossary.md#term-checkpoint) ownership and separation.
- `docs/architecture/{module-boundaries,dependency-rules,lifecycle}.md` — affected explanation.
- `docs/design/decisions/0031-model-checkpoint-extension-boundary.md` — significant decision.
- `settings.gradle.kts`, `extensions/checkpoint/build.gradle.kts` — project and direct edges.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java` — exact edges and forbidden reverse dependencies.
- This brief, Checkpoint master plan, and roadmap — synchronized status and frontier.

## Acceptance criteria

- The authoritative root and one extensions scoped contract agree on a model-only Checkpoint
  owner, allowed NN/public Engine edges, forbidden lower-layer and reverse edges, and future
  optional Training adapter ownership. No seventh contract or public persistence API appears.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [persistence glossary](../../../../glossary.md#term-persistence); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Gradle](../../../../glossary.md#term-gradle) includes only `extensions/checkpoint` now; its direct project dependencies are exactly
  `:extensions:nn` and `:modules:engine`. Training and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) do not depend on [Checkpoint](../../../../glossary.md#term-checkpoint).
- Focused automated architecture tests enforce the included module and direction; tests and
  Gradle project build pass.
- [ADR](../../../../glossary.md#term-architecture-decision-record-adr) and explanatory docs distinguish current [module](../../../../glossary.md#term-module) boundary from future [artifact](../../../../glossary.md#term-artifact) behavior.

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :testing:architecture-tests:test :extensions:checkpoint:check
```

Integration/repository validation: `./gradlew test javadoc` once after review, because the change
adds a project and modifies architecture/[dependency](../../../../glossary.md#term-dependency) boundaries.

## Documentation and review impact

- Planning brief uses the planning profile; the contract, explanations and ADR use their own
  architecture/ADR profiles and targeted glossary search. No Java production type or Javadoc is
  introduced; any Java test helper receives contract-compliant Javadoc.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A clean Class C implementation context and an independent targeted documentation/architecture
  review context are mandatory. The review reuses successful executable-test evidence unless it
  changes behavior.

## Result

Implemented the empty [model](../../../../glossary.md#term-model)-only [Checkpoint](../../../../glossary.md#term-checkpoint) [Gradle](../../../../glossary.md#term-gradle) project with exactly [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and public [Engine](../../../../glossary.md#term-engine)
direct dependencies; updated the architecture root and owning extensions contract, focused
explanations and glossary, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0031, and an architecture [dependency](../../../../glossary.md#term-dependency)-direction test. No [persistence](../../../../glossary.md#term-persistence)
[API](../../../../glossary.md#term-application-programming-interface-api) or Training Checkpoint project was added. The current exact two-edge build does not preclude
a later justified direct Model edge when a public API names Model types.

Focused [validation](../../../../glossary.md#term-validation): `./gradlew :testing:architecture-tests:test :extensions:checkpoint:check`
passed after the final test edit (`BUILD SUCCESSFUL in 1s`; 37 actionable tasks: 2 executed,
35 up-to-date; `:extensions:checkpoint:test` reported `NO-SOURCE`). The project remains
source-empty by acceptance inspection rather than a durable architecture-test assertion.
Documentation validation: 136 local Markdown links in nine changed boundary documents resolved;
`git diff --check` passed and new files had no trailing whitespace. The glossary's existing
checkpoint term was updated for the empty project. No production Java type was added; the new
architecture-test class and methods have contract-compliant [Javadoc](../../../../glossary.md#term-javadoc).
Independent Class C architecture/documentation review found no blocking issue and made no edit;
it reused the focused test result. Initial unconfigured `./gradlew test javadoc` was stopped after
[Metal](../../../../glossary.md#term-metal) tests reported the missing required `SYNAPTIK_METAL_TEST_LIBRARY`. After rebuilding the
source-matched [native library](../../../../glossary.md#term-native-library) with `./native/metal-macos-arm64/build.sh`, the repository gate
`SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
passed (`BUILD SUCCESSFUL in 5m 11s`; 101 actionable tasks, 3 executed, 98 up-to-date). The
five-[fork](../../../../glossary.md#term-fork-meanings) [CPU](../../../../glossary.md#term-cpu) evidence test completed; the new Checkpoint project reported `NO-SOURCE` for Java
and Javadoc. No unresolved issue or follow-up is required within 0001. `Status: Complete`.
