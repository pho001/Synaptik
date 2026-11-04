# [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0025B — Export [Model](../../../../glossary.md#term-model) types required by the public NN [API](../../../../glossary.md#term-application-programming-interface-api)

> **Editorial reading context — not a status update.** Gradle api makes Model's Tensor,
> Shape and DataType available when an application depends on NN's public signatures.
> An implementation dependency would be hidden from that consumer's compile classpath.
> This visibility repair neither adds a new semantic owner nor makes NN execute tensors.
> Today's [typed Model boundary](../../../../reference/nn/composition.md#current-nn-typed-model-composition-contract)
> shows the public types crossing from the application into NN, while [state references](../../../../reference/nn/state.md#current-nn-state-dictionary-contract)
> show that access to a type is unrelated to copying values or acquiring storage ownership.

## Status

Complete

## Goal and class

Class C build/[module](../../../../glossary.md#term-module)-boundary [regression](../../../../glossary.md#term-regression) repair. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s public [signatures](../../../../glossary.md#term-signature) expose [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor),
[Shape](../../../../glossary.md#shape) and [DataType](../../../../glossary.md#data-type--datatype), but its [Gradle](../../../../glossary.md#term-gradle) declaration uses `implementation`. Change this existing
permitted Model edge to `api` and prove a consumer with only an NN project [dependency](../../../../glossary.md#term-dependency) compiles.
Do not add a new conceptual dependency, [API](../../../../glossary.md#term-application-programming-interface-api), production class, or layer capability.

## [Scope](../../../../glossary.md#term-scope) and required contracts

Own `extensions/nn/build.gradle.kts`, the relevant [dependency](../../../../glossary.md#term-dependency) assertion in
`testing/architecture-tests`, that project's build file and one minimal isolated [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) consumer
source set/[fixture](../../../../glossary.md#term-fixture). Preserve the existing isolated Training consumer. Review affected NN guide
and [Javadoc](../../../../glossary.md#term-javadoc); public Java [signatures](../../../../glossary.md#term-signature) remain unchanged.
Read AGENTS; root scope index, Core invariants, Dependency rules, Testing requirements;
`extensions-training.md` heading `extensions/nn`; NN [Parameter](../../../../glossary.md#term-parameter)/[Model](../../../../glossary.md#term-model) public signatures and
`NnTrainingDependencyContractTest`. Use existing consumer-fixture conventions, no new framework.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Fixture's only declared project dependency is NN; it calls ordinary Parameter/Model APIs that
  require Model types. It fails with baseline implementation-only visibility and passes with api.<span data-glossary-navigation="true"> ([Fixture glossary](../../../../glossary.md#term-fixture); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model))</span>
- Existing allowed/forbidden edges remain enforced. Do not weaken [dependency](../../../../glossary.md#term-dependency) assertions broadly.
- Run focused architecture validation and NN test/Javadoc once. Coordinator runs final full
  checkpoint because published build metadata changes. Separate review confirms exposed edge,
  fixture isolation, and documentation; no duplicate full run.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Commit](../../../../glossary.md#term-commit) owned paths, no push/plans. Report files/commit/tests/doc impact and completion status.

## Dependencies and integration

- Depends on: existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public [facade](../../../../glossary.md#term-facade); project-review E2; no [dependency](../../../../glossary.md#term-dependency) on 0021B–0024.
- Conflicts with: architecture-tests build/dependency-[fixture](../../../../glossary.md#term-fixture) writers and NN build writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea` plus [planning](../../../../glossary.md#term-planning)-only repair briefs.
- Integration order: independent of other repair code; shared build [checkpoint](../../../../glossary.md#term-checkpoint) after all merges.
- Integration validation: isolated consumer compilation, architecture tests, NN evidence,
  independent review, final repository checkpoint and diff check.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator.

Launch verified 2026-10-13: source/[API](../../../../glossary.md#term-application-programming-interface-api) and negative/positive consumer diagnostics confirm the
missing exported edge; [Model](../../../../glossary.md#term-model) is already the sole allowed NN dependency. User-authorized repair,
dependencies complete, no other build-fixture writer. Use isolated worktree and clean context.

## Integration record

Implementation `6eefd08d` integrated as `7905036f` after independent review.
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-only and Training-only consumers compiled, 306 NN tests, focused [architecture test](../../../../glossary.md#term-architecture-test) and [Javadoc](../../../../glossary.md#term-javadoc); independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
