# NN 0025B — Export Model types required by the public NN API

## Status

Review needed

## Goal and class

Class C build/module-boundary regression repair. NN's public signatures expose Model Tensor,
Shape and DataType, but its Gradle declaration uses `implementation`. Change this existing
permitted Model edge to `api` and prove a consumer with only an NN project dependency compiles.
Do not add a new conceptual dependency, API, production class, or layer capability.

## Scope and required contracts

Own `extensions/nn/build.gradle.kts`, the relevant dependency assertion in
`testing/architecture-tests`, that project's build file and one minimal isolated NN consumer
source set/fixture. Preserve the existing isolated Training consumer. Review affected NN guide
and Javadoc; public Java signatures remain unchanged.
Read AGENTS; root scope index, Core invariants, Dependency rules, Testing requirements;
`extensions-training.md` heading `extensions/nn`; NN Parameter/Model public signatures and
`NnTrainingDependencyContractTest`. Use existing consumer-fixture conventions, no new framework.

## Acceptance and validation

- Fixture's only declared project dependency is NN; it calls ordinary Parameter/Model APIs that
  require Model types. It fails with baseline implementation-only visibility and passes with api.
- Existing allowed/forbidden edges remain enforced. Do not weaken dependency assertions broadly.
- Run focused architecture validation and NN test/Javadoc once. Coordinator runs final full
  checkpoint because published build metadata changes. Separate review confirms exposed edge,
  fixture isolation, and documentation; no duplicate full run.
- Commit owned paths, no push/plans. Report files/commit/tests/doc impact and completion status.

## Dependencies and integration

- Depends on: existing NN public facade; project-review E2; no dependency on 0021B–0024.
- Conflicts with: architecture-tests build/dependency-fixture writers and NN build writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea` plus planning-only repair briefs.
- Integration order: independent of other repair code; shared build checkpoint after all merges.
- Integration validation: isolated consumer compilation, architecture tests, NN evidence,
  independent review, final repository checkpoint and diff check.
- Shared-document integration owner: coordinator.

Launch verified 2026-10-13: source/API and negative/positive consumer diagnostics confirm the
missing exported edge; Model is already the sole allowed NN dependency. User-authorized repair,
dependencies complete, no other build-fixture writer. Use isolated worktree and clean context.

## Integration record

Implementation `6eefd08d` integrated as `7905036f` after independent review.
NN-only and Training-only consumers compiled, 306 NN tests, focused architecture test and Javadoc; independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. Coordinator's combined repository checkpoint and planning closure remain
outstanding; this status does not imply an outstanding code-review finding.
