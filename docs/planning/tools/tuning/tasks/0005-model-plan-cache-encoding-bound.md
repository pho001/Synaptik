# Tuning 0005 — Bound [model](../../../../glossary.md#term-model)-plan [cache](../../../../glossary.md#term-cache) encoding before allocation

## Status

Complete

## Change class and goal

Class C: persistent-[artifact](../../../../glossary.md#term-artifact) resource safety, without a wire/[schema](../../../../glossary.md#term-schema) or public [API](../../../../glossary.md#term-application-programming-interface-api) change.
Reject an oversized [model](../../../../glossary.md#term-model)-plan [cache](../../../../glossary.md#term-cache) before allocating/encoding its complete [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). Preserve
[canonical](../../../../glossary.md#term-canonical-form) bytes for every accepted schema-1 artifact and preserve the destination on failure.

## Finding and [scope](../../../../glossary.md#term-scope)

`ModelPlanCacheFile.encode` grows a `ByteArrayOutputStream`, copies its [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), and computes its
[checksum](../../../../glossary.md#term-checksum) before checking `MAX_FILE_BYTES`. Individually valid bounded entries can exceed the
aggregate 16 MiB limit; the current check therefore does not bound encoder memory use.
`WorkloadCacheFile.encode` already checks aggregate size before allocating its output.

Owned paths: `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java`
and focused tests under that [module](../../../../glossary.md#term-module). Review the affected implementation [Javadoc](../../../../glossary.md#term-javadoc) and
`docs/architecture/performance-evidence-and-tuning.md`; update explanation only if needed.
No changes to candidate selection, [cache](../../../../glossary.md#term-cache) [identities](../../../../glossary.md#term-identity-meanings), [schema](../../../../glossary.md#term-schema), file [publication](../../../../glossary.md#term-publication) policy, or [lifecycle](../../../../glossary.md#lifecycle).

## Required contracts

Read `AGENTS.md`; the root `ARCHITECTURE.md` [scope](../../../../glossary.md#term-scope) index, Core invariants, [Module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings) routing,
and [Dependency](../../../../glossary.md#term-dependency) rules; and `docs/architecture/contracts/backend-execution.md`, heading
`Performance evidence and optimization tooling`. Read current encoder/decoder, the Phase-1
bounded encoder as a local comparison, and `CompletePlanTuningCacheTest`.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Checked preflight accounts for every fixed/length-prefixed field and the checksum before any
  complete output allocation. Oversize is a controlled failure, not an allocation failure.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- Accepted [schema](../../../../glossary.md#term-schema)-1 bytes remain [canonical](../../../../glossary.md#term-canonical-form) and decoder/inspection compatible.
- Focused regression covers aggregate overflow of the limit using individually legal fields,
  boundary accounting, and unchanged target/no temporary artifacts on rejection.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [target glossary](../../../../glossary.md#term-target); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Do not require allocating gigabytes or timing a failure to prove the bound; use a directly
  testable bounded-size calculation or equivalent deterministic test.
- [Run](../../../../glossary.md#term-run) focused [regression](../../../../glossary.md#term-regression), then `./gradlew :tools:tuning:test :tools:tuning:javadoc` once after
  stabilization. Independent review reuses successful tests unless it finds a specific risk.
- Report changed files, evidence, [Javadoc](../../../../glossary.md#term-javadoc)/documentation impact, unresolved items, and
  `Status: Complete` or `Status: Incomplete` with specific follow-up. Do not mark the brief
  Complete; the coordinator owns planning acceptance.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Dependencies and integration

- Depends on: completed Tuning 0002–0004; project review 2026-10-13 verified finding.
- Conflicts with: other tuning [cache](../../../../glossary.md#term-cache)-format/[persistence](../../../../glossary.md#term-persistence) writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (disjoint [owners](../../../../glossary.md#term-owner) only).
- Common base revision: `728680eaabc44a485da9f32050914ca4f09b295c` plus this [planning](../../../../glossary.md#term-planning)-only brief.
- Integration order: independent cache repair; no [dependency](../../../../glossary.md#term-dependency) on other review repairs.
- Integration validation: module evidence, independent Class C review, final project checkpoint
  if multiple executable modules change, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`git diff --check`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordinator; executor does not edit roadmap/master plans.

Launch verification (2026-10-13): user authorized confirmed repairs in the repository audit;
the existing bounded-[artifact](../../../../glossary.md#term-artifact) contract suffices, dependencies are Complete, source confirms the
late limit check, and no other tuning writer is active. Use an isolated worktree and clean
implementation context, followed by independent targeted review/documentation.

## Integration record

Implementation `ea9059dc` integrated as `d324ee13` after independent review.
20 focused / 73 [module](../../../../glossary.md#term-module) tests and [Javadoc](../../../../glossary.md#term-javadoc); independent code/documentation APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
