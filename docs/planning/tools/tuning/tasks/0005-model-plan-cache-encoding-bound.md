# Tuning 0005 — Bound model-plan cache encoding before allocation

## Status

Review needed

## Change class and goal

Class C: persistent-artifact resource safety, without a wire/schema or public API change.
Reject an oversized model-plan cache before allocating/encoding its complete payload. Preserve
canonical bytes for every accepted schema-1 artifact and preserve the destination on failure.

## Finding and scope

`ModelPlanCacheFile.encode` grows a `ByteArrayOutputStream`, copies its payload, and computes its
checksum before checking `MAX_FILE_BYTES`. Individually valid bounded entries can exceed the
aggregate 16 MiB limit; the current check therefore does not bound encoder memory use.
`WorkloadCacheFile.encode` already checks aggregate size before allocating its output.

Owned paths: `tools/tuning/src/main/java/io/github/pho001/synaptik/tools/tuning/ModelPlanCacheFile.java`
and focused tests under that module. Review the affected implementation Javadoc and
`docs/architecture/performance-evidence-and-tuning.md`; update explanation only if needed.
No changes to candidate selection, cache identities, schema, file publication policy, or lifecycle.

## Required contracts

Read `AGENTS.md`; the root `ARCHITECTURE.md` scope index, Core invariants, Module-ownership routing,
and Dependency rules; and `docs/architecture/contracts/backend-execution.md`, heading
`Performance evidence and optimization tooling`. Read current encoder/decoder, the Phase-1
bounded encoder as a local comparison, and `CompletePlanTuningCacheTest`.

## Acceptance and validation

- Checked preflight accounts for every fixed/length-prefixed field and the checksum before any
  complete output allocation. Oversize is a controlled failure, not an allocation failure.
- Accepted schema-1 bytes remain canonical and decoder/inspection compatible.
- Focused regression covers aggregate overflow of the limit using individually legal fields,
  boundary accounting, and unchanged target/no temporary artifacts on rejection.
- Do not require allocating gigabytes or timing a failure to prove the bound; use a directly
  testable bounded-size calculation or equivalent deterministic test.
- Run focused regression, then `./gradlew :tools:tuning:test :tools:tuning:javadoc` once after
  stabilization. Independent review reuses successful tests unless it finds a specific risk.
- Report changed files, evidence, Javadoc/documentation impact, unresolved items, and
  `Status: Complete` or `Status: Incomplete` with specific follow-up. Do not mark the brief
  Complete; the coordinator owns planning acceptance.

## Dependencies and integration

- Depends on: completed Tuning 0002–0004; project review 2026-10-13 verified finding.
- Conflicts with: other tuning cache-format/persistence writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (disjoint owners only).
- Common base revision: `728680eaabc44a485da9f32050914ca4f09b295c` plus this planning-only brief.
- Integration order: independent cache repair; no dependency on other review repairs.
- Integration validation: module evidence, independent Class C review, final project checkpoint
  if multiple executable modules change, and `git diff --check`.
- Shared-document integration owner: main coordinator; executor does not edit roadmap/master plans.

Launch verification (2026-10-13): user authorized confirmed repairs in the repository audit;
the existing bounded-artifact contract suffices, dependencies are Complete, source confirms the
late limit check, and no other tuning writer is active. Use an isolated worktree and clean
implementation context, followed by independent targeted review/documentation.

## Integration record

Implementation `ea9059dc` integrated as `d324ee13` after independent review.
20 focused / 73 module tests and Javadoc; independent code/documentation APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. Coordinator's combined repository checkpoint and planning closure remain
outstanding; this status does not imply an outstanding code-review finding.
