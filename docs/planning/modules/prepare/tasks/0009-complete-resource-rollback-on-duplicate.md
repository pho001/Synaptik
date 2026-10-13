# Prepare 0009 — Complete rollback after duplicate-resource rejection

## Status

Review needed

## Goal and class

Class C, existing persistent-resource ownership contract. When a finalizer returns `[A,A,B]`,
or earlier finalizers returned `A` and a later one returns `[A,B]`, reject the duplicate but close
every newly transferred unique resource, including trailing `B`, exactly once in reverse order.
Current `BackendPartitionFinalizationHandoff` throws while scanning and never records trailing
owners. Source review and a read-only diagnostic reproduced the leak.

## Scope and contracts

Own `modules/prepare/.../BackendPartitionFinalizationHandoff.java`, its focused test, and affected
Javadoc/explanation only. No finalizer or Runtime ownership/API change.
Read AGENTS, root scope index/Core invariants/Module-ownership routing/Dependency rules, and
`runtime-prepare-engine.md` headings `modules/prepare`, `Prepare lifecycle`, and persistent
prepared-resource ownership under `modules/runtime`. Inspect finalization result validation and
rollback, duplicate-resource tests, and the existing resource-lifecycle explanation.

## Acceptance and validation

- Collect all unique resources in each already returned result before rejecting its first
  duplicate. Do not call later finalizers or validate/use a rejected executable.
- Preserve first duplicate diagnostics, reverse unique-identity cleanup, exact-once close,
  suppressed distinct cleanup failures, and self-suppression avoidance.
- Regress same-result and cross-result duplicates with trailing fresh owners and throwing close.
- Run focused tests, then `./gradlew :modules:prepare:test :modules:prepare:javadoc` once after
  stabilization. No full repository run in worker. Independent Class C review reuses evidence.
- Review/update Javadoc, report files/tests/documentation impact and Complete/Incomplete status.

## Dependencies and integration

- Depends on: Prepare 0006/0008 Complete; project-review B1.
- Conflicts with: persistent-resource/finalization writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea`, plus planning-only repair briefs.
- Integration order: independent of Compiler/Tuning/Training repairs.
- Integration validation: focused/module evidence, independent review, coordinator's combined
  repository checkpoint and diff check.
- Shared-document integration owner: coordinator; no worker roadmap/master-plan edits.

Launch verified 2026-10-13: user-authorized repair, complete dependencies, existing explicit
ownership contract, disjoint write and semantic scope. Use clean context and isolated worktree.

## Integration record

Implementation `ec2e7ddf` integrated as `d7de5721` after independent review.
18 focused / 65 module tests and Javadoc; independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. Coordinator's combined repository checkpoint and planning closure remain
outstanding; this status does not imply an outstanding code-review finding.
