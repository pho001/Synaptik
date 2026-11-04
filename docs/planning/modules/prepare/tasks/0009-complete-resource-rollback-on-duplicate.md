# [Prepare](../../../../glossary.md#prepare) 0009 — Complete [rollback](../../../../glossary.md#term-rollback) after duplicate-resource rejection

## Status

Complete

## Goal and class

Class C, existing persistent-resource [ownership](../../../../glossary.md#term-ownership) contract. When a finalizer returns `[A,A,B]`,
or earlier finalizers returned `A` and a later one returns `[A,B]`, reject the duplicate but close
every newly transferred unique resource, including trailing `B`, exactly once in reverse order.
Current `BackendPartitionFinalizationHandoff` throws while scanning and never records trailing
[owners](../../../../glossary.md#term-owner). Source review and a read-only diagnostic reproduced the leak.

## [Scope](../../../../glossary.md#term-scope) and contracts

Own `modules/prepare/.../BackendPartitionFinalizationHandoff.java`, its focused test, and affected
[Javadoc](../../../../glossary.md#term-javadoc)/explanation only. No finalizer or [Runtime](../../../../glossary.md#term-runtime) [ownership](../../../../glossary.md#term-ownership-meanings)/[API](../../../../glossary.md#term-application-programming-interface-api) change.
Read AGENTS, root scope index/Core invariants/[Module](../../../../glossary.md#term-module)-ownership routing/[Dependency](../../../../glossary.md#term-dependency) rules, and
`runtime-prepare-engine.md` headings `modules/prepare`, `Prepare lifecycle`, and persistent
prepared-[resource ownership](../../../../glossary.md#term-resource-ownership) under `modules/runtime`. Inspect finalization result [validation](../../../../glossary.md#term-validation) and
[rollback](../../../../glossary.md#term-rollback), duplicate-resource tests, and the existing resource-[lifecycle](../../../../glossary.md#lifecycle) explanation.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- Collect all unique resources in each already returned result before rejecting its first
  duplicate. Do not call later finalizers or validate/use a rejected executable.
- Preserve first duplicate diagnostics, reverse unique-identity cleanup, exact-once close,
  suppressed distinct cleanup failures, and self-suppression avoidance.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Regress same-result and cross-result duplicates with trailing fresh [owners](../../../../glossary.md#term-owner) and throwing close.
- [Run](../../../../glossary.md#term-run) focused tests, then `./gradlew :modules:prepare:test :modules:prepare:javadoc` once after
  stabilization. No full repository run in worker. Independent Class C review reuses evidence.
- Review/update [Javadoc](../../../../glossary.md#term-javadoc), report files/tests/documentation impact and Complete/Incomplete status.

## Dependencies and integration

- Depends on: [Prepare](../../../../glossary.md#prepare) 0006/0008 Complete; project-review B1.
- Conflicts with: persistent-resource/finalization writers.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS.
- Common base revision: production `728680ea`, plus [planning](../../../../glossary.md#term-planning)-only repair briefs.
- Integration order: independent of [Compiler](../../../../glossary.md#term-compiler)/Tuning/Training repairs.
- Integration validation: focused/module evidence, independent review, coordinator's combined
  repository checkpoint and diff check.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator; no worker roadmap/master-plan edits.

Launch verified 2026-10-13: user-authorized repair, complete dependencies, existing explicit
[ownership](../../../../glossary.md#term-ownership-meanings) contract, disjoint write and semantic [scope](../../../../glossary.md#term-scope). Use clean context and isolated worktree.

## Integration record

Implementation `ec2e7ddf` integrated as `d7de5721` after independent review.
18 focused / 65 [module](../../../../glossary.md#term-module) tests and [Javadoc](../../../../glossary.md#term-javadoc); independent Class C APPROVE.
Affected Javadoc and explanatory documentation reviewed in the same change. No unresolved
review findings. The coordinator's combined repository [checkpoint](../../../../glossary.md#term-checkpoint) passed after ordered integration.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
