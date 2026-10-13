# Metal 0079 — Align scalar provider admission and program preflight

## Status

In progress

## Goal and class

Class C correctness repair for project-review D5. Make already-advertised scalar RELU and
scalar-MUL occurrences executable through the existing qualified routes, without extending the
provider's dtype/geometry domain or bypassing anchor authentication.

## Finding and scope

Engine 0019's device fixture exposed `FLOAT32` scalar `x.exp().relu()` with Metal registered
before CPU: ownership `[cpu, metal]` succeeds, but `engine.session` rejects the scalar RELU in
Java preflight. Provider `supportsCanonicalUnary` explicitly allows scalar RELU; its test asserts
true. `MetalNativeApi` instead permits scalar RELU only in an authenticated anchor epilogue.
The same source mismatch affects scalar MUL: the provider allows it, preflight requires an anchor.
The later native per-operation shape checks accept matching scalars, but the earlier schema-20
decoder also requires rank-zero RELU/MUL to be cleared by authenticated anchor membership.
Two existing actual-device raw-ABI tests confirm native UNSUPPORTED_SHAPE, not successful execution.

Own `MetalNativeApi` validation, the corresponding native decoder guards, coordinated Metal
identity constants/tests, active source/package evidence and the smallest focused tests/Javadoc,
a new bounded actual-device/public integration regression and backend conformance. Audit the
adjacent scalar CLAMP path as a control and separately test each currently advertised low dtype;
do not infer their eligibility from FLOAT32. No new operation, route, provider admission, fusion
eligibility, wire schema, kernel algorithm or generalized scalar-unary support is authorized.
Remove only the obsolete standalone-admission dependency on anchor membership, not validation
of actual fused instruction records, shape/type consistency, memory bounds or native resources.

If the current kernel cannot execute an advertised case within Model semantics after the bounded
decoder correction, stop and report the conflict rather than widening capability, changing Model
or hiding it with an unrelated fixture. Advance coordinated Metal identity 35 → 36 to distinguish
the corrected admitted program domain; retain ABI 7/schema 20. Coordinator owns all shared
root/scoped/API/explanatory/glossary version facts and integrates them serially with CPU 0022.
Worker owns only Metal-specific Javadoc/backend guide/native README and code/evidence.

## Contracts and acceptance

Read AGENTS; root scope index, Core invariants, Module-ownership routing, Dependency rules and
Testing requirements; `backend-execution.md` Concrete backend modules, Metal backend, Low-precision
capability evidence, Profile-free backend capability and identity; Model unary RELU/scalar MUL
Javadocs; actual provider, preflight, anchor and native shape/dispatch paths.

- Reproduce provider-true/prepare-false in a focused test before repair.
- For each already-admitted dtype, scalar RELU/MUL and CLAMP control pass ordinary analysis,
  preparation and actual-device execution. Include finite, NaN/Inf and signed-zero boundaries
  relevant to each existing contract, gradient metadata and direct/composed scalar results.
- Public scalar `x.exp().relu()` with Metal-first registration prepares and computes correctly;
  owner sequence remains `[cpu, metal]`. Use a new integration class, not Engine 0019's file.
- Positive-rank cases and valid rank-zero anchor epilogues remain legal. Forged anchor records,
  mismatched shapes/types, unsupported operations and unsupported geometries still reject.
- Rebuild/sign/package the changed native source and renew active source evidence whose hashes
  changed; preserve unaffected shader-source/AIR results rather than rerunning historical proofs.
  Run focused regressions, one Metal test/Javadoc pass and affected conformance/integration
  tests against that source-matched identity-36 package. No worker full repository run.
- Independent Class C code/documentation review checks the admission boundary and evidence.
  Update affected explanation/Javadoc; preserve numerical/route contracts and historical records.
- Report changed files, commit, exact commands/results, docs impact and limitations; no push.

## Dependencies and integration

- Depends on: independently reviewed and integrated Metal 0078; current Model semantics; D5.
- Conflicts with: Metal provider/preflight/anchor/native/schema writers; any shared version
  document edit until serialized by coordinator.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS, serial Metal successor.
- Common base revision: `b3c48901` plus this revised brief (0078 integrated at `1db53756`).
- Integration order: 0078 → 0079 → combined checkpoint; independent of CPU code ownership.
- Integration validation: focused/module/device evidence, independent review and final checkpoint.
- Shared-document integration owner: coordinator; no worker plans or shared version edits.

Launch verified 2026-10-13: 0078 implementation, actual-device/module/package validation and
independent Class C review are approved and integrated at `1db53756`. Only the repair group's
combined repository checkpoint remains deferred until all serial successors are integrated;
it is not a prerequisite for this disjoint Java preflight repair. No other Metal writer is active.
The provider already advertises these scalars. Subsequent raw-ABI diagnosis identified the second
native decoder guard, so the coordinator explicitly authorizes its bounded correction and native
rebuild within the user's correctness-repair request. This adds no operation, provider domain or
numerical contract. Resume the same clean isolated implementation context only after it receives
this revised brief. Shared version documents remain coordinator-owned.
