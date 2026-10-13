# Metal 0079 — Align scalar provider admission and program preflight

## Status

Ready

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
Native RELU and scalar arithmetic shape validation already accept matching rank-zero shapes.

Own `MetalNativeApi` validation and the smallest associated focused tests/Javadoc, a new bounded
actual-device/public integration regression and backend conformance as appropriate. Audit the
adjacent scalar CLAMP path as a control and separately test each currently advertised low dtype;
do not infer their eligibility from FLOAT32. No new operation, route, provider admission, fusion
eligibility, schema, native kernel or generalized scalar-unary support is authorized.

If the current native route cannot execute an advertised case within Model semantics, stop and
report the precise conflict rather than widening capability, changing Model or hiding it with
an unrelated test fixture. Any necessary compatibility-identity change requires coordination
of shared version documents; justify no bump if formats and interpretation are unchanged.

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
- Run focused regressions, one Metal test/Javadoc pass and affected conformance/integration
  tests using the source-matched identity-35 library from 0078. No native rebuild if source is
  unchanged and no duplicated full repository run. Coordinator owns combined checkpoint.
- Independent Class C code/documentation review checks the admission boundary and evidence.
  Update affected explanation/Javadoc; preserve numerical/route contracts and historical records.
- Report changed files, commit, exact commands/results, docs impact and limitations; no push.

## Dependencies and integration

- Depends on: independently reviewed and integrated Metal 0078; current Model semantics; D5.
- Conflicts with: Metal provider/preflight/anchor/native/schema writers; any shared version
  document edit until serialized by coordinator.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS, serial Metal successor.
- Common base revision: `1db53756` (0078 code `6e8d6b26` plus reviewed version-document repair).
- Integration order: 0078 → 0079 → combined checkpoint; independent of CPU code ownership.
- Integration validation: focused/module/device evidence, independent review and final checkpoint.
- Shared-document integration owner: coordinator; no worker plans or shared version edits.

Launch verified 2026-10-13: 0078 implementation, actual-device/module/package validation and
independent Class C review are approved and integrated at `1db53756`. Only the repair group's
combined repository checkpoint remains deferred until all serial successors are integrated;
it is not a prerequisite for this disjoint Java preflight repair. No other Metal writer is active.
The current native checks already admit matching scalar RELU/MUL, and the provider advertises
them; the user-authorized audit repair needs no new operation or numerical contract. Use an
isolated worktree and clean implementation context. Shared version documents remain coordinator-owned.
