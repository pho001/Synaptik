# [Metal](../../../../glossary.md#term-metal) 0079 — Align [scalar](../../../../glossary.md#term-scalar) provider [admission](../../../../glossary.md#term-admission) and program [preflight](../../../../glossary.md#term-preflight)

## Status

Complete

## Goal and class

Class C correctness repair for project-review D5. Make already-advertised [scalar](../../../../glossary.md#term-scalar) RELU and
scalar-MUL occurrences executable through the existing qualified routes, without extending the
provider's [dtype](../../../../glossary.md#data-type--datatype)/geometry domain or bypassing anchor [authentication](../../../../glossary.md#term-authentication).

## Finding and [scope](../../../../glossary.md#term-scope)

[Engine](../../../../glossary.md#term-engine) 0019's [device](../../../../glossary.md#term-device) [fixture](../../../../glossary.md#term-fixture) exposed [`FLOAT32`](../../../../glossary.md#term-float32) [scalar](../../../../glossary.md#term-scalar) `x.exp().relu()` with [Metal](../../../../glossary.md#term-metal) registered
before [CPU](../../../../glossary.md#term-cpu): [ownership](../../../../glossary.md#term-ownership-meanings) `[cpu, metal]` succeeds, but `engine.session` rejects the scalar RELU in
Java [preflight](../../../../glossary.md#term-preflight). Provider `supportsCanonicalUnary` explicitly allows scalar RELU; its test asserts
true. `MetalNativeApi` instead permits scalar RELU only in an authenticated anchor [epilogue](../../../../glossary.md#term-epilogue).
The same source mismatch affects scalar MUL: the provider allows it, preflight requires an anchor.
The later native per-[operation](../../../../glossary.md#operation) [shape](../../../../glossary.md#shape) checks accept matching scalars, but the earlier [schema](../../../../glossary.md#term-schema)-20
decoder also requires [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) RELU/MUL to be cleared by authenticated anchor membership.
Two existing actual-device raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi) tests confirm native UNSUPPORTED_SHAPE, not successful execution.
After removing those guards, actual-device tests expose one further part of the same mismatch:
FLOAT32 scalar MUL lowers its exact literal as [rank](../../../../glossary.md#term-rank) one, making its result incompatible with the
advertised rank-zero output. The bounded rank-zero MUL literal/result-shape correction is in scope;
do not change the shared scalar helper's behavior for unrelated operations.

Own `MetalNativeApi` [validation](../../../../glossary.md#term-validation), the corresponding native decoder guards, coordinated Metal
[identity](../../../../glossary.md#term-identity-meanings) [constants](../../../../glossary.md#term-constant)/tests, active source/package evidence and the smallest focused tests/[Javadoc](../../../../glossary.md#term-javadoc),
a new bounded actual-device/public integration [regression](../../../../glossary.md#term-regression) and [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance). Audit the
adjacent scalar CLAMP path as a control and separately test each currently advertised low [dtype](../../../../glossary.md#data-type--datatype);
do not infer their eligibility from FLOAT32. No new operation, route, provider [admission](../../../../glossary.md#term-admission), [fusion](../../../../glossary.md#term-fusion)
eligibility, wire schema, [kernel](../../../../glossary.md#kernel) algorithm or generalized scalar-[unary](../../../../glossary.md#term-unary) support is authorized.
Remove only the obsolete standalone-admission [dependency](../../../../glossary.md#term-dependency) on anchor membership, not validation
of actual [fused](../../../../glossary.md#specialization-and-fusion) instruction records, shape/type consistency, memory bounds or native resources.

If the current kernel cannot execute an advertised case within [Model](../../../../glossary.md#term-model) semantics after the bounded
decoder correction, stop and report the conflict rather than [widening](../../../../glossary.md#term-widening) capability, changing Model
or hiding it with an unrelated fixture. Advance coordinated Metal identity 35 → 36 to distinguish
the corrected admitted program domain; retain ABI 7/schema 20. Coordinator owns all shared
root/scoped/[API](../../../../glossary.md#term-application-programming-interface-api)/explanatory/glossary version facts and integrates them serially with CPU 0022.
Worker owns only Metal-specific Javadoc/backend guide/native README and code/evidence.

## Contracts and acceptance

Read AGENTS; root [scope](../../../../glossary.md#term-scope) index, Core invariants, [Module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings) routing, [Dependency](../../../../glossary.md#term-dependency) rules and
Testing requirements; `backend-execution.md` [Concrete backend](../../../../glossary.md#concrete-backend) modules, [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend), Low-[precision](../../../../glossary.md#term-precision)
capability evidence, Profile-free [backend capability](../../../../glossary.md#backend-capability) and [identity](../../../../glossary.md#term-identity-meanings); [Model](../../../../glossary.md#term-model) [unary](../../../../glossary.md#term-unary) RELU/[scalar](../../../../glossary.md#term-scalar) MUL
Javadocs; actual provider, [preflight](../../../../glossary.md#term-preflight), anchor and native [shape](../../../../glossary.md#shape)/[dispatch](../../../../glossary.md#term-dispatch) paths.

- Reproduce provider-true/[prepare](../../../../glossary.md#prepare)-false in a focused test before repair.
- For each already-admitted dtype, scalar RELU/MUL and CLAMP control pass ordinary analysis,
  preparation and actual-device execution. Include finite, NaN/Inf and signed-zero boundaries
  relevant to each existing contract, gradient metadata and direct/composed scalar results.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [device glossary](../../../../glossary.md#term-device); [finite glossary](../../../../glossary.md#term-finite); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Public scalar `x.exp().relu()` with Metal-first registration prepares and computes correctly;
  owner sequence remains <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`[cpu, metal]`. Use a new integration class, not [Engine](../../../../glossary.md#term-engine) 0019's file.
- Positive-rank cases and valid rank-zero anchor epilogues remain legal. Forged anchor records,
  mismatched shapes/types, unsupported operations and unsupported geometries still reject.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Rebuild/sign/package the changed native source and renew active source evidence whose hashes
  changed; preserve unaffected shader-source/AIR results rather than rerunning historical proofs.
  Run focused regressions, one Metal test/Javadoc pass and affected conformance/integration
  tests against that source-matched identity-36 package. No worker full repository run.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([shader glossary](../../../../glossary.md#term-shader); [AIR glossary](../../../../glossary.md#term-air); [Run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Independent Class C code/documentation review checks the admission boundary and evidence.
  Update affected explanation/Javadoc; preserve numerical/route contracts and historical records.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Report changed files, [commit](../../../../glossary.md#term-commit), exact commands/results, docs impact and limitations; no push.

## Dependencies and integration

- Depends on: independently reviewed and integrated [Metal](../../../../glossary.md#term-metal) 0078; current [Model](../../../../glossary.md#term-model) semantics; D5.
- Conflicts with: Metal provider/preflight/anchor/native/schema writers; any shared version
  document edit until serialized by coordinator.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS, serial Metal successor.
- Common base revision: `b3c48901` plus this revised brief (0078 integrated at `1db53756`).
- Integration order: 0078 → 0079 → combined [checkpoint](../../../../glossary.md#term-checkpoint); independent of [CPU](../../../../glossary.md#term-cpu) code [ownership](../../../../glossary.md#term-ownership-meanings).
- Integration [validation](../../../../glossary.md#term-validation): focused/[module](../../../../glossary.md#term-module)/[device](../../../../glossary.md#term-device) evidence, independent review and final checkpoint.
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator; no worker plans or shared version edits.

Launch verified 2026-10-13: 0078 implementation, actual-device/module/package validation and
independent Class C review are approved and integrated at `1db53756`. Only the repair group's
combined repository checkpoint remains deferred until all serial successors are integrated;
it is not a prerequisite for this disjoint Java [preflight](../../../../glossary.md#term-preflight) repair. No other Metal writer is active.
The provider already advertises these [scalars](../../../../glossary.md#term-scalar). Subsequent raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi) diagnosis identified the second
native decoder [guard](../../../../glossary.md#term-guard), so the coordinator explicitly authorizes its bounded correction and native
rebuild within the user's correctness-repair request. Subsequent device validation passed six
[admission](../../../../glossary.md#term-admission) tests and eight of ten integration cases, with no skips; the two [FLOAT32](../../../../glossary.md#term-float32) MUL failures
identify a [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) literal versus [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) result, not a numerical-policy conflict. Coordinator
source review confirms `exact_float32_scalar` creates `shape:@[@1]` and MUL currently performs no
[rank](../../../../glossary.md#term-rank) repair. Its bounded standalone rank-zero correction is authorized under the same advertised
capability contract. This adds no [operation](../../../../glossary.md#operation), provider domain, [fusion](../../../../glossary.md#term-fusion) or numerical contract.
[Resume](../../../../glossary.md#schema-serialization-and-checkpoint) the same clean isolated context with this revised brief. Shared version documents remain
coordinator-owned.

## Integration record

Implementation `1f9a5833` integrated as `22a2f1be` after independent Class C APPROVE.
25 focused tests passed without skips; full [Metal](../../../../glossary.md#term-metal) 346 passed / 10 optional [observer](../../../../glossary.md#term-observer) skips;
21 [conformance](../../../../glossary.md#term-conformance) and 10 [scalar](../../../../glossary.md#term-scalar) integration tests passed. [Javadoc](../../../../glossary.md#term-javadoc) passed with 46 existing warnings.
Native source/package [hashes](../../../../glossary.md#term-hash) were independently checked. Exact commands and source-matched
[identity](../../../../glossary.md#term-identity-meanings)-36 library: `backends/metal/evidence/0079/validation.json`.
Worker guide/README/Javadocs and coordinator shared version/glossary corrections were reviewed.
No unresolved 0079 finding; shared-document finalization and the group [checkpoint](../../../../glossary.md#term-checkpoint) passed.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete
