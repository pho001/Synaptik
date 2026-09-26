# Task 0057: Exact BOOL Classification, Logic, and WHERE

## Status

In progress

Authorized as the sole serialized Metal production frontier after Task 0053 reached its external
constructive-real proof blocker. This task starts from planning base `ca450b03` and must preserve
Task 0053 fail-closed source and evidence unchanged.

## Change class

Class C. This is an atomic Metal capability, route, native execution, lifecycle, publication,
test, and documentation cutover. It changes no public API or Model semantics.

## Goal

Implement exact profile-common Metal support for schema-13 operation wires:

- `40 IS_FINITE`, `41 IS_NAN`, and `42 IS_INF` over canonical `FLOAT32`;
- `43 LOGICAL_AND`, `44 LOGICAL_OR`, and `45 LOGICAL_NOT` over canonical `BOOL`; and
- `51 WHERE` with canonical `BOOL` condition and canonical `FLOAT32` branches.

Implement both systematic route candidates where structurally valid:

1. the named direct MPSGraph selectors as independently forceable structural candidates; and
2. an auditable custom raw/integer route as the deterministic production route.

MPSGraph selector presence, successful creation, or bounded execution is not `DOMAIN-PASS`. The
custom route earns correctness from exact bit/integer semantics and becomes the fixed production
route without timing, benchmarking, runtime selection, fallback, retry, or autotuning.

## Exact capability domain

Every descriptor is resolved dense-contiguous, zero-offset, non-view, fully static, positive, and
rank `1..16`; all checked byte and launch geometry must fit before resource creation.

- Classification has one `FLOAT32` input and same-Shape `BOOL` output; both are no-grad at the
  occurrence boundary except that input `requiresGrad` may be either value because classification
  is nondifferentiable; output is always `false`.
- `AND`/`OR` have two no-grad `BOOL` inputs, the exact right-aligned broadcast output Shape, and a
  no-grad `BOOL` output. `NOT` has one no-grad `BOOL` input and same-Shape no-grad `BOOL` output.
- `WHERE` has an ordered no-grad `BOOL` condition followed by two same-typed `FLOAT32` branches.
  Branches first broadcast with each other, then the condition broadcasts to that result. Output
  has that final Shape and `requiresGrad == (trueBranch.requiresGrad || falseBranch.requiresGrad)`.
- Caller BOOL ingress is accepted only as canonical bytes `0` or `1`. All BOOL outputs are canonical.
  Existing local comparison/ONE_HOT BOOL values may feed the new nodes. Cross-owner BOOL transfer
  remains within the existing typed-transfer contract and receives no unrelated widening.
- Wire `73 SELECT` is a scalar-index operation and is explicitly excluded.

Both numerical profiles admit this exact discrete/selection domain. No scalar/rank-zero, dynamic or
zero extent, view, noncanonical BOOL, non-`FLOAT32` classification/branch payload, dtype promotion,
or new gradient topology is added.

## Exact custom semantics

- Classification reads each FLOAT32 element as an unmodified `uint32` word. `IS_FINITE` is exponent
  not all ones; `IS_NAN` is exponent all ones and fraction nonzero; `IS_INF` is exponent all ones
  and fraction zero. Sign, quiet/signaling bit, payload, zero sign, and subnormal magnitude never
  enter floating arithmetic.
- Logic reads canonical BOOL bytes and writes exactly byte `0` or `1`; `AND`, `OR`, and `NOT` are
  integer truth-table operations.
- `WHERE` computes right-aligned broadcast source ordinals and copies the selected branch's raw
  `uint32` word verbatim. It must preserve every NaN payload/sign/quiet bit, signed zero, subnormal,
  infinity, and ordinary finite word.
- Every operand, result, and internal logical value is present in the declared whole-program value
  table and resource liveness. Mixing new custom nodes with already approved Task-0052 custom nodes
  and existing exact MPSGraph steps must remain one honest custom-program recipe; no hidden buffer,
  undeclared intermediate, per-node fallback, or route switch is allowed.

## Schema, route, and identity cutover

- Reuse ABI 5, exactly thirteen exports, schema 13, operation wires `1..115`, type wires `1..6`, and
  attribute wires `0..41`. Add no export, public enum, operation/type/attribute wire, or dual decoder.
- Lower exactly wires `40..45` and `51` through the existing typed schema-13 program.
- Change the custom catalog state for those seven wires from `PENDING` to `AVAILABLE` with a closed
  Task-0057 reason, while retaining each direct MPSGraph row as `DIRECT / MD-PRED`.
- Replace the Task-0052-specific whole-program route naming with one exact custom-program identity
  only if required by the now-shared executor. Migrate every caller; add no alias or compatibility
  shim. Keep stable private route wire `3` if its semantics remain the existing whole-program custom
  family.
- Bump every persisted workload, exact-policy, candidate, compatibility, route, and codec identity
  whose accepted graph/domain or canonical bytes change. Old identities fail closed.
- Candidate generation must expose custom for the exact domain and MPSGraph only where its complete
  structural recipe is implemented. Production always selects custom. Package-private forcing may
  select only a freshly regenerated authenticated candidate.

## Implementation order

1. Re-audit current Model metadata, Compiler capture, schema-13 lowering, route catalog/identity,
   native decoder, generated custom source, typed ingress/publication, and Engine composition.
2. Add exact capability and Java lowering/preflight validation, including arity, dtype, Shape,
   broadcasting, gradient flags, canonical BOOL ingress, and wire-73 rejection.
3. Extend the shared custom raw/integer source and native whole-program builder for all seven nodes,
   exact input counts, typed resources, broadcast metadata, and liveness.
4. Add direct MPSGraph selector lowering for the seven nodes and keep it structural-only.
5. Update fixed route generation/authentication/codec versions, preparation/finalization/resource
   declaration, publication, trace identity, catalog counts, and exact no-fallback behavior.
6. Prove the public path through a real locally built dylib and public `Engine` composition, then
   update contracts/docs and obtain independent Class C review. Remediate every P0/P1/P2 finding.

## Required verification

- Native build plus exact export audit: ABI remains 5 with the same thirteen exports.
- Focused Metal unit and schema/native malformed-control tests.
- Capability and backend-conformance matrix with `41 true / 74 false = 115`, both profiles for the
  seven exact rows, and explicit wire-73 exclusion.
- Classification raw-word corpus covering both zeros, normals, minimum/ordinary subnormals,
  infinities, and signed quiet/signaling NaNs with payloads.
- Full BOOL truth tables, noncanonical caller-ingress rejection, unary/binary Shape checks, operand
  order, right-aligned broadcast including rank 16, and canonical output bytes.
- WHERE raw-bit preservation for signed zeros, subnormals, infinities, and multiple NaN payloads;
  condition/branch broadcast order and mixed branch `requiresGrad` cases.
- Mixed comparison/classification/logic/WHERE programs, existing exact nodes around the new nodes,
  all declared intermediate buffers, fresh outputs, no alias, no fallback, and unsupported-mixture
  fail-closed checks.
- Independent forced custom and forced MPSGraph structural candidate tests. MPSGraph results may be
  recorded only as regression evidence and never promoted to domain proof.
- Repeated runs, concurrent independent runs, close-vs-run behavior, rollback, malformed resources,
  authenticated route handoff, stale identity rejection, and trace route identity.
- Public no-skip `Engine` smoke against the freshly built real dylib, with no CPU ownership or skip.
- Focused architecture/Javadoc/docs checks and `git diff --check`. Do not run a final full repository
  build and do not run timing or benchmarks.

## Dependencies and conflicts

Depends on Complete Tasks 0052, 0055, and 0056; current exact Model classification/logical/WHERE
semantics; current Compiler forward/gradient-role capture; current Prepare/Runtime/Engine lifecycle;
and ADR 0022's deterministic correctness-first route policy. Historical Task 0010 supplies blocker
context only and is not a dependency or authority for the new custom route.

Conflicts with every concurrent Metal capability, schema, native preflight, generated-kernel,
candidate/codec/route, preparation/finalization, buffer, publication, trace, test, package, and
shared documentation edit, and with any resumed Task-0053 production/proof integration. Parallel
group: none. The implementation owner integrates all shared files serially.

## Architecture and documentation

Update the Metal backend contract, API/backend guides, native README where its executable domain is
stated, master plan, and roadmap. No ADR is required because the task follows the existing private
schema, typed-transfer, synchronous lifecycle, and deterministic fixed-route architecture rather
than changing an architecture decision.

## Completion gate

Complete only after the exact custom route, structural MPSGraph route, capability, schema/native
validation, honest mixed-program resources, publication, lifecycle/concurrency, no-fallback proof,
real public Engine smoke, documentation, clean commit, and independent Class C approval all pass.
Any missing exactness or lifecycle obligation leaves the seven rows fail-closed.
