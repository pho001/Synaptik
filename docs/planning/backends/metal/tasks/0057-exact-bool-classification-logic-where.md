# Task 0057: Exact BOOL Classification, Logic, and WHERE

## Status

Complete

Planned at `8927a0f2` from base `ca450b03` and implemented at `e5d9d5c7` after Task 0053 reached
its external constructive-real proof blocker. The cutover preserves Task 0053 fail-closed source
and evidence unchanged.

## Change class

Class C. This is an atomic Metal capability, route, native execution, lifecycle, publication,
test, and documentation cutover. It changes no public API or Model semantics.

## Goal

Implement exact profile-common Metal support for private operation wires:

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

- Retain ABI 5, exactly thirteen exports, operation wires `1..115`, type wires `1..6`, and
  attribute wires `0..41`. Advance node schema `13 -> 14` because the fixed route is now embedded
  at header byte offset 40; old schema-13 images and zero/unknown routes fail closed with no dual
  decoder or compatibility shim.
- Lower exactly wires `40..45` and `51` through the typed schema-14 program.
- Change the custom catalog state for those seven wires from `PENDING` to `AVAILABLE` with a closed
  Task-0057 reason, while retaining each direct MPSGraph row as `DIRECT / MD-PRED`.
- Replace the Task-0052-specific whole-program route name with `CUSTOM_PROGRAM`, migrate every
  caller, and retain stable private route wire `3`.
- Advance persisted workload, exact-policy, candidate, compatibility, route, and codec identity
  `14 -> 15`; old identities fail closed.
- Candidate generation exposes custom first for the exact domain and MPSGraph second only for an
  exact single-node new operation whose complete direct selector recipe is implemented. Production
  always selects custom. Package-private forcing can select only a freshly regenerated
  authenticated candidate.

## Implementation order

1. Re-audit current Model metadata, Compiler capture, schema lowering, route catalog/identity,
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

## Completion evidence

- Implementation `e5d9d5c7` adds capability/lowering for wires `40..45` and `51`, the shared
  `CUSTOM_PROGRAM` recipe, fixed raw/integer kernels, direct audited selectors, schema-14 route
  authentication, identity 15, strict BOOL ingress, complete declared liveness, and exact
  publication. Remediation `7a3bb072` closes rank-zero Java/native image acceptance and expands
  boundary, route, and conformance proof.
- Native build, fixed ad-hoc signing, local packaging, the native package verifier, and Gradle's
  opt-in `verifyMetalNativePackage` passed. ABI remains 5 with exactly thirteen exports; the
  manifest records node schema 14.
- The complete `:backends:metal:test` task and Metal Javadoc passed against the fresh packaged
  dylib. Focused native coverage executes both production custom and forced direct routes across
  both profiles, both zero signs, minimum/largest subnormals, finite extrema, both infinities, and
  signed quiet/signaling NaNs with payloads; logic covers every truth-table pair and rank-16
  right-aligned broadcasting; WHERE preserves selected raw bits.
- Focused backend conformance passed for both profiles, rank-zero exclusion, and explicit wire-73
  rejection. The public no-skip Engine smoke passed with sole owner `metal`, fixed
  `CUSTOM_KERNEL` trace route, custom/MPSGraph mixing, exact canonical BOOL and raw FLOAT32
  publications, pre-invocation rejection of noncanonical BOOL bytes, and a subsequent successful
  run proving session liveness.
- The supplied independent P1/P2 audit was remediated by the clean schema-14 cutover, explicit
  rank-zero capability tests, public Engine proof, and documentation of the closed owned internal
  BOOL-producer contract. A subsequent cumulative Class C review found three P2 validation and
  documentation defects; remediation `7a3bb072` closed all three, and the same reviewer approved
  the live tree with zero P0/P1/P2 findings. Focused architecture/backend/API/native/planning
  documentation and `git diff --check` passed. No timing, benchmark, or final full repository build
  was run.
