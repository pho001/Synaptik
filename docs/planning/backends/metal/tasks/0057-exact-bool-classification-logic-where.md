# Task 0057: Exact [BOOL](../../../../glossary.md#data-type--datatype) Classification, Logic, and WHERE

## Reader context

Classification reads FLOAT32 encodings and produces Boolean values; logic consumes canonical bytes; WHERE chooses and copies a branch's raw word. The condition is not a numerical multiplier: unselected NaNs and signed zeros must not be transformed by arithmetic. Java checks descriptor relationships; native execution validates resources and writes declared targets.

Application BOOL feeds contain values, but Metal borrows their input storage rather than owning its lifetime. Results use fresh run storage. This opened a bounded FLOAT32/BOOL slice; current carrier, rank and gradient coverage has evolved beyond its counts.

- [Current classification logic and WHERE domains](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection) — Locates typed roles and branch-gradient relationships.
- [Current borrowed inputs and fresh targets](../../../../backend-guide/metal-backend.md#cold-run-setup-and-binding) — Clarifies storage ownership.

## Status

Complete

Planned at `8927a0f2` from base `ca450b03` and implemented at `e5d9d5c7` after Task 0053 reached
its external constructive-real proof blocker. The cutover preserves Task 0053 [fail-closed](../../../../glossary.md#term-fail-closed) source
and evidence unchanged.

## Change class

Class C. This is an [atomic](../../../../glossary.md#resource-transaction-and-rollback) [Metal](../../../../glossary.md#term-metal) capability, route, native execution, [lifecycle](../../../../glossary.md#lifecycle), [publication](../../../../glossary.md#term-publication),
test, and documentation cutover. It changes no public [API](../../../../glossary.md#term-application-programming-interface-api) or [Model](../../../../glossary.md#term-model) semantics.

## Goal

Implement exact profile-common [Metal](../../../../glossary.md#term-metal) support for private [operation](../../../../glossary.md#operation) wires:

- `40 IS_FINITE`, `41 IS_NAN`, and `42 IS_INF` over [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32);
- `43 LOGICAL_AND`, `44 LOGICAL_OR`, and `45 LOGICAL_NOT` over canonical [`BOOL`](../../../../glossary.md#data-type--datatype); and
- `51 WHERE` with canonical `BOOL` condition and canonical `FLOAT32` branches.

Implement both systematic route candidates where structurally valid:

1. the named direct [MPSGraph](../../../../glossary.md#term-mpsgraph) selectors as independently forceable structural candidates; and
2. an auditable custom raw/integer route as the deterministic production route.

MPSGraph selector presence, successful creation, or bounded execution is not `DOMAIN-PASS`. The
custom route earns correctness from exact bit/integer semantics and becomes the fixed production
route without timing, [benchmarking](../../../../glossary.md#term-benchmarking), [runtime](../../../../glossary.md#term-runtime) selection, fallback, retry, or [autotuning](../../../../glossary.md#autotuning--model-autotuning).

## Exact capability domain

Every [descriptor](../../../../glossary.md#term-descriptor) is resolved dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings), fully static, positive, and
[rank](../../../../glossary.md#term-rank) `1..16`; all checked byte and launch geometry must fit before resource creation.

- Classification has one [`FLOAT32`](../../../../glossary.md#term-float32) input and same-[Shape](../../../../glossary.md#shape) [`BOOL`](../../../../glossary.md#data-type--datatype) output; both are no-grad at the
  occurrence boundary except that input <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>`requiresGrad` may be either value because classification
  is nondifferentiable; output is always `false`.
- `AND`/`OR` have two no-grad `BOOL` inputs, the exact right-aligned broadcast output Shape, and a
  no-grad `BOOL` output. `NOT` has one no-grad `BOOL` input and same-Shape no-grad `BOOL` output.
- `WHERE` has an ordered no-grad `BOOL` condition followed by two same-typed `FLOAT32` branches.
  Branches first broadcast with each other, then the condition broadcasts to that result. Output
  has that final Shape and `requiresGrad == (trueBranch.requiresGrad || falseBranch.requiresGrad)`.
- Caller BOOL ingress is accepted only as [canonical](../../../../glossary.md#term-canonical-form) bytes `0` or `1`. All BOOL outputs are canonical.
  Existing local comparison/ONE_HOT BOOL values may feed the new nodes. Cross-owner BOOL transfer
  remains within the existing typed-transfer contract and receives no unrelated widening.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner); [widening glossary](../../../../glossary.md#term-widening))</span>
- Wire `73 SELECT` is a [scalar](../../../../glossary.md#term-scalar)-index [operation](../../../../glossary.md#operation) and is explicitly excluded.

Both numerical profiles admit this exact discrete/selection domain. No scalar/[rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), dynamic or
zero [extent](../../../../glossary.md#term-extent), view, noncanonical BOOL, non-`FLOAT32` classification/branch [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), [dtype](../../../../glossary.md#data-type--datatype) promotion,
or new [gradient](../../../../glossary.md#term-gradient) [topology](../../../../glossary.md#term-topology) is added.

## Exact custom semantics

- Classification reads each [FLOAT32](../../../../glossary.md#term-float32) [element](../../../../glossary.md#term-element) as an unmodified [`uint32`](../../../../glossary.md#index-mappings-and-cost-dominance) word. `IS_FINITE` is exponent
  not all ones; `IS_NAN` is [exponent](../../../../glossary.md#term-exponent) all ones and fraction nonzero; `IS_INF` is exponent all ones
  and fraction zero. Sign, quiet/signaling bit, payload, zero sign, and subnormal magnitude never
  enter floating arithmetic.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [subnormal glossary](../../../../glossary.md#term-subnormal))</span>
- Logic reads [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) bytes and writes exactly byte `0` or `1`; `AND`, `OR`, and `NOT` are
  integer truth-table operations.
- `WHERE` computes right-aligned broadcast source ordinals and copies the selected branch's raw
  `uint32` word verbatim. It must preserve every NaN payload/sign/quiet bit, signed zero, subnormal,
  infinity, and ordinary finite word.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite))</span>
- Every operand, result, and internal logical value is present in the declared whole-program value
  table and resource liveness. Mixing new custom nodes with already approved Task-0052 custom nodes
  and existing exact MPSGraph steps must remain one honest custom-program recipe; no hidden buffer,
  undeclared intermediate, per-node fallback, or route switch is allowed.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [nodes glossary](../../../../glossary.md#node); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [buffer glossary](../../../../glossary.md#term-buffer))</span>

## [Schema](../../../../glossary.md#term-schema), route, and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Retain [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, exactly thirteen exports, [operation](../../../../glossary.md#operation) wires `1..115`, type wires `1..6`, and
  attribute wires `0..41`. Advance [node](../../../../glossary.md#node) schema `13 -> 14` because the fixed route is now embedded
  at header byte offset 40; old schema-13 images and zero/unknown routes fail closed with no dual
  decoder or compatibility shim.
- Lower exactly wires `40..45` and `51` through the typed schema-14 program.
- Change the custom catalog state for those seven wires from `PENDING` to `AVAILABLE` with a closed
  Task-0057 reason, while retaining each direct MPSGraph row as <span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`DIRECT / MD-PRED`.
- Replace the Task-0052-specific whole-program route name with `CUSTOM_PROGRAM`, migrate every
  caller, and retain stable private route wire `3`.
- Advance persisted workload, exact-policy, candidate, compatibility, route, and [codec](../../../../glossary.md#term-codec) identity
  `14 -> 15`; old identities [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- Candidate generation exposes custom first for the exact domain and MPSGraph second only for an
  exact single-node new operation whose complete direct selector recipe is implemented. Production
  always selects custom. Package-private forcing can select only a freshly regenerated
  authenticated candidate.

## Implementation order

1. Re-audit current Model metadata, Compiler capture, schema lowering, route catalog/identity,
   native decoder, generated custom source, typed ingress/publication, and Engine composition.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [publication glossary](../../../../glossary.md#term-publication))</span>
2. Add exact capability and Java lowering/preflight validation, including arity, dtype, Shape,
   broadcasting, gradient flags, canonical BOOL ingress, and wire-73 rejection.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [validation glossary](../../../../glossary.md#term-validation); [arity glossary](../../../../glossary.md#term-arity); [dtype glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting); [gradient glossary](../../../../glossary.md#term-gradient); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
3. Extend the shared custom raw/integer source and native whole-program builder for all seven nodes,
   exact input counts, typed resources, broadcast metadata, and liveness.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
4. Add direct [MPSGraph](../../../../glossary.md#term-mpsgraph) selector [lowering](../../../../glossary.md#lowering) for the seven [nodes](../../../../glossary.md#node) and keep it structural-only.
5. Update fixed route generation/authentication/codec versions, preparation/finalization/resource
   declaration, publication, trace identity, catalog counts, and exact no-fallback behavior.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [codec glossary](../../../../glossary.md#term-codec); [trace glossary](../../../../glossary.md#trace))</span>
6. Prove the public path through a real locally built [dylib](../../../../glossary.md#term-dylib) and public [`Engine`](../../../../glossary.md#term-engine) composition, then
   update contracts/docs and obtain independent Class C review. Remediate every P0/P1/P2 finding.

## Required verification

- Native build plus exact export audit: [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains 5 with the same thirteen exports.
- Focused [Metal](../../../../glossary.md#term-metal) unit and [schema](../../../../glossary.md#term-schema)/native malformed-control tests.
- Capability and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) [matrix](../../../../glossary.md#term-matrix-meanings) with `41 true / 74 false = 115`, both profiles for the
  seven exact rows, and explicit wire-73 exclusion.
- Classification raw-word corpus covering both zeros, normals, minimum/ordinary subnormals,
  infinities, and signed quiet/signaling NaNs with payloads.<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Full BOOL truth tables, noncanonical caller-ingress rejection, unary/binary Shape checks, operand
  order, right-aligned broadcast including rank 16, and canonical output bytes.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [Shape glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- WHERE raw-bit preservation for signed zeros, subnormals, infinities, and multiple NaN payloads;
  condition/branch broadcast order and mixed branch <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>`requiresGrad` cases.
- Mixed comparison/classification/logic/WHERE programs, existing exact nodes around the new nodes,
  all declared intermediate buffers, fresh outputs, no alias, no fallback, and unsupported-mixture
  fail-closed checks.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [buffers glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Independent forced custom and forced MPSGraph structural candidate tests. MPSGraph results may be
  recorded only as regression evidence and never promoted to domain proof.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [regression glossary](../../../../glossary.md#term-regression))</span>
- Repeated runs, concurrent independent runs, close-vs-run behavior, rollback, malformed resources,
  authenticated route handoff, stale identity rejection, and trace route identity.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [rollback glossary](../../../../glossary.md#term-rollback); [handoff glossary](../../../../glossary.md#term-handoff); [identity glossary](../../../../glossary.md#term-identity-meanings); [trace glossary](../../../../glossary.md#trace))</span>
- Public no-skip [`Engine`](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test) against the freshly built real [dylib](../../../../glossary.md#term-dylib), with no [CPU](../../../../glossary.md#term-cpu) [ownership](../../../../glossary.md#term-ownership-meanings) or skip.
- Focused architecture/[Javadoc](../../../../glossary.md#term-javadoc)/docs checks and `git diff --check`. Do not run a final full repository
  build and do not run timing or benchmarks.

## Dependencies and conflicts

Depends on Complete Tasks 0052, 0055, and 0056; current exact [Model](../../../../glossary.md#term-model) classification/logical/WHERE
semantics; current [Compiler](../../../../glossary.md#term-compiler) forward/[gradient](../../../../glossary.md#term-gradient)-role [capture](../../../../glossary.md#term-capture-meanings); current [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle);
and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0022's deterministic correctness-first route policy. Historical Task 0010 supplies blocker
context only and is not a [dependency](../../../../glossary.md#term-dependency) or authority for the new custom route.

Conflicts with every concurrent [Metal](../../../../glossary.md#term-metal) capability, [schema](../../../../glossary.md#term-schema), native [preflight](../../../../glossary.md#term-preflight), generated-[kernel](../../../../glossary.md#kernel),
candidate/[codec](../../../../glossary.md#term-codec)/route, preparation/finalization, [buffer](../../../../glossary.md#term-buffer), [publication](../../../../glossary.md#term-publication), [trace](../../../../glossary.md#trace), test, package, and
shared documentation edit, and with any resumed Task-0053 production/proof integration. Parallel
group: none. The implementation [owner](../../../../glossary.md#term-owner) integrates all shared files serially.

## Architecture and documentation

Update the [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend) contract, [API](../../../../glossary.md#term-application-programming-interface-api)/backend guides, native README where its executable domain is
stated, master plan, and roadmap. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr) is required because the task follows the existing private
[schema](../../../../glossary.md#term-schema), typed-transfer, [synchronous](../../../../glossary.md#term-synchronous) [lifecycle](../../../../glossary.md#lifecycle), and deterministic fixed-route architecture rather
than changing an architecture decision.

## Completion gate

Complete only after the exact custom route, structural [MPSGraph](../../../../glossary.md#term-mpsgraph) route, capability, [schema](../../../../glossary.md#term-schema)/native
[validation](../../../../glossary.md#term-validation), honest mixed-program resources, [publication](../../../../glossary.md#term-publication), [lifecycle](../../../../glossary.md#lifecycle)/[concurrency](../../../../glossary.md#term-concurrency), no-fallback proof,
real public [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test), documentation, clean [commit](../../../../glossary.md#term-commit), and independent Class C approval all pass.
Any missing exactness or lifecycle obligation leaves the seven rows [fail-closed](../../../../glossary.md#term-fail-closed).

## Completion evidence

- Implementation `e5d9d5c7` adds capability/[lowering](../../../../glossary.md#lowering) for wires `40..45` and `51`, the shared
  `CUSTOM_PROGRAM` recipe, fixed raw/integer kernels, direct audited selectors, schema-14 route
  authentication, identity 15, strict BOOL ingress, complete declared liveness, and exact
  publication. Remediation <span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [schema glossary](../../../../glossary.md#term-schema); [authentication glossary](../../../../glossary.md#term-authentication); [identity glossary](../../../../glossary.md#term-identity-meanings); [BOOL glossary](../../../../glossary.md#data-type--datatype); [liveness glossary](../../../../glossary.md#term-liveness); [publication glossary](../../../../glossary.md#term-publication))</span>`7a3bb072` closes rank-zero Java/native image acceptance and expands
  boundary, route, and conformance proof.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Native build, fixed ad-hoc signing, local packaging, the native package verifier, and Gradle's
  opt-in <span data-glossary-navigation="true"> ([ad-hoc signing glossary](../../../../glossary.md#term-ad-hoc-signing); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`verifyMetalNativePackage` passed. ABI remains 5 with exactly thirteen exports; the
  manifest records node schema 14.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [manifest glossary](../../../../glossary.md#term-manifest); [node glossary](../../../../glossary.md#node))</span>
- The complete `:backends:metal:test` task and Metal Javadoc passed against the fresh packaged
  dylib. Focused native coverage executes both production custom and forced direct routes across
  both profiles, both zero signs, minimum/largest subnormals, finite extrema, both infinities, and
  signed quiet/signaling NaNs with payloads; logic covers every truth-table pair and rank-16
  right-aligned broadcasting; WHERE preserves selected raw bits.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [dylib glossary](../../../../glossary.md#term-dylib); [subnormals glossary](../../../../glossary.md#term-subnormal); [finite glossary](../../../../glossary.md#term-finite); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rank glossary](../../../../glossary.md#term-rank); [broadcasting glossary](../../../../glossary.md#broadcasting); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Focused backend conformance passed for both profiles, rank-zero exclusion, and explicit wire-73
  rejection. The public no-skip Engine smoke passed with sole owner <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Engine glossary](../../../../glossary.md#term-engine); [smoke glossary](../../../../glossary.md#term-smoke-test); [owner glossary](../../../../glossary.md#term-owner))</span>[`metal`](../../../../glossary.md#term-metal), fixed
  `CUSTOM_KERNEL` trace route, custom/MPSGraph mixing, exact canonical BOOL and raw FLOAT32
  publications, pre-invocation rejection of noncanonical BOOL bytes, and a subsequent successful
  run proving session liveness.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../../glossary.md#term-float32); [run glossary](../../../../glossary.md#term-run))</span>
- The supplied independent P1/P2 audit was remediated by the clean schema-14 cutover, explicit
  rank-zero capability tests, public Engine proof, and documentation of the closed owned internal
  BOOL-producer contract. A subsequent cumulative Class C review found three P2 validation and
  documentation defects; remediation <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [validation glossary](../../../../glossary.md#term-validation))</span>`7a3bb072` closed all three, and the same reviewer approved
  the live tree with zero P0/P1/P2 findings. Focused architecture/backend/API/native/planning
  documentation and <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check` passed. No timing, benchmark, or final full repository build
  was run.
