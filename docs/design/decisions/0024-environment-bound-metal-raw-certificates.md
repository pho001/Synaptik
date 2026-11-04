# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0024: Environment-Bound [Metal](../../glossary.md#term-metal) Raw-Route Certificates


## Editorial reading context

A certificate in this withdrawn route mechanism bound evidence to an exact program and environment. The Java loader and native context supplied that identity; a friendly operation name was insufficient. This is history, not a setup procedure: the successor removed this certificate mechanism and alternate low-precision vendor route. Do not rebuild old probes to imply current qualification.

Use the [certificate removal and custom-only route](0025-custom-only-metal-low-precision.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Superseded — 2026-09-29 by
[ADR 0025](0025-custom-only-metal-low-precision.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>.

This document retains the historical decision and evidence narrative. Its [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) store,
environment export, probes, and alternate low [MPSGraph](../../glossary.md#term-mpsgraph) route are no longer active.

## Context

[Metal](../../glossary.md#term-metal) has a complete custom-program implementation for every admitted [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16)
[occurrence](../../glossary.md#term-occurrence). The custom route is therefore the safe baseline. [MPSGraph](../../glossary.md#term-mpsgraph) can also preserve raw
sixteen-bit words for some non-arithmetic [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) programs, but a framework route name does
not bind the [device](../../glossary.md#term-device), operating system, [toolchain](../../glossary.md#term-toolchain), loaded framework and bridge bytes, [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)
options, or exact [graph](../../glossary.md#graph) image. Generic [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) would not establish that [identity](../../glossary.md#term-identity-meanings) or prevent a
stale result from silently enabling a different program.

[Certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) [schema](../../glossary.md#term-schema) 1 already freezes the required key fields. It previously had no active loader,
certificate corpus, or route effect. The [native bridge](../../glossary.md#term-native-bridge) was [ABI](../../glossary.md#term-application-binary-interface-abi) 5 and had no way to report the exact
context environment that cold preparation would use.

## Decision

### [Canonical](../../glossary.md#term-canonical-form) no-[accumulator](../../glossary.md#accumulator-type) [identity](../../glossary.md#term-identity-meanings)

A non-arithmetic [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) key uses the literal `NONE` for `accumulator-dtype`. Numeric
accumulator names remain valid for arithmetic families. Empty strings, [BOOL](../../glossary.md#data-type--datatype), and other sentinels
are invalid.

### [Immutable](../../glossary.md#term-immutable) context environment

[Metal](../../glossary.md#term-metal) native [ABI](../../glossary.md#term-application-binary-interface-abi) 6 adds
`synaptik_metal_context_certification_environment`. One live context returns a [canonical](../../glossary.md#term-canonical-form) six-line
record containing:

- the highest supported Apple [GPU](../../glossary.md#term-gpu) family;
- the exact Darwin OS build;
- the macOS [SDK](../../glossary.md#term-sdk) and loaded [MPSGraph](../../glossary.md#term-mpsgraph) framework versions;
- [host](../../glossary.md#term-host) [compiler](../../glossary.md#term-compiler), Xcode build, and Metal compiler versions; and
- explicit graph options, optimization level, synchronous compilation choice, and
  reduced-precision-fast-math setting.<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph); [synchronous glossary](../../glossary.md#term-synchronous); [compilation glossary](../../glossary.md#compiler-engine-and-execution-stage-names); [precision glossary](../../glossary.md#term-precision))</span>

Java [hashes](../../glossary.md#term-hash) the exact caller-selected [dylib](../../glossary.md#term-dylib) immediately before loading and after context creation.
A changed file rejects construction. The immutable Java [identity](../../glossary.md#term-identity-meanings) adds that [digest](../../glossary.md#term-digest) and the canonical
capability-ledger [hash](../../glossary.md#term-hash). Preparation reads only this retained identity; it performs no later [device](../../glossary.md#term-device),
filesystem, framework, or [toolchain](../../glossary.md#term-toolchain) discovery.

### [Atomic](../../glossary.md#resource-transaction-and-rollback) [schema](../../glossary.md#term-schema)-one store

The bundled [UTF-8](../../glossary.md#term-utf-8) TSV store is parsed once when the context opens. Missing data, forbidden bytes,
unknown schema/family/route/enums, malformed or [heterogeneous](../../glossary.md#term-heterogeneous) [dtype](../../glossary.md#data-type--datatype) tuples, a non-`NONE`
[accumulator](../../glossary.md#accumulator-type), [detached](../../glossary.md#detached-model-results) domain/program or determinism [hashes](../../glossary.md#term-hash), duplicate keys, and oversize input
reject the complete store. A rejected store is empty for eligibility.

[Lookup](../../glossary.md#term-lookup) is exact complete-key equality. There is no prefix, nearest-environment, compatible-[GPU](../../glossary.md#term-gpu),
program-family, stale-[binary](../../glossary.md#term-binary-descriptor-and-metadata), or [shape](../../glossary.md#shape) fallback. Accuracy `PASS` is required for eligibility.
Determinism remains reported independently and cannot turn an accuracy failure into a qualified
route.

### Narrow raw-preserving [MPSGraph](../../glossary.md#term-mpsgraph) candidate

`CUSTOM_PROGRAM` remains the first candidate for every low [partition](../../glossary.md#partition). A second `MPSGRAPH`
candidate exists only for a [homogeneous](../../glossary.md#term-homogeneous) no-[gradient](../../glossary.md#term-gradient) [BFLOAT16](../../glossary.md#term-bfloat16) or [FLOAT16](../../glossary.md#term-float16) partition composed solely
of:

- `RESHAPE`;
- simple `PERMUTE`;
- materializing [`CONTIGUOUS`](../../glossary.md#term-contiguous);
- [`SLICE`](../../glossary.md#slice);
- `CONCAT`; or
- `TILE`.

Boundary physical [layouts](../../glossary.md#layout) must be non-empty, zero-offset, [contiguous storage](../../glossary.md#contiguous-request). The [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) key
contains the exact [schema](../../glossary.md#term-schema)-19 MPSGraph program-image [SHA-256](../../glossary.md#term-sha-256), so a certificate covers one complete
static [Shape](../../glossary.md#shape)/layout/value/[node](../../glossary.md#node)/feed/[target](../../glossary.md#term-target)/profile image rather than an [operation](../../glossary.md#operation) name. External
offsets, holes, zero extents, mixed carriers, gradients, arithmetic, a different environment, and
an unlisted program remain custom-only.

The selected route is fixed during cold analysis before shared resource declarations escape.
Finalization and execution cannot retry, fall back, or replace it. A [graph](../../glossary.md#graph) plan declares only feeds,
targets, and its address workspace; custom internal materializations do not leak into the graph
plan. [Trace](../../glossary.md#trace) [PREPARE](../../glossary.md#prepare) [metadata](../../glossary.md#term-metadata) reports the final route and, for this certificate family, the schema,
exact key [digest](../../glossary.md#term-digest), accuracy status, and separate determinism record.

### Evidence and [activation](../../glossary.md#term-activation)

The checked-in `backends/metal/evidence/0072` accuracy and determinism reports were generated by the
production [ABI](../../glossary.md#term-application-binary-interface-abi)-6 bridge on the environment named in those reports. For both profiles and both low
carriers, the harness executes every one of the 65,536 raw words through each of the six listed
programs, poisons supplied [targets](../../glossary.md#term-target), repeats each executable three times, and requires byte-exact
expected output and repeat [identity](../../glossary.md#term-identity-meanings). It separately exercises 257-word odd counts. Offset/hole and
zero-count boundaries remain ineligible rather than being generalized from the dense evidence.

Only after the base RESHAPE/PERMUTE/[CONTIGUOUS](../../glossary.md#term-contiguous) [matrix](../../glossary.md#term-matrix-meanings) passed did the same run admit
[SLICE](../../glossary.md#slice)/CONCAT/TILE rows. The active store contains only exhaustive program rows and binds the exact
evidence-file digests. The identity ledger activates [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) [schema](../../glossary.md#term-schema) 1 and ABI 6 together.

### Positive and qualified-negative vendor-route closure

The active store contains exactly 24 positive raw-preserving [MPSGraph](../../glossary.md#term-mpsgraph) certificates: six exact
[schema](../../glossary.md#term-schema)-19 programs, two low carriers, and two profiles. Neither classic [Metal Performance Shaders](../../glossary.md#mps-mpsgraph-and-openblas)
([MPS](../../glossary.md#term-mps)) nor [Metal Performance Primitives](../../glossary.md#mps-mpsgraph-and-openblas) ([MPP](../../glossary.md#term-mpp)) is positively certified.

`native/metal-macos-arm64/qualify-low-precision-vendor-routes.sh` rebuilds and executes the
checked-in probes with the installed Xcode 27 [SDK](../../glossary.md#term-sdk)/[toolchain](../../glossary.md#term-toolchain) and the exact packaged [ABI](../../glossary.md#term-application-binary-interface-abi)-6 [dylib](../../glossary.md#term-dylib). The
[canonical](../../glossary.md#term-canonical-form) `backends/metal/evidence/0072/vendor-route-qualification.tsv` binds that probe source,
dylib, [capability ledger](../../glossary.md#capability-ledger), [GPU](../../glossary.md#term-gpu), OS, SDK/framework, [compiler](../../glossary.md#term-compiler), and option [identity](../../glossary.md#term-identity-meanings). Every row has
[qualification](../../glossary.md#term-qualification) `REJECTED` and one stable reason.

The [classic-MPS](../../glossary.md#mps-mpsgraph-and-openblas) probe confirms `MPSSupportsMTLDevice`, executes one [FLOAT16](../../glossary.md#term-float16)
`MPSCNNConvolution` over an actual `MPSImage`, and reads back
`MPSNNConvolutionAccumulatorPrecisionOptionFloat`. Those gates do not prove the complete exact
[Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) [occurrence](../../glossary.md#term-occurrence) domain or the [ADR](../../glossary.md#term-architecture-decision-record-adr)-0023 family envelope. The installed public data-source
contract excludes [BFLOAT16](../../glossary.md#term-bfloat16); the isolated [runtime](../../glossary.md#term-runtime) construction probe rejects it, and no public
BFLOAT16 [accumulator](../../glossary.md#accumulator-type) guarantee exists.

The MPP probe compiles `matmul2d` and `convolution2d` for FLOAT16 and BFLOAT16 inputs with [FLOAT32](../../glossary.md#term-float32)
destinations. The source requires `__HAVE_TENSOR__`, statically fixes `relaxed_precision=false`,
compiles with [Metal](../../glossary.md#term-metal) language 4.0 and deployment [target](../../glossary.md#term-target) 26.2, links a metallib, and creates all four
pipelines on the named Apple GPU. Those successful availability gates still expose no public
internal-[accumulator type](../../glossary.md#accumulator-type). A low-input/FLOAT32-destination [signature](../../glossary.md#term-signature) specifies the destination
[element](../../glossary.md#term-element) type, not FLOAT32 internal accumulation, so it cannot satisfy ADR 0023.

Qualified-negative evidence creates no candidate, route family, route wire, or [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions).
Classic MPS and MPP therefore remain custom-only, and their probe failures or missing guarantees
cannot trigger runtime fallback.

## Consequences

- The capability provider and planning ownership do not change; this is a backend-private
  prepare-time route decision.<span data-glossary-navigation="true"> ([planning glossary](../../glossary.md#term-planning); [ownership glossary](../../glossary.md#term-ownership-meanings); [backend glossary](../../glossary.md#backend); [prepare glossary](../../glossary.md#prepare))</span>
- Custom execution remains complete and independently selectable for every qualified raw [graph](../../glossary.md#graph).
- Deleting, corrupting, or copying the store to another environment reduces candidates to custom;
  it does not make preparation or execution less safe.
- Rebuilding or resigning the dylib, changing OS/framework/toolchain/options, changing schema-19
  bytes, or changing the capability ledger requires new evidence and new exact certificate rows.<span data-glossary-navigation="true"> ([dylib glossary](../../glossary.md#term-dylib); [toolchain glossary](../../glossary.md#term-toolchain); [schema glossary](../../glossary.md#term-schema); [capability ledger glossary](../../glossary.md#capability-ledger))</span>
- The native ABI has fourteen exports. Existing export signatures, program/operation/attribute/type
  wires, and route wires remain unchanged.<span data-glossary-navigation="true"> ([ABI glossary](../../glossary.md#term-application-binary-interface-abi); [signatures glossary](../../glossary.md#term-signature); [operation glossary](../../glossary.md#operation))</span>
- [Runtime](../../glossary.md#term-runtime) gains no [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) [lookup](../../glossary.md#term-lookup), [device](../../glossary.md#term-device) query, timing, retry, or fallback work.

## Rejected alternatives

### Qualify by [operation](../../glossary.md#operation) kind

Kind-only [qualification](../../glossary.md#term-qualification) loses [Shape](../../glossary.md#shape), [layout](../../glossary.md#layout), value table, feed/[target](../../glossary.md#term-target) ordering, profile, and exact
program bytes. Rejected.

### Match compatible or newer environments

The evidence does not prove monotonicity across [GPU](../../glossary.md#term-gpu) families, OS/framework builds, toolchains,
bridge bytes, or [graph](../../glossary.md#graph) options. Rejected.

### Remove the custom candidate after certification

The [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) adds an eligible [opaque](../../glossary.md#term-opaque) route; it does not invalidate the complete safe baseline.
Rejected.

### Retry custom after a [graph](../../glossary.md#graph) execution failure

That would change a [prepared execution](../../glossary.md#prepared-execution--preparedexecution)'s implementation and failure contract on the [hot path](../../glossary.md#term-hot-path).
Rejected.

### Use determinism as an accuracy gate

Repeat [identity](../../glossary.md#term-identity-meanings) and numerical/raw correctness answer different questions. Determinism remains
separate [metadata](../../glossary.md#term-metadata). Rejected.

## Related documents

- [Architecture root](../../../ARCHITECTURE.md)
- [Backend execution contract](../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [Metal backend guide](../../backend-guide/metal-backend.md)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal))</span>
- [ADR 0023](0023-low-precision-accelerator-parity.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
