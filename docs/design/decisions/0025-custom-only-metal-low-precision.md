# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0025: Custom-Only [Metal](../../glossary.md#term-metal) Low [Precision](../../glossary.md#term-precision)

## Status

Accepted — 2026-09-29

Supersedes [ADR 0024](0024-environment-bound-metal-raw-certificates.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>. The [Model](../../glossary.md#term-model)-owned
low-[precision](../../glossary.md#term-precision) numerical contract remains in the [foundational-modules contract](../../architecture/contracts/foundational-modules.md#model-low-precision-contract)<span data-glossary-navigation="true"> ([modules glossary](../../glossary.md#term-module))</span>;
[ADR 0023](0023-low-precision-accelerator-parity.md) records historical rationale and provider
evidence. [ADR 0026](0026-profile-free-numerical-semantics.md) supersedes only this record's
profile-dependent [trace](../../glossary.md#trace) and [identity](../../glossary.md#term-identity-meanings) provisions. The custom-only [Metal](../../glossary.md#term-metal) route remains accepted.

## Context

[Metal](../../glossary.md#term-metal) has a complete custom-program implementation for every admitted [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16) [occurrence](../../glossary.md#term-occurrence). Arithmetic uses [FLOAT32](../../glossary.md#term-float32) working values and accumulators with one final [ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing). Raw-preserving movement and [layout](../../glossary.md#layout) operations copy represented sixteen-bit words exactly.

[ADR](../../glossary.md#term-architecture-decision-record-adr) 0024 added environment-bound certificates for six raw-preserving program families. The [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) layer had one benefit: an exact [GPU](../../glossary.md#term-gpu), OS, [SDK](../../glossary.md#term-sdk)/framework, [compiler](../../glossary.md#term-compiler), bridge-[binary](../../glossary.md#term-binary-descriptor-and-metadata), option, capability-ledger, and [schema](../../glossary.md#term-schema)-19 program-key match could add [MPSGraph](../../glossary.md#term-mpsgraph) as a second candidate for that exact image. It did not widen [Model](../../glossary.md#term-model) capability, improve the numerical contract, supply a missing implementation, or provide retry or fallback.

Maintaining that alternate candidate required a native environment export, [dylib](../../glossary.md#term-dylib) hashing, a certificate schema and store, accuracy and determinism records, vendor probes, [trace](../../glossary.md#trace) certificate fields, and environment-aware tests. Those layers made the active route contract depend on machine-specific [qualification](../../glossary.md#term-qualification) even though the custom route already covered the complete admitted domain.

## Decision

Every [Metal](../../glossary.md#term-metal) [partition](../../glossary.md#partition) containing a [BFLOAT16](../../glossary.md#term-bfloat16) or [FLOAT16](../../glossary.md#term-float16) value selects `CUSTOM_PROGRAM` and exposes only the `CUSTOM_PROGRAM` candidate. This includes arithmetic and exact raw-preserving RESHAPE, simple PERMUTE, materializing [CONTIGUOUS](../../glossary.md#term-contiguous), [SLICE](../../glossary.md#slice), CONCAT, and TILE.

The route restriction applies to the partition candidate, not every internal step. A BFLOAT16- or
FLOAT16-valued [operation](../../glossary.md#operation) uses a custom [kernel](../../glossary.md#kernel), with no hidden BFLOAT16-to-FLOAT16 substitution.
Any [FLOAT32](../../glossary.md#term-float32)-only operation may use an internal [MPSGraph](../../glossary.md#term-mpsgraph) boundary step under the existing FLOAT32
policy without changing the enclosing `CUSTOM_PROGRAM` route. This applies both to an independent
FLOAT32 branch and to one following an explicit low-to-FLOAT32 cast.
Generated [pointwise](../../glossary.md#term-pointwise) units and anchor-[epilogue](../../glossary.md#term-epilogue) [fusion](../../glossary.md#term-fusion) remain forbidden throughout the partition.
Classic [Metal Performance Shaders](../../glossary.md#mps-mpsgraph-and-openblas), [Metal Performance Primitives](../../glossary.md#mps-mpsgraph-and-openblas), [CPU](../../glossary.md#term-cpu) execution, retry, and fallback
are not low-[precision](../../glossary.md#term-precision) route alternatives.

Low-precision [PREPARE](../../glossary.md#prepare) [trace](../../glossary.md#trace) reports execution facts only: the selected custom route, ordered
feed-then-[target](../../glossary.md#term-target) logical [dtype](../../glossary.md#data-type--datatype) tuple, and numerical profile. It has no [accumulator](../../glossary.md#accumulator-type) or working-type
field and carries no [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) status, key, accuracy, determinism, environment, or hypothetical
candidate. The separate [Model](../../glossary.md#term-model) arithmetic guarantee remains FLOAT32 working values and accumulators
with one final [ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing).

The certificate store, certificate DTOs, environment [identity](../../glossary.md#term-identity-meanings) and [dylib](../../glossary.md#term-dylib) hashing, [qualification](../../glossary.md#term-qualification) probes and evidence, and native environment export are removed without compatibility [aliases](../../glossary.md#aliasing-and-arena). Native [ABI](../../glossary.md#term-application-binary-interface-abi) advances from 6 to 7 and its exact export set decreases from fourteen to thirteen. Program [schema](../../glossary.md#term-schema) 19 and the existing type, operation, attribute, and route wire allocations remain unchanged. The coordinated Metal workload, exact-policy, candidate, compatibility, route-policy, and [codec](../../glossary.md#term-codec) identities advance from 28 to 29.

## Consequences

The route contract is smaller and machine-independent: low [precision](../../glossary.md#term-precision) has one implementation family, one candidate, no alternate-route [qualification](../../glossary.md#term-qualification) state, and no retry path. Preparation and [trace](../../glossary.md#trace) cannot accidentally expose evidence for an unselected route.

No admitted low-precision capability or numerical guarantee is lost. Custom arithmetic, reductions, scans, [matrix](../../glossary.md#term-matrix-meanings) operations, [convolution](../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../glossary.md#term-pooling), [dropout](../../glossary.md#dropout--inverted-dropout), ordering, selection, state, casts, and raw-preserving operations remain available under their existing profile and geometry predicates. Exact raw-bit behavior, [FLOAT32](../../glossary.md#term-float32) working/[accumulator](../../glossary.md#accumulator-type) semantics, final [ties-to-even](../../glossary.md#term-ties-to-even) [narrowing](../../glossary.md#term-narrowing), generated backward closure, all-seven-[carrier](../../glossary.md#term-storage-carrier) transfer, and [fail-closed](../../glossary.md#term-fail-closed) unsupported cases remain unchanged. Existing FLOAT32 [MPSGraph](../../glossary.md#term-mpsgraph) behavior remains unchanged.

The removed capability is only the optional second MPSGraph candidate for the previously certified exact raw images. Historical rationale and measurements remain documented in superseded [ADR](../../glossary.md#term-architecture-decision-record-adr) 0024, but its machine-bound [artifacts](../../glossary.md#term-artifact) are not active repository contracts.

## Rejected alternatives

Keeping dormant [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) DTOs, loaders, exports, probes, or compatibility [aliases](../../glossary.md#aliasing-and-arena) would preserve a second inactive architecture and invite future accidental routing through it. Retaining raw-only [MPSGraph](../../glossary.md#term-mpsgraph) while making arithmetic custom-only would keep the same policy split and environment machinery. [Runtime](../../glossary.md#term-runtime) retry from custom to a vendor route would violate fixed cold route selection and obscure failures.

## Related decisions

- [ADR 0023](0023-low-precision-accelerator-parity.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0024](0024-environment-bound-metal-raw-certificates.md)
- [ADR 0020](0020-synchronous-single-default-device-metal-execution.md)
