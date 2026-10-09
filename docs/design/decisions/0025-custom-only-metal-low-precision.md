# ADR 0025: Custom-Only Metal Low Precision

## Status

Accepted — 2026-09-29

Supersedes [ADR 0024](0024-environment-bound-metal-raw-certificates.md). The Model-owned
low-precision numerical contract remains in the [foundational-modules contract](../../architecture/contracts/foundational-modules.md#model-low-precision-contract);
[ADR 0023](0023-low-precision-accelerator-parity.md) records historical rationale and provider
evidence. [ADR 0026](0026-profile-free-numerical-semantics.md) supersedes only this record's
profile-dependent trace and identity provisions. The custom-only Metal route remains accepted.

## Context

Metal has a complete custom-program implementation for every admitted BFLOAT16 and FLOAT16 occurrence. Arithmetic uses FLOAT32 working values and accumulators with one final ties-to-even narrowing. Raw-preserving movement and layout operations copy represented sixteen-bit words exactly.

ADR 0024 added environment-bound certificates for six raw-preserving program families. The certificate layer had one benefit: an exact GPU, OS, SDK/framework, compiler, bridge-binary, option, capability-ledger, and schema-19 program-key match could add MPSGraph as a second candidate for that exact image. It did not widen Model capability, improve the numerical contract, supply a missing implementation, or provide retry or fallback.

Maintaining that alternate candidate required a native environment export, dylib hashing, a certificate schema and store, accuracy and determinism records, vendor probes, trace certificate fields, and environment-aware tests. Those layers made the active route contract depend on machine-specific qualification even though the custom route already covered the complete admitted domain.

## Decision

Every Metal partition containing a BFLOAT16 or FLOAT16 value selects `CUSTOM_PROGRAM` and exposes only the `CUSTOM_PROGRAM` candidate. This includes arithmetic and exact raw-preserving RESHAPE, simple PERMUTE, materializing CONTIGUOUS, SLICE, CONCAT, and TILE.

The route restriction applies to the partition candidate, not every internal step. A BFLOAT16- or
FLOAT16-valued operation uses a custom kernel, with no hidden BFLOAT16-to-FLOAT16 substitution.
Any FLOAT32-only operation may use an internal MPSGraph boundary step under the existing FLOAT32
policy without changing the enclosing `CUSTOM_PROGRAM` route. This applies both to an independent
FLOAT32 branch and to one following an explicit low-to-FLOAT32 cast.
Generated pointwise units and anchor-epilogue fusion remain forbidden throughout the partition.
Classic Metal Performance Shaders, Metal Performance Primitives, CPU execution, retry, and fallback
are not low-precision route alternatives.

Low-precision PREPARE trace reports execution facts only: the selected custom route, ordered
feed-then-target logical dtype tuple, and numerical profile. It has no accumulator or working-type
field and carries no certificate status, key, accuracy, determinism, environment, or hypothetical
candidate. The separate Model arithmetic guarantee remains FLOAT32 working values and accumulators
with one final ties-to-even narrowing.

The certificate store, certificate DTOs, environment identity and dylib hashing, qualification probes and evidence, and native environment export are removed without compatibility aliases. Native ABI advances from 6 to 7 and its exact export set decreases from fourteen to thirteen. Program schema 19 and the existing type, operation, attribute, and route wire allocations remain unchanged. The coordinated Metal workload, exact-policy, candidate, compatibility, route-policy, and codec identities advance from 28 to 29.

## Consequences

The route contract is smaller and machine-independent: low precision has one implementation family, one candidate, no alternate-route qualification state, and no retry path. Preparation and trace cannot accidentally expose evidence for an unselected route.

No admitted low-precision capability or numerical guarantee is lost. Custom arithmetic, reductions, scans, matrix operations, convolution/pooling, dropout, ordering, selection, state, casts, and raw-preserving operations remain available under their existing profile and geometry predicates. Exact raw-bit behavior, FLOAT32 working/accumulator semantics, final ties-to-even narrowing, generated backward closure, all-seven-carrier transfer, and fail-closed unsupported cases remain unchanged. Existing FLOAT32 MPSGraph behavior remains unchanged.

The removed capability is only the optional second MPSGraph candidate for the previously certified exact raw images. Historical rationale and measurements remain documented in superseded ADR 0024, but its machine-bound artifacts are not active repository contracts.

## Rejected alternatives

Keeping dormant certificate DTOs, loaders, exports, probes, or compatibility aliases would preserve a second inactive architecture and invite future accidental routing through it. Retaining raw-only MPSGraph while making arithmetic custom-only would keep the same policy split and environment machinery. Runtime retry from custom to a vendor route would violate fixed cold route selection and obscure failures.

## Related decisions

- [ADR 0023](0023-low-precision-accelerator-parity.md)
- [ADR 0024](0024-environment-bound-metal-raw-certificates.md)
- [ADR 0020](0020-synchronous-single-default-device-metal-execution.md)
