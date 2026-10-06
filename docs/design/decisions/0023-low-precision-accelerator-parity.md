# ADR 0023: Low-Precision ACCELERATOR Parity and P0 Evidence

## Status

Accepted — 2026-09-28

Qualifies the non-`FLOAT32` scope of
[ADR 0021](0021-total-recursive-accelerator-numerical-floor.md). ADR 0021 records the rationale
for its `FLOAT32` recursive floors and existing `STRICT_IEEE` result sets; the current normative
rules are in the [architecture root](../../../ARCHITECTURE.md#core-invariants) and its incorporated
[Model contract](../../architecture/contracts/foundational-modules.md#modulesmodel).

[ADR 0025](0025-custom-only-metal-low-precision.md) supersedes this decision's certificate-schema
and Metal opaque-route qualification provisions. This ADR preserves the historical rationale for
the Model-owned low-precision result sets, provider ledger, and append-only identity allocation;
their current obligations are in the architecture root and incorporated
[Model](../../architecture/contracts/foundational-modules.md#model-low-precision-contract-and-active-accelerator-extension)
and [backend execution](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
contracts.

## Context

The public model currently has seven data types. `FLOAT16` is appended at ordinal 6 after `BOOL`
with Model-owned value, promotion, cast, and factory semantics; it has no active CPU generator or
Metal wire, schema, identity, route, or executable capability. `ACCELERATOR` currently widens
backend capability and allowed results only for documented `FLOAT32` occurrences. Existing
`BFLOAT16` occurrences retain their current family-specific behavior, and low-precision backend
coverage remains unsupported where not explicitly reported. Metal currently uses program schema
18, data-type wires 1 through 6, operation wires 1 through 115, attribute wires 0 through 41,
route wires 1 through 3, backend-local identity version 27, and native ABI 5. The CPU generator
schema is 67.

Adding low precision without first freezing this state would make it possible to mistake a route,
device, generated-backward, runtime, or certificate fact for provider capability, silently widen
an existing strict result set, or reuse a serialized identity. Those non-provider facts do not yet
have P0 evidence sources and therefore cannot be invented in this manifest.

## Decision drivers

- Preserve every current `STRICT_IEEE` result and capability snapshot.
- Preserve the current `ACCELERATOR FLOAT32` result and capability snapshots.
- Preserve append-only `FLOAT16` identity allocation across every durable boundary.
- Define one Model-owned low-precision arithmetic contract without making backend capability active
  in P0.
- Permit both deductive implementations and bounded transformed algorithms certified against
  public cancellation/size-aware family envelopes.
- Keep accuracy qualification separate from determinism metadata.
- Derive F32 parity targets and corresponding BF16 audit answers from actual current CPU and Metal
  providers, not a hand-written support table or inference.
- Keep provider capability and target mapping separate from route selection, certification,
  runtime availability, device identity, and generated-backward ownership.
- Reserve append-only backend identity values without activating their later consumers.

## Decision

### P0 freezes evidence; it enables no capability

P0 adds checked-in canonical manifests and validation only. It leaves the separately landed
model-only `FLOAT16` allocation active but does not change a provider answer, select a route,
modify runtime behavior, widen an existing allowed-result set, activate a backend wire/schema/ABI,
or claim generated-backward ownership.

The capability ledger is generated from actual `CpuCapabilityProvider` and
`MetalCapabilityProvider` queries over a stable representative basis. Each row records only the
provider query identity — backend, profile, operation kind and attributes, ordered input/output
descriptors, type, Shape, layout, gradient metadata, arity — and the boolean F32 provider answer.
For an F32-role witness it also queries the corresponding BF16 occurrence by replacing every F32
descriptor and F32-typed scalar or cast target in that witness with BF16 while retaining every
other field. Target and exclusion columns are P0 mapping decisions, not facts returned by the
provider. They do not encode route, runtime, device, certificate, or backward ownership.

The manifest is canonical UTF-8 TSV with LF endings, explicit fields, stable ordering, and unique
keys. A repeated key is valid only if the complete row bytes are identical, and is rejected even
then as duplicate input. Validation reports the first missing, added, or byte-changed row. Both
supported rows and explicit unsupported rows are retained. `BackendCapabilityProvider` remains the
existing deterministic boolean predicate; it gains no enumeration or explanation API.

The frozen basis is representative evidence rather than a claim to enumerate every possible
shape, layout, attribute, gradient, or occurrence. Normal provider tests provide conformance
evidence across the complete current predicate domain; neither those tests nor this historical ADR
are normative authority. Any later low-precision implementation must regenerate the same canonical
basis through the actual providers, review every diff, and preserve every existing true and false
row unless a rule change updates the architecture root and the owning incorporated scoped contract
together. An ADR may record that decision but cannot change the frozen rows on its own.

### Low-precision target mapping

A row can create a low-precision arithmetic target only when its frozen `ACCELERATOR FLOAT32`
provider answer is true. A false row stays an explicit exclusion. Exact movement, shape/layout,
indexing, selection, state, ordering, and cast rows are profile-common exact targets and gain no
arithmetic relaxation. Existing `STRICT_IEEE` rows remain snapshots only and are never a source of
new low-precision accelerator capability.

For the CPU target, `FLOAT16` must first reach parity with the frozen CPU `FLOAT32` row. Existing
CPU `BFLOAT16` behavior is preserved; the only planned CPU `BFLOAT16` residual in this tranche is
dropout. For Metal, both `BFLOAT16` and `FLOAT16` must reach parity with the frozen Metal
`ACCELERATOR FLOAT32` row. “Parity” means the same occurrence domain and semantic result-set
obligations. It does not require the same route, instruction sequence, performance, or bit pattern,
and it does not widen any already-supported `BFLOAT16` occurrence.

A later capability flip must be complete for the mapped occurrence: provider answer, preparation,
execution, publication, failure behavior, and any separately supported generated graph must agree.
A forward row never implies generated-backward ownership.

### Model-owned low-precision arithmetic contract

The low-precision contract applies only to future arithmetic occurrences newly admitted under
`ACCELERATOR` after their capability work lands. It does not retroactively change any current
`BFLOAT16` occurrence and never changes `STRICT_IEEE`.

`FLOAT16` and `BFLOAT16` are distinct, incomparable storage types. A mixed `FLOAT16`/`BFLOAT16`
floating operation promotes to `FLOAT32`; either low type combined with `FLOAT32` promotes to
`FLOAT32`, and any combination containing `FLOAT64` promotes to `FLOAT64`. A homogeneous low-
precision occurrence retains its declared low output type.

Kind, attributes, arity, descriptors, Shape/layout mapping, raw storage and movement, contributor
domain, guards, predicates, indices, masks, state transitions, selection, ordering, casts, public
outputs, and saved values remain exact. Casts retain their declared conversion. Each declared low
arithmetic output performs one final round-to-nearest, ties-to-even narrowing; an implementation
may not pre-narrow an input, contributor, accumulator, public value, or saved value.

Every newly admitted family first requires a complete custom implementation whose low arithmetic
uses `FLOAT32` working and accumulator values. The family contract explicitly declares where low
operands may use DAZ, low results may use FTZ, and an arithmetic exact-zero result may choose
either sign. Arithmetic NaN must retain NaN class and the Model formula's domain behavior, but its
payload and sign are not accuracy requirements. These freedoms never change a predicate, index,
mask, selection, state transition, raw copy, or cast.

Within one operation, arithmetic may reassociate and use corresponding `FLOAT32` FMA. Fusion may
cross only an intermediate that is semantically unobservable and has exactly one consumer. It may
never cross a public output, saved value, fan-out, predicate, index, mask, selection, or state
boundary. This is an observability rule rather than a blanket ban on cross-node contraction or
algebraic transformation.

Model owns a public accuracy envelope for each admitted low-precision family. The envelope fixes
the exact domain and special-class rules, reference result, accumulator and final narrowing, and a
forward-error bound whose scale depends explicitly on family size and cancellation sensitivity.
It is not a generic absolute/relative `allclose` tolerance. Every route, including an opaque or
transformed vendor route, must qualify through either deductive complete-output-set proof against
the envelope or versioned certification against that envelope. A literal primitive tree can
usually take the proof path; an algorithm that has no such tree can take the certification path.
An operation name, undocumented vendor claim, passing examples, or `allclose` sample alone is
insufficient.

### Inactive route-certificate schema

P0 defines no active route, generated-backward, runtime, or device fact, and the provider ledger
contains none. It does freeze certificate schema 1 for later opaque-route qualification. The
canonical certificate key contains the profile, operation/fusion family, ordered dtype tuple,
accumulator dtype, exact Shape/layout domain, route identity, GPU family, OS build, SDK/framework
version, compiler version, native binary digest, shader/program digest, compile/math flags and
options, and exact capability-manifest hash.

The keyed accuracy record names the Model-owned envelope, selects `DEDUCTIVE_PROOF` or
`CERTIFIED_ENVELOPE`, retains the immutable evidence digest, and records the verdict. Determinism
metadata is a separate record keyed back to the certificate; it is never an accuracy field or
verdict input. The checked-in schema contains field definitions only and activates no route.
Unknown, stale, or mismatched evidence cannot qualify an opaque route. A later P5/P9 route manifest
must be derived separately from its owning evidence.

### Append-only identity allocation

The following table records the active Model allocation and the still-inactive backend cutovers:

| Identity | Current | Reserved cutover |
|---|---:|---:|
| `DataType.FLOAT16` | ordinal 6, active after `BOOL` | activated in Model only |
| Metal `FLOAT16` type wire | absent | 7 |
| Metal program schema | 18 | 19 |
| Metal route wires | 1..3 | unchanged |
| Metal candidate/compatibility/route-policy/workload/codec identities | 27 | 28 |
| CPU generator schema | 67 | 68 |
| route-certificate schema | field schema 1, no certificate | 1 |
| Metal native ABI | 5 | 5, unchanged |

Operation wires 1 through 115 and attribute wires 0 through 41 remain frozen. No current identity
may be renumbered or reused. The Model enum allocation is activated; it does not consume any
backend reservation. The integrated backend implementation cutover must activate the remaining
wires, schemas, codecs, generated artifacts, caches, tests, and documentation together and reject
old or unknown artifacts at their existing fail-closed boundaries. Native ABI 5 remains valid only
because this reservation adds no export, signature, count, or C-layout change; any such change
requires a separate ABI decision.

## Consequences

- Current runtime, native behavior, provider answers, capabilities, routes, `STRICT_IEEE` snapshots,
  and `ACCELERATOR FLOAT32` snapshots remain byte-for-byte or behaviorally unchanged.
- The checked-in provider ledger exposes drift in either provider and separately exposes target-map
  edits. The identity-allocation and certificate-schema manifests freeze cross-phase contracts
  without pretending any route certificate or route selection already exists.
- Low-precision capability work can be partitioned by backend and operation family, but every
  admitted row has an explicit current `FLOAT32` reference and exclusion reason.
- A newly admitted custom baseline may use operation-local arithmetic freedom and bounded
  observable-single-use fusion. An opaque transformed route may qualify by deductive proof or
  versioned certification against the public cancellation/size-aware family envelope.
- Determinism evidence remains independent from accuracy evidence.
- ADR 0021's statement that non-`FLOAT32` behavior remains strict describes the current system and
  every pre-existing occurrence. This decision owns the only future exception: newly admitted
  low-precision arithmetic under the contract above.

## Rejected alternatives

### Infer low-precision support from an operation kind

Capability is occurrence-specific. Kind-only inference loses attributes, ordered descriptors,
layouts, gradients, and arity. Rejected.

### Put route or certificate facts in the provider manifest

A provider answers backend semantic ownership. Route, runtime, device, certificate, and generated-
backward facts have different owners and lifecycles. Combining them would create a second planning
or preparation authority. Rejected.

### Add a capability-enumeration API

The stable representative basis can query the actual predicate directly. Redesigning
`BackendCapabilityProvider` would add public weight without improving the frozen evidence.
Rejected.

### Accept vendor primitives by operation name or generic tolerance

A name does not bind environment, options, program bytes, or complete occurrence domain. A
passing example or generic final-output `allclose` tolerance does not establish a Model-owned
family envelope. Eligibility requires deductive proof or a versioned certificate against that
envelope. Rejected.

## Related documents

- [Foundational-module contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Backend execution contract](../../architecture/contracts/backend-execution.md)
- [Capability-provider guide](../../backend-guide/capability-provider.md)
- [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)
- [ADR 0022](0022-auditable-custom-metal-route-cost-evidence.md)
