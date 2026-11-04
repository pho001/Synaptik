# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0023: Low-[Precision](../../glossary.md#term-precision) [ACCELERATOR](../../glossary.md#term-accelerator) Parity and P0 Evidence


## Editorial reading context

A provider query asks whether a fully described operation occurrence is supported; a ledger records answers over a representative basis. A true FLOAT32 answer is not automatically an answer for FLOAT16 or BFLOAT16, nor proof of backward support. The manifest and identity reservations below belong to the stated historical cutover; they are not new allocation instructions.

Use the [capability questions and qualification scope](../../backend-guide/capability-provider.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Superseded — 2026-10-08 by [ADR 0026](0026-profile-free-numerical-semantics.md) for the
profile-indexed low-[precision](../../glossary.md#term-precision) numerical contract, envelope/proof choice, and current-state
claims. Originally accepted 2026-09-28. Its provider-ledger and append-only [identity](../../glossary.md#term-identity-meanings) rationale
remain historical evidence; the current profile-free ledger and identities are specified by the
architecture root and incorporated [backend](../../glossary.md#backend) contract.

Historically qualified the non-[`FLOAT32`](../../glossary.md#term-float32) [scope](../../glossary.md#term-scope) of
[ADR 0021](0021-total-recursive-accelerator-numerical-floor.md). [ADR](../../glossary.md#term-architecture-decision-record-adr) 0021 records the rationale
for its `FLOAT32` recursive floors and existing `STRICT_IEEE` result sets; current [normative](../../glossary.md#term-normative)
rules are in the [architecture root](../../../ARCHITECTURE.md#core-invariants) and its incorporated
[profile-free Model contract](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).

[ADR 0025](0025-custom-only-metal-low-precision.md) superseded this decision's [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions)-[schema](../../glossary.md#term-schema)
and [Metal](../../glossary.md#term-metal) [opaque](../../glossary.md#term-opaque)-route [qualification](../../glossary.md#term-qualification) provisions. This ADR preserves the historical rationale for
the former [Model](../../glossary.md#term-model)-owned low-precision result sets, provider ledger, and append-only identity allocation;
their current obligations are in the architecture root and incorporated
[Model](../../architecture/contracts/foundational-modules.md#model-low-precision-contract)
and [backend execution](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
contracts.

All "current", future, and inactive-allocation statements below are a [snapshot](../../glossary.md#term-snapshot) of the P0
pre-cutover decision, not present capability or an implementation instruction.

## Context

At the P0 decision, the public [model](../../glossary.md#term-model) had seven data types. [`FLOAT16`](../../glossary.md#term-float16) was appended at ordinal 6 after [`BOOL`](../../glossary.md#data-type--datatype)
with Model-owned value, promotion, cast, and factory semantics; it has no active [CPU](../../glossary.md#term-cpu) generator or
[Metal](../../glossary.md#term-metal) wire, [schema](../../glossary.md#term-schema), [identity](../../glossary.md#term-identity-meanings), route, or executable capability. [`ACCELERATOR`](../../glossary.md#term-accelerator) then widened
[backend capability](../../glossary.md#backend-capability) and allowed results only for documented [`FLOAT32`](../../glossary.md#term-float32) occurrences. Existing
[`BFLOAT16`](../../glossary.md#term-bfloat16) occurrences retain their current family-specific behavior, and low-[precision](../../glossary.md#term-precision) [backend](../../glossary.md#backend)
coverage remained unsupported where not explicitly reported. Metal then used program schema
18, data-type wires 1 through 6, [operation](../../glossary.md#operation) wires 1 through 115, attribute wires 0 through 41,
route wires 1 through 3, backend-local identity version 27, and native [ABI](../../glossary.md#term-application-binary-interface-abi) 5. The CPU generator
schema is 67.

Adding low precision without first freezing this state would make it possible to mistake a route,
[device](../../glossary.md#term-device), generated-backward, [runtime](../../glossary.md#term-runtime), or [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) fact for provider capability, silently widen
an existing strict result set, or reuse a serialized identity. Those non-provider facts do not yet
have P0 evidence sources and therefore cannot be invented in this [manifest](../../glossary.md#term-manifest).

## Decision drivers

- Preserve every then-current `STRICT_IEEE` result and capability [snapshot](../../glossary.md#term-snapshot).
- Preserve the then-current `ACCELERATOR FLOAT32` result and capability snapshots.
- Preserve append-only [`FLOAT16`](../../glossary.md#term-float16) [identity](../../glossary.md#term-identity-meanings) allocation across every durable boundary.
- Define one Model-owned low-precision arithmetic contract without making backend capability active
  in P0.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [precision glossary](../../glossary.md#term-precision); [backend capability glossary](../../glossary.md#backend-capability))</span>
- Permit both deductive implementations and bounded transformed algorithms certified against
  public cancellation/size-aware family envelopes.<span data-glossary-navigation="true"> ([cancellation glossary](../../glossary.md#term-cancellation))</span>
- Keep accuracy [qualification](../../glossary.md#term-qualification) separate from determinism [metadata](../../glossary.md#term-metadata).
- Derive F32 parity targets and corresponding BF16 audit answers from actual current CPU and Metal
  providers, not a hand-written support table or inference.<span data-glossary-navigation="true"> ([F32 glossary](../../glossary.md#floating-point-precision-and-rounding); [targets glossary](../../glossary.md#term-target); [BF16 glossary](../../glossary.md#floating-point-precision-and-rounding); [CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [inference glossary](../../glossary.md#term-inference))</span>
- Keep provider capability and target mapping separate from route selection, certification,
  runtime availability, device identity, and generated-backward ownership.<span data-glossary-navigation="true"> ([runtime glossary](../../glossary.md#term-runtime); [device glossary](../../glossary.md#term-device); [ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- Reserve append-only [backend identity](../../glossary.md#backend-identity--backendid) values without activating their later consumers.

## Decision

The following was the P0 pre-cutover decision, retained as historical evidence rather than
present numerical authority or an active [identity](../../glossary.md#term-identity-meanings)-allocation instruction.

### P0 freezes evidence; it enables no capability

P0 adds checked-in [canonical](../../glossary.md#term-canonical-form) manifests and [validation](../../glossary.md#term-validation) only. It leaves the separately landed
[model](../../glossary.md#term-model)-only [`FLOAT16`](../../glossary.md#term-float16) allocation active but does not change a provider answer, select a route,
modify [runtime](../../glossary.md#term-runtime) behavior, widen an existing allowed-result set, activate a [backend](../../glossary.md#backend) wire/[schema](../../glossary.md#term-schema)/[ABI](../../glossary.md#term-application-binary-interface-abi),
or claim generated-backward [ownership](../../glossary.md#term-ownership-meanings).

The [capability ledger](../../glossary.md#capability-ledger) is generated from actual `CpuCapabilityProvider` and
`MetalCapabilityProvider` queries over a stable representative basis. Each row records only the
provider query [identity](../../glossary.md#term-identity-meanings) — backend, profile, [operation](../../glossary.md#operation) kind and attributes, ordered input/output
[descriptors](../../glossary.md#term-descriptor), type, [Shape](../../glossary.md#shape), [layout](../../glossary.md#layout), [gradient](../../glossary.md#term-gradient) [metadata](../../glossary.md#term-metadata), [arity](../../glossary.md#term-arity) — and the [boolean](../../glossary.md#data-type--datatype) [F32](../../glossary.md#floating-point-precision-and-rounding) provider answer.
For an F32-role witness it also queries the corresponding [BF16](../../glossary.md#floating-point-precision-and-rounding) [occurrence](../../glossary.md#term-occurrence) by replacing every F32
descriptor and F32-typed [scalar](../../glossary.md#term-scalar) or cast [target](../../glossary.md#term-target) in that witness with BF16 while retaining every
other field. Target and exclusion columns are P0 mapping decisions, not facts returned by the
provider. They do not encode route, runtime, [device](../../glossary.md#term-device), [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), or backward ownership.

The [manifest](../../glossary.md#term-manifest) is canonical [UTF-8](../../glossary.md#term-utf-8) TSV with LF endings, explicit fields, stable ordering, and unique
keys. A repeated key is valid only if the complete row bytes are identical, and is rejected even
then as duplicate input. Validation reports the first missing, added, or byte-changed row. Both
supported rows and explicit unsupported rows are retained. [`BackendCapabilityProvider`](../../glossary.md#backend-capability-provider--backendcapabilityprovider) remains the
existing deterministic boolean predicate; it gains no enumeration or explanation [API](../../glossary.md#term-application-programming-interface-api).

The frozen basis is representative evidence rather than a claim to enumerate every possible
shape, layout, attribute, gradient, or occurrence. Normal provider tests provide [conformance](../../glossary.md#term-conformance)
evidence across the complete current predicate domain; neither those tests nor this historical [ADR](../../glossary.md#term-architecture-decision-record-adr)
are [normative](../../glossary.md#term-normative) authority. Any later low-[precision](../../glossary.md#term-precision) implementation must regenerate the same canonical
basis through the actual providers, review every diff, and preserve every existing true and false
row unless a rule change updates the architecture root and the owning incorporated scoped contract
together. An ADR may record that decision but cannot change the frozen rows on its own.

### Low-[precision](../../glossary.md#term-precision) [target](../../glossary.md#term-target) mapping

A row can create a low-precision arithmetic target only when its frozen `ACCELERATOR FLOAT32`
provider answer is true. A false row stays an explicit exclusion. Exact movement, [shape](../../glossary.md#shape)/[layout](../../glossary.md#layout),
indexing, selection, state, ordering, and cast rows are profile-common exact targets and gain no
arithmetic relaxation. Existing `STRICT_IEEE` rows remain snapshots only and are never a source of
new low-precision [accelerator](../../glossary.md#term-accelerator) capability.

For the [CPU](../../glossary.md#term-cpu) target, [`FLOAT16`](../../glossary.md#term-float16) must first reach parity with the frozen CPU [`FLOAT32`](../../glossary.md#term-float32) row. Existing
CPU [`BFLOAT16`](../../glossary.md#term-bfloat16) behavior is preserved; the only planned CPU `BFLOAT16` [residual](../../glossary.md#scan-residual-port-and-history-demand) in this tranche is
[dropout](../../glossary.md#dropout--inverted-dropout). For [Metal](../../glossary.md#term-metal), both `BFLOAT16` and `FLOAT16` must reach parity with the frozen Metal
`ACCELERATOR FLOAT32` row. “Parity” means the same [occurrence](../../glossary.md#term-occurrence) domain and semantic result-set
obligations. It does not require the same route, instruction sequence, performance, or bit pattern,
and it does not widen any already-supported `BFLOAT16` occurrence.

A later capability flip must be complete for the mapped occurrence: provider answer, preparation,
execution, [publication](../../glossary.md#term-publication), failure behavior, and any separately supported generated [graph](../../glossary.md#graph) must agree.
A forward row never implies generated-backward [ownership](../../glossary.md#term-ownership-meanings).

### [Model](../../glossary.md#term-model)-owned low-[precision](../../glossary.md#term-precision) arithmetic contract

The low-precision contract applies only to future arithmetic occurrences newly admitted under
[`ACCELERATOR`](../../glossary.md#term-accelerator) after their capability work lands. It does not retroactively change any current
[`BFLOAT16`](../../glossary.md#term-bfloat16) [occurrence](../../glossary.md#term-occurrence) and never changes `STRICT_IEEE`.

[`FLOAT16`](../../glossary.md#term-float16) and `BFLOAT16` are distinct, incomparable storage types. A mixed `FLOAT16`/`BFLOAT16`
floating [operation](../../glossary.md#operation) promotes to [`FLOAT32`](../../glossary.md#term-float32); either low type combined with `FLOAT32` promotes to
`FLOAT32`, and any combination containing [`FLOAT64`](../../glossary.md#term-float64) promotes to `FLOAT64`. A [homogeneous](../../glossary.md#term-homogeneous) low-
precision occurrence retains its declared low output type.

Kind, attributes, [arity](../../glossary.md#term-arity), [descriptors](../../glossary.md#term-descriptor), [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) mapping, raw storage and movement, contributor
domain, guards, predicates, indices, [masks](../../glossary.md#term-boolean-mask), state transitions, selection, ordering, casts, public
outputs, and saved values remain exact. Casts retain their declared conversion. Each declared low
arithmetic output performs one final [round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../glossary.md#term-narrowing); an implementation
may not pre-narrow an input, contributor, [accumulator](../../glossary.md#accumulator-type), public value, or saved value.

Every newly admitted family first requires a complete custom implementation whose low arithmetic
uses `FLOAT32` working and accumulator values. The family contract explicitly declares where low
operands may use [DAZ](../../glossary.md#term-daz), low results may use [FTZ](../../glossary.md#term-ftz), and an arithmetic exact-zero result may choose
either sign. Arithmetic [NaN](../../glossary.md#term-nan) must retain NaN class and the Model formula's domain behavior, but its
[payload](../../glossary.md#artifact-envelope-and-atomic-publication) and sign are not accuracy requirements. These freedoms never change a predicate, index,
mask, selection, state transition, raw copy, or cast.

Within one operation, arithmetic may reassociate and use corresponding `FLOAT32` [FMA](../../glossary.md#term-fused-multiply-add-fma). [Fusion](../../glossary.md#term-fusion) may
cross only an intermediate that is semantically unobservable and has exactly one consumer. It may
never cross a public output, saved value, [fan-out](../../glossary.md#term-fan-out), predicate, index, mask, selection, or state
boundary. This is an observability rule rather than a blanket ban on cross-[node](../../glossary.md#node) [contraction](../../glossary.md#contraction-dimension) or
algebraic transformation.

Model owns a public accuracy envelope for each admitted low-precision family. The envelope fixes
the exact domain and special-class rules, reference result, accumulator and final narrowing, and a
forward-error bound whose scale depends explicitly on family size and [cancellation](../../glossary.md#term-cancellation) [sensitivity](../../glossary.md#term-sensitivity).
It is not a generic absolute/relative `allclose` [tolerance](../../glossary.md#term-tolerance). Every route, including an [opaque](../../glossary.md#term-opaque) or
transformed vendor route, must qualify through either deductive complete-output-set proof against
the envelope or versioned certification against that envelope. A literal primitive tree can
usually take the proof path; an algorithm that has no such tree can take the certification path.
An operation name, undocumented vendor claim, passing examples, or `allclose` [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) alone is
insufficient.

### Inactive route-[certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) [schema](../../glossary.md#term-schema)

P0 defines no active route, generated-backward, [runtime](../../glossary.md#term-runtime), or [device](../../glossary.md#term-device) fact, and the provider ledger
contains none. It does freeze certificate schema 1 for later [opaque](../../glossary.md#term-opaque)-route [qualification](../../glossary.md#term-qualification). The
[canonical](../../glossary.md#term-canonical-form) certificate key contains the profile, [operation](../../glossary.md#operation)/[fusion](../../glossary.md#term-fusion) family, ordered [dtype](../../glossary.md#data-type--datatype) tuple,
[accumulator](../../glossary.md#accumulator-type) dtype, exact [Shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) domain, route [identity](../../glossary.md#term-identity-meanings), [GPU](../../glossary.md#term-gpu) family, OS build, [SDK](../../glossary.md#term-sdk)/framework
version, [compiler](../../glossary.md#term-compiler) version, native [binary](../../glossary.md#term-binary-descriptor-and-metadata) [digest](../../glossary.md#term-digest), [shader](../../glossary.md#term-shader)/program digest, [compile](../../glossary.md#compile)/math flags and
options, and exact capability-[manifest](../../glossary.md#term-manifest) [hash](../../glossary.md#term-hash).

The keyed accuracy record names the [Model](../../glossary.md#term-model)-owned envelope, selects `DEDUCTIVE_PROOF` or
`CERTIFIED_ENVELOPE`, retains the [immutable](../../glossary.md#term-immutable) evidence digest, and records the verdict. Determinism
[metadata](../../glossary.md#term-metadata) is a separate record keyed back to the certificate; it is never an accuracy field or
verdict input. The checked-in schema contains field definitions only and activates no route.
Unknown, stale, or mismatched evidence cannot qualify an opaque route. A later P5/P9 route manifest
must be derived separately from its owning evidence.

### Append-only [identity](../../glossary.md#term-identity-meanings) allocation

The following table records the active [Model](../../glossary.md#term-model) allocation and the still-inactive [backend](../../glossary.md#backend) cutovers:

| Identity | Current | Reserved cutover |
|---|---:|---:|
| `DataType.FLOAT16` | ordinal 6, active after [`BOOL`](../../glossary.md#data-type--datatype) | activated in Model only |
| [Metal](../../glossary.md#term-metal) [`FLOAT16`](../../glossary.md#term-float16) type wire | absent | 7 |
| Metal program [schema](../../glossary.md#term-schema) | 18 | 19 |
| Metal route wires | 1..3 | unchanged |
| Metal candidate/compatibility/route-policy/workload/[codec](../../glossary.md#term-codec) identities | 27 | 28 |
| [CPU](../../glossary.md#term-cpu) generator schema | 67 | 68 |
| route-[certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) schema | field schema 1, no certificate | 1 |
| Metal native [ABI](../../glossary.md#term-application-binary-interface-abi) | 5 | 5, unchanged |

[Operation](../../glossary.md#operation) wires 1 through 115 and attribute wires 0 through 41 remain frozen. No current identity
may be renumbered or reused. The Model enum allocation is activated; it does not consume any
backend reservation. The integrated backend implementation cutover must activate the remaining
wires, schemas, codecs, generated [artifacts](../../glossary.md#term-artifact), caches, tests, and documentation together and reject
old or unknown artifacts at their existing [fail-closed](../../glossary.md#term-fail-closed) boundaries. Native ABI 5 remains valid only
because this reservation adds no export, [signature](../../glossary.md#term-signature), count, or C-[layout](../../glossary.md#layout) change; any such change
requires a separate ABI decision.

## Consequences

- Current [runtime](../../glossary.md#term-runtime), native behavior, provider answers, capabilities, routes, `STRICT_IEEE` snapshots,
  and `ACCELERATOR FLOAT32` snapshots remain byte-for-byte or behaviorally unchanged.
- The checked-in provider ledger exposes drift in either provider and separately exposes target-map
  edits. The identity-allocation and certificate-schema manifests freeze cross-phase contracts
  without pretending any route certificate or route selection already exists.<span data-glossary-navigation="true"> ([target glossary](../../glossary.md#term-target); [identity glossary](../../glossary.md#term-identity-meanings); [certificate glossary](../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [schema glossary](../../glossary.md#term-schema))</span>
- Low-precision capability work can be partitioned by backend and operation family, but every
  admitted row has an explicit current <span data-glossary-navigation="true"> ([precision glossary](../../glossary.md#term-precision); [backend glossary](../../glossary.md#backend); [operation glossary](../../glossary.md#operation))</span>[`FLOAT32`](../../glossary.md#term-float32) reference and exclusion reason.
- A newly admitted custom baseline may use operation-local arithmetic freedom and bounded
  observable-single-use fusion. An opaque transformed route may qualify by deductive proof or
  versioned certification against the public cancellation/size-aware family envelope.<span data-glossary-navigation="true"> ([fusion glossary](../../glossary.md#term-fusion); [opaque glossary](../../glossary.md#term-opaque); [cancellation glossary](../../glossary.md#term-cancellation))</span>
- Determinism evidence remains independent from accuracy evidence.
- [ADR](../../glossary.md#term-architecture-decision-record-adr) 0021's statement that non-`FLOAT32` behavior remained strict described the P0 system and
  every occurrence then present. At the time, this decision reserved the future exception for
  newly admitted low-precision arithmetic; ADR 0026 now governs present numerical semantics.<span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence))</span>

## Rejected alternatives

### Infer low-[precision](../../glossary.md#term-precision) support from an [operation](../../glossary.md#operation) kind

Capability is [occurrence](../../glossary.md#term-occurrence)-specific. Kind-only [inference](../../glossary.md#term-inference) loses attributes, ordered [descriptors](../../glossary.md#term-descriptor),
[layouts](../../glossary.md#layout), [gradients](../../glossary.md#term-gradient), and [arity](../../glossary.md#term-arity). Rejected.

### Put route or [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) facts in the provider [manifest](../../glossary.md#term-manifest)

A provider answers [backend](../../glossary.md#backend) semantic [ownership](../../glossary.md#term-ownership-meanings). Route, [runtime](../../glossary.md#term-runtime), [device](../../glossary.md#term-device), certificate, and generated-
backward facts have different [owners](../../glossary.md#term-owner) and lifecycles. Combining them would create a second [planning](../../glossary.md#term-planning)
or preparation authority. Rejected.

### Add a capability-enumeration [API](../../glossary.md#term-application-programming-interface-api)

The stable representative basis can query the actual predicate directly. Redesigning
[`BackendCapabilityProvider`](../../glossary.md#backend-capability-provider--backendcapabilityprovider) would add public [weight](../../glossary.md#term-weight) without improving the frozen evidence.
Rejected.

### Accept vendor primitives by [operation](../../glossary.md#operation) name or generic [tolerance](../../glossary.md#term-tolerance)

A name does not bind environment, options, program bytes, or complete [occurrence](../../glossary.md#term-occurrence) domain. A
passing example or generic final-output `allclose` tolerance does not establish a [Model](../../glossary.md#term-model)-owned
family envelope. Eligibility requires deductive proof or a versioned [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) against that
envelope. Rejected.

## Related documents

- [Foundational-module contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module))</span>
- [Backend execution contract](../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../glossary.md#backend))</span>
- [Capability-provider guide](../../backend-guide/capability-provider.md)
- [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0022](0022-auditable-custom-metal-route-cost-evidence.md)
