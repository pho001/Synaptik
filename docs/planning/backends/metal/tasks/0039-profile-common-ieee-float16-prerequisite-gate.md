# Task 0039: Profile-Common [IEEE](../../../../glossary.md#term-ieee) [FLOAT16](../../../../glossary.md#term-float16) Prerequisite Gate

## Status

Blocked

[Model](../../../../glossary.md#term-model) 0026 is an independent Draft without a detailed brief. It must first define and deliver true
[IEEE](../../../../glossary.md#term-ieee) [binary16](../../../../glossary.md#floating-point-precision-and-rounding) Model semantics before any [backend](../../../../glossary.md#backend) can advertise [`FLOAT16`](../../../../glossary.md#term-float16). Task 0040 failed its
independent [BFLOAT16](../../../../glossary.md#term-bfloat16) gate; no [Metal](../../../../glossary.md#term-metal) task is [Ready](../../../../glossary.md#term-ready).

## Change class

Class A — [planning](../../../../glossary.md#term-planning) and evidence only. This task changes no observable behavior, public [API](../../../../glossary.md#term-application-programming-interface-api),
production source, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), test, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), transfer, or capability.

## Goal

Resolve the historical [Metal](../../../../glossary.md#term-metal) 0028 [`FLOAT16`](../../../../glossary.md#term-float16) reservation honestly. Record the missing [Model](../../../../glossary.md#term-model)-owned
semantic prerequisite, the current closed Metal representation boundaries, and one bounded
post-[dependency](../../../../glossary.md#term-dependency) candidate without authorizing it or reserving [schema](../../../../glossary.md#term-schema) values. Do not infer [backend](../../../../glossary.md#backend)
semantics from a [device](../../../../glossary.md#term-device) data-type name or a shared two-byte [carrier](../../../../glossary.md#term-storage-carrier).

## [Dependency](../../../../glossary.md#term-dependency) blocker

The current public [Model](../../../../glossary.md#term-model) has exactly [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and
[`BOOL`](../../../../glossary.md#data-type--datatype). It has no `DataType.FLOAT16`, [binary16](../../../../glossary.md#floating-point-precision-and-rounding) bit/conversion helper, exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) variant,
unambiguous public [carrier](../../../../glossary.md#term-storage-carrier) construction, promotion position, cross-type conversion policy, or
[operation](../../../../glossary.md#operation)-family [`FLOAT16`](../../../../glossary.md#term-float16) input, accumulation/intermediate, output, [rounding](../../../../glossary.md#term-rounding), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero),
[subnormal](../../../../glossary.md#term-subnormal), [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [identity](../../../../glossary.md#term-identity-meanings), and [gradient](../../../../glossary.md#term-gradient)-[constant](../../../../glossary.md#term-constant) contracts.

[Model 0026](../../../modules/model/master-plan.md#task-list) remains Draft, has no task file, and is
required before any [backend](../../../../glossary.md#backend) advertises `FLOAT16`. It must preserve `BFLOAT16` as a distinct format;
equal storage width does not establish promotion, arithmetic, conversion, or route equivalence.
Repository-wide downstream enum adoption must also [compile](../../../../glossary.md#compile) while every unselected backend row stays
[fail-closed](../../../../glossary.md#term-fail-closed).

## Numerical-profile boundary

[`NumericalProfile`](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
already supplies the only two [graph](../../../../glossary.md#graph)-wide [identities](../../../../glossary.md#term-identity-meanings): `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator). Config owns
[identity](../../../../glossary.md#term-identity-meanings) only and requires no new profile, [dtype](../../../../glossary.md#data-type--datatype) policy, parser, default, or route surface.

The current allowed-result table grants additional results only to named [`FLOAT32`](../../../../glossary.md#term-float32) rows and states
that future [`FLOAT16`](../../../../glossary.md#term-float16) remains independently [Model](../../../../glossary.md#term-model)-owned. Unless Model 0026 and a coordinated
architecture change explicitly add a [binary16](../../../../glossary.md#floating-point-precision-and-rounding) row, any later [Metal](../../../../glossary.md#term-metal) `FLOAT16` capability must have
identical exact behavior under both profiles. Strict capability is then an accelerator subset by
equality. `FLOAT32` [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), [reassociation](../../../../glossary.md#term-reassociation), [FMA](../../../../glossary.md#term-fused-multiply-add-fma), reduction-root, or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-[publication](../../../../glossary.md#term-publication) freedoms do
not transfer by width or [device class](../../../../glossary.md#device-class--deviceclass).

## Representation evidence only

The installed [SDK](../../../../glossary.md#term-sdk) declares `MPSDataTypeFloat16` as the [IEEE](../../../../glossary.md#term-ieee)-754-2008 [binary16](../../../../glossary.md#floating-point-precision-and-rounding) exchange format and
makes it distinct from `MPSDataTypeBFloat16`. This establishes an available [MPS](../../../../glossary.md#term-mps) representation
vocabulary, not Synaptik [operation](../../../../glossary.md#operation) results, promotion, accumulation, conversion, capability, or
profile permission. No [device](../../../../glossary.md#term-device) execution is justified while the [Model](../../../../glossary.md#term-model) [oracle](../../../../glossary.md#term-test-oracle) is absent.

Current [Metal](../../../../glossary.md#term-metal) remains deliberately closed:

- Java and native inferred value types contain only [`FLOAT32`](../../../../glossary.md#term-float32), [`INT32`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype);
- the MPSGraph create ABI carries ranks, dimensions, and operation records but no per-value dtype
  table, so operation wires currently imply type;<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [dimensions glossary](../../../../glossary.md#dimension); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- native width and `MPSDataType` maps have no two-byte [`FLOAT16`](../../../../glossary.md#term-float16) row;
- caller ingress accepts only `FLOAT32` and `INT32`;
- local [publication](../../../../glossary.md#term-publication) accepts [canonical](../../../../glossary.md#term-canonical-form) `FLOAT32`/`BOOL` and converts numeric output in four-byte
  cells;
- schedule assembly and workload [identity](../../../../glossary.md#term-identity-meanings) recognize only the three current Metal value types; and
- [CPU](../../../../glossary.md#term-cpu)/Metal transfer remains canonical positive-[rank](../../../../glossary.md#term-rank) `FLOAT32` only.

Generic byte [buffers](../../../../glossary.md#term-buffer), upload/download copies, and [descriptor](../../../../glossary.md#term-descriptor)-derived resource sizes do not by
themselves authorize a type or operation.

## Conditional post-[dependency](../../../../glossary.md#term-dependency) candidate

After [Model](../../../../glossary.md#term-model) 0026 is Complete and independently reviewed, re-audit [Compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) and public [Engine](../../../../glossary.md#term-engine)
[materialization](../../../../glossary.md#materialization). If those [owners](../../../../glossary.md#term-owner) can [carry](../../../../glossary.md#term-carry) the new [descriptor](../../../../glossary.md#term-descriptor) while failing closed elsewhere, the
smallest candidate is one profile-common, forward-only, no-[gradient](../../../../glossary.md#term-gradient) `FLOAT16 CONTIGUOUS`
[occurrence](../../../../glossary.md#term-occurrence):

- one fully static [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) `1..16` [`FLOAT16`](../../../../glossary.md#term-float16) feed;
- one same-[Shape](../../../../glossary.md#shape), same-type fresh canonical output;
- caller-[host](../../../../glossary.md#term-host) ingress and local [detached](../../../../glossary.md#detached-model-results) [publication](../../../../glossary.md#term-publication) only; and
- identical exact [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) behavior under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator).

This candidate excludes arithmetic, [unary](../../../../glossary.md#term-unary) numeric operations, affine-[view](../../../../glossary.md#term-view-meanings) input/publication,
[scalars](../../../../glossary.md#term-scalar)/[constants](../../../../glossary.md#term-constant), casts, [mixed precision](../../../../glossary.md#term-mixed-precision), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), gradients, cross-[owner](../../../../glossary.md#term-owner) transfer, dynamic or
zero extents, rank zero, unresolved [layout](../../../../glossary.md#layout), and every other [operation](../../../../glossary.md#operation) family.

Blocked Task 0040 implemented no [schema](../../../../glossary.md#term-schema) and consumes no successor value. A later separately
authorized FLOAT16 implementation may consider [node](../../../../glossary.md#node) schema 12, a type-specific operation wire 20,
internal value-type wire 4, and version-thirteen [identities](../../../../glossary.md#term-identity-meanings) only after revalidating them against the
then-current frontier. The fixed 160-byte record, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, thirteen exports, and attribute wires
`0..6` could remain unchanged. These values are conditional and unreserved; this Blocked task
allocates none of them.

Caller-host ingress would need exact `FLOAT16` storage [admission](../../../../glossary.md#term-admission) and local publication would need
two-byte [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) canonical output. [CPU](../../../../glossary.md#term-cpu) capability and CPU/[Metal](../../../../glossary.md#term-metal) `FLOAT16` transfer remain
separate future scopes.

## [Device](../../../../glossary.md#term-device) gate

[Run](../../../../glossary.md#term-run) no probe now. There is no public [Model](../../../../glossary.md#term-model) [oracle](../../../../glossary.md#term-test-oracle) and several independent contracts are missing.

Only after the [dependency](../../../../glossary.md#term-dependency) supplies the complete exact contract, and only if one isolated fact
remains, a separately authorized implementation task may run one production-settings [MPSGraph](../../../../glossary.md#term-mpsgraph)
execution over a small representative raw-bit corpus derived from that delivered contract. It must
not enumerate all 65,536 patterns or add a [Shape](../../../../glossary.md#shape)/[operation](../../../../glossary.md#operation) [matrix](../../../../glossary.md#term-matrix-meanings). Failure blocks that task and
removes the probe; success can authorize only its separately revalidated bounded candidate.

## Non-goals

- Selecting, specifying, implementing, or changing [Model](../../../../glossary.md#term-model) 0026.
- Adding `DataType.FLOAT16`, a Config profile, profile relaxation, public [manifest](../../../../glossary.md#term-manifest), or fallback.
- Editing [Compiler](../../../../glossary.md#term-compiler), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), [CPU](../../../../glossary.md#term-cpu), [Metal](../../../../glossary.md#term-metal) production, or native source.
- Reserving a [schema](../../../../glossary.md#term-schema), wire, type, [identity](../../../../glossary.md#term-identity-meanings), [ABI](../../../../glossary.md#term-application-binary-interface-abi), export, [cache](../../../../glossary.md#term-cache), or transfer value.
- Probing a [device](../../../../glossary.md#term-device), building [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), or adding/updating tests.
- Claiming historical Metal 0028, Blocked Task 0040, or any existing Blocked [operation](../../../../glossary.md#operation) complete.

## Dependencies and integration

- Hard dependency: Model 0026 Complete with public binary16 representation, promotion, conversion,
  family semantics, profile result sets, and reviewed downstream fail-closed enum adoption.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [binary16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Existing completed context: [Model](../../../../glossary.md#term-model) 0027–0029, Config 0006, [Engine](../../../../glossary.md#term-engine) 0018, and [Metal](../../../../glossary.md#term-metal) 0038.
- Conflicts with: every concurrent Model dtype/profile task and Metal capability/schema/native/type/
  ingress/publication/identity/transfer scope.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema); [publication glossary](../../../../glossary.md#term-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None.
- Common base revision: `1d489623b1a4e2699ef791f4b901368fbc5a4057`.
- Integration validation: planning links/anchors/fences/final newlines, status consistency,
  documentation-only path scope, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check`.

## Files and [ownership](../../../../glossary.md#term-ownership-meanings)

This blocker owns only:

- this task brief;
- `docs/planning/backends/metal/master-plan.md`;
- `docs/planning/roadmap.md`; and
- correction of the two stale [Metal](../../../../glossary.md#term-metal)-0021 [Ready](../../../../glossary.md#term-ready)-frontier sentences in
  `docs/planning/modules/model/master-plan.md` while preserving Model 0026 Draft and no Model task
  Ready.

Current public and Metal guides already state that [`FLOAT16`](../../../../glossary.md#term-float16) is unsupported and [Model](../../../../glossary.md#term-model) 0026 is the
prerequisite; they remain unchanged.

## Acceptance criteria

- Task 0039 is Blocked on Draft Model 0026, Task 0040 is independently Blocked, and no Metal task
  is Ready.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- The absence of public binary16 type/semantic/oracle contracts and every current Metal plumbing gap
  above is explicit.<span data-glossary-navigation="true"> ([binary16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- `MPSDataTypeFloat16` is representation evidence only, not capability evidence.
- Any later capability is profile-common unless Model explicitly changes the result-set contract;
  Config gains no profile.
- Its bounded candidate and all possible schema, operation, type, and identity values remain
  conditional and unreserved.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- No [device](../../../../glossary.md#term-device) probe, source/test/native/[schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/transfer change, or executable [validation](../../../../glossary.md#term-validation) runs.
- [Model](../../../../glossary.md#term-model) 0026 remains Draft with no detailed brief and no Model task [Ready](../../../../glossary.md#term-ready).

## Result

Blocked before executable work. No [FLOAT16](../../../../glossary.md#term-float16) probe or production change is permitted until [Model](../../../../glossary.md#term-model)
0026 is Complete and the candidate is revalidated against its delivered contracts and the
then-current [Metal](../../../../glossary.md#term-metal) [schema](../../../../glossary.md#term-schema).
