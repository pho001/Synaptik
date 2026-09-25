# Task 0039: Profile-Common IEEE FLOAT16 Prerequisite Gate

## Status

Blocked

Model 0026 is an independent Draft without a detailed brief. It must first define and deliver true
IEEE binary16 Model semantics before any backend can advertise `FLOAT16`. No Metal task is Ready.

## Change class

Class A — planning and evidence only. This task changes no observable behavior, public API,
production source, native code, test, schema, identity, transfer, or capability.

## Goal

Resolve the historical Metal 0028 `FLOAT16` reservation honestly. Record the missing Model-owned
semantic prerequisite, the current closed Metal representation boundaries, and one bounded
post-dependency candidate without authorizing it or reserving schema values. Do not infer backend
semantics from a device data-type name or a shared two-byte carrier.

## Dependency blocker

The current public Model has exactly `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and
`BOOL`. It has no `DataType.FLOAT16`, binary16 bit/conversion helper, exact `ScalarValue` variant,
unambiguous public carrier construction, promotion position, cross-type conversion policy, or
operation-family `FLOAT16` input, accumulation/intermediate, output, rounding, signed-zero,
subnormal, NaN, infinity, identity, and gradient-constant contracts.

[Model 0026](../../../modules/model/master-plan.md#task-list) remains Draft, has no task file, and is
required before any backend advertises `FLOAT16`. It must preserve `BFLOAT16` as a distinct format;
equal storage width does not establish promotion, arithmetic, conversion, or route equivalence.
Repository-wide downstream enum adoption must also compile while every unselected backend row stays
fail-closed.

## Numerical-profile boundary

[`NumericalProfile`](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
already supplies the only two graph-wide identities: `STRICT_IEEE` and `ACCELERATOR`. Config owns
identity only and requires no new profile, dtype policy, parser, default, or route surface.

The current allowed-result table grants additional results only to named `FLOAT32` rows and states
that future `FLOAT16` remains independently Model-owned. Unless Model 0026 and a coordinated
architecture change explicitly add a binary16 row, any later Metal `FLOAT16` capability must have
identical exact behavior under both profiles. Strict capability is then an accelerator subset by
equality. `FLOAT32` DAZ/FTZ, reassociation, FMA, reduction-root, or MATMUL-publication freedoms do
not transfer by width or device class.

## Representation evidence only

The installed SDK declares `MPSDataTypeFloat16` as the IEEE-754-2008 binary16 exchange format and
makes it distinct from `MPSDataTypeBFloat16`. This establishes an available MPS representation
vocabulary, not Synaptik operation results, promotion, accumulation, conversion, capability, or
profile permission. No device execution is justified while the Model oracle is absent.

Current Metal remains deliberately closed:

- Java and native inferred value types contain only `FLOAT32`, `INT32`, and `BOOL`;
- the MPSGraph create ABI carries ranks, dimensions, and operation records but no per-value dtype
  table, so operation wires currently imply type;
- native width and `MPSDataType` maps have no two-byte `FLOAT16` row;
- caller ingress accepts only `FLOAT32` and `INT32`;
- local publication accepts canonical `FLOAT32`/`BOOL` and converts numeric output in four-byte
  cells;
- schedule assembly and workload identity recognize only the three current Metal value types; and
- CPU/Metal transfer remains canonical positive-rank `FLOAT32` only.

Generic byte buffers, upload/download copies, and descriptor-derived resource sizes do not by
themselves authorize a type or operation.

## Conditional post-dependency candidate

After Model 0026 is Complete and independently reviewed, re-audit Compiler capture and public Engine
materialization. If those owners can carry the new descriptor while failing closed elsewhere, the
smallest candidate is one profile-common, forward-only, no-gradient `FLOAT16 CONTIGUOUS`
occurrence:

- one fully static canonical positive-rank `1..16` `FLOAT16` feed;
- one same-Shape, same-type fresh canonical output;
- caller-host ingress and local detached publication only; and
- identical exact represented-bit behavior under `STRICT_IEEE` and `ACCELERATOR`.

This candidate excludes arithmetic, unary numeric operations, affine-view input/publication,
scalars/constants, casts, mixed precision, `BFLOAT16`, gradients, cross-owner transfer, dynamic or
zero extents, rank zero, unresolved layout, and every other operation family.

A later separately authorized implementation may consider node schema 12, one type-specific
`FLOAT16_CONTIGUOUS` operation wire 20, internal value-type wire 4 with two-byte width,
`MPSDataTypeFloat16`, and version-thirteen workload/policy/candidate/compatibility/route/codec
identities. The fixed 160-byte record, ABI v4, thirteen exports, and attribute wires `0..6` could
remain unchanged. These numbers are conditional and unreserved; this Blocked task allocates none of
them and does not consume attribute wire 7.

Caller-host ingress would need exact `FLOAT16` storage admission and local publication would need
two-byte big-endian canonical output. CPU capability and CPU/Metal `FLOAT16` transfer remain
separate future scopes.

## Device gate

Run no probe now. There is no public Model oracle and several independent contracts are missing.

Only after the dependency supplies the complete exact contract, and only if one isolated fact
remains, a separately authorized implementation task may run one production-settings MPSGraph
execution: pass every one of the 65,536 binary16 bit patterns through the candidate
`CONTIGUOUS` reshape/materialization path and compare every output pattern bit-for-bit. Failure
blocks that task and removes the probe; success can authorize only the bounded candidate above.

## Non-goals

- Selecting, specifying, implementing, or changing Model 0026.
- Adding `DataType.FLOAT16`, a Config profile, profile relaxation, public manifest, or fallback.
- Editing Compiler, Planning, Prepare, Runtime, Engine, CPU, Metal production, or native source.
- Reserving a schema, wire, type, identity, ABI, export, cache, or transfer value.
- Probing a device, building native code, or adding/updating tests.
- Claiming historical Metal 0028, reserved Metal 0029, or any existing Blocked operation complete.

## Dependencies and integration

- Hard dependency: Model 0026 Complete with public binary16 representation, promotion, conversion,
  family semantics, profile result sets, and reviewed downstream fail-closed enum adoption.
- Existing completed context: Model 0027–0029, Config 0006, Engine 0018, and Metal 0038.
- Conflicts with: every concurrent Model dtype/profile task and Metal capability/schema/native/type/
  ingress/publication/identity/transfer scope.
- Parallel group: None.
- Common base revision: `1d489623b1a4e2699ef791f4b901368fbc5a4057`.
- Integration validation: planning links/anchors/fences/final newlines, status consistency,
  documentation-only path scope, and `git diff --check`.

## Files and ownership

This blocker owns only:

- this task brief;
- `docs/planning/backends/metal/master-plan.md`;
- `docs/planning/roadmap.md`; and
- correction of the two stale Metal-0021 Ready-frontier sentences in
  `docs/planning/modules/model/master-plan.md` while preserving Model 0026 Draft and no Model task
  Ready.

Current public and Metal guides already state that `FLOAT16` is unsupported and Model 0026 is the
prerequisite; they remain unchanged.

## Acceptance criteria

- Task 0039 is Blocked on Draft Model 0026 and no Metal task is Ready.
- The absence of public binary16 type/semantic/oracle contracts and every current Metal plumbing gap
  above is explicit.
- `MPSDataTypeFloat16` is representation evidence only, not capability evidence.
- Any later capability is profile-common unless Model explicitly changes the result-set contract;
  Config gains no profile.
- The bounded candidate and all possible schema/type/identity values remain conditional and
  unreserved.
- No device probe, source/test/native/schema/identity/transfer change, or executable validation runs.
- Model 0026 remains Draft with no detailed brief and no Model task Ready.

## Result

Blocked before executable work. No probe or production change is permitted until Model 0026 is
Complete and the conditional candidate is revalidated against its delivered contracts.
