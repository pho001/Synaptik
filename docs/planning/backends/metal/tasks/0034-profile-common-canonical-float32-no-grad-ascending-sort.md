# Task 0034: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) No-Grad Ascending SORT

## Reader context

The selector's one-input, one-output structure fit SORT, but its existence did not establish
Model's complete ordering policy. That policy separately constrained signed-zero order,
NaNs, stable equal-value ordering and preservation of selected raw representations. A passing
ordinary vector could not authorize every admitted length and value.

Shape, type, layout and gradient flags described the proposed occurrence; the input buffer's
words determined its ordering cases. Model owned the policy and Metal would need to qualify
the route. No probe or implementation followed this assessment, and the listed protocol changes
remained conditional.
The [current stable ordering and exact-value domain](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
connects to later SORT, ARGSORT and TOP_K support without turning this Blocked record into delivery.

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add stable ordering to both [Metal](../../../../glossary.md#term-metal) profile [matrices](../../../../glossary.md#term-matrix-meanings),
evolve private [schema](../../../../glossary.md#term-schema)/route [identities](../../../../glossary.md#term-identity-meanings), and change native [graph](../../../../glossary.md#graph) construction. This [planning](../../../../glossary.md#term-planning)-only
blocker changes no executable behavior and ran no [device](../../../../glossary.md#term-device) probe.

## Goal

Assess the smallest structurally coherent [Metal](../../../../glossary.md#term-metal) ordering [slice](../../../../glossary.md#slice): [canonical](../../../../glossary.md#term-canonical-form) positive static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
[`FLOAT32`](../../../../glossary.md#term-float32), no-grad, ascending `SORT` on [axis](../../../../glossary.md#term-axis) zero with one output under both numerical profiles.
[MPSGraph](../../../../glossary.md#term-mpsgraph) exposes a direct one-output selector, but its documentation omits multiple independent
stability, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [subnormal](../../../../glossary.md#term-subnormal), exact-bit, and [shape](../../../../glossary.md#shape)-algorithm properties. Ordering has no
[`ACCELERATOR`](../../../../glossary.md#term-accelerator) relaxation. One execution cannot close the gaps, so Task 0034 is Blocked before any
probe or production edit.

## Candidate domain if unblocked

- Admit exactly `OrderingKind.SORT` with `new SortAttrs(0, false)`, one ordered input, and only
  output slot zero.
- Report identical exact capability under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); accelerator adds no
  ordering result freedom and may not remove strict capability.
- Require [`FLOAT32`](../../../../glossary.md#term-float32), `requiresGrad=false`, fully static canonical dense-contiguous zero-offset
  non-view input/output with the same positive rank-one Shape <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Shape glossary](../../../../glossary.md#shape))</span>`[N]`; all existing checked element,
  byte, dimension, native-integer, and materialization bounds must pass.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [dimension glossary](../../../../glossary.md#dimension); [materialization glossary](../../../../glossary.md#materialization))</span>
- Produce one fresh non-aliasing canonical output containing the exact input representations in
  Model's stable ascending order. Capability is value-independent over the complete FLOAT32 domain.<span data-glossary-navigation="true"> ([aliasing glossary](../../../../glossary.md#term-aliasing); [Model glossary](../../../../glossary.md#term-model))</span>
- Preserve the current whole-partition lifecycle, FLOAT32-only CPU/Metal transfer, canonical
  publication, and profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lifecycle glossary](../../../../glossary.md#lifecycle); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>

## Non-goals

- Descending order, another axis/rank/type/layout, scalar/empty/dynamic Shape, affine input,
  in-place sort, named axes, user comparators, unstable sort, or algorithm options.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>
- `ARGSORT`, `TOP_K`, values-plus-indices results, INT64 support, cast repair, K/largest/sorted
  options, partial selection, or a hidden output.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- `requiresGrad=true`, matching ARGSORT construction, scatter pullback, training, or [fusion](../../../../glossary.md#term-fusion).
- Host sorting/repair, fallback, retry, value scan, composed comparison/indexing graph, tolerance,
  or imported binary/comparison/reduction/MATMUL result freedom.<span data-glossary-navigation="true"> ([Host glossary](../../../../glossary.md#term-host); [scan glossary](../../../../glossary.md#term-cumulative-scan); [graph glossary](../../../../glossary.md#graph); [tolerance glossary](../../../../glossary.md#term-tolerance); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- A Shape, axis, direction, type, optimization, context, repetition, or value corpus matrix;
  narrowing capability to one passing geometry or selected value set.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  gives ordering no [`ACCELERATOR`](../../../../glossary.md#term-accelerator) relaxation and preserves its bit, selected-value, index, and tie
  contracts.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  keeps route/resource selection cold and Runtime free of operation-policy lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md) and
  [top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md) own kind, attributes,
  ordering, Shape, type, output-role, and metadata contracts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compiler ordering gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  owns pullbacks; [CPU ordering](../../cpu/tasks/0006c-portable-stable-ordering-and-selection.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  is exact atomic-operation precedent, not Metal authorization.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019C/0019C1; Compiler 0005C/0006B11; CPU 0006C
  precedent; Config 0006; Engine 0018. Earlier Metal blockers are independent evidence.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/
  materialization/public Engine scope; ordering/top-K/INT64/gradient work; any 0016–0018 restart.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [INT64 glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Parallel group: None.
- Common base revision: `977cdfbc5917b369d021c512ac54a063f2b8815d`.
- Integration order: no implementation before one recorded complete-domain unblock.
- Integration [validation](../../../../glossary.md#term-validation): Markdown/link/anchor/fence/whitespace validation and `git diff --check`;
  no executable validation or device probe while Blocked.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0034 planner; later Class C work needs a new owner/review.

## Exact [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), and [CPU](../../../../glossary.md#term-cpu) boundary

Model applies ordering independently to each logical-[axis](../../../../glossary.md#term-axis) [slice](../../../../glossary.md#slice). Ascending non-NaNs use numerical
order, including [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and ordinary [finite](../../../../glossary.md#term-finite) [subnormals](../../../../glossary.md#term-subnormal). [Negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) strictly precedes [positive
zero](../../../../glossary.md#nan-infinity-and-signed-zero). Every [NaN](../../../../glossary.md#term-nan) forms one final class without sign, [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), or signaling distinctions. Equal
finite values and multiple NaNs retain increasing original [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). `SORT` copies the exact
selected input representation, including [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/signaling bits, [signed zero](../../../../glossary.md#term-signed-zero), and subnormal
bits; it does not canonicalize or recreate a value.

`SORT` has one input and one output, preserving input type, [Shape](../../../../glossary.md#shape), and `requiresGrad`. `ARGSORT` is a
separate one-output [occurrence](../../../../glossary.md#term-occurrence) whose result is always [INT64](../../../../glossary.md#data-type--datatype) and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). `TOP_K` always
has two ordered outputs from one [producer](../../../../glossary.md#term-producer): values at slot zero and mandatory INT64 indices at slot
one. Its values retain input [gradient](../../../../glossary.md#term-gradient) eligibility.

Compiler [inference](../../../../glossary.md#term-inference) reconstructs those exact roles. A differentiable floating `SORT` pullback
constructs matching `ARGSORT` and replacement scatter; a TOP_K values pullback requires its
original slot-one indices. Requiring `requiresGrad=false` avoids neither a forward semantic rule nor
an output, but keeps this candidate independent of unsupported INT64 and gradient [topology](../../../../glossary.md#term-topology).

CPU 0006C realizes all three families atomically with stable primitive-index merge scratch, exact
represented-value copies, and INT64 logical indices. Its algorithm and evidence prove the semantic
boundary is implementable; they do not prove [MPSGraph](../../../../glossary.md#term-mpsgraph).

## Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) surface and availability

The installed macOS 27.0 [SDK](../../../../glossary.md#term-sdk)'s `MPSGraphSortOps.h` declares
`sortWithTensor:axis:descending:name:` from macOS 13.0 and returns one [tensor](../../../../glossary.md#tensor). The current [runtime](../../../../glossary.md#term-runtime) is
macOS 26.6.2, so the selector is available. The header states only that it sorts along the [axis](../../../../glossary.md#term-axis) and
reverses direction when requested; it gives no stable-order, special-value, bit-preservation,
supported-type, or [shape](../../../../glossary.md#shape)-algorithm contract.

The same header declares one-output `argSortWithTensor:axis:descending:name:` from macOS 13.0 and
explicitly returns [INT32](../../../../glossary.md#data-type--datatype). That cannot implement [Model](../../../../glossary.md#term-model)'s fixed [INT64](../../../../glossary.md#data-type--datatype) ARGSORT. The current native and
Java [Metal](../../../../glossary.md#term-metal) value vocabularies contain only [FLOAT32](../../../../glossary.md#term-float32), INT32, and [BOOL](../../../../glossary.md#data-type--datatype); workload type wires,
[materialization](../../../../glossary.md#materialization), and [CPU](../../../../glossary.md#term-cpu)/Metal transfer have no INT64 path. INT32 ingress/local plumbing is not an
INT64 [publication](../../../../glossary.md#term-publication) substitute.

`MPSGraphTopKOps.h` declares minor-[dimension](../../../../glossary.md#dimension) TOP_K from macOS 12.0 and axis TOP_K/BOTTOM_K from
macOS 14.0, returning two tensors. Model likewise mandates two outputs, but current
`MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and [schema](../../../../glossary.md#term-schema)-v11 [node](../../../../glossary.md#node) record admit exactly one
output per node. The Apple header also omits indices type, stable ties, special ordering, exact bits,
and output-order guarantees and exposes no equivalent of Model's `sorted=false` policy. TOP_K is
not impossible after a broad multi-output/type cutover, but it is not representable by the current
Metal route and is not this smallest [slice](../../../../glossary.md#slice).

## Documentation-first blocker

Ordering's [normative](../../../../glossary.md#term-normative) profile row permits no relaxation. Comparison [DAZ](../../../../glossary.md#term-daz) and other [operation](../../../../glossary.md#operation) rows are
operation-scoped and cannot change an ordering input, comparator, selected representation, tie, or
[publication](../../../../glossary.md#term-publication). A direct selector avoids [graph](../../../../glossary.md#graph) composition but inherits no adjacent freedom.

For the complete [FLOAT32](../../../../glossary.md#term-float32) candidate, the direct SORT header independently omits:

1. stable increasing logical-index order for equal keys and multiple represented NaNs;
2. NaNs-last placement in ascending output and preservation of each selected [NaN](../../../../glossary.md#term-nan) representation;
3. strict negative-zero-before-positive-zero ordering and exact zero-bit copying;
4. numerical ordering of [subnormals](../../../../glossary.md#term-subnormal) without DAZ/[FTZ](../../../../glossary.md#term-ftz) and exact subnormal-bit copying;
5. exact [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) movement for every selected [finite](../../../../glossary.md#term-finite)/infinite value; and
6. one conforming behavior across every admitted `N` and [opaque](../../../../glossary.md#term-opaque) [shape](../../../../glossary.md#shape)-dependent sort algorithm.

A single adversarial [vector](../../../../glossary.md#term-vector) could observe several properties for one `N` and compiled algorithm,
but a pass would not prove the other independent contracts or all admitted Shapes. The one-[run](../../../../glossary.md#term-run) gate
is permitted only when exactly one concrete undocumented property remains; it is therefore not
sound here. The prohibited [matrix](../../../../glossary.md#term-matrix-meanings) and arbitrary one-geometry/value [narrowing](../../../../glossary.md#term-narrowing) are not alternatives.
No probe ran and no [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), capability, [lowering](../../../../glossary.md#lowering), native, test, or source change exists.

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0034 changes and reserves no schema. If its direct-selector route were separately unblocked and
landed first from the current state, it could advance [node](../../../../glossary.md#node) schema `11 -> 12`, append `SORT=20`, add
`AttributeKind.ORDERING=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`. Retain wires `1..19`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, the 160-byte
record, and thirteen exports.

Its conditional record uses `firstInput=input`, absent second input, one output,
`attributeCount=2`, `axis=0`, `auxiliary=0` for `descending=false`, and zero attribute cells
`0..15`. Java and native [preflight](../../../../glossary.md#term-preflight) reject every other [operation](../../../../glossary.md#operation), attribute, type, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout),
[gradient](../../../../glossary.md#term-gradient), [arity](../../../../glossary.md#term-arity), sentinel, or [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) form and validate selector availability plus returned
[FLOAT32](../../../../glossary.md#term-float32) Shape/type before [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names).

Task 0033 conditionally names the same next schema/wire/attribute/version numbers but explicitly
reserves none. Whichever separately authorized implementation lands first owns the actual next
identities; the other must rebase and choose new append-only values. While both remain Blocked,
schema 11, wires `1..19`, attribute kinds `0..6`, version-twelve identities, ABI v4, and thirteen
exports remain unchanged and unreserved. A custom [kernel](../../../../glossary.md#kernel) needs its own typed route.

## Exact unblocking options

Task 0034 can unblock only through one separately authorized complete-domain path:

1. authoritative Apple documentation for the exact direct selector that closes supported FLOAT32
   type/output, unconditional stability, NaNs-last and represented-NaN behavior, signed-zero order,
   non-DAZ subnormal comparison/copy, exact represented-bit movement, and all admitted Shapes; only
   one genuinely remaining property may receive a later one-run raw-bit gate; or<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [DAZ glossary](../../../../glossary.md#term-daz); [subnormal glossary](../../../../glossary.md#term-subnormal); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [run glossary](../../../../glossary.md#term-run))</span>
2. an exact custom [Metal](../../../../glossary.md#term-metal) stable-sort [kernel](../../../../glossary.md#kernel) with a complete proof for every admitted `N`, explicit
   raw-bit NaN classification, negative/positive-zero order, ordinary subnormal comparison,
   left-logical-index tie selection, and represented-bit copy.

Afterward, structural proof must cover duplicate Java/native kind, [arity](../../../../glossary.md#term-arity), type, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout),
[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), availability, [alias](../../../../glossary.md#aliasing-and-arena), result-Shape/type, stale [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), [publication](../../../../glossary.md#term-publication), and
[CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) checks. Those checks cannot substitute for the prior ordering proof.

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

- Record the exact candidate, exclusions, Model/profile contract, installed-selector gaps,
  ARGSORT type mismatch, TOP_K arity boundary, exact unblocks, and unreserved conditional schema.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [arity glossary](../../../../glossary.md#term-arity); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep Task 0034 `Blocked`, expose no [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, and
  final diff. Run no probe, test, formatter, build, schema reservation, or production edit.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

## Documentation and review impact

Only this task, the [Metal](../../../../glossary.md#term-metal) master plan, and roadmap change. Architecture, APIs, [backend](../../../../glossary.md#backend)/native
guides, glossary, [Javadoc](../../../../glossary.md#term-javadoc), and sources remain accurate because behavior does not change. Later
Class C implementation needs synchronized documentation and independent targeted review.

## Result

[Planning](../../../../glossary.md#term-planning) from clean `977cdfbc5917b369d021c512ac54a063f2b8815d` found direct [FLOAT32](../../../../glossary.md#term-float32) SORT the only
structurally aligned one-output ordering candidate, but its documentation leaves multiple exact
gaps. ARGSORT has an [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) mismatch and TOP_K cannot fit the current one-output route. No
probe or executable change occurred. Task 0034 is Blocked; no [Metal](../../../../glossary.md#term-metal) task is [Ready](../../../../glossary.md#term-ready).
