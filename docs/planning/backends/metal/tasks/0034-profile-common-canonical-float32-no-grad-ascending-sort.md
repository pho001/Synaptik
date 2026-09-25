# Task 0034: Profile-Common Canonical FLOAT32 No-Grad Ascending SORT

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add stable ordering to both Metal profile matrices,
evolve private schema/route identities, and change native graph construction. This planning-only
blocker changes no executable behavior and ran no device probe.

## Goal

Assess the smallest structurally coherent Metal ordering slice: canonical positive static rank-one
`FLOAT32`, no-grad, ascending `SORT` on axis zero with one output under both numerical profiles.
MPSGraph exposes a direct one-output selector, but its documentation omits multiple independent
stability, NaN, signed-zero, subnormal, exact-bit, and shape-algorithm properties. Ordering has no
`ACCELERATOR` relaxation. One execution cannot close the gaps, so Task 0034 is Blocked before any
probe or production edit.

## Candidate domain if unblocked

- Admit exactly `OrderingKind.SORT` with `new SortAttrs(0, false)`, one ordered input, and only
  output slot zero.
- Report identical exact capability under `STRICT_IEEE` and `ACCELERATOR`; accelerator adds no
  ordering result freedom and may not remove strict capability.
- Require `FLOAT32`, `requiresGrad=false`, fully static canonical dense-contiguous zero-offset
  non-view input/output with the same positive rank-one Shape `[N]`; all existing checked element,
  byte, dimension, native-integer, and materialization bounds must pass.
- Produce one fresh non-aliasing canonical output containing the exact input representations in
  Model's stable ascending order. Capability is value-independent over the complete FLOAT32 domain.
- Preserve the current whole-partition lifecycle, FLOAT32-only CPU/Metal transfer, canonical
  publication, and profile-free Runtime/Trace boundary.

## Non-goals

- Descending order, another axis/rank/type/layout, scalar/empty/dynamic Shape, affine input,
  in-place sort, named axes, user comparators, unstable sort, or algorithm options.
- `ARGSORT`, `TOP_K`, values-plus-indices results, INT64 support, cast repair, K/largest/sorted
  options, partial selection, or a hidden output.
- `requiresGrad=true`, matching ARGSORT construction, scatter pullback, training, or fusion.
- Host sorting/repair, fallback, retry, value scan, composed comparison/indexing graph, tolerance,
  or imported binary/comparison/reduction/MATMUL result freedom.
- A Shape, axis, direction, type, optimization, context, repetition, or value corpus matrix;
  narrowing capability to one passing geometry or selected value set.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  gives ordering no `ACCELERATOR` relaxation and preserves its bit, selected-value, index, and tie
  contracts.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route/resource selection cold and Runtime free of operation-policy lookup.
- [Model sort/argsort](../../../modules/model/tasks/0019c-sort-and-argsort.md) and
  [top-K](../../../modules/model/tasks/0019c1-top-k-values-and-indices.md) own kind, attributes,
  ordering, Shape, type, output-role, and metadata contracts.
- [Compiler ordering gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  owns pullbacks; [CPU ordering](../../cpu/tasks/0006c-portable-stable-ordering-and-selection.md)
  is exact atomic-operation precedent, not Metal authorization.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019C/0019C1; Compiler 0005C/0006B11; CPU 0006C
  precedent; Config 0006; Engine 0018. Earlier Metal blockers are independent evidence.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/
  materialization/public Engine scope; ordering/top-K/INT64/gradient work; any 0016–0018 restart.
- Parallel group: None.
- Common base revision: `977cdfbc5917b369d021c512ac54a063f2b8815d`.
- Integration order: no implementation before one recorded complete-domain unblock.
- Integration validation: Markdown/link/anchor/fence/whitespace validation and `git diff --check`;
  no executable validation or device probe while Blocked.
- Shared-document integration owner: Task 0034 planner; later Class C work needs a new owner/review.

## Exact Model, Compiler, and CPU boundary

Model applies ordering independently to each logical-axis slice. Ascending non-NaNs use numerical
order, including infinities and ordinary finite subnormals. Negative zero strictly precedes positive
zero. Every NaN forms one final class without sign, payload, or signaling distinctions. Equal
finite values and multiple NaNs retain increasing original logical index. `SORT` copies the exact
selected input representation, including NaN payload/signaling bits, signed zero, and subnormal
bits; it does not canonicalize or recreate a value.

`SORT` has one input and one output, preserving input type, Shape, and `requiresGrad`. `ARGSORT` is a
separate one-output occurrence whose result is always INT64 and non-differentiable. `TOP_K` always
has two ordered outputs from one producer: values at slot zero and mandatory INT64 indices at slot
one. Its values retain input gradient eligibility.

Compiler inference reconstructs those exact roles. A differentiable floating `SORT` pullback
constructs matching `ARGSORT` and replacement scatter; a TOP_K values pullback requires its
original slot-one indices. Requiring `requiresGrad=false` avoids neither a forward semantic rule nor
an output, but keeps this candidate independent of unsupported INT64 and gradient topology.

CPU 0006C realizes all three families atomically with stable primitive-index merge scratch, exact
represented-value copies, and INT64 logical indices. Its algorithm and evidence prove the semantic
boundary is implementable; they do not prove MPSGraph.

## Installed MPSGraph surface and availability

The installed macOS 27.0 SDK's `MPSGraphSortOps.h` declares
`sortWithTensor:axis:descending:name:` from macOS 13.0 and returns one tensor. The current runtime is
macOS 26.6.2, so the selector is available. The header states only that it sorts along the axis and
reverses direction when requested; it gives no stable-order, special-value, bit-preservation,
supported-type, or shape-algorithm contract.

The same header declares one-output `argSortWithTensor:axis:descending:name:` from macOS 13.0 and
explicitly returns INT32. That cannot implement Model's fixed INT64 ARGSORT. The current native and
Java Metal value vocabularies contain only FLOAT32, INT32, and BOOL; workload type wires,
materialization, and CPU/Metal transfer have no INT64 path. INT32 ingress/local plumbing is not an
INT64 publication substitute.

`MPSGraphTopKOps.h` declares minor-dimension TOP_K from macOS 12.0 and axis TOP_K/BOTTOM_K from
macOS 14.0, returning two tensors. Model likewise mandates two outputs, but current
`MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and schema-v11 node record admit exactly one
output per node. The Apple header also omits indices type, stable ties, special ordering, exact bits,
and output-order guarantees and exposes no equivalent of Model's `sorted=false` policy. TOP_K is
not impossible after a broad multi-output/type cutover, but it is not representable by the current
Metal route and is not this smallest slice.

## Documentation-first blocker

Ordering's normative profile row permits no relaxation. Comparison DAZ and other operation rows are
operation-scoped and cannot change an ordering input, comparator, selected representation, tie, or
publication. A direct selector avoids graph composition but inherits no adjacent freedom.

For the complete FLOAT32 candidate, the direct SORT header independently omits:

1. stable increasing logical-index order for equal keys and multiple represented NaNs;
2. NaNs-last placement in ascending output and preservation of each selected NaN representation;
3. strict negative-zero-before-positive-zero ordering and exact zero-bit copying;
4. numerical ordering of subnormals without DAZ/FTZ and exact subnormal-bit copying;
5. exact represented-bit movement for every selected finite/infinite value; and
6. one conforming behavior across every admitted `N` and opaque shape-dependent sort algorithm.

A single adversarial vector could observe several properties for one `N` and compiled algorithm,
but a pass would not prove the other independent contracts or all admitted Shapes. The one-run gate
is permitted only when exactly one concrete undocumented property remains; it is therefore not
sound here. The prohibited matrix and arbitrary one-geometry/value narrowing are not alternatives.
No probe ran and no schema, identity, capability, lowering, native, test, or source change exists.

## Conditional schema only

Task 0034 changes and reserves no schema. If its direct-selector route were separately unblocked and
landed first from the current state, it could advance node schema `11 -> 12`, append `SORT=20`, add
`AttributeKind.ORDERING=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and codec identities together `12 -> 13`. Retain wires `1..19`, ABI v4, the 160-byte
record, and thirteen exports.

Its conditional record uses `firstInput=input`, absent second input, one output,
`attributeCount=2`, `axis=0`, `auxiliary=0` for `descending=false`, and zero attribute cells
`0..15`. Java and native preflight reject every other operation, attribute, type, Shape, layout,
gradient, arity, sentinel, or payload form and validate selector availability plus returned
FLOAT32 Shape/type before compilation.

Task 0033 conditionally names the same next schema/wire/attribute/version numbers but explicitly
reserves none. Whichever separately authorized implementation lands first owns the actual next
identities; the other must rebase and choose new append-only values. While both remain Blocked,
schema 11, wires `1..19`, attribute kinds `0..6`, version-twelve identities, ABI v4, and thirteen
exports remain unchanged and unreserved. A custom kernel needs its own typed route.

## Exact unblocking options

Task 0034 can unblock only through one separately authorized complete-domain path:

1. authoritative Apple documentation for the exact direct selector that closes supported FLOAT32
   type/output, unconditional stability, NaNs-last and represented-NaN behavior, signed-zero order,
   non-DAZ subnormal comparison/copy, exact represented-bit movement, and all admitted Shapes; only
   one genuinely remaining property may receive a later one-run raw-bit gate; or
2. an exact custom Metal stable-sort kernel with a complete proof for every admitted `N`, explicit
   raw-bit NaN classification, negative/positive-zero order, ordinary subnormal comparison,
   left-logical-index tie selection, and represented-bit copy.

Afterward, structural proof must cover duplicate Java/native kind, arity, type, Shape, layout,
payload, availability, alias, result-Shape/type, stale identity, lifecycle, publication, and
CPU-free Engine checks. Those checks cannot substitute for the prior ordering proof.

## Acceptance criteria and validation

- Record the exact candidate, exclusions, Model/profile contract, installed-selector gaps,
  ARGSORT type mismatch, TOP_K arity boundary, exact unblocks, and unreserved conditional schema.
- Keep Task 0034 `Blocked`, expose no Ready Metal frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, and
  final diff. Run no probe, test, formatter, build, schema reservation, or production edit.

## Documentation and review impact

Only this task, the Metal master plan, and roadmap change. Architecture, APIs, backend/native
guides, glossary, Javadoc, and sources remain accurate because behavior does not change. Later
Class C implementation needs synchronized documentation and independent targeted review.

## Result

Planning from clean `977cdfbc5917b369d021c512ac54a063f2b8815d` found direct FLOAT32 SORT the only
structurally aligned one-output ordering candidate, but its documentation leaves multiple exact
gaps. ARGSORT has an INT32/INT64 mismatch and TOP_K cannot fit the current one-output route. No
probe or executable change occurred. Task 0034 is Blocked; no Metal task is Ready.
