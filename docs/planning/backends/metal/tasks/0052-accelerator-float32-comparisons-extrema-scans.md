# Task 0052: ACCELERATOR FLOAT32 Comparisons, Extrema, and Scans

## Status

Completed

Gates 1 through 4 and the atomic production cutover are complete. Every opaque direct candidate
remains `DOMAIN-BLOCKED`; every auditable custom candidate is `DOMAIN-PASS` and numerical `PASS`.
Fourteen operation rows had exactly one survivor and therefore needed no comparative cost gate.
CLAMP had two identical-semantics/domain survivors; fused custom CLAMP strictly structurally
dominates composed custom CLAMP with one versus two dispatches and zero versus 4,194,304
route-owned temporary bytes. The retained Apple M3 Max timings are diagnostic history only and did
not authorize those routes. Task 0055 later migrated production to ABI 5/schema 13 and identity 14
without changing Task-0052 capability or fixed routes. Focused Java/native/conformance/
Compiler-contract and no-skip real-dylib public Engine proof passed; the final full repository
build remains deliberately deferred.

## Change class

Class C — implementation extends profile-qualified capability, exact BOOL publication,
scalar-attribute and scan schema meaning, typed native lowering, candidate identity, prepared
execution, and public Engine evidence. It does not change Model semantics, public API, Compiler
generation, Runtime policy, native ABI exports, or the completed package contract.

The earlier evidence slice changed only planning documents and non-production review evidence,
ran exactly one disposable numerical invocation, and later recovered and hash-validated that
evidence without rerun. The completed production slice consumes only the independently approved
proof and Gate-3 route decision; it does not rerun either oracle.

## Goal

Add one bounded, forward-only, canonical `FLOAT32` operation slice under
`NumericalProfile.ACCELERATOR`:

- all six `BinaryComparisonKind` values;
- tensor `BinaryArithmeticKind.MIN` and `MAX`;
- scalar `ScalarElementwiseKind.MIN`, `MAX`, and `CLAMP`;
- ordinary `AggregateReductionKind.MIN` and `MAX`; and
- `CumulativeScanKind.CUM_SUM` and `CUM_PROD`.

Every operation must independently pass structural/API and complete-domain proof, bounded numerical
regression, survivor adjudication under ADR 0022, and fixed production-route gates. Capability
remains false until those gates pass for the complete authorized domain. A failure does not
authorize a narrower shape, attribute, mode, or edge-value subset.

Every candidate was predeclared before the sole numerical invocation. Every candidate must
independently prove complete-domain coverage and pass the bounded regression corpus before survivor
adjudication. Opaque direct and MPSGraph-composed candidates remain `DOMAIN-BLOCKED`. A row with one
survivor requires no comparative cost gate. Multiple identical-semantics/domain survivors may close
without timing only through strict structural dominance; nondominated alternatives require an
authorized controlled environment. Runtime selection, retry, fallback, and input-dependent routing
are forbidden.

## Exact operation inventory and partition

### In scope

| Partition | Exact kinds | Model attributes and result |
|---|---|---|
| comparisons | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | `NoOperationAttrs`; exact right-aligned broadcast; canonical `BOOL`; ordered relations are false on NaN, `NOT_EQUAL` complements equality, and signed zeros compare equal |
| tensor extrema | `BinaryArithmeticKind.MIN`, `MAX` | `NoOperationAttrs`; exact right-aligned broadcast; NaN propagation, numeric order, and operation-fixed signed-zero winner |
| scalar extrema | `ScalarElementwiseKind.MIN`, `MAX` | exact `ScalarValueAttrs(FLOAT32)`; same Shape; scalar is an exact represented candidate |
| scalar clamp | `ScalarElementwiseKind.CLAMP` | exact `ClampRangeAttrs(FLOAT32,FLOAT32)`; one occurrence whose value is ordered `MIN(MAX(x,min),max)` with the Model NaN and signed-zero rules |
| aggregate extrema | `AggregateReductionKind.MIN`, `MAX` | full `NoOperationAttrs`, `AxisReductionAttrs`, and `MultiAxisReductionAttrs`; exact contributor membership, NaN propagation, signed-zero winner, and selected original candidate |
| cumulative scans | `CUM_SUM`, `CUM_PROD` | exact `CumulativeScanAttrs(axis,exclusive,reverse)`; same Shape; exact direction, placement, identity, and per-prefix contributor membership |

The comparison and extrema result sets are exact/discrete. An accelerator comparison may read each
subnormal operand as its same-signed zero. An extrema site may treat DAZ-normalized values as tied,
but its NaN and signed-zero rules remain exact and a selected result remains an original candidate.
There is no ULP, relative, absolute, or epsilon acceptance for these operations.

For a scan, first form the exact ordered traversal sequence for the selected axis: preserve logical
axis order for forward mode and reverse that sequence before taking any prefix for reverse mode.
Inclusive output uses the ordered prefix through the current visited element; exclusive output uses
the ordered prefix before it. The result set contains only order-preserving binary parenthesizations
formed by recursively splitting a prefix into two contiguous nonempty intervals. Each contributor
appears exactly once, with one binary32 round-to-nearest-even operation per arithmetic site,
independent operand DAZ, and exact-subnormal-or-either-signed-zero FTZ at each result. No
permutation, multiset tree, dropped/duplicated/invented/pretruncated contributor, wrong-prefix
contributor, or final-output tolerance envelope is permitted.

### Exact capability domain

An occurrence is eligible only when all of the following hold:

- the graph profile is exactly `ACCELERATOR`;
- kind and attributes are exactly one row above, with exact Model arity and one output;
- every input and output is fully static, uses checked element/byte geometry, and has only positive
  extents;
- each non-scalar descriptor is canonical dense-contiguous, zero-offset, and not a view;
- every tensor payload is `FLOAT32`; comparison outputs alone are canonical `BOOL`;
- every input and output has `requiresGrad == false`;
- comparison and tensor-extrema output Shape is the exact right-aligned broadcast;
- scalar extrema and scans preserve input Shape;
- scan axis is normalized and in range, and all four exclusive/reverse combinations are accepted;
- aggregate input rank is `1..16`, output rank is `0..16`, axes are normalized, unique, and in
  declared order, and output Shape exactly reflects full/single/multi-axis plus keep-dimensions;
- an empty multi-axis extrema reduction is an exact identity/copy and does not call an accidental
  reduction selector; and
- topology, value types/states, feeds, targets, and ownership are completely validated before native
  resource creation.

Comparison BOOL outputs may be direct public results through the already-proved local canonical
BOOL materialization path. At the original Task-0052 cutover, CPU/Metal transfer remained
FLOAT32-only, so whole-partition analysis rejected comparison BOOL consumers, cross-owner BOOL
edges, and BOOL feeds. Task 0055 later enabled exact all-six-carrier transfer without adding any
BOOL-consuming Metal operation. This task adds no Metal logical operation, classification,
`WHERE`, or BOOL-consuming node. Existing supported FLOAT32 nodes may precede or follow in-scope
FLOAT32 extrema/scans where current canonical-state rules already permit it.

### Explicit exclusions

Exclude `ARG_MIN`/`ARG_MAX`: they produce Model `INT64`, have FIRST/LAST tie policy, and require an
unimplemented INT64 local/ingress/publication path. Also exclude `PROD` reduction, `ALL`, `ANY`,
masked/statistical/advanced reductions, cumulative MIN/MAX SDK conveniences, classification,
Boolean logic, `WHERE`, casts, non-FLOAT32 scalar attributes, strict-profile occurrences,
rank-zero inputs, zero/dynamic extents, affine/view inputs, and malformed metadata.

This task is forward/no-grad only. Current Compiler rules remain unchanged: comparisons are
non-differentiable; tensor/scalar extrema gradients introduce comparisons, `WHERE`, arithmetic, and
broadcast reductions; aggregate-extrema gradients introduce `EQUAL`, `WHERE`, `SUM`, expansion,
and division; `CUM_SUM` gradients generate a reverse-direction scan; and `CUM_PROD` gradients add
comparisons, `WHERE`, scans, multiplication, and division. None of those generated topologies is a
capability claim here.

## Pre-Task-0052 Metal architecture and caller inventory

The pre-cutover provider routes accelerator aggregate kinds to `supportsReduction`, MATMUL separately,
and all other accelerator candidates to `supportsBinary`. `supportsBinary` admits only tensor
`ADD/SUB/MUL/DIV`; `supportsReduction` admits only `SUM/MEAN/SUM_TO_SHAPE`. It has no scalar,
comparison, or cumulative-scan capability branch. The preparer and native lowering likewise contain
no node for this task.

The pre-cutover private program is ABI `4`, fixed record size `160`, schema `11`, operation wires
`1..19`, attribute wires `0..6`, value types `FLOAT32`/`INT32`/`BOOL`, and canonical/affine value
states. It already publishes canonical positive-rank BOOL through ONE_HOT and rank-zero FLOAT32
through reduction work, but accepts no general BOOL feed. Workload signature, exact-policy,
candidate, compatibility, route-policy, and codec identities are version `12`; the complete-plan
wrapper remains schema `1`. The native boundary retains thirteen exports.

Compiler inference already captures every in-scope kind and validates its typed attributes. CPU is
the complete behavioral fallback. No new Tensor API or Compiler rewrite is needed. Generated
backward graphs are deliberately outside this forward/no-grad slice as described above.

Installed SDK headers expose all six direct comparison selectors, NaN-propagating pointwise
minimum/maximum selectors, `clampWithTensor:minValueTensor:maxValueTensor:name:`, NaN-propagating
reduction minimum/maximum selectors, exact-data graph constants, and cumulative sum/product
selectors with axis/exclusive/reverse available from macOS 13. Header names establish only API
shape, not a complete-domain numerical contract, Model conformance, dispatch count, or temporary
memory. A bounded run cannot fill that proof gap.

Historical evidence is retained only as regression-corpus motivation, not as a current pass or
complete-domain proof:

- Task 0010 found minimum subnormals comparing equal to signed zero. Model 0030 now permits the
  observed DAZ alternative, but all predicates still need a complete-domain route proof and the
  fresh bounded regression gate.
- Task 0012 found NaN-propagating reduction extrema order-dependent for opposite zeros and for
  zero/minimum-subnormal pairs. DAZ can admit some subnormal ties, but it does not relax NaN or the
  fixed MIN/MAX signed-zero winner. Direct extrema therefore needed exact custom candidates in the
  same sole numerical invocation and survivor adjudication.
- Task 0013 found cumulative SUM/PRODUCT flushing subnormal results. The current recursive aggregate
  floor permits per-site DAZ/FTZ and reassociation, but the old sample does not prove the opaque
  selectors for every rank, extent, axis, mode, ordered prefix, or parenthesization.

## Per-operation four-gate ledger

No row advances by family analogy. The Result records structural/API, complete-domain, and numerical
verdicts separately for every operation kind and candidate. Gate 3 applies ADR 0022 to the 16 custom
survivors; Gate 4 freezes one authorized production route per kind.

| Operation rows | Gate 1: structural/API plus complete-domain candidates | Gate 2: bounded regression rule | Gate 3: survivor adjudication | Gate 4: fixed-route rule |
|---|---|---|---|---|
| six comparisons | corresponding direct selector and auditable exact custom predicate kernel | exact predicate over represented/DAZ operand alternatives; canonical byte `0/1` | each has one survivor; no comparison required | freeze the sole survivor |
| tensor MIN/MAX | direct NaN-propagating selector and auditable exact custom pointwise kernel | exact candidate membership, NaN class, and operation-fixed zero sign in both operand orders | each has one survivor; no comparison required | freeze the sole survivor |
| scalar MIN/MAX | direct NaN-propagating selector against exact-data scalar constant and auditable exact custom scalar kernel | same exact extrema rule including scalar raw-bit identity and broadcast | each has one survivor; no comparison required | freeze the sole survivor |
| CLAMP | direct clamp; auditable fused custom; two-extrema MPSGraph composition if both primitive routes are `DOMAIN-PASS`; two-step custom-extrema composition | recursive exact `MIN(MAX(x,min),max)` membership with NaN and both zero-bound orders | fused strictly dominates composed: `1` versus `2` dispatches and `0` versus `4,194,304` route-owned bytes | freeze fused solely on structural dominance |
| reduction MIN/MAX | direct NaN-propagating selector and auditable exact custom contributor traversal | exact mapped contributors, NaN, signed-zero winner, original-candidate selection, and identity form | each has one survivor; no comparison required | freeze the sole survivor |
| CUM_SUM/CUM_PROD | corresponding direct cumulative selector and auditable exact custom ordered scan | recursive ordered-prefix/contiguous-split result set with exact identity/placement and abstract NaNs | each has one survivor; no comparison required | freeze the sole survivor |

Custom candidates are fixed typed kernels, not a general framework. Predicate/extrema kernels use
auditable bit classification/order and preserve selected source bits. Reduction kernels use exact
checked coordinate mapping and a documented deterministic contributor traversal. Scan kernels use
checked rank/axis indexing and a documented sequential ordered traversal, which is one legal
left-associated contiguous-split parenthesization in every mode. Both concrete CLAMP compositions
retain the two required FLOAT32 extrema sites. All listed candidates are defined before Gate 1B;
every `DOMAIN-PASS` candidate is prepared before the sole oracle. A candidate that cannot obtain
`DOMAIN-PASS` is recorded as blocked and never reaches numerical evidence or survivor adjudication.

## Gate 1A: structural/API and schema proof

Before any device execution, prove in disposable host construction and Java/native preflight:

1. every named installed selector exists on the supported deployment floor and maps operand order,
   broadcast, axis, keep-dimensions realization, exclusive/reverse flag, identity, and output type
   exactly;
2. exact scalar constants are created from four raw FLOAT32 bytes, never through a lossy decimal,
   string, `double`, or host arithmetic conversion;
3. direct reduction output reshaping changes only Shape metadata needed for exact keep-dimensions;
4. every typed custom and composed candidate compiles with safe math and no fast-relaxed option,
   preserves its declared arithmetic sites/source candidates, and exposes a complete resource
   ledger;
5. every malformed type, rank, axis, flag, attr, sentinel, padding, topology, or geometry case is
   rejected independently in Java and native preflight; and
6. capability remains false and no staged candidate is reachable through public preparation.

## Gate 1B: distinct complete-domain proof prerequisite

Before the bounded device corpus, each candidate must have an independently reviewable proof that
its complete output set is a subset of the Model set for every authorized descriptor and attribute:
all ranks `1..16`, every checked positive extent, every valid broadcast, every normalized reduction
axis set/order/keep-dimensions form, and every scan axis plus exclusive/reverse mode.

- A direct opaque selector needs an authoritative, version-applicable selector contract that states
  all required mapping, special-value, ordering/winner, arithmetic-site, and shape-dependent
  behavior. Installed declarations, operation names, historical runs, and a new bounded sample are
  not that contract. Without it the direct candidate is `DOMAIN-BLOCKED`, even if every sample
  passes.
- A custom route needs auditable source plus a complete mapping/algorithm proof: checked launch and
  index arithmetic, rank/axis/broadcast coverage, bounds, exact contributor order/membership,
  primitive sites, DAZ/FTZ allowances, NaN abstraction, selected-bit behavior, and the absence of an
  opaque library call whose result set is unproved.
- A composed route needs a site-by-site proof from already `DOMAIN-PASS` primitive candidates,
  including exact FLOAT32 intermediate materialization/rounding. An opaque fusion claim or final
  output sample is insufficient.

Record a proof artifact identity and `DOMAIN-PASS`/`DOMAIN-BLOCKED` verdict for every candidate and
operation before Gate 2. If an operation has no `DOMAIN-PASS` candidate, that operation remains
Blocked and capability-false; do not narrow its rank, extent, axis, mode, or edge-value domain
inside this task. The Gate-2 corpus is regression evidence only and can reject a proof or
implementation, never establish complete-domain conformance.

Against the current unchanged baseline, the intended cutover is schema `11 -> 12`, retaining wires
`1..19` and appending:

```text
20 GT                 21 GE                 22 LT
23 LE                 24 EQUAL              25 NOT_EQUAL
26 TENSOR_MIN         27 TENSOR_MAX         28 SCALAR_MIN
29 SCALAR_MAX         30 CLAMP              31 REDUCTION_MIN
32 REDUCTION_MAX      33 CUM_SUM            34 CUM_PROD
```

Append attribute wires `SCALAR_VALUE=7`, `CLAMP_RANGE=8`, and `SCAN=9`:

- scalar value: `attribute_count=1`, one zero-extended raw uint32 FLOAT32 word;
- clamp range: `attribute_count=2`, ordered lower/upper raw uint32 words;
- scan: normalized axis in the axis cell, `attribute_count=2`, exact Boolean words for exclusive
  then reverse; and
- comparison/tensor extrema use NONE; aggregate extrema reuse REDUCTION.

Keep ABI `4`, 160-byte records, thirteen exports, value types/states, and complete-plan wrapper `1`.
Advance workload signature, exact default policy, candidate/compatibility/route-policy schemas, and
codec together `12 -> 13`; stale bytes fail closed with no dual decoder or alias.

These wire/version values are not reserved by this blocked plan. Task 0051 implemented none of its
provisional schema-12/wire-20/version-13 values. If Task 0051 lands before this task is authorized,
0052 must be revised from the new baseline before any implementation: retain landed EXP/SIGMOID
wires, append this vocabulary after them, and advance identities once. The tasks must never race or
ship two meanings for one wire/version.

## Gate 2: exactly one fresh real-device numerical invocation

After Gates 1A and 1B pass and capability is still false, compile every listed `DOMAIN-PASS`
candidate before launch. Run exactly one real-device process invocation using one default-device
context, production-default MPSGraph optimization, read-back
`MPSGraphReducedPrecisionFastMathNone`, safe custom math, direct supplied targets, input identity
controls, output canaries, and exact Shape/type checks. There is no retry, second context,
optimization matrix, independent-executable matrix, per-failure rerun, or additional device-oracle
execution beyond the one recorded in Result. This bounded corpus is regression/device/toolchain
evidence only; it does not authorize ranks, extents, broadcasts, axes, or modes absent from the
sample.

### Frozen pointwise corpus

Use these ordered raw-word pairs for all six comparisons and tensor MIN/MAX:

```text
00000000/80000000  80000000/00000000  00000001/00000000  00000000/00000001
80000001/80000000  80000000/80000001  007fffff/00800000  00800000/007fffff
807fffff/80800000  80800000/807fffff  3f800000/3f800000  3f800000/3f800001
bf800000/bf7fffff  7f7fffff/7f800000  ff7fffff/ff800000  7f800000/ff800000
7fc12345/3f800000  3f800000/7fc12345  7fa12345/bf800000  bf800000/7fa12345
ffc54321/00000000  00000000/ffc54321  7fc12345/ffc54321  40000000/c0000000
```

In the same executable group add one exact ordinary finite broadcast witness with Shapes `[2,1,3]`
and `[1,4,3]` and target `[2,4,3]`. The validator enumerates independent same-signed-zero DAZ
choices only at subnormal operands. Comparison outputs must belong to the resulting exact Boolean
set and be canonical bytes. Extrema outputs must satisfy the Model winner rule and equal an original
candidate raw word; NaN requires classification only.

For scalar MIN/MAX, apply each scalar word
`00000000,80000000,00000001,80000001,3f800000,bf800000,7f800000,7fc12345`
to the ordered de-duplicated words from the pair corpus. For CLAMP, use exactly these ordered bound
pairs:

```text
bf800000/3f800000  80000000/00000000  00000000/80000000  00000001/3f800000
bf800000/80000001  7fc12345/3f800000  bf800000/7fc12345  7f800000/7f800000
```

Each scalar/bound case is a separate graph output in the same one process invocation, not a device,
optimization, or retry matrix.

### Frozen aggregate-extrema corpus

Use one `[2,3,8]` FLOAT32 tensor whose six logical rows are:

```text
00000000 80000000 00000001 80000001 00800000 80800000 3f800000 bf800000
80000000 00000000 007fffff 807fffff 7f7fffff ff7fffff 7f800000 ff800000
7fc12345 3f800000 bf800000 7fa12345 00000000 80000000 40000000 c0000000
bf800000 3f800000 80800000 00800000 80000001 00000001 80000000 00000000
ff800000 7f800000 ff7fffff 7f7fffff 807fffff 007fffff 00000000 80000000
c0000000 40000000 80000000 00000000 ffc54321 bf800000 3f800000 ffa54321
```

For both MIN and MAX, create full, axis `1` with each keep-dimensions value, ordered multi-axis
`[2,0]` with each keep-dimensions value, and empty multi-axis identity outputs. Validate contributor
mapping, output Shape, canaries, NaN classification, fixed zero sign, and exact source-word
membership. DAZ may affect comparisons/ties but may not synthesize a non-source result.

In the same invocation, create distinct full-reduction inputs and direct outputs for both MIN and
MAX over the exact two-element tensors `[00000000,80000000]` and
`[80000000,00000000]`. These cells contain no dominant nonzero value and therefore directly test
both opposite-zero operand orders, including MAX's required positive-zero result and MIN's required
negative-zero result. Also create separate rank-one singleton full reductions for each word
`00000000,80000000,00000001,80000001,7fc12345,7fa12345,ffc54321,ffa54321`.
Require exact raw signed-zero/subnormal output and NaN class output for both MIN and MAX. These are
actual candidate targets, not values merely present inside a larger dominated reduction.

### Frozen scan corpus

Use the same `[2,3,8]` words so each logical scan line covers signed zeros, both subnormal signs,
normal boundaries, cancellation, overflow candidates, opposing infinities, and NaNs in distinct
positions. For both CUM_SUM and CUM_PROD, emit all four exclusive/reverse combinations on axis `2`
and one additional inclusive-forward output on axis `1` as the non-innermost mapping witness.
Require exact Shape, identities (`+0` for exclusive sum and `+1` for exclusive product), direction,
placement, input preservation, canaries, and recursive result-set membership for every prefix.

For each slice the validator reverses the contributor sequence first when `reverse=true`, then maps
each logical output to its exact inclusive or exclusive ordered prefix. It memoizes only
order-preserving parenthesizations generated by contiguous interval splits; it never treats a
prefix as a multiset and never permutes leaves.

NaN is an abstract class at every scan arithmetic site. If an operand is NaN or the addition/
multiplication itself produces NaN (including opposite infinities for SUM or zero times infinity
for PRODUCT), that site may produce any binary32 NaN: payload, sign, and quiet/signaling state are
unconstrained. Any later site consuming that abstract NaN remains abstract NaN. A NaN final output
therefore requires class only, while its exact prefix position is still mandatory. Every non-NaN
site and result retains the exact ordered-tree raw-bit DAZ/round/FTZ rules.

The host validator computes reachable sets from exact binary32 raw words plus that single abstract
NaN token. It never compares to a sequential host fold, a whole-prefix ULP envelope, `double`, or
native host fast math.

The raw report must name each operation and candidate, complete-domain proof identity/verdict,
corpus and output SHA-256, structural/control verdicts, maximum reachable-set size, DAZ/FTZ
observations where applicable, first exact failure, and `PASS`/`FAIL`. Record the unedited report
text and SHA-256 in this task. Preserve the exact custom source, host source, validator, audit,
wrapper, and proof as non-production review evidence; remove the compiled executable, raw output,
bytecode/cache, and temporary workspace artifacts.

## Gate 3: custom-survivor structural adjudication and retained timing diagnostics

ADR 0022 admits exact source-bound route declarations for the surviving explicit custom routes.
Their retained source owns every encoder dispatch and resource used by a synchronous hot invocation:

- every comparison, tensor/scalar-extrema, fused-CLAMP, reduction, and scan candidate declares one
  compute dispatch and zero route-owned temporary bytes above steady prepared inputs/outputs;
- composed-custom CLAMP declares two compute dispatches and one `FLOAT32` intermediate of exactly
  `4 * elementCount` bytes; its frozen 1,048,576-element workload therefore declares `4,194,304`
  route-owned temporary bytes.

Any framework call, unowned encoder/resource, hidden allocation, or source mismatch invalidates the
declaration. No direct selector, MPSGraph composition, graph-node count, command-buffer count,
inferred fusion, heap estimate, or process resident memory substitutes for actual opaque-route
facts.

The historical Apple M3 Max process and all raw samples remain retained unchanged. It prepared all
custom candidates, exact-checked results, used the documented workloads, ran four alternating-order
warmup rounds and eight retained rounds to the 25 ms floor, discarded and retried nothing, and
recorded the original durations and environment. Those timings are local-device diagnostics only.
They do not qualify a candidate, select a route, change tuning identity, populate a production
decision, or break a structural tie.

## Gate 4: fixed production route

Apply ADR 0022 independently to each operation kind:

1. remove every candidate without structural/API `PASS`, `DOMAIN-PASS`, or numerical `PASS`;
2. if one candidate remains, freeze it without a comparative cost gate;
3. if multiple candidates with identical proven semantics/domain remain, select without timing only
   when one has no more dispatches and no more route-owned temporary bytes than every alternative
   and is strictly better in at least one dimension;
4. if multiple nondominated candidates remain, leave route selection pending an explicitly
   authorized controlled environment; and
5. encode an authorized survivor as one immutable production route and remove unselected
   production implementations after retaining required evidence.

Fourteen rows had one survivor. CLAMP had two: fused custom strictly dominates composed custom by
one versus two dispatches and zero versus 4,194,304 route-owned temporary bytes. No timing fact is
needed or authoritative. Do not aggregate gates across operations: one kind never authorizes
another.


## Production lifecycle and proof

For operations that pass all four gates, preserve current whole-partition analysis, stable ordered
values/nodes/feeds/targets, declarations-before-assignment, transactional finalization, immutable
splats, context child leases, direct assigned targets, run-owned outputs/address workspace,
synchronous invocation, reuse, concurrent/session isolation, rollback, close rejection,
publication, transfer, trace, and tuning behavior. Add no hot-path check, retry, fallback, host
staging, hidden materialization, per-node native call, or general custom-kernel framework.

Focused proof must cover capability positives/negatives; Java/native schema agreement; stale
identity rejection; exact scalar raw-word round trips; every reduction form and scan mode; mixed
partitions with existing canonical nodes; canonical BOOL publication; rank-zero FLOAT32 extrema
publication; malformed geometry/topology rejection; repeat/concurrent/session isolation; and one
real public Engine forward scenario for each completed partition with no CPU owner or skip.

## Dependencies and integration

- Depends on: completed and independently approved Model 0030/0031; Metal 0050 packaged baseline;
  retained Metal 0010/0012/0013 evidence; current Compiler inference/autograd capture; Config 0006;
  Engine 0018.
- Gate-3 prerequisite: resolved by ADR 0022 for only the closed custom survivors. Opaque routes
  would require supported actual observation for structural facts; none reached adjudication.
- Complete-domain gate: closed for every auditable custom candidate and still `DOMAIN-BLOCKED` for
  every opaque direct candidate; the per-candidate proof identities and verdicts are in Result.
- Independent of: Task 0051/0053 numerical candidates, timing, route, schema implementation, and
  result. Task 0053 remains separately Blocked.
- Planning base: `e9f5595f`; proof/oracle execution base:
  `bd5c40d3f08487cddec4284ac1588ce276f1c979`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: none.
- Integration order: architecture/authorization; one custom-only Gate-3 process and raw evidence;
  Gate-4 route freeze/loser deletion; production schema/lifecycle/capability; focused proof;
  documentation; logical commits; independent Class C review.
- Integration validation: focused Metal tests, backend conformance, Compiler contract checks, and
  no-skip real-dylib public Engine scenarios; do not run the final full repository build in this
  task execution.
- Shared-document integration owner: Task-0052 implementer.

## Expected future files and symbols

The authorized implementation may update:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  `MetalNegPartitionPreparer`, preparation/finalization/execution types, and affected package
  Javadocs;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, decisions, and
  complete-plan compatibility tests;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and its README;
- focused capability/schema/native/prepared/candidate/codec tests, backend conformance, Compiler
  contract checks, and real public Engine integration; and
- current architecture/Metal/backend/API/glossary explanatory docs, this task, Metal master, and
  roadmap.

No Task-0051/0053 task file, oracle evidence, candidate source, or result is an implementation
input. The exact Task-0052 Gate-1/Gate-2 source/proof package remains immutable under
[`../evidence/0052/`](../evidence/0052/README.md). Gate-3 raw evidence is retained separately;
compiled probes, executables, caches, and temporary workspaces are removed after adjudication.

## Acceptance criteria

1. Gate 1A/API and Gate 1B were recorded independently for every candidate. Opaque direct routes are
   `DOMAIN-BLOCKED`; all auditable custom routes are `DOMAIN-PASS`.
2. Exactly one fresh real-device numerical process invocation ran over the frozen bounded group
   with no retry or matrix, including dedicated opposite-zero and singleton aggregate outputs plus
   ordered-prefix scan validation with abstract NaNs.
3. Every `DOMAIN-PASS` custom candidate compiled before command-buffer creation and passed its own
   numerical verdict. The bounded sample remains regression evidence and does not establish the
   complete-domain proof.
4. Gate 3 retains actual M3 Max timings as non-authoritative diagnostics and exact source-owned
   dispatch/temporary-byte declarations. Opaque candidates receive no inferred facts.
5. Gate 4 fixes each sole survivor without comparison and fixes fused CLAMP solely because its
   `1` dispatch/`0` route bytes strictly dominate composed CLAMP's `2` dispatches/`4,194,304`
   route bytes. Raw timing does not authorize a route.
6. The original cutover appended schema-12 operation wires `20..34` and attributes `7..9` and
   advanced backend-local identities to 13; Complete Task 0055 later migrated them unchanged into
   ABI 5/schema 13 and identity 14.
7. Capability, Java/native preflight and lowering, cold preparation, persistent resources, exact
   BOOL publication, scalar raw words, reductions, scans, and synchronous execution cover all 15
   kinds over the exact domain. Every exclusion remains false.
8. Focused Metal, conformance, Compiler-contract, and no-skip real-dylib public Engine scenarios
   pass. The final full repository build is deliberately deferred.

## Documentation and review impact

The authorization and evidence commits update architecture, this task, the Metal master frontier,
the roadmap frontier, and permanent Task-0052 Gate-3 evidence. They do not yet change production
capability, schema/identity, public Engine behavior, native guide, or package Javadoc. The
production commits must synchronize those surfaces while preserving every historical blocked
record. Independent Class C review must inspect every surviving custom candidate's
structural/domain/numerical evidence, exact route facts, diagnostic timing record, retained indexing
and numerical invariants, ordered-prefix/abstract-NaN oracle, fixed routes, false exclusions,
stale-data rejection, lifecycle, real no-skip Engine proof, and disposable-artifact boundaries.

## Result

Gates 1 through 4 and production implementation are complete. Gate 1A/API, Gate 1B complete-domain
proof, and the sole Gate-2 numerical invocation completed from clean execution base
`bd5c40d3f08487cddec4284ac1588ce276f1c979`. Gate 3 retained exact structural facts plus local
timing diagnostics; Gate 4 fixed all 15 routes under ADR 0022 without using timing authority.

The installed SDK declarations gave API `PASS` to every opaque direct comparison, pointwise/scalar
extrema, clamp, reduction-extrema, and cumulative SUM/PRODUCT candidate. They do not specify the
complete Task-0052 raw-word domain: NaN class behavior at every site, signed-zero winner, DAZ/FTZ
freedom, exact selected bits, reduction membership/order, or scan parenthesization. Every opaque
direct candidate is therefore `DOMAIN-BLOCKED` and was not staged or numerically run. The MPSGraph
CLAMP composition is also `DOMAIN-BLOCKED` because both opaque primitive dependencies are blocked.

Every auditable custom candidate is `DOMAIN-PASS`. The retained proof covers checked positive
geometry, rank-`1..16` coordinate decoding, 64-bit right-aligned broadcast offsets, checked
three-dimensional launch decomposition, total raw-bit predicate ordering, source-bit-preserving
NaN/zero extrema, two exact CLAMP sites and intermediate materialization, exact reduction
contributor mapping/traversal, and scan line/direction/prefix placement. The scan loop is one legal
left-associated contiguous-split tree, compiles with `MTLMathModeSafe`, copies singleton
contributors without an arithmetic site, and uses exact exclusive identities. The independent
validator recursively enumerates every ordered contiguous interval split, independent operand DAZ,
binary32 round-to-nearest-even, independent subnormal-result FTZ to either signed zero, and a single
abstract NaN class. Thus the bounded run checked membership in the complete permitted set rather
than one CPU fold. The exact proof artifact is
[`domain-proof.txt`](../evidence/0052/domain-proof.txt), SHA-256
`589cb804a00c35ea4589818eeedafcc238d4b91c2ec07961b561eabda9a69153`; the
[review guide](../evidence/0052/README.md) makes the aggregate/scan formulas,
one-to-one/bounds/contributor-order arguments, and ordered-prefix DAZ/RNE/FTZ/NaN invariants
explicit.

Before the device run, the audited package compiled cleanly with warnings as errors and proved one
`MTLCreateSystemDefaultDevice`, one queue, one command buffer, one commit/wait, all 15 custom entry
points compiled before command-buffer creation, safe math, frozen-corpus agreement, checked
indexing, input snapshots, and guarded outputs. The exact review artifacts were later recovered
from the original execution-session writes/edits and every recomputed hash matched its recorded
pre-run identity:

| Artifact | SHA-256 |
|---|---|
| [custom Metal source](../evidence/0052/oracle.metal) | `ff15f63c9d2e54d62fa2157e1c424cc00acb90d5355101b18a07175e789a40c0` |
| [Objective-C++ staging harness](../evidence/0052/oracle.mm) | `6db1c52113482753fcc96d3c14ffcee0479d9ca7b616e3a69f603d7ccb4c9683` |
| [exact offline validator](../evidence/0052/validate.py) | `94fd86bb6bf16b00195f990979ff892e22c8fac6cfe024173bae0bdf27f83e08` |
| [pre-run source/corpus/control audit](../evidence/0052/audit.py) | `0c522be1e4f93f82576ad674a07d411c0c68a1c181ea88ac0724f56cfefc1d14` |
| [audited single-invocation wrapper](../evidence/0052/run-once.py) | `ec73cd5f6cef831f10fbd2300ca3bf6d3e0796920aa58e402dd56ac0b6481046` |
| compiled oracle executable (identity only; not retained) | `9840a0987391a8d7ddfb6a8fbd2ea9923681ca0e911d5f8bece129267e06fc51` |

Exactly one fresh Apple M3 Max oracle process invocation then ran. It used one default-device
context, one command buffer and compute encoder, safe custom math, 15 precompiled pipelines, 16
snapshotted inputs, and 98 guarded raw outputs. It exited `0`; standard error was empty; all inputs
were preserved; every canary passed; and there was no retry, second process, context, command
buffer, optimization/capability/independent-executable matrix, timing, or capture. Corpus SHA-256
was `8b44313d15bf054d2f05430383258f6f0b19ab03743810c071b0a9be54f7ea5f`;
raw device-output SHA-256 was
`f316906cd30cfb907b5a5005aaaea90e2f7b77381887e86ff7f62ee9af5a33d2`; the empty standard-error
SHA-256 was `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
and the exact raw offline-verdict SHA-256 was
`7d551adbdcd149015921c9d5197111cb4cb2291c008fbed2af095ed5031ef5b4`.
The custom comparison/extrema/CLAMP/reduction routes use no floating arithmetic sites and returned
exact represented source/Boolean members, so DAZ/FTZ observation is not applicable to their
implementation path. The scan validator admitted the frozen per-site DAZ/FTZ choices and observed
maximum reachable-set sizes `9` for SUM and `2` for PRODUCT.

The unedited offline verdict was:

```text
VALIDATOR=task0052-exact-v1
CORPUS_SHA256=8b44313d15bf054d2f05430383258f6f0b19ab03743810c071b0a9be54f7ea5f
RAW_OUTPUT_SHA256=f316906cd30cfb907b5a5005aaaea90e2f7b77381887e86ff7f62ee9af5a33d2
RAW_OUTPUT_COUNT=98
CANDIDATE=direct_cmp_gt API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_cmp_ge API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_cmp_lt API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_cmp_le API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_cmp_eq API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_cmp_ne API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_tensor_min API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_tensor_max API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_scalar_min API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_scalar_max API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_clamp API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=composed_mpsgraph_clamp API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_reduction_min API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_reduction_max API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_scan_sum API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=direct_scan_prod API=PASS DOMAIN=BLOCKED NUMERICAL=NOT_RUN REASON=opaque_no_authoritative_complete_domain_contract
CANDIDATE=custom_cmp_gt API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=e9ef9d3352b0f89e1d22b11f78c8dcbe61912f132bd3688a9f5a5e792db5025e CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_cmp_ge API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=40b8269968eb1f3f39c560181587c4ee3c52a072306c5cee3cb190cb1da11da4 CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_cmp_lt API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=1ba3a8d639b0cf8add279b98644d2e9cbd1c6d4f57a00c303a98dd8304941a56 CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_cmp_le API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=6fc3db8432839bd5735410188889d3350ca7701ea0ed97be3ea5d83f85dcd551 CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_cmp_eq API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=c766a88c1715ed358d7822bceba7c174448f9a320e1b7b2c261f1a5c9ce9f94d CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_cmp_ne API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=334a2e4459c07c670e2cab40a23adfe8fc715a108c76c1fee5feaa3008f3d52b CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_tensor_min API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=00091d91318e8d3afa00c51b114669b8d452421c6663717e29fd8a4c9eccf0a7 CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_tensor_max API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=06d221248d15a16bb69c35dde57d2f161bae3e4756f3cc493231b11eaf6a1ed5 CASE_OUTPUTS=2 FAILURES=0
CANDIDATE=custom_scalar_min API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=7ceb233e0015dc48a27dd7135d64449e533b76dd64bbed11a493783a5e9bcd1b CASE_OUTPUTS=8 FAILURES=0
CANDIDATE=custom_scalar_max API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=77efd14123cd6c7ead0b14ac83c36f400f6856354d2c6aada6a20d2d7363f85e CASE_OUTPUTS=8 FAILURES=0
CANDIDATE=custom_clamp_fused API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=2df4b16ad0a3a2ca8ca43c80bdace3f3ee87072219620a4f46eb79251c8896d7 CASE_OUTPUTS=8 FAILURES=0
CANDIDATE=composed_custom_clamp API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=c9457207006154d40137d743b453fb07dbccb4f3e8d993b7f13d5a36ed19b496 CASE_OUTPUTS=16 FAILURES=0
CANDIDATE=custom_reduction_min API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=8a50400a1a0e4b40b0341160f7bfa438209e2ba7ea8d3d1ed9108ca193f35229 CASE_OUTPUTS=16 FAILURES=0
CANDIDATE=custom_reduction_max API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=4650fb36d913d0adc983c75d06632611a2e22c49bc89bb5441e6678945bddb64 CASE_OUTPUTS=16 FAILURES=0
CANDIDATE=custom_scan_sum API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=8babefbff077ab0834eaef266d3c9c3202a4a154a81d3835ee6268494e9d80c7 CASE_OUTPUTS=5 MAX_REACHABLE_SET=9 FAILURES=0
CANDIDATE=custom_scan_prod API=PASS DOMAIN=PASS NUMERICAL=PASS OUTPUT_SHA256=c87a30f1218d651fd7eb9d311ede39874e6188e1f6290cb49291e4453e6aa316 CASE_OUTPUTS=5 MAX_REACHABLE_SET=2 FAILURES=0
ORACLE_VERDICT=PASS FAILURES=0
```

ADR 0022 applies to the 16 closed custom survivors. The permanent
[Gate-3 evidence](../evidence/0052-gate3/README.md) retains source audit, the one-run wrapper, all
128 raw timing samples, and exact source-owned dispatch/resource declarations. One Apple M3 Max
process prepared all survivors, exact-checked the cost workload, preserved every input, ran four
alternating-order warmup rounds and eight retained rounds to the 25 ms batch floor, exited zero,
and wrote no standard error. It did not rerun the numerical oracle. The raw evidence SHA-256 is
`4ec49dac639f27389bf276886b9f82eac28fbcf85ea0d1bf73ddbda6d3e92f64`. Those raw timing values are
preserved historical diagnostics and are not qualification, route-selection, or tuning-identity
authority.

Fourteen operations had one surviving route, so no comparative cost gate applied. CLAMP had two
identical-semantics/domain survivors. Fused custom CLAMP is the production route solely because its
exact one dispatch and zero route-owned temporary bytes strictly dominate composed custom CLAMP's
two dispatches and `4,194,304` route-owned temporary bytes. The recorded medians are not part of
that justification. Opaque direct and MPSGraph candidates remain `DOMAIN-BLOCKED` and received no
inferred facts. The disposable Gate-3 executable was removed before production work.

The original production cutover appended schema-12 operation wires `20..34` and attributes `7..9`,
advanced backend-local identities to 13, and retained ABI 4. Task 0055 later migrated those exact
routes into ABI 5/schema 13 and identity 14. The whole-partition custom route retains declared
run-owned logical-value buffers, direct assigned targets, one synchronous Java/native invocation,
and no host staging, hidden materialization, retry, fallback, per-node downcall, or hot compilation.

Focused capability, schema/native, prepared-execution, conformance, Compiler-contract, lifecycle,
stale-identity, malformed-state, concurrent-session, and CPU-free public Engine checks passed.
Review remediation rejects duplicate physical handles across distinct live value-table entries at
both Java binding and the native boundary. The warnings-as-errors bridge and local package verify
with exactly thirteen exports. The final full repository build was not run, as required by this
task.
