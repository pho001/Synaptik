# Task 0052: ACCELERATOR FLOAT32 Comparisons, Extrema, and Scans

## Status

Blocked

Planning is complete, but execution is not authorized while supported installed tooling cannot read
actual GPU compute-dispatch and peak transient-resource facts. This task does not depend on Task
0051's EXP/SIGMOID implementation, oracle verdict, or route choice. It shares only that external
instrumentation prerequisite and must not edit, reinterpret, or resume Task 0051.

## Change class

Class C — eventual implementation would extend profile-qualified capability, exact BOOL
publication, scalar-attribute and scan schema meaning, typed native lowering, candidate identity,
prepared execution, and public Engine evidence. It must not change Model semantics, public API,
Compiler generation, Runtime policy, native ABI exports, or the completed package contract.

This planning slice changes only this task, the Metal master plan, and the roadmap. It runs no
device oracle, timing workload, capture, native build, or production test and leaves every new
capability false.

## Goal

Add one bounded, forward-only, canonical `FLOAT32` operation slice under
`NumericalProfile.ACCELERATOR`:

- all six `BinaryComparisonKind` values;
- tensor `BinaryArithmeticKind.MIN` and `MAX`;
- scalar `ScalarElementwiseKind.MIN`, `MAX`, and `CLAMP`;
- ordinary `AggregateReductionKind.MIN` and `MAX`; and
- `CumulativeScanKind.CUM_SUM` and `CUM_PROD`.

Every operation must independently pass four gates in order: structural/API with a distinct
complete-domain proof prerequisite, numerical regression evidence, cost with actual dispatch and
transient-resource evidence, and fixed production route. Capability remains false for an operation
until all four gates pass for its complete authorized domain. A failure does not authorize a
narrower shape, attribute, mode, or edge-value subset.

Stage every predeclared direct, custom, and composed candidate needed for a real cost comparison
before the sole numerical invocation. Every candidate must independently prove complete-domain
coverage and pass the bounded regression corpus before cost. Route family is only the last cost
tie-breaker: direct MPSGraph before exact custom before certified composition. It is never a
preselection rule. Runtime selection, retry, fallback, and input-dependent routing are forbidden.

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
BOOL materialization path. CPU/Metal transfer remains FLOAT32-only: whole-partition analysis must
reject any comparison BOOL consumer, cross-owner BOOL edge, or BOOL feed. This task adds no Metal
BOOL ingress, logical operation, classification, `WHERE`, or BOOL-consuming node. Existing
supported FLOAT32 nodes may precede or follow in-scope FLOAT32 extrema/scans where current
canonical-state rules already permit it.

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

## Current Metal architecture and caller inventory

The current provider routes accelerator aggregate kinds to `supportsReduction`, MATMUL separately,
and all other accelerator candidates to `supportsBinary`. `supportsBinary` admits only tensor
`ADD/SUB/MUL/DIV`; `supportsReduction` admits only `SUM/MEAN/SUM_TO_SHAPE`. It has no scalar,
comparison, or cumulative-scan capability branch. The preparer and native lowering likewise contain
no node for this task.

The current private program is ABI `4`, fixed record size `160`, schema `11`, operation wires
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
  fixed MIN/MAX signed-zero winner. Direct extrema therefore need exact custom candidates in the
  same sole numerical invocation and cost comparison.
- Task 0013 found cumulative SUM/PRODUCT flushing subnormal results. The current recursive aggregate
  floor permits per-site DAZ/FTZ and reassociation, but the old sample does not prove the opaque
  selectors for every rank, extent, axis, mode, ordered prefix, or parenthesization.

## Per-operation four-gate ledger

No row advances by family analogy. `DOMAIN-PASS`, numerical verdict, cost facts, and the fixed route
are recorded separately for every operation kind and candidate.

| Operation rows | Gate 1: structural/API plus complete-domain candidates | Gate 2: bounded regression rule | Gate 3: measured facts for every surviving candidate | Gate 4: fixed-route rule |
|---|---|---|---|---|
| six comparisons | corresponding direct selector and auditable exact custom predicate kernel | exact predicate over represented/DAZ operand alternatives; canonical byte `0/1` | hot median, actual compute dispatches, peak transient bytes per candidate and predicate | all `DOMAIN-PASS`/numerical-pass candidates compete; route family only breaks a complete cost tie |
| tensor MIN/MAX | direct NaN-propagating selector and auditable exact custom pointwise kernel | exact candidate membership, NaN class, and operation-fixed zero sign in both operand orders | same three facts per kind/candidate | same common ranking |
| scalar MIN/MAX | direct NaN-propagating selector against exact-data scalar constant and auditable exact custom scalar kernel | same exact extrema rule including scalar raw-bit identity and broadcast | same three facts per kind/candidate | same common ranking |
| CLAMP | direct clamp; auditable fused custom; two-extrema MPSGraph composition if both primitive routes are `DOMAIN-PASS`; two-step custom-extrema composition | recursive exact `MIN(MAX(x,min),max)` membership with NaN and both zero-bound orders | same three facts for every complete passing candidate | same common ranking; direct/custom/composed is only the final tie-break order |
| reduction MIN/MAX | direct NaN-propagating selector and auditable exact custom contributor traversal | exact mapped contributors, NaN, signed-zero winner, original-candidate selection, and identity form | same three facts per kind/candidate on fixed multi-axis workload | same common ranking; no pointwise reduction decomposition |
| CUM_SUM/CUM_PROD | corresponding direct cumulative selector and auditable exact custom ordered scan | recursive ordered-prefix/contiguous-split result set with exact identity/placement and abstract NaNs | same three facts per kind/candidate on one fixed reverse-exclusive workload | same common ranking |

Custom candidates are fixed typed kernels, not a general framework. Predicate/extrema kernels use
auditable bit classification/order and preserve selected source bits. Reduction kernels use exact
checked coordinate mapping and a documented deterministic contributor traversal. Scan kernels use
checked rank/axis indexing and a documented sequential ordered traversal, which is one legal
left-associated contiguous-split parenthesization in every mode. Both concrete CLAMP compositions
retain the two required FLOAT32 extrema sites. All listed candidates are defined before Gate 1B;
every `DOMAIN-PASS` candidate is prepared before the sole oracle. A candidate that cannot obtain
`DOMAIN-PASS` is recorded as blocked and never reaches numerical or cost evidence.

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
optimization matrix, independent-executable matrix, per-failure rerun, or device-oracle execution
in this planning commit. This bounded corpus is regression/device/toolchain evidence only; it does
not authorize ranks, extents, broadcasts, axes, or modes absent from the sample.

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
text and SHA-256 in this task, then remove probe source, binary, raw output, validator, and generated
artifacts.

## Gate 3: one bounded cost-evidence group

Only `DOMAIN-PASS` plus numerical-`PASS` candidates enter this gate. Every surviving direct,
custom, and composed candidate runs in the same bounded comparison; route family does not filter or
preselect candidates. Run one process containing a predeclared standalone workload per operation
kind/candidate, not an input-size, axis, mode, optimization, or device matrix:

- comparisons, tensor extrema, and scalar extrema/clamp: canonical 1,048,576-element inputs;
- reductions: canonical `[64,128,128]`, ordered axes `[2,0]`, keep-dimensions false;
- scans: canonical `[1024,1024]`, axis `1`, exclusive/reverse true; and
- deterministic ordinary values seeded with the frozen edge corpus, with every retained result
  exact-checked before timing.

Prepare every route first. Use four alternating-order warmup rounds and eight alternating-order
retained rounds; each retained batch runs to a 25 ms floor with a 1,000,000-execution ceiling.
Discard and retry nothing. The synchronous hot boundary is route execution plus result closure;
exclude preparation, upload, download, validation, and reporting.

After uncaptured timing, capture exactly one hot logical execution for every passing candidate and
read the trace with supported installed tooling. Record raw batch durations/iterations, normalized
medians, trace SHA-256, actual framework-internal compute-dispatch records, and peak transient Metal
resource bytes above steady prepared inputs/outputs. A backend-call count, graph-node count,
command-buffer count, inferred fusion, heap estimate, process RSS, or custom resource ledger is not
a substitute. Missing actual dispatch or transient facts blocks that operation and leaves its
capability false.

## Gate 4: fixed production route

Apply this predeclared rule independently to each operation kind:

1. remove every candidate without structural/API `PASS`, `DOMAIN-PASS`, or numerical `PASS`;
2. require complete timing, actual compute-dispatch, and peak transient-byte facts for every
   remaining direct, custom, and composed candidate; missing facts block the operation;
3. select lexicographically by lowest retained normalized-duration median, then fewer actual
   compute dispatches, then fewer peak transient bytes;
4. only when all three measured facts tie, prefer direct MPSGraph, then exact custom, then certified
   composition, then lexical candidate ID;
5. no valid complete candidate, a partial-domain route, or any unmeasured survivor blocks that
   operation and leaves capability false; and
6. encode the winner as one immutable production route and delete every unselected custom kernel,
   composition, probe, benchmark, capture, and report artifact.

There is no direct-route baseline, coverage-only custom tier, 10% override, or family
preselection. Every numerically complete survivor can win on measured cost; route family is only
the final exact-tie discriminator.

Do not aggregate gates across operations. For example, passing `LESS_THAN` does not authorize
`GREATER_THAN`; passing MIN does not authorize MAX; passing CUM_SUM does not authorize CUM_PROD.
One operation may complete while another remains false only if schema, identity, documentation, and
public claims name that exact completed subset and independent review approves the split.

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
- External blocker: supported installed tooling that reads actual framework-internal GPU compute
  dispatches and peak transient Metal resource bytes from each route capture.
- Complete-domain blocker: no candidate yet has a recorded authoritative-selector or
  auditable-custom/composition proof for the complete Task-0052 domain; sample execution cannot
  substitute for it.
- Independent of: Task 0051 numerical candidates, timing, route, schema implementation, and result.
- Common planning base: `f2debef31f382db77c8c2758e56d646858838d0f`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: none.
- Integration order: this planning/remediation commit; instrumentation prerequisite; Gate 1A
  structural proof and separate Gate 1B complete-domain proof with capability false; exactly one
  Gate-2 regression process invocation; artifact removal/evidence record; one Gate-3 cost group over
  every complete passing candidate; Gate-4 route freeze/loser deletion; production
  schema/lifecycle/capability; focused proof; documentation; implementation commit; independent
  Class C review.
- Shared-document integration owner: future task implementer. Task 0051 and 0052 may cross-link only
  in the Metal master/roadmap until one is explicitly authorized as the sole production frontier.

## Expected future files and symbols

After the blocker resolves and this task is revised/authorized from the then-current base:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  `MetalNegPartitionPreparer`, and affected package Javadocs;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, decisions, and
  complete-plan compatibility tests;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and its README;
- focused capability/schema/native/prepared/candidate/codec tests, backend conformance, Compiler
  contract checks, and real public Engine integration; and
- current Metal/backend/API/glossary explanatory docs, this task, Metal master, and roadmap.

No Task-0051 task file, oracle evidence, candidate source, or result is an implementation input.
No probe, oracle output, capture, custom kernel, benchmark artifact, or cache remains unless Gate 4
selects code as the production route; raw evidence is transcribed into this task before deletion.

## Acceptance criteria

1. This planning/remediation commit inventories the complete exact operation partition, current
   Metal architecture/callers, all staged candidates, complete-domain prerequisite, corrected
   frozen corpus, four gates, identity cutover, and blockers; it runs no device oracle and changes
   no production file.
2. Every candidate first receives an authoritative-selector or auditable-custom/composition
   complete-domain proof over the full authorized ranks/extents/broadcasts/axes/modes. A bounded
   sample is rejection/regression evidence only; no proof means `DOMAIN-BLOCKED`.
3. Future execution then performs exactly one fresh real-device numerical process invocation over
   the frozen bounded group with no retry or matrix, including dedicated opposite-zero and
   singleton aggregate outputs plus ordered-prefix scan validation with abstract NaNs.
4. Every operation kind receives its own structural, domain-proof, numerical,
   actual-dispatch/transient cost, and fixed-route verdict; no family analogy or partial-domain
   narrowing authorizes capability.
5. Every complete numerical-pass direct/custom/composed candidate enters the same cost comparison;
   measured median/dispatch/bytes decide before direct/custom/composed is used only as an exact tie
   breaker. Production contains one fixed route with no runtime choice/fallback.
6. Schema/identity cutover is single-version, Java/native exact, and revised before implementation
   if Task 0051 consumes the current next values first.
7. All explicit exclusions remain false, especially strict profile, gradients, BOOL consumers,
   ARG_MIN/ARG_MAX/INT64, PROD reduction, affine inputs, and malformed rows.
8. Focused native/backend/conformance/Engine/documentation validation and independent Class C
   review pass before any operation status becomes Complete.

## Planning validation

Validate only planning scope:

```bash
git diff --check
```

Also validate changed Markdown links, anchors, fences, final newlines, status consistency, exactly
zero production/native/test/probe changes, and no device/native/build/test invocation. Do not run a
full repository build.

## Documentation and review impact

This commit updates only the new task, Metal master frontier, and roadmap frontier. Future
implementation synchronizes current capability, schema/identity, complete-domain proofs, route
decisions, gradient exclusions, public Engine evidence, native guide, package Javadocs, master plan,
and roadmap while preserving every historical blocked record. Independent Class C review must
inspect every candidate's structural/domain/numerical/cost evidence, ordered-prefix/abstract-NaN
oracle, fixed routes, false exclusions, stale-data rejection, lifecycle, real no-skip Engine proof,
and removed artifacts.

## Result

Planning-only and Blocked on supported readable GPU dispatch/transient-resource instrumentation;
no candidate also has a recorded Task-0052 complete-domain proof verdict yet. No real-device
oracle, timing, capture, native build, test, or production action ran. No schema, wire, attribute,
identity, capability, route, Java/native source, package, or generated artifact changed. All
comparison, extrema, and scan capability named here remains false pending future complete-domain
proof and four-gate execution.
