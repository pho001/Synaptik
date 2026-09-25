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

Every operation must independently pass four gates in order: structural/API, numerical, cost with
actual dispatch and transient-resource evidence, and fixed production route. Capability remains
false for an operation until all four gates pass for its complete authorized domain. A failure does
not authorize a narrower shape, attribute, mode, or edge-value subset.

Prefer a direct MPSGraph selector when it is structurally exact and passes the numerical gate. A
custom Metal route is eligible only where direct MPSGraph cannot provide exactness or complete
domain coverage. A composition is the last resort after direct and exact custom routes. Runtime
selection, retry, fallback, and input-dependent routing are forbidden.

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

Each scan output prefix uses exactly its declared contributors once in the declared traversal and
exclusive/inclusive placement. The oracle admits every result from any legal binary tree over that
exact prefix contributor multiset, with one binary32 round-to-nearest-even operation per arithmetic
site, independent operand DAZ, exact subnormal or either signed-zero FTZ at each result, and no
operation other than the corresponding addition or multiplication. It admits no dropped,
duplicated, invented, pretruncated, or wrong-prefix contributor and no final-output tolerance
envelope.

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
shape, not Model conformance, dispatch count, or temporary memory.

Historical evidence is retained, not treated as a current pass:

- Task 0010 found minimum subnormals comparing equal to signed zero. Model 0030 now permits the
  observed DAZ alternative, but all predicates still need a fresh bounded gate.
- Task 0012 found NaN-propagating reduction extrema order-dependent for opposite zeros and for
  zero/minimum-subnormal pairs. DAZ can admit some subnormal ties, but it does not relax NaN or the
  fixed MIN/MAX signed-zero winner. Direct extrema therefore need a custom exact contingency in the
  same sole numerical invocation.
- Task 0013 found cumulative SUM/PRODUCT flushing subnormal results. The current recursive aggregate
  floor permits per-site DAZ/FTZ and reassociation, so the direct selectors are semantically
  reachable only after a fresh all-contributors-once membership gate.

## Per-operation four-gate ledger

No row advances by family analogy. Evidence and the fixed route are recorded separately for each
operation kind.

| Operation rows | Gate 1: structural/API candidates | Gate 2: numerical rule | Gate 3: measured facts | Gate 4: fixed-route rule |
|---|---|---|---|---|
| six comparisons | corresponding direct MPSGraph selector | exact predicate over represented/DAZ operand alternatives; canonical byte `0/1` | hot median, actual compute dispatches, peak transient bytes for each selector | direct only; any failure blocks that predicate |
| tensor MIN/MAX | direct NaN-propagating selector; exact custom pointwise contingency | exact candidate membership, NaN class, and operation-fixed zero sign in both operand orders | same three facts for every passing candidate per kind | direct if valid; custom only if direct is invalid/incomplete |
| scalar MIN/MAX | direct NaN-propagating selector against exact-data scalar constant; exact custom contingency | same exact extrema rule including scalar raw-bit identity and broadcast | same three facts per kind/candidate | direct if valid; otherwise custom |
| CLAMP | direct clamp selector; exact fused custom contingency; ordered two-extrema MPSGraph composition last | recursive exact `MIN(MAX(x,min),max)` membership with NaN and both zero-bound orders | same three facts for every passing candidate | direct, then custom, then composition; never skip a valid higher-preference tier for speed |
| reduction MIN/MAX | direct NaN-propagating reduction selector; exact custom contributor traversal contingency | exact mapped contributors, NaN, signed-zero winner, original-candidate selection, and identity form | same three facts per kind/candidate on fixed multi-axis workload | direct if valid; otherwise custom; no pointwise decomposition |
| CUM_SUM/CUM_PROD | corresponding direct cumulative selector | recursive per-prefix all-contributors-once set with exact identity/placement | same three facts per kind on one fixed reverse-exclusive workload | direct only; failure blocks that scan kind |

A custom candidate must use exact bit classification/order for extrema and preserve selected source
bits. A custom reduction may use a deterministic traversal; it must not invent an approximate
numeric sentinel or canonicalize a selected non-NaN candidate. Custom scans are not staged: the
matching direct selectors have complete structural coverage and the current floor specifically
reopens their historical DAZ/FTZ behavior. If either direct scan fails membership, this task blocks
that kind rather than creating a second oracle cycle.

## Gate 1: structural/API and schema proof

Before any device execution, prove in disposable host construction and Java/native preflight:

1. every named installed selector exists on the supported deployment floor and maps operand order,
   broadcast, axis, keep-dimensions realization, exclusive/reverse flag, identity, and output type
   exactly;
2. exact scalar constants are created from four raw FLOAT32 bytes, never through a lossy decimal,
   string, `double`, or host arithmetic conversion;
3. direct reduction output reshaping changes only Shape metadata needed for exact keep-dimensions;
4. custom extrema candidates compile with safe math and no fast-relaxed option, preserve raw source
   candidates, and expose a complete resource ledger;
5. every malformed type, rank, axis, flag, attr, sentinel, padding, topology, or geometry case is
   rejected independently in Java and native preflight; and
6. capability remains false and no staged candidate is reachable through public preparation.

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

After Gate 1 passes and capability is still false, compile every direct candidate and all listed
extrema contingencies before launch. Run exactly one real-device process invocation using one
default-device context, production-default MPSGraph optimization, read-back
`MPSGraphReducedPrecisionFastMathNone`, safe custom math, direct supplied targets, input identity
controls, output canaries, and exact Shape/type checks. There is no retry, second context,
optimization matrix, independent-executable matrix, per-failure rerun, or device-oracle execution
in this planning commit.

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

### Frozen scan corpus

Use the same `[2,3,8]` words so each logical scan line covers signed zeros, both subnormal signs,
normal boundaries, cancellation, overflow candidates, opposing infinities, and NaNs in distinct
positions. For both CUM_SUM and CUM_PROD, emit all four exclusive/reverse combinations on axis `2`
and one additional inclusive-forward output on axis `1` as the non-innermost mapping witness.
Require exact Shape, identities (`+0` for exclusive sum and `+1` for exclusive product), direction,
placement, input preservation, canaries, and recursive result-set membership for every prefix.

The host validator computes reachable sets from exact binary32 raw words. For scans it memoizes all
legal full binary trees over each exact declared prefix contributor multiset and enumerates only
Model-authorized DAZ/round/FTZ choices at each arithmetic node. It never compares to a sequential
host fold, a whole-prefix ULP envelope, `double`, or native host fast math.

The raw report must name each operation and candidate, corpus and output SHA-256, structural/control
verdicts, maximum reachable-set size, DAZ/FTZ observations where applicable, first exact failure,
and `PASS`/`FAIL`. Record the unedited report text and SHA-256 in this task, then remove probe source,
binary, raw output, validator, and generated artifacts.

## Gate 3: one bounded cost-evidence group

Only numerical `PASS` candidates enter this gate. Run one process containing a predeclared
standalone workload per operation kind/candidate, not an input-size, axis, mode, optimization, or
device matrix:

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

1. remove every structural or numerical failure;
2. select a direct MPSGraph route whenever it is valid and complete;
3. consider an exact custom route only when direct is invalid or incomplete for the authorized
   domain;
4. consider composition only when neither direct nor custom supplies exact complete coverage;
5. within the first eligible preference tier, an alternative replaces its stable baseline only when
   retained median is at least 10% lower, actual dispatch count is no greater, and peak transient
   bytes are no greater; ties choose fewer dispatches, then fewer bytes, then lexical candidate ID;
6. missing cost facts, no valid candidate, or a route that covers only a subset blocks that operation;
   and
7. encode the winner as one immutable production route and delete every unselected custom kernel,
   composition, probe, benchmark, capture, and report artifact.

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
- Independent of: Task 0051 numerical candidates, timing, route, schema implementation, and result.
- Common planning base: `f2debef31f382db77c8c2758e56d646858838d0f`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.
- Parallel group: none.
- Integration order: this planning commit; instrumentation prerequisite; Gate 1 with capability
  false; exactly one Gate-2 process invocation; artifact removal/evidence record; one Gate-3 cost
  group; Gate-4 route freeze/loser deletion; production schema/lifecycle/capability; focused proof;
  documentation; implementation commit; independent Class C review.
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

1. This planning commit inventories the complete exact operation partition, current Metal
   architecture/callers, candidates, frozen corpus, four gates, identity cutover, and blockers; it
   runs no device oracle and changes no production file.
2. Future execution performs Gate 1 while all new capability is false, then exactly one fresh
   real-device numerical process invocation over the frozen bounded group with no retry or matrix.
3. Every operation kind receives its own structural, numerical, actual-dispatch/transient cost, and
   fixed-route verdict; no family analogy or partial-domain narrowing authorizes capability.
4. Direct MPSGraph is preferred when exact, custom exists only for exactness/coverage, composition is
   last, and production contains one fixed route with no runtime choice/fallback.
5. Schema/identity cutover is single-version, Java/native exact, and revised before implementation if
   Task 0051 consumes the current next values first.
6. All explicit exclusions remain false, especially strict profile, gradients, BOOL consumers,
   ARG_MIN/ARG_MAX/INT64, PROD reduction, affine inputs, and malformed rows.
7. Focused native/backend/conformance/Engine/documentation validation and independent Class C review
   pass before any operation status becomes Complete.

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
implementation synchronizes current capability, schema/identity, route decisions, gradient
exclusions, public Engine evidence, native guide, package Javadocs, master plan, and roadmap while
preserving every historical blocked record. Independent Class C review must inspect each operation's
four evidence gates, raw oracle/cost facts, fixed routes, false exclusions, stale-data rejection,
lifecycle, real no-skip Engine proof, and removed artifacts.

## Result

Planning-only and Blocked on supported readable GPU dispatch/transient-resource instrumentation.
No real-device oracle, timing, capture, native build, test, or production action ran. No schema,
wire, attribute, identity, capability, route, Java/native source, package, or generated artifact
changed. All comparison, extrema, and scan capability named here remains false pending future
four-gate execution.
