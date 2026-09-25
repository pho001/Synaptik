# Task 0054: Current ACCELERATOR 115-Kind Completeness Audit

## Status

Complete

This is a read-only documentation/audit result from exact clean base
`93b3d37927c54cf96c76a2053dc2d1d925852947`. It changes no Model or Compiler contract, Metal
capability, schema, identity, route, production source, native binary, test, package, or evidence.
It ran no build, test, compiler, proof tool, native process, device oracle, timing workload, or trace
capture.

The audit used the prior independent inventory at `history://metal-remaining-coverage` as input, then
re-derived counts and current status from the post-Task-0053 Model enums, Metal capability/provider
and private-schema source, Compiler gradient registry/rules, Tasks 0051–0053, and the retained
reviewed Task-0052 evidence. Historical selector/header observations are context only; they do not
create capability.

## Result summary

The current Model declares exactly **40 `OperationKind` enum families and 115 operation-kind
constants**. Under `NumericalProfile.ACCELERATOR`:

| Partition | Distinct operation-kind constants | Current capability truth |
|---|---:|---|
| at least one current bounded Metal domain | **19** | some occurrences `true`; every unlisted dtype/shape/layout/attribute/gradient topology remains false |
| Task 0051 historical / Task 0053 successor | **2** | `EXP` and `SIGMOID` remain false; no Task-0053 candidate is `DOMAIN-PASS` |
| Task 0052 | **15** | all were false in production at the exact audit base; the then-current record had stopped after reviewed custom `DOMAIN-PASS` and numerical `PASS` |
| every other operation-kind constant | **79** | false for every occurrence |
| **total** | **115** | `19 + 2 + 15 + 79 = 115` |

Task 0052 has **15 operation-kind constants but 16 custom candidates** because `CLAMP` has both
fused-custom and two-step-custom candidates. Candidate count is not operation-kind count.

At the exact audited base there was no Metal task Ready. Task 0053 was blocked on concrete
source/constants plus pinned machine-checkable complete-domain proof and independent review, and on
supported full-Xcode tooling for actual dispatch/peak-transient evidence. Task 0052 had stopped after
Gate 2. That Task-0052 status is historical: ADR 0022 subsequently authorized its auditable custom
route-cost method, and retained Gate-3 evidence then passed and selected fused custom `CLAMP`.

## Authority and counting method

Current capability truth is `MetalCapabilityProvider.supportsOccurrence`, followed by private Java
and native preflight. Planning records cannot make a rejected occurrence true. Conversely, a kind
with one accepted occurrence is counted once in the 19-kind current partition even though most of
its Model dtype/shape/layout/gradient rows remain false.

The count was regenerated from every source enum that directly `implements OperationKind`. Enum
constants are qualified by family below so duplicate spellings such as `ADD`, `MIN`, and `MAX` are
not conflated.

Legend:

- **T** — at least one current capability-true bounded Metal ACCELERATOR occurrence;
- **E** — Task-0051 historical / Task-0053 successor `EXP`/`SIGMOID`, currently false;
- **X** — Task-0052 operation, currently false after proof/regression but before cost/route/production;
- **F** — every other kind, currently false for every occurrence.

## Exact 115-constant ledger

| OperationKind enum | Count | T | E | X | F |
|---|---:|---|---|---|---|
| `ScaledDotProductAttentionKind` | 1 | — | — | — | `SCALED_DOT_PRODUCT_ATTENTION` |
| `Conv2dKind` | 1 | — | — | — | `CONV2D` |
| `Conv3dKind` | 1 | — | — | — | `CONV3D` |
| `BinaryArithmeticKind` | 7 | `ADD`, `SUB`, `MUL`, `DIV` | — | `MIN`, `MAX` | `POW` |
| `CastKind` | 1 | — | — | — | `CAST` |
| `FloatingClassificationKind` | 3 | — | — | — | `IS_FINITE`, `IS_NAN`, `IS_INF` |
| `BinaryComparisonKind` | 6 | — | — | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | — |
| `BooleanLogicalKind` | 3 | — | — | — | `AND`, `OR`, `NOT` |
| `ScalarElementwiseKind` | 8 | — | — | `MIN`, `MAX`, `CLAMP` | `ADD`, `SUB`, `MUL`, `DIV`, `POW` |
| `WhereSelectionKind` | 1 | — | — | — | `WHERE` |
| `UnaryElementwiseKind` | 19 | `ABS`, `NEG` | `EXP`, `SIGMOID` | — | `RECIPROCAL`, `LOG`, `LOG1P`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `TANH`, `GELU`, `GELU_TANH_APPROXIMATION`, `SILU` |
| `AxisGatherKind` | 2 | `GATHER` | — | — | `GATHER_ELEMENTS` |
| `AxisScatterKind` | 2 | `SCATTER_ELEMENTS` | — | — | `SCATTER_ADD` |
| `GatherNdKind` | 1 | — | — | — | `GATHER_ND` |
| `OneHotKind` | 1 | `ONE_HOT` | — | — | — |
| `ScatterNdKind` | 1 | — | — | — | `SCATTER_ND` |
| `SelectKind` | 1 | — | — | — | `SELECT` |
| `AxisTransformKind` | 3 | `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE` | — | — | — |
| `ContiguousKind` | 1 | `CONTIGUOUS` | — | — | — |
| `PadKind` | 1 | — | — | — | `PAD` |
| `ShapeTransformKind` | 2 | `RESHAPE`, `EXPAND` | — | — | — |
| `SliceKind` | 2 | — | — | — | `SLICE`, `SLICE_UPDATE` |
| `TensorCompositionKind` | 2 | — | — | — | `CONCAT`, `STACK` |
| `TileKind` | 1 | — | — | — | `TILE` |
| `WindowTransformKind` | 6 | `UNFOLD_AXIS` | — | — | `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, `FOLD3D` |
| `MatmulKind` | 1 | `MATMUL` | — | — | — |
| `LossKind` | 3 | — | — | — | `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` |
| `BatchNormKind` | 2 | — | — | — | `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` |
| `LayerNormKind` | 1 | — | — | — | `LAYER_NORM` |
| `RmsNormKind` | 1 | — | — | — | `RMS_NORM` |
| `SoftmaxKind` | 2 | — | — | — | `SOFTMAX`, `LOG_SOFTMAX` |
| `OrderingKind` | 2 | — | — | — | `SORT`, `ARGSORT` |
| `TopKKind` | 1 | — | — | — | `TOP_K` |
| `Pool2dKind` | 2 | — | — | — | `MAX_POOL2D`, `AVERAGE_POOL2D` |
| `Pool3dKind` | 2 | — | — | — | `MAX_POOL3D`, `AVERAGE_POOL3D` |
| `DropoutKind` | 1 | — | — | — | `DROPOUT` |
| `GraphRngKind` | 1 | — | — | — | `INITIAL_STATE` |
| `RecurrentScanKind` | 3 | — | — | — | `RNN_TANH`, `GRU_RESET_AFTER`, `LSTM` |
| `AggregateReductionKind` | 14 | `SUM`, `MEAN` | — | `MIN`, `MAX` | `PROD`, `ALL`, `ANY`, `ARG_MAX`, `ARG_MIN`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` |
| `CumulativeScanKind` | 2 | — | — | `CUM_SUM`, `CUM_PROD` | — |
| **Totals** | **115** | **19** | **2** | **15** | **79** |

The same 115 constants group by semantic family as follows:

| Family | Count |
|---|---:|
| elementwise, comparison, classification, logical, selection, cast | 48 |
| aggregate reductions | 14 |
| cumulative scans | 2 |
| layout/window transforms | 18 |
| indexing | 8 |
| linear algebra | 1 |
| normalization | 6 |
| losses | 3 |
| convolution | 2 |
| pooling | 4 |
| attention | 1 |
| ordering | 3 |
| random/state | 2 |
| recurrent | 3 |
| **total** | **115** |

## Current 19-kind capability-true bounded domains

Every row below is partial relative to its complete Model family. “False remainder” is normative:
anything not explicitly accepted remains false.

| Kinds | Count | Current ACCELERATOR true domain | False remainder, including dtype/shape/layout/gradient rows |
|---|---:|---|---|
| unary `ABS`, `NEG` | 2 | canonical dense zero-offset `FLOAT32`, static positive rank `1..16`, exact same Shape and matching input/output `requiresGrad` | every other dtype; rank zero, zero/dynamic extent, view/offset/noncanonical layout, Shape or grad mismatch; `ABS` all-Metal backward is false because Compiler emits comparisons and nested `WHERE`; `NEG` backward can remain `NEG` |
| tensor `ADD`, `SUB`, `MUL`, `DIV` | 4 | ACCELERATOR-only canonical `FLOAT32` rank `1..16`, exact right-aligned broadcast, all three grad flags equal | STRICT; integer and other floating dtypes; rank zero/zero/dynamic/view; mixed input grad flags allowed by broader Model formulas; any generated topology leaving current binary/reduction/layout closure |
| reductions `SUM`, `MEAN` | 2 | canonical `FLOAT32` rank `1..16` input; full, normalized single-axis, ordered unique multi-axis including empty, keep/drop dimensions; `SUM_TO_SHAPE`; canonical output may be rank zero; grad flags equal | integral/wider floating, masked forms, malformed/duplicate axes, dynamic/zero/view; rank-zero inputs and rank-zero intermediates; any backward occurrence outside current EXPAND/SUM_TO_SHAPE/binary closure |
| `MATMUL` | 1 | positive static rank-two `FLOAT32`; each operand canonical or authenticated local rank-two transpose; canonical output; output grad is operand-grad OR | batched/broadcast/general-rank contraction, non-FLOAT32, arbitrary views/offsets, zero/dynamic extents, and any generated gradient leaving the bounded rank-two route |
| `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, `CONTIGUOUS` | 6 | fully static positive `FLOAT32` geometry; exact local zero-offset affine views; `CONTIGUOUS` materializes canonical output; matching grad flags | all other dtypes, rank zero, zero/dynamic extents, offset/general-strided views, cross-owner view ingress/publication, malformed mappings, and generated gradients whose inverse topology leaves the bounded layout/reduction set |
| `GATHER` | 1 | canonical positive-rank `FLOAT32` data, canonical positive-rank `INT32` indices, exact axis/output mapping, data/output grad equality, indices no-grad | INT64 indices, non-FLOAT32 data, rank zero, zero/dynamic/view, malformed/out-of-bounds indices; all-Metal backward is false because Compiler emits `SCATTER_ADD` |
| `ONE_HOT` | 1 | canonical positive-rank `INT32` indices to canonical positive-rank `BOOL`, rank below 16 before trailing depth, both no-grad | INT64, BOOL ingress/consumer chains, rank zero/rank-16 input, zero/dynamic/view, malformed depth; operation is intentionally non-differentiable |
| `SCATTER_ELEMENTS` | 1 | replacement only (`ScatterReduction.NONE`), canonical `FLOAT32` data/updates and `INT32` indices, positive equal-rank geometry, valid unique targets, data-shaped canonical output, output grad is data/update OR | ADD/MUL/MIN/MAX reductions, INT64, other dtypes, duplicates/OOB, rank zero/zero/dynamic/view; full all-Metal backward is false because update cotangents require unsupported `GATHER_ELEMENTS` |
| `UNFOLD_AXIS` | 1 | canonical `FLOAT32`, input rank `1..15`, output rank+1, size `1..16`, positive step, no padding/dilation, positive static extents, exact canonical materialization, matching grad flags | every other dtype/form, rank zero/rank16 input, size above 16, zero/dynamic/view, padding/dilation/ceil variants; all-Metal backward is false because Compiler emits `FOLD_AXIS` |
| **total** | **19** | at least one true occurrence each | no family is complete over all Model rows |

`requiresGrad=true` on a forward occurrence is not an all-Metal generated-backward guarantee.
Capability classifies each generated occurrence independently. Current source and completed task
records explicitly preserve the `ABS`, `GATHER`, `SCATTER_ELEMENTS`, and `UNFOLD_AXIS` backward gaps
above. Task-0052 and Task-0053 authorized future slices are forward/no-grad. `ONE_HOT` is
non-differentiable.

The Compiler's current `FirstOrderGradientCoverage` source is intentionally fail-closed but its
Javadoc still describes **38 enum families / 111 constants / 133 fingerprints**. The Model now has
40/115: `CONV3D` plus the three `RecurrentScanKind` constants are absent from that closed signature
inventory and have no family owner, so differentiable roles fail closed. This audit records that
source/documentation drift; it does not repair Compiler production in a documentation-only task.

## Tasks 0051 and 0053: two false constants

`UnaryElementwiseKind.EXP` and `SIGMOID` are counted only here.

Task 0051 is the immutable consumed-oracle record. Its 83-word sample gave bounded numerical passes
to direct EXP and an inherited stable SIGMOID composition, but direct EXP lacks authoritative
complete-domain proof and is `DOMAIN-BLOCKED`; the composition inherits that blocker. Original
custom EXP, custom SIGMOID, and direct SIGMOID retain numerical failures. No Task-0051 route may run
again.

Task 0053 supersedes 0051 for all future work. No source/constants/certificate exist and neither
custom candidate is `DOMAIN-PASS`. Before its one new oracle, it requires exact retained source,
pinned Lean or Coq/Flocq/Gappa proof over all raw words and ranks `1..16`, an independent exhaustive
checker, read-only approval, and the full-Xcode trace prerequisite. Both operation capabilities
remain false; schema/wires/identities remain unreserved; non-FLOAT32, excluded geometry, and
gradient topologies remain false even if its future no-grad canonical slice completes.

## Task 0052: 15 false constants at the audit base; Gate-3 result now supersedes the stop reason

Task 0052 owns exactly:

- six `BinaryComparisonKind` constants;
- `BinaryArithmeticKind.MIN` and `MAX`;
- `ScalarElementwiseKind.MIN`, `MAX`, and `CLAMP`;
- `AggregateReductionKind.MIN` and `MAX`; and
- `CumulativeScanKind.CUM_SUM` and `CUM_PROD`.

This corrects the old inventory as of the exact audit base: Task 0052 was not merely planned and it
was not missing complete-domain verdicts. Every opaque direct candidate was `DOMAIN-BLOCKED`; all
**16** auditable custom candidates were reviewed `DOMAIN-PASS` and numerical `PASS`, with exact
source/validator/audit/proof retained under [`../evidence/0052/`](../evidence/0052/README.md). At
that base Gate 3 had not run, so there was then no cost result, Gate-4 route, schema/identity
cutover, production implementation, or capability change, and all 15 kinds were production false.
ADR 0022 and [`../evidence/0052-gate3/`](../evidence/0052-gate3/README.md) now supersede only that
historical stop reason: the one authorized actual-M3 Gate-3 run passed, selected fused custom
`CLAMP`, and left Task 0052 in production implementation. This read-only audit's base, 115-kind
partition, and counts are intentionally unchanged.

Task 0052's future authorized domain is canonical positive-static rank-`1..16` `FLOAT32`, forward and
no-grad, with its exact broadcast/axis/scan forms and terminal comparison BOOL publication. Integer,
other floating dtype, rank-zero input, zero/dynamic/view, BOOL-consumer/transfer, masked/advanced
reduction, and generated-gradient rows remain outside 0052 and false.

## Remaining 79 false kinds: exact blocker partition

The 79 constants not assigned to current true capability, Tasks 0051/0053, or Task 0052 are
partitioned exactly once below. These are primary serial workstreams; a kind may also depend on an
earlier schema or primitive prerequisite.

| Primary blocker/workstream | Count | Exact kinds | Dominant blockers |
|---|---:|---|---|
| exact BOOL/classification/selection/reduction | **9** | `IS_FINITE`, `IS_NAN`, `IS_INF`, `AND`, `OR`, `NOT`, `WHERE`, `ALL`, `ANY` | BOOL ingress/consumer/transfer absent; complete raw truth/selection/reduction proof |
| typed CAST | **1** | `CAST` | all 36 ordered pairs over six Model dtypes; exact conversion/canonicalization; missing carriers/transfers |
| exact layout/index/window movement | **16** | `PAD`, `SLICE`, `SLICE_UPDATE`, `CONCAT`, `STACK`, `TILE`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, `FOLD3D`, `GATHER_ELEMENTS`, `GATHER_ND`, `SCATTER_ADD`, `SCATTER_ND`, `SELECT` | variable arity, bounds/mapping, collision/overlap arithmetic, INT64, wider layouts/dtypes |
| remaining elementary/tensor/scalar arithmetic | **21** | `RECIPROCAL`, `LOG`, `LOG1P`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `TANH`, `GELU`, `GELU_TANH_APPROXIMATION`, `SILU`, tensor `POW`, scalar `ADD`, `SUB`, `MUL`, `DIV`, `POW` | complete-domain exact/primitive/composite proof, scalar attrs/constants, known class/sign failures |
| remaining numerical aggregates | **6** | `PROD`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | contributor/order proof, primitive dependencies, correction/divisor attrs, gradients |
| normalization | **6** | `SOFTMAX`, `LOG_SOFTMAX`, `LAYER_NORM`, `RMS_NORM`, `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | recursive numerical proof, >3 inputs/multi-output BN, state slots and gradients |
| losses | **3** | `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | reductions/softmax/log guards, target/index roles, loss attrs and gradients |
| exact ordering/arg extrema | **5** | `SORT`, `ARGSORT`, `TOP_K`, `ARG_MAX`, `ARG_MIN` | stable NaN/zero/tie ordering, INT64 outputs, TOP_K multi-output |
| convolution/pooling | **6** | `CONV2D`, `CONV3D`, `MAX_POOL2D`, `AVERAGE_POOL2D`, `MAX_POOL3D`, `AVERAGE_POOL3D` | opaque contraction/window algorithms, known max-pool winner failures, geometry/divisor and gradient proof |
| attention | **1** | `SCALED_DOT_PRODUCT_ATTENTION` | 3–4 inputs/1–2 outputs, BOOL mask, contractions/softmax, weights slot and gradients |
| explicit RNG/dropout | **2** | `INITIAL_STATE`, `DROPOUT` | zero input, INT64 state, three outputs, explicit auditable RNG; opaque framework randomness incompatible |
| recurrent | **3** | `RNN_TANH`, `GRU_RESET_AFTER`, `LSTM` | 5–7 inputs/2–3 outputs, INT64 valid lengths, state publication, recursive proof, Compiler FC gradient ownership |
| **total** | **79** | — | `9+1+16+21+6+6+3+5+6+1+2+3 = 79` |

## Current private schema, type, cardinality, and evidence blockers

| Blocker class | Current exact boundary | Consequences |
|---|---|---|
| node schema and identity | private schema `11`, ABI `4`, fixed 160-byte node record, operation wires `1..19`, attribute wires `0..6`, version-12 workload/policy/candidate/compatibility/route/codec identities | no EXP/SIGMOID, comparison/extrema/scan, scalar/clamp, normalization/loss/random/recurrent vocabulary; 0052/0053 provisional wires are unreserved and conflict until serialized |
| per-node cardinality | capability requires exactly one output; node records encode one required first input, optional second, and one special auxiliary third input; one output index | blocks zero-input `INITIAL_STATE`; variadic `CONCAT`/`STACK`; 4+ input attention/normalization/recurrent; multi-output BN/TOP_K/dropout/attention/recurrent |
| executable cardinality distinction | compiled executable/run ABI already accepts variable feed and target counts | this does not remove the per-node schema/capability blocker; feed/target arrays are not multi-output node semantics |
| local value types | private value table supports only `FLOAT32`, `INT32`, `BOOL`; caller ingress only canonical FLOAT32/INT32; BOOL may publish but has no ingress/general consumer; CPU/Metal transfer remains canonical non-view FLOAT32 | blocks FLOAT64/BFLOAT16, INT64 indices/state/results, BOOL graphs/masks, typed CAST closure, and mixed-owner typed execution |
| public types | Model already exposes `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, `BOOL` | none of the 115 constants needs a new public type; FLOAT16 is a separate future Model project and is not in this audit |
| geometry/layout | fully static positive extents, maximum rank 16, mostly canonical zero-offset values; only local zero-offset affine FLOAT32 views; rank zero only direct local SUM/MEAN target and cannot feed another node | blocks scalar/rank-zero inputs/intermediates, zero/dynamic extents, general views/offsets, wider typed materialization and many composite formulas |
| attributes | closed NONE/target-shape/permutation/axis/reduction/depth/window-axis vocabulary | blocks scalar/clamp/scan, epsilon/correction, mask, sort/top-k, random, convolution/pooling, attention, normalization, loss and recurrent state |
| complete-domain proof | selector declaration/sample is not authorization | opaque direct EXP, all Task-0052 direct routes, and many remaining direct-looking APIs stay `DOMAIN-BLOCKED`; custom/composed routes need auditable complete proofs |
| measured cost | installed tooling cannot read required actual dispatch and peak transient Metal facts | Task 0052 cannot run Gate 3; Task 0053 cannot consume its new oracle or later adjudicate under its conjunctive prerequisites |
| Compiler gradient closure | current registry text is stale at 38/111 while Model is 40/115; Conv3d and three recurrent constants fail closed; several current true forward kinds generate false dependencies | no global “requiresGrad means all-Metal” claim; per-occurrence generated graph must remain entirely inside current true domains or fail/mix according to existing ownership rules |

## Serial dependency and task partition

No row below is Ready merely because it is listed. Every operation slice retains the four gates:
structural/API, complete-domain proof, one bounded regression oracle, measured all-survivor cost with
actual dispatch/transient facts, then one fixed production route.

1. **Resolve external evidence prerequisites — 0 kinds.** Provide pinned Task-0053 proof tooling and
   supported full-Xcode trace reading. Do not substitute inferred dispatch/memory.
2. **Plan one private schema/type/cardinality foundation — 0 kinds.** From the then-current base,
   define one clean no-capability cutover for closed 115-kind needs: variable node cardinality and
   outputs, all six existing Model carriers, BOOL/INT consumers/transfers, legal rank-zero values,
   and closed attributes. Avoid repeated incompatible schema bumps and retain no dual decoder.
3. **Execute Task 0053 — 2 kinds.** Concrete certified custom EXP and recursive stable SIGMOID,
   read-only proof approval, one new oracle, measured cost, fixed route, production review. Never
   rerun Task 0051.
4. **Finish Task 0052 — 15 kinds.** Preserve reviewed custom proof/regression evidence; run Gate 3
   over every survivor, freeze per-kind routes, then implement atomically. Do not reinterpret its 16
   candidates as 16 kinds or widen its no-grad canonical domain.
5. **Exact BOOL pipeline — 9 kinds.** Classification, logic, WHERE, ALL, ANY plus end-to-end BOOL
   ingress/consumption/publication.
6. **Typed CAST closure — 1 kind.** All 36 ordered current-type pairs with per-pair exact verdicts;
   no FLOAT16 conflation.
7. **Exact layout/index/window movement — 16 kinds.** Bounds-first mapping, variable arity,
   INT32/INT64 indices, collision/overlap semantics, and raw-bit preservation.
8. **Remaining elementary/tensor/scalar arithmetic — 21 kinds.** Complete-domain primitive and
   recursive-site proofs; this is the dependency base for later composites.
9. **Remaining numerical aggregates — 6 kinds.** Auditable contributors/order/axes/corrections and
   generated-gradient closure.
10. **Normalization — 6 kinds.** Consume proved elementary/aggregate/BOOL/schema primitives.
11. **Losses — 3 kinds.** Consume normalization/reduction/guard primitives with exact target roles.
12. **Ordering and arg extrema — 5 kinds.** Consume INT64 and multi-output schema plus exact stable
    order/tie/value-copy proof.
13. **Convolution and pooling — 6 kinds.** Auditable contraction/window/winner/divisor routes;
    separately close Conv3d Compiler ownership before gradient claims.
14. **Attention — 1 kind.** Consume MATMUL/softmax/BOOL/multi-output closure.
15. **Explicit RNG/dropout — 2 kinds.** Zero-input/three-output INT64-state lifecycle and exact
    counter/key semantics.
16. **Recurrent — 3 kinds.** Consume schema, MATMUL, elementary and valid-length primitives; add
    separate Compiler first-order design before any gradient claim.
17. **Bounded-domain closure for the existing 19 kinds — no new constant count.** Revisit each
    remaining Model dtype/shape/layout/gradient row only after its carrier and primitive dependencies
    exist; a true kind name never authorizes an unlisted row.
18. **Regenerate the 115-kind closure audit.** Require `19 + 2 + 15 + 79 = 115` (or a consciously
    changed Model total), every constant exactly once, every true row fixed to one route, every false
    row fail-closed, and every generated gradient entirely owned or explicitly outside scope.

The 12 remaining-kind workstreams in steps 5–16 sum exactly to 79:
`9+1+16+21+6+6+3+5+6+1+2+3`.

## Files and scope

This audit changes only:

- this Task-0054 file;
- the Metal master-plan audit/frontier summary; and
- the repository roadmap Metal audit/frontier summary.

It does not edit Tasks 0051–0053, Task-0052 evidence, production source, tests, native source,
package/Javadoc, generated files, or external history.

## Acceptance criteria

1. The current 40 enum families and 115 operation-kind constants are regenerated from source and
   each constant appears exactly once in the T/E/X/F ledger.
2. Exact totals are `T=19`, `E=2`, `X=15`, `F=79`, and the semantic-family total is 115.
3. Every current true kind is described only with its bounded dtype/shape/layout/attribute/grad-flag
   domain; all unlisted rows and known generated-gradient gaps remain false.
4. Task 0051 is historical, Task 0053 supersedes it, and their two kinds remain false.
5. Task 0052 is corrected to reviewed custom `DOMAIN-PASS` plus numerical `PASS`, with 15 kinds/16
   custom candidates, but all capability remains false before Gate 3/4/production.
6. The 79 remaining kinds are partitioned exactly into 12 blocker/workstream classes summing to 79.
7. Current private schema/type/cardinality/geometry/attribute/proof/trace/gradient blockers and the
   serial dependency partition are explicit.
8. No device, build, test, compile, proof, timing, capture, production/schema/capability, or evidence
   change occurs.
9. Changed Markdown links, headings/anchors, fences, final newlines, count equations, status/frontier
   consistency, path scope, and `git diff --check` pass.
10. The three-document audit is committed for independent read-only review.

## Validation

Validate documentation and arithmetic only:

```bash
git diff --check
```

A documentation validator must check all changed links, duplicate headings/anchors, balanced fences,
final newlines, Task-0054 `Complete` status, Metal/roadmap status agreement, the exact enum/ledger and
blocker sums, exactly three changed planning files, and absence of production/native/test/evidence
artifacts. Do not run a build, test, compiler, proof tool, device process, timing workload, or trace.

## Documentation and review impact

The Metal master and roadmap record Task 0054 as a completed read-only audit without making any
blocked task Ready. Future plans may use this ledger as current inventory but must re-read source if
the Model enums, capability provider, schema, gradient registry, or Tasks 0051–0053 change.

## Result

Complete read-only audit: **115 = 19 current bounded-true + 2 Task-0051/0053 false + 15 Task-0052
false + 79 remaining false**. Current operation capability is not complete for any Model family over
all dtype/shape/layout/gradient rows. The dominant blockers are closed private cardinality/type/
attribute schema, complete-domain numerical proof, missing actual-dispatch/peak-transient trace
evidence, and generated-gradient dependencies outside the current 19-kind bounded set.
