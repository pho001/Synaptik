# Task 0054: Current [ACCELERATOR](../../../../glossary.md#term-accelerator) 115-Kind Completeness Audit

## Reader context

The counted unit is an operation-kind constant, not an executable graph, signature, dtype or shader. A kind counts true here if at least one bounded occurrence was supported at the audit base; other descriptor combinations still needed separate admission. Likewise, 16 CLAMP-inclusive candidates do not mean 16 distinct kinds.

The auditor inspected Model, Compiler-generated graphs and Metal provider/schema without execution. “Current” below is relative to the recorded revision. Later capability/profile changes do not rewrite this inventory; use live occurrence admission rather than carrying historical totals into a new integration.

- [Current occurrence-level admission](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection) — Explains why kind counts are weaker than complete graph or gradient support.

## Status

Complete

This is a read-only documentation/audit result from exact clean base
`93b3d37927c54cf96c76a2053dc2d1d925852947`. It changes no [Model](../../../../glossary.md#term-model) or [Compiler](../../../../glossary.md#term-compiler) contract, [Metal](../../../../glossary.md#term-metal)
capability, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), route, production source, native [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), test, package, or evidence.
It ran no build, test, compiler, proof tool, native process, [device](../../../../glossary.md#term-device) [oracle](../../../../glossary.md#term-test-oracle), timing workload, or [trace](../../../../glossary.md#trace)
[capture](../../../../glossary.md#term-capture-meanings).

The audit used the prior independent inventory at `history://metal-remaining-coverage` as input, then
re-derived counts and current status from the post-Task-0053 Model enums, Metal capability/provider
and private-schema source, Compiler [gradient](../../../../glossary.md#term-gradient) [registry](../../../../glossary.md#term-registry)/rules, Tasks 0051–0053, and the retained
reviewed Task-0052 evidence. Historical selector/header observations are context only; they do not
create capability.

## Result summary

The current [Model](../../../../glossary.md#term-model) declares exactly **40 [`OperationKind`](../../../../glossary.md#operationkind) enum families and 115 [operation](../../../../glossary.md#operation)-kind
[constants](../../../../glossary.md#term-constant)**. Under `NumericalProfile.ACCELERATOR`:

| [Partition](../../../../glossary.md#partition) | Distinct operation-kind constants | Current capability truth |
|---|---:|---|
| at least one current bounded [Metal](../../../../glossary.md#term-metal) domain | **19** | some occurrences `true`; every unlisted [dtype](../../../../glossary.md#data-type--datatype)/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/attribute/[gradient](../../../../glossary.md#term-gradient) [topology](../../../../glossary.md#term-topology) remains false |
| Task 0051 historical / Task 0053 successor | **2** | [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) remain false; no Task-0053 candidate is `DOMAIN-PASS` |
| Task 0052 | **15** | all were false in production at the exact audit base; the then-current record had stopped after reviewed custom `DOMAIN-PASS` and numerical `PASS` |
| every other operation-kind constant | **79** | false for every [occurrence](../../../../glossary.md#term-occurrence) |
| **total** | **115** | `19 + 2 + 15 + 79 = 115` |

Task 0052 has **15 operation-kind constants but 16 custom candidates** because `CLAMP` has both
[fused](../../../../glossary.md#specialization-and-fusion)-custom and two-step-custom candidates. Candidate count is not operation-kind count.

At the exact audited base there was no Metal task [Ready](../../../../glossary.md#term-ready). Task 0053 was blocked on concrete
source/constants plus pinned machine-checkable complete-domain proof and independent review, and on
supported full-Xcode tooling for actual [dispatch](../../../../glossary.md#term-dispatch)/peak-transient evidence. Task 0052 had stopped after
Gate 2. That Task-0052 status is historical: [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0022 subsequently authorized its auditable custom
route-cost method, and retained Gate-3 evidence then passed and selected fused custom `CLAMP`.

## Authority and counting method

Current capability truth is `MetalCapabilityProvider.supportsOccurrence`, followed by private Java
and native [preflight](../../../../glossary.md#term-preflight). [Planning](../../../../glossary.md#term-planning) records cannot make a rejected [occurrence](../../../../glossary.md#term-occurrence) true. Conversely, a kind
with one accepted occurrence is counted once in the 19-kind current [partition](../../../../glossary.md#partition) even though most of
its [Model](../../../../glossary.md#term-model) [dtype](../../../../glossary.md#data-type--datatype)/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) rows remain false.

The count was regenerated from every source enum that directly `implements OperationKind`. Enum
[constants](../../../../glossary.md#term-constant) are qualified by family below so duplicate spellings such as `ADD`, `MIN`, and `MAX` are
not conflated.

Legend:

- **T** — at least one current capability-true bounded [Metal](../../../../glossary.md#term-metal) [ACCELERATOR](../../../../glossary.md#term-accelerator) occurrence;
- **E** — Task-0051 historical / Task-0053 successor [`EXP`](../../../../glossary.md#term-expx)/[`SIGMOID`](../../../../glossary.md#term-sigmoid), currently false;
- **X** — Task-0052 [operation](../../../../glossary.md#operation), currently false after proof/[regression](../../../../glossary.md#term-regression) but before cost/route/production;
- **F** — every other kind, currently false for every occurrence.

## Exact 115-[constant](../../../../glossary.md#term-constant) ledger

| [OperationKind](../../../../glossary.md#operationkind) enum | Count | T | E | X | F |
|---|---:|---|---|---|---|
| `ScaledDotProductAttentionKind` | 1 | — | — | — | `SCALED_DOT_PRODUCT_ATTENTION` |
| `Conv2dKind` | 1 | — | — | — | [`CONV2D`](../../../../glossary.md#convolution--conv2d) |
| `Conv3dKind` | 1 | — | — | — | [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) |
| `BinaryArithmeticKind` | 7 | `ADD`, `SUB`, `MUL`, `DIV` | — | `MIN`, `MAX` | `POW` |
| `CastKind` | 1 | — | — | — | `CAST` |
| `FloatingClassificationKind` | 3 | — | — | — | `IS_FINITE`, `IS_NAN`, `IS_INF` |
| `BinaryComparisonKind` | 6 | — | — | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | — |
| `BooleanLogicalKind` | 3 | — | — | — | `AND`, `OR`, `NOT` |
| `ScalarElementwiseKind` | 8 | — | — | `MIN`, `MAX`, `CLAMP` | `ADD`, `SUB`, `MUL`, `DIV`, `POW` |
| `WhereSelectionKind` | 1 | — | — | — | `WHERE` |
| `UnaryElementwiseKind` | 19 | `ABS`, `NEG` | [`EXP`](../../../../glossary.md#term-expx), [`SIGMOID`](../../../../glossary.md#term-sigmoid) | — | `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, [`TANH`](../../../../glossary.md#term-tanh), [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU` |
| `AxisGatherKind` | 2 | [`GATHER`](../../../../glossary.md#gather) | — | — | `GATHER_ELEMENTS` |
| `AxisScatterKind` | 2 | `SCATTER_ELEMENTS` | — | — | `SCATTER_ADD` |
| `GatherNdKind` | 1 | — | — | — | `GATHER_ND` |
| `OneHotKind` | 1 | `ONE_HOT` | — | — | — |
| `ScatterNdKind` | 1 | — | — | — | `SCATTER_ND` |
| `SelectKind` | 1 | — | — | — | `SELECT` |
| `AxisTransformKind` | 3 | `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE` | — | — | — |
| `ContiguousKind` | 1 | [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) | — | — | — |
| `PadKind` | 1 | — | — | — | [`PAD`](../../../../glossary.md#padding) |
| `ShapeTransformKind` | 2 | `RESHAPE`, `EXPAND` | — | — | — |
| `SliceKind` | 2 | — | — | — | [`SLICE`](../../../../glossary.md#slice), `SLICE_UPDATE` |
| `TensorCompositionKind` | 2 | — | — | — | `CONCAT`, `STACK` |
| `TileKind` | 1 | — | — | — | `TILE` |
| `WindowTransformKind` | 6 | `UNFOLD_AXIS` | — | — | `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, `FOLD3D` |
| `MatmulKind` | 1 | [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) | — | — | — |
| `LossKind` | 3 | — | — | — | `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` |
| `BatchNormKind` | 2 | — | — | — | `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` |
| `LayerNormKind` | 1 | — | — | — | `LAYER_NORM` |
| `RmsNormKind` | 1 | — | — | — | `RMS_NORM` |
| `SoftmaxKind` | 2 | — | — | — | [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX` |
| `OrderingKind` | 2 | — | — | — | `SORT`, `ARGSORT` |
| `TopKKind` | 1 | — | — | — | `TOP_K` |
| `Pool2dKind` | 2 | — | — | — | `MAX_POOL2D`, `AVERAGE_POOL2D` |
| `Pool3dKind` | 2 | — | — | — | `MAX_POOL3D`, `AVERAGE_POOL3D` |
| `DropoutKind` | 1 | — | — | — | [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) |
| `GraphRngKind` | 1 | — | — | — | `INITIAL_STATE` |
| `RecurrentScanKind` | 3 | — | — | — | `RNN_TANH`, `GRU_RESET_AFTER`, [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) |
| `AggregateReductionKind` | 14 | `SUM`, [`MEAN`](../../../../glossary.md#term-mean) | — | `MIN`, `MAX` | `PROD`, `ALL`, `ANY`, `ARG_MAX`, `ARG_MIN`, `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` |
| `CumulativeScanKind` | 2 | — | — | `CUM_SUM`, `CUM_PROD` | — |
| **Totals** | **115** | **19** | **2** | **15** | **79** |

The same 115 constants group by semantic family as follows:

| Family | Count |
|---|---:|
| [elementwise](../../../../glossary.md#term-elementwise), comparison, classification, logical, selection, cast | 48 |
| aggregate reductions | 14 |
| cumulative scans | 2 |
| [layout](../../../../glossary.md#layout)/window transforms | 18 |
| indexing | 8 |
| [linear](../../../../glossary.md#term-linear-meanings) algebra | 1 |
| [normalization](../../../../glossary.md#term-normalization) | 6 |
| [losses](../../../../glossary.md#loss) | 3 |
| [convolution](../../../../glossary.md#term-channels-first-convolution-layers) | 2 |
| [pooling](../../../../glossary.md#term-pooling) | 4 |
| attention | 1 |
| ordering | 3 |
| random/state | 2 |
| recurrent | 3 |
| **total** | **115** |

## Current 19-kind capability-true bounded domains

Every row below is partial relative to its complete [Model](../../../../glossary.md#term-model) family. “False remainder” is [normative](../../../../glossary.md#term-normative):
anything not explicitly accepted remains false.

| Kinds | Count | Current [ACCELERATOR](../../../../glossary.md#term-accelerator) true domain | False remainder, including [dtype](../../../../glossary.md#data-type--datatype)/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) rows |
|---|---:|---|---|
| [unary](../../../../glossary.md#term-unary) `ABS`, `NEG` | 2 | [canonical](../../../../glossary.md#term-canonical-form) dense zero-offset [`FLOAT32`](../../../../glossary.md#term-float32), static positive [rank](../../../../glossary.md#term-rank) `1..16`, exact same Shape and matching input/output `requiresGrad` | every other dtype; rank zero, zero/dynamic [extent](../../../../glossary.md#term-extent), [view](../../../../glossary.md#term-view-meanings)/offset/noncanonical layout, Shape or grad mismatch; `ABS` all-[Metal](../../../../glossary.md#term-metal) backward is false because [Compiler](../../../../glossary.md#term-compiler) emits comparisons and nested `WHERE`; `NEG` backward can remain `NEG` |
| [tensor](../../../../glossary.md#tensor) `ADD`, `SUB`, `MUL`, `DIV` | 4 | ACCELERATOR-only canonical `FLOAT32` rank `1..16`, exact right-aligned broadcast, all three grad flags equal | STRICT; integer and other floating dtypes; rank zero/zero/dynamic/view; mixed input grad flags allowed by broader Model formulas; any generated [topology](../../../../glossary.md#term-topology) leaving current [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/reduction/layout closure |
| reductions `SUM`, [`MEAN`](../../../../glossary.md#term-mean) | 2 | canonical `FLOAT32` rank `1..16` input; full, normalized single-[axis](../../../../glossary.md#term-axis), ordered unique multi-axis including empty, keep/drop [dimensions](../../../../glossary.md#dimension); `SUM_TO_SHAPE`; canonical output may be rank zero; grad flags equal | integral/wider floating, masked forms, malformed/duplicate axes, dynamic/zero/view; [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) inputs and rank-zero intermediates; any backward [occurrence](../../../../glossary.md#term-occurrence) outside current EXPAND/SUM_TO_SHAPE/binary closure |
| [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) | 1 | positive static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `FLOAT32`; each operand canonical or authenticated local rank-two [transpose](../../../../glossary.md#term-transpose); canonical output; output grad is operand-grad OR | batched/broadcast/general-rank [contraction](../../../../glossary.md#contraction-dimension), non-FLOAT32, arbitrary views/offsets, zero/dynamic extents, and any generated gradient leaving the bounded rank-two route |
| `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) | 6 | fully static positive `FLOAT32` geometry; exact local zero-offset affine views; `CONTIGUOUS` materializes canonical output; matching grad flags | all other dtypes, rank zero, zero/dynamic extents, offset/general-strided views, cross-[owner](../../../../glossary.md#term-owner) view ingress/[publication](../../../../glossary.md#term-publication), malformed mappings, and generated gradients whose inverse topology leaves the bounded layout/reduction set |
| [`GATHER`](../../../../glossary.md#gather) | 1 | canonical positive-rank `FLOAT32` data, canonical positive-rank [`INT32`](../../../../glossary.md#data-type--datatype) indices, exact axis/output mapping, data/output grad equality, indices no-grad | [INT64](../../../../glossary.md#data-type--datatype) indices, non-FLOAT32 data, rank zero, zero/dynamic/view, malformed/out-of-bounds indices; all-Metal backward is false because Compiler emits `SCATTER_ADD` |
| `ONE_HOT` | 1 | canonical positive-rank `INT32` indices to canonical positive-rank [`BOOL`](../../../../glossary.md#data-type--datatype), rank below 16 before trailing depth, both no-grad | INT64, BOOL ingress/consumer chains, rank zero/rank-16 input, zero/dynamic/view, malformed depth; [operation](../../../../glossary.md#operation) is intentionally non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) |
| `SCATTER_ELEMENTS` | 1 | replacement only (`ScatterReduction.NONE`), canonical `FLOAT32` data/updates and `INT32` indices, positive equal-rank geometry, valid unique [targets](../../../../glossary.md#term-target), data-shaped canonical output, output grad is data/update OR | ADD/MUL/MIN/MAX reductions, INT64, other dtypes, duplicates/OOB, rank zero/zero/dynamic/view; full all-Metal backward is false because update cotangents require unsupported `GATHER_ELEMENTS` |
| `UNFOLD_AXIS` | 1 | canonical `FLOAT32`, input rank `1..15`, output rank+1, size `1..16`, positive step, no [padding](../../../../glossary.md#padding)/[dilation](../../../../glossary.md#term-dilation), positive static extents, exact canonical [materialization](../../../../glossary.md#materialization), matching grad flags | every other dtype/form, rank zero/rank16 input, size above 16, zero/dynamic/view, padding/dilation/ceil variants; all-Metal backward is false because Compiler emits `FOLD_AXIS` |
| **total** | **19** | at least one true occurrence each | no family is complete over all Model rows |

`requiresGrad=true` on a forward occurrence is not an all-Metal generated-backward guarantee.
Capability classifies each generated occurrence independently. Current source and completed task
records explicitly preserve the `ABS`, `GATHER`, `SCATTER_ELEMENTS`, and `UNFOLD_AXIS` backward gaps
above. Task-0052 and Task-0053 authorized future slices are forward/no-grad. `ONE_HOT` is
non-differentiable.

The Compiler's current `FirstOrderGradientCoverage` source is intentionally [fail-closed](../../../../glossary.md#term-fail-closed) but its
[Javadoc](../../../../glossary.md#term-javadoc) still describes **38 enum families / 111 [constants](../../../../glossary.md#term-constant) / 133 fingerprints**. The Model now has
40/115: [`CONV3D`](../../../../glossary.md#conv3d--ncdhw) plus the three `RecurrentScanKind` constants are absent from that closed [signature](../../../../glossary.md#term-signature)
inventory and have no family owner, so differentiable roles [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). This audit records that
source/documentation drift; it does not repair Compiler production in a documentation-only task.

## Tasks 0051 and 0053: two false [constants](../../../../glossary.md#term-constant)

`UnaryElementwiseKind.EXP` and [`SIGMOID`](../../../../glossary.md#term-sigmoid) are counted only here.

Task 0051 is the [immutable](../../../../glossary.md#term-immutable) consumed-[oracle](../../../../glossary.md#term-test-oracle) record. Its 83-word [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) gave bounded numerical passes
to direct [EXP](../../../../glossary.md#term-expx) and an inherited stable SIGMOID composition, but direct EXP lacks authoritative
complete-domain proof and is `DOMAIN-BLOCKED`; the composition inherits that blocker. Original
custom EXP, custom SIGMOID, and direct SIGMOID retain numerical failures. No Task-0051 route may [run](../../../../glossary.md#term-run)
again.

Task 0053 supersedes 0051 for all future work. No source/constants/[certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) exist and neither
custom candidate is `DOMAIN-PASS`. Before its one new oracle, it requires exact retained source,
pinned [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) or [Coq](../../../../glossary.md#proof-assistant-and-model-checking-tools)/[Flocq](../../../../glossary.md#proof-assistant-and-model-checking-tools)/[Gappa](../../../../glossary.md#proof-assistant-and-model-checking-tools) proof over all raw words and ranks `1..16`, an independent exhaustive
checker, read-only approval, and the full-Xcode [trace](../../../../glossary.md#trace) prerequisite. Both [operation](../../../../glossary.md#operation) capabilities
remain false; [schema](../../../../glossary.md#term-schema)/wires/[identities](../../../../glossary.md#term-identity-meanings) remain unreserved; non-[FLOAT32](../../../../glossary.md#term-float32), excluded geometry, and
[gradient](../../../../glossary.md#term-gradient) topologies remain false even if its future no-grad [canonical](../../../../glossary.md#term-canonical-form) [slice](../../../../glossary.md#slice) completes.

## Task 0052: 15 false [constants](../../../../glossary.md#term-constant) at the audit base; Gate-3 result now supersedes the stop reason

Task 0052 owns exactly:

- six `BinaryComparisonKind` constants;
- `BinaryArithmeticKind.MIN` and `MAX`;
- `ScalarElementwiseKind.MIN`, `MAX`, and `CLAMP`;
- `AggregateReductionKind.MIN` and `MAX`; and
- `CumulativeScanKind.CUM_SUM` and `CUM_PROD`.

This corrects the old inventory as of the exact audit base: Task 0052 was not merely planned and it
was not missing complete-domain verdicts. Every [opaque](../../../../glossary.md#term-opaque) direct candidate was `DOMAIN-BLOCKED`; all
**16** auditable custom candidates were reviewed `DOMAIN-PASS` and numerical `PASS`, with exact
source/validator/audit/proof retained under [`../evidence/0052/`](../evidence/0052/README.md). At
that base Gate 3 had not [run](../../../../glossary.md#term-run), so there was then no cost result, Gate-4 route, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)
cutover, production implementation, or capability change, and all 15 kinds were production false.
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0022 and [`../evidence/0052-gate3/`](../evidence/0052-gate3/README.md) now supersede only that
historical stop reason: the one authorized actual-M3 Gate-3 run passed, selected [fused](../../../../glossary.md#specialization-and-fusion) custom
`CLAMP`, and left Task 0052 in production implementation. This read-only audit's base, 115-kind
[partition](../../../../glossary.md#partition), and counts are intentionally unchanged.

Task 0052's future authorized domain is [canonical](../../../../glossary.md#term-canonical-form) positive-static [rank](../../../../glossary.md#term-rank)-`1..16` [`FLOAT32`](../../../../glossary.md#term-float32), forward and
no-grad, with its exact broadcast/[axis](../../../../glossary.md#term-axis)/[scan](../../../../glossary.md#term-cumulative-scan) forms and terminal comparison [BOOL](../../../../glossary.md#data-type--datatype) [publication](../../../../glossary.md#term-publication). Integer,
other floating [dtype](../../../../glossary.md#data-type--datatype), [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) input, zero/dynamic/[view](../../../../glossary.md#term-view-meanings), BOOL-consumer/transfer, masked/advanced
reduction, and generated-[gradient](../../../../glossary.md#term-gradient) rows remain outside 0052 and false.

## Remaining 79 false kinds: exact blocker [partition](../../../../glossary.md#partition)

The 79 [constants](../../../../glossary.md#term-constant) not assigned to current true capability, Tasks 0051/0053, or Task 0052 are
partitioned exactly once below. These are primary serial workstreams; a kind may also depend on an
earlier [schema](../../../../glossary.md#term-schema) or primitive prerequisite.

| Primary blocker/workstream | Count | Exact kinds | Dominant blockers |
|---|---:|---|---|
| exact [BOOL](../../../../glossary.md#data-type--datatype)/classification/selection/reduction | **9** | `IS_FINITE`, `IS_NAN`, `IS_INF`, `AND`, `OR`, `NOT`, `WHERE`, `ALL`, `ANY` | BOOL ingress/consumer/transfer absent; complete raw truth/selection/reduction proof |
| typed CAST | **1** | `CAST` | all 36 ordered pairs over six [Model](../../../../glossary.md#term-model) [dtypes](../../../../glossary.md#data-type--datatype); exact conversion/[canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form); missing carriers/transfers |
| exact [layout](../../../../glossary.md#layout)/index/window movement | **16** | [`PAD`](../../../../glossary.md#padding), [`SLICE`](../../../../glossary.md#slice), `SLICE_UPDATE`, `CONCAT`, `STACK`, `TILE`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, `FOLD3D`, `GATHER_ELEMENTS`, `GATHER_ND`, `SCATTER_ADD`, `SCATTER_ND`, `SELECT` | variable [arity](../../../../glossary.md#term-arity), bounds/mapping, collision/overlap arithmetic, [INT64](../../../../glossary.md#data-type--datatype), wider layouts/dtypes |
| remaining elementary/[tensor](../../../../glossary.md#tensor)/[scalar](../../../../glossary.md#term-scalar) arithmetic | **21** | `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, [`TANH`](../../../../glossary.md#term-tanh), [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, `SILU`, tensor `POW`, scalar `ADD`, `SUB`, `MUL`, `DIV`, `POW` | complete-domain exact/primitive/composite proof, scalar attrs/constants, known class/sign failures |
| remaining numerical aggregates | **6** | `PROD`, `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM` | contributor/order proof, primitive dependencies, correction/divisor attrs, [gradients](../../../../glossary.md#term-gradient) |
| [normalization](../../../../glossary.md#term-normalization) | **6** | [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax), `LOG_SOFTMAX`, `LAYER_NORM`, `RMS_NORM`, `BATCH_NORM_INFERENCE`, `BATCH_NORM_TRAINING` | recursive numerical proof, >3 inputs/multi-output BN, state slots and gradients |
| [losses](../../../../glossary.md#loss) | **3** | `MEAN_SQUARED_ERROR`, `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS`, `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | reductions/softmax/log guards, [target](../../../../glossary.md#term-target)/index roles, loss attrs and gradients |
| exact ordering/arg [extrema](../../../../glossary.md#saturation-pole-and-extremum) | **5** | `SORT`, `ARGSORT`, `TOP_K`, `ARG_MAX`, `ARG_MIN` | stable [NaN](../../../../glossary.md#term-nan)/zero/tie ordering, INT64 outputs, TOP_K multi-output |
| [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling) | **6** | [`CONV2D`](../../../../glossary.md#convolution--conv2d), [`CONV3D`](../../../../glossary.md#conv3d--ncdhw), `MAX_POOL2D`, `AVERAGE_POOL2D`, `MAX_POOL3D`, `AVERAGE_POOL3D` | [opaque](../../../../glossary.md#term-opaque) [contraction](../../../../glossary.md#contraction-dimension)/window algorithms, known max-pool winner failures, geometry/divisor and gradient proof |
| attention | **1** | `SCALED_DOT_PRODUCT_ATTENTION` | 3–4 inputs/1–2 outputs, BOOL [mask](../../../../glossary.md#term-boolean-mask), contractions/softmax, [weights](../../../../glossary.md#term-weight) slot and gradients |
| explicit [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout) | **2** | `INITIAL_STATE`, `DROPOUT` | zero input, INT64 state, three outputs, explicit auditable RNG; opaque framework randomness incompatible |
| recurrent | **3** | `RNN_TANH`, `GRU_RESET_AFTER`, [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) | 5–7 inputs/2–3 outputs, INT64 valid lengths, state [publication](../../../../glossary.md#term-publication), recursive proof, [Compiler](../../../../glossary.md#term-compiler) FC gradient [ownership](../../../../glossary.md#term-ownership-meanings) |
| **total** | **79** | — | `9+1+16+21+6+6+3+5+6+1+2+3 = 79` |

## Current private [schema](../../../../glossary.md#term-schema), type, [cardinality](../../../../glossary.md#term-cardinality), and evidence blockers

| Blocker class | Current exact boundary | Consequences |
|---|---|---|
| [node](../../../../glossary.md#node) schema and [identity](../../../../glossary.md#term-identity-meanings) | private schema `11`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) `4`, fixed 160-byte node record, [operation](../../../../glossary.md#operation) wires `1..19`, attribute wires `0..6`, version-12 workload/policy/candidate/compatibility/route/[codec](../../../../glossary.md#term-codec) identities | no [EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid), comparison/[extrema](../../../../glossary.md#saturation-pole-and-extremum)/[scan](../../../../glossary.md#term-cumulative-scan), [scalar](../../../../glossary.md#term-scalar)/clamp, [normalization](../../../../glossary.md#term-normalization)/[loss](../../../../glossary.md#loss)/random/recurrent vocabulary; 0052/0053 provisional wires are unreserved and conflict until serialized |
| per-node cardinality | capability requires exactly one output; node records encode one required first input, optional second, and one special auxiliary third input; one output index | blocks zero-input `INITIAL_STATE`; [variadic](../../../../glossary.md#term-variadic) `CONCAT`/`STACK`; 4+ input attention/normalization/recurrent; multi-output BN/TOP_K/[dropout](../../../../glossary.md#dropout--inverted-dropout)/attention/recurrent |
| executable cardinality distinction | compiled executable/[run](../../../../glossary.md#term-run) ABI already accepts variable feed and [target](../../../../glossary.md#term-target) counts | this does not remove the per-node schema/capability blocker; feed/target arrays are not multi-output node semantics |
| local value types | private value table supports only [`FLOAT32`](../../../../glossary.md#term-float32), [`INT32`](../../../../glossary.md#data-type--datatype), [`BOOL`](../../../../glossary.md#data-type--datatype); caller ingress only [canonical](../../../../glossary.md#term-canonical-form) FLOAT32/INT32; BOOL may publish but has no ingress/general consumer; [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) transfer remains canonical non-[view](../../../../glossary.md#term-view-meanings) FLOAT32 | blocks [FLOAT64](../../../../glossary.md#term-float64)/[BFLOAT16](../../../../glossary.md#term-bfloat16), [INT64](../../../../glossary.md#data-type--datatype) indices/state/results, BOOL [graphs](../../../../glossary.md#graph)/[masks](../../../../glossary.md#term-boolean-mask), typed CAST closure, and mixed-[owner](../../../../glossary.md#term-owner) typed execution |
| public types | [Model](../../../../glossary.md#term-model) already exposes `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, `BOOL` | none of the 115 [constants](../../../../glossary.md#term-constant) needs a new public type; [FLOAT16](../../../../glossary.md#term-float16) is a separate future Model project and is not in this audit |
| geometry/[layout](../../../../glossary.md#layout) | fully static positive extents, maximum [rank](../../../../glossary.md#term-rank) 16, mostly canonical zero-offset values; only local zero-offset affine FLOAT32 views; rank zero only direct local SUM/[MEAN](../../../../glossary.md#term-mean) target and cannot feed another node | blocks scalar/[rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) inputs/intermediates, zero/dynamic extents, general views/offsets, wider typed [materialization](../../../../glossary.md#materialization) and many composite formulas |
| attributes | closed NONE/target-[shape](../../../../glossary.md#shape)/[permutation](../../../../glossary.md#permutation)/[axis](../../../../glossary.md#term-axis)/reduction/depth/window-axis vocabulary | blocks scalar/clamp/scan, [epsilon](../../../../glossary.md#epsilon)/correction, mask, sort/top-k, random, [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling), attention, normalization, loss and [recurrent state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) |
| complete-domain proof | selector declaration/[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) is not authorization | [opaque](../../../../glossary.md#term-opaque) direct EXP, all Task-0052 direct routes, and many remaining direct-looking APIs stay `DOMAIN-BLOCKED`; custom/composed routes need auditable complete proofs |
| measured cost | installed tooling cannot read required actual [dispatch](../../../../glossary.md#term-dispatch) and peak transient Metal facts | Task 0052 cannot run Gate 3; Task 0053 cannot consume its new [oracle](../../../../glossary.md#term-test-oracle) or later adjudicate under its conjunctive prerequisites |
| [Compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient) closure | current [registry](../../../../glossary.md#term-registry) text is stale at 38/111 while Model is 40/115; [Conv3d](../../../../glossary.md#conv3d--ncdhw) and three recurrent constants [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation); several current true forward kinds generate false dependencies | no global “requiresGrad means all-Metal” claim; per-[occurrence](../../../../glossary.md#term-occurrence) generated graph must remain entirely inside current true domains or fail/mix according to existing [ownership](../../../../glossary.md#term-ownership-meanings) rules |

## Serial [dependency](../../../../glossary.md#term-dependency) and task [partition](../../../../glossary.md#partition)

No row below is [Ready](../../../../glossary.md#term-ready) merely because it is listed. Every [operation](../../../../glossary.md#operation) [slice](../../../../glossary.md#slice) retains the four gates:
structural/[API](../../../../glossary.md#term-application-programming-interface-api), complete-domain proof, one bounded [regression](../../../../glossary.md#term-regression) [oracle](../../../../glossary.md#term-test-oracle), measured all-survivor cost with
actual [dispatch](../../../../glossary.md#term-dispatch)/transient facts, then one fixed production route.

1. **Resolve external evidence prerequisites — 0 kinds.** Provide pinned Task-0053 proof tooling and
   supported full-Xcode trace reading. Do not substitute inferred dispatch/memory.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>
2. **Plan one private [schema](../../../../glossary.md#term-schema)/type/[cardinality](../../../../glossary.md#term-cardinality) foundation — 0 kinds.** From the then-current base,
   define one clean no-capability cutover for closed 115-kind needs: variable node cardinality and
   outputs, all six existing Model carriers, BOOL/INT consumers/transfers, legal rank-zero values,
   and closed attributes. Avoid repeated incompatible schema bumps and retain no dual decoder.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [Model glossary](../../../../glossary.md#term-model); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
3. **Execute Task 0053 — 2 kinds.** Concrete certified custom EXP and recursive stable SIGMOID,
   read-only proof approval, one new oracle, measured cost, fixed route, production review. Never
   rerun Task 0051.<span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx); [SIGMOID glossary](../../../../glossary.md#term-sigmoid))</span>
4. **Finish Task 0052 — 15 kinds.** Preserve reviewed custom proof/regression evidence; run Gate 3
   over every survivor, freeze per-kind routes, then implement atomically. Do not reinterpret its 16
   candidates as 16 kinds or widen its no-grad canonical domain.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
5. **Exact [BOOL](../../../../glossary.md#data-type--datatype) pipeline — 9 kinds.** Classification, logic, WHERE, ALL, ANY plus end-to-end BOOL
   ingress/consumption/publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
6. **Typed CAST closure — 1 kind.** All 36 ordered current-type pairs with per-pair exact verdicts;
   no FLOAT16 conflation.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
7. **Exact [layout](../../../../glossary.md#layout)/index/window movement — 16 kinds.** Bounds-first mapping, variable arity,
   INT32/INT64 indices, collision/overlap semantics, and raw-bit preservation.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
8. **Remaining elementary/[tensor](../../../../glossary.md#tensor)/[scalar](../../../../glossary.md#term-scalar) arithmetic — 21 kinds.** Complete-domain primitive and
   recursive-site proofs; this is the dependency base for later composites.
9. **Remaining numerical aggregates — 6 kinds.** Auditable contributors/order/axes/corrections and
   generated-gradient closure.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [gradient glossary](../../../../glossary.md#term-gradient))</span>
10. **[Normalization](../../../../glossary.md#term-normalization) — 6 kinds.** Consume proved elementary/aggregate/BOOL/schema primitives.
11. **[Losses](../../../../glossary.md#loss) — 3 kinds.** Consume normalization/reduction/[guard](../../../../glossary.md#term-guard) primitives with exact [target](../../../../glossary.md#term-target) roles.
12. **Ordering and arg [extrema](../../../../glossary.md#saturation-pole-and-extremum) — 5 kinds.** Consume INT64 and multi-output schema plus exact stable
    order/tie/value-copy proof.
13. **[Convolution](../../../../glossary.md#term-channels-first-convolution-layers) and [pooling](../../../../glossary.md#term-pooling) — 6 kinds.** Auditable contraction/window/winner/divisor routes;
    separately close Conv3d Compiler ownership before gradient claims.<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
14. **Attention — 1 kind.** Consume [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[softmax](../../../../glossary.md#softmax--log-softmax)/BOOL/multi-output closure.
15. **Explicit [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[dropout](../../../../glossary.md#dropout--inverted-dropout) — 2 kinds.** Zero-input/three-output INT64-state lifecycle and exact
    counter/key semantics.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
16. **Recurrent — 3 kinds.** Consume schema, MATMUL, elementary and valid-length primitives; add
    separate Compiler first-order design before any gradient claim.
17. **Bounded-domain closure for the existing 19 kinds — no new [constant](../../../../glossary.md#term-constant) count.** Revisit each
    remaining Model dtype/shape/layout/gradient row only after its carrier and primitive dependencies
    exist; a true kind name never authorizes an unlisted row.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
18. **Regenerate the 115-kind closure audit.** Require `19 + 2 + 15 + 79 = 115` (or a consciously
    changed Model total), every constant exactly once, every true row fixed to one route, every false
    row fail-closed, and every generated gradient entirely owned or explicitly outside scope.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [scope glossary](../../../../glossary.md#term-scope))</span>

The 12 remaining-kind workstreams in steps 5–16 sum exactly to 79:
`9+1+16+21+6+6+3+5+6+1+2+3`.

## Files and [scope](../../../../glossary.md#term-scope)

This audit changes only:

- this Task-0054 file;
- the [Metal](../../../../glossary.md#term-metal) master-plan audit/frontier summary; and
- the repository roadmap Metal audit/frontier summary.

It does not edit Tasks 0051–0053, Task-0052 evidence, production source, tests, native source,
package/[Javadoc](../../../../glossary.md#term-javadoc), generated files, or external history.

## Acceptance criteria

1. The current 40 enum families and 115 operation-kind constants are regenerated from source and
   each constant appears exactly once in the T/E/X/F ledger.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [constants glossary](../../../../glossary.md#term-constant))</span>
2. Exact totals are `T=19`, `E=2`, `X=15`, `F=79`, and the semantic-family total is 115.
3. Every current true kind is described only with its bounded dtype/shape/layout/attribute/grad-flag
   domain; all unlisted rows and known generated-gradient gaps remain false.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
4. Task 0051 is historical, Task 0053 supersedes it, and their two kinds remain false.
5. Task 0052 is corrected to reviewed custom `DOMAIN-PASS` plus numerical `PASS`, with 15 kinds/16
   custom candidates, but all capability remains false before Gate 3/4/production.
6. The 79 remaining kinds are partitioned exactly into 12 blocker/workstream classes summing to 79.
7. Current private schema/type/cardinality/geometry/attribute/proof/trace/gradient blockers and the
   serial dependency partition are explicit.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cardinality glossary](../../../../glossary.md#term-cardinality); [trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency); [partition glossary](../../../../glossary.md#partition))</span>
8. No device, build, test, compile, proof, timing, capture, production/schema/capability, or evidence
   change occurs.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
9. Changed Markdown links, headings/anchors, fences, final newlines, count equations, status/frontier
   consistency, path scope, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.
10. The three-document audit is committed for independent read-only review.

## [Validation](../../../../glossary.md#term-validation)

Validate documentation and arithmetic only:

```bash
git diff --check
```

A documentation validator must check all changed links, duplicate headings/anchors, balanced fences,
final newlines, Task-0054 `Complete` status, [Metal](../../../../glossary.md#term-metal)/roadmap status agreement, the exact enum/ledger and
blocker sums, exactly three changed [planning](../../../../glossary.md#term-planning) files, and absence of production/native/test/evidence
[artifacts](../../../../glossary.md#term-artifact). Do not [run](../../../../glossary.md#term-run) a build, test, [compiler](../../../../glossary.md#term-compiler), proof tool, [device](../../../../glossary.md#term-device) process, timing workload, or [trace](../../../../glossary.md#trace).

## Documentation and review impact

The [Metal](../../../../glossary.md#term-metal) master and roadmap record Task 0054 as a completed read-only audit without making any
blocked task [Ready](../../../../glossary.md#term-ready). Future plans may use this ledger as current inventory but must re-read source if
the [Model](../../../../glossary.md#term-model) enums, capability provider, [schema](../../../../glossary.md#term-schema), [gradient](../../../../glossary.md#term-gradient) [registry](../../../../glossary.md#term-registry), or Tasks 0051–0053 change.

## Result

Complete read-only audit: **115 = 19 current bounded-true + 2 Task-0051/0053 false + 15 Task-0052
false + 79 remaining false**. Current [operation](../../../../glossary.md#operation) capability is not complete for any [Model](../../../../glossary.md#term-model) family over
all [dtype](../../../../glossary.md#data-type--datatype)/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient) rows. The dominant blockers are closed private [cardinality](../../../../glossary.md#term-cardinality)/type/
attribute [schema](../../../../glossary.md#term-schema), complete-domain numerical proof, missing actual-[dispatch](../../../../glossary.md#term-dispatch)/peak-transient [trace](../../../../glossary.md#trace)
evidence, and generated-gradient dependencies outside the current 19-kind bounded set.
