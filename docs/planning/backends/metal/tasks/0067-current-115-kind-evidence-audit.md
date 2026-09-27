# Task 0067: Current Metal 115-Kind Evidence Audit and Reconciliation

## Status

Audit plan and baseline ledger drafted from exact clean approved base
`209ba28a973f9e59bd70ec4fe14d3402d2c793a9`. No production source, capability, route, schema,
identity, ABI, native binary, test expectation, or historical evidence has changed. Independent plan
review is required before remediation.

## Change class

Audit-first Class C. The audit is exhaustive over the current 115 stable operation wires. It may
repair a demonstrated mismatch inside the already-published boundary, but the baseline ledger is not
capability authorization. Any finding that would move a production-false row to true, broaden a true
row beyond its current exact occurrence domain, add a wire/attribute/route/export, or change Model
semantics requires a concrete plan amendment and a fresh independent review before production edits.

## Goal

Reconcile every current Model operation kind through the complete Metal path: Model semantic family,
attributes/cardinality/carriers/output and gradient relations; Compiler inference, first-order
classification, saved roles, and generated graph reachability; profile-qualified capability;
private schema, route, and identity; MPSGraph/custom catalogs; Java and native preflight; native
encoding/dispatch/execution; direct and generated backward behavior; public Engine reachability;
behavioral tests; and active documentation.

The result must classify exactly 115 distinct stable `wire + kind` rows once each. A production-true
row means at least one exact bounded occurrence is admitted, never the complete Model family. A
production-false row stays false unless a separately reviewed amendment proves a complete exact
domain and implements it end to end.

## Non-goals and fixed boundaries

No timing, benchmark, autotune experiment, fallback, retry, route race, hot measurement, opaque
selector inference, historical oracle rerun, public API, new Model kind/type, schema bump, ABI/export
change, or compatibility reader. Historical Tasks 0006–0066 and retained evidence remain immutable;
this task links them only as provenance and updates current statements separately.

## Authority and evidence order

Current source is authoritative in this order:

1. Model enum semantics and `OperationSignature` declarations.
2. Compiler captured inference, `FirstOrderGradientCoverage`, `AutogradPreflight`, formula owners, and
   saved-role rules.
3. `MetalCapabilityProvider.supportsOccurrence` for the exact profile/descriptor occurrence.
4. `MetalMpsGraphProgram.NodeKind`, `MetalOperationRouteCatalog`, preparation/lowering, route policy,
   schema-15 validation, and identity-22 authentication.
5. Native schema decoder, semantic/type/shape/layout checks, resource preflight, fixed dispatch, and
   kernel/MPSGraph execution.
6. Real behavioral tests and public Engine reachability.
7. Active documentation. Planning text never turns rejected source into capability.

Task 0056's selector/source ledger remains immutable structural provenance. Task 0054 remains its
historical pre-cutover audit. Neither substitutes for this current source-backed reconciliation.

## Ledger notation

- `S`: private schema/native structural executable flag (`Y/N`).
- `P`: at least one current production capability occurrence (`T/F`).
- `Catalog M/C`: MPSGraph `D` direct, `C` composed, or `U` unavailable; custom `A` available or `P`
  pending. Catalog state is descriptive, not capability.
- Prepared route: exact current production route family. `DUAL` is only the bounded singleton-NEG
  candidate set; `BY-DOMAIN` is MATMUL's occurrence-selected fixed MPSGraph/custom boundary.
- Compiler: `D` conditional differentiable formula owner, `ND` intentional non-differentiability,
  `D/ND/FC` role/type/cardinality-dependent classification, and `FC-B` an intentional forward-only
  boundary excluded from the supported-formula registry and rejected by explicit pre-allocation
  preflight.
- Domain/blocker codes below are normative evidence pointers for the baseline claim.

## Canonical 115-row baseline ledger

This table is generated from the current wire order plus current Model signatures and then reconciled
against current catalog and capability partitions. It is the sole Task-0067 row key; later findings
and remediations must cite a wire from this table.

| Wire | Stable node / Model kind | Model attrs; cardinality | S | P | Catalog M/C | Prepared route | Compiler | Domain or blocker |
|---:|---|---|:---:|:---:|:---:|---|---|---|
| 1 | `NEG` / `UnaryElementwiseKind.NEG` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `DUAL` | `D` | `D22` |
| 2 | `ADD` / `BinaryArithmeticKind.ADD` | `NoOperationAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D` | `D15` |
| 3 | `SUB` / `BinaryArithmeticKind.SUB` | `NoOperationAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D` | `D15` |
| 4 | `MUL` / `BinaryArithmeticKind.MUL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D` | `D15` |
| 5 | `DIV` / `BinaryArithmeticKind.DIV` | `NoOperationAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D` | `D15` |
| 6 | `RESHAPE` / `ShapeTransformKind.RESHAPE` | `TargetShapeAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 7 | `EXPAND` / `ShapeTransformKind.EXPAND` | `TargetShapeAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 8 | `PERMUTE` / `AxisTransformKind.PERMUTE` | `PermutationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 9 | `EXPAND_DIMS` / `AxisTransformKind.EXPAND_DIMS` | `AxisTransformAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 10 | `SQUEEZE` / `AxisTransformKind.SQUEEZE` | `AxisTransformAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 11 | `CONTIGUOUS` / `ContiguousKind.CONTIGUOUS` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-AFF` |
| 12 | `ABS` / `UnaryElementwiseKind.ABS` | `NoOperationAttrs; 1 -> 1` | Y | T | D/P | `MPSGRAPH` | `D` | `D22` |
| 13 | `SUM` / `AggregateReductionKind.SUM` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs/SumToShapeAttrs; 1 -> 1 / MaskedReductionAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D/ND/FC` | `D20-RED` |
| 14 | `MEAN` / `AggregateReductionKind.MEAN` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1 / MaskedReductionAttrs; 2 -> 1` | Y | T | D/P | `MPSGRAPH` | `D/ND/FC` | `D20-RED` |
| 15 | `MATMUL` / `MatmulKind.MATMUL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `BY-DOMAIN` | `D` | `D61-MM` |
| 16 | `GATHER` / `AxisGatherKind.GATHER` | `IndexAxisAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-IDX` |
| 17 | `ONE_HOT` / `OneHotKind.ONE_HOT` | `OneHotAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-IDX` |
| 18 | `SCATTER_ELEMENTS` / `AxisScatterKind.SCATTER_ELEMENTS` | `ScatterElementsAttrs; 3 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-IDX` |
| 19 | `UNFOLD_AXIS` / `WindowTransformKind.UNFOLD_AXIS` | `UnfoldAxisAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-IDX` |
| 20 | `GT` / `BinaryComparisonKind.GREATER_THAN` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 21 | `GE` / `BinaryComparisonKind.GREATER_OR_EQUAL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 22 | `LT` / `BinaryComparisonKind.LESS_THAN` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 23 | `LE` / `BinaryComparisonKind.LESS_OR_EQUAL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 24 | `EQ` / `BinaryComparisonKind.EQUAL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 25 | `NE` / `BinaryComparisonKind.NOT_EQUAL` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D52` |
| 26 | `TENSOR_MIN` / `BinaryArithmeticKind.MIN` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 27 | `TENSOR_MAX` / `BinaryArithmeticKind.MAX` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 28 | `SCALAR_MIN` / `ScalarElementwiseKind.MIN` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 29 | `SCALAR_MAX` / `ScalarElementwiseKind.MAX` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 30 | `CLAMP` / `ScalarElementwiseKind.CLAMP` | `ClampRangeAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 31 | `REDUCTION_MIN` / `AggregateReductionKind.MIN` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 32 | `REDUCTION_MAX` / `AggregateReductionKind.MAX` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D52` |
| 33 | `CUM_SUM` / `CumulativeScanKind.CUM_SUM` | `CumulativeScanAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D52` |
| 34 | `CUM_PROD` / `CumulativeScanKind.CUM_PROD` | `CumulativeScanAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D52` |
| 35 | `SCALED_DOT_PRODUCT_ATTENTION` / `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` | `ScaledDotProductAttentionAttrs; 3..4 -> 1..2` | N | F | D/P | `BLOCKED` | `D/ND/FC` | `B-ATTN` |
| 36 | `CONV2D` / `Conv2dKind.CONV2D` | `Conv2dAttrs; 2..3 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D64-CONV` |
| 37 | `CONV3D` / `Conv3dKind.CONV3D` | `Conv3dAttrs; 2..3 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `FC-B` | `D64-CONV` |
| 38 | `TENSOR_POW` / `BinaryArithmeticKind.POW` | `NoOperationAttrs; 2 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-POWER` |
| 39 | `CAST` / `CastKind.CAST` | `CastAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-CAST` |
| 40 | `IS_FINITE` / `FloatingClassificationKind.IS_FINITE` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 41 | `IS_NAN` / `FloatingClassificationKind.IS_NAN` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 42 | `IS_INF` / `FloatingClassificationKind.IS_INF` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 43 | `LOGICAL_AND` / `BooleanLogicalKind.AND` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 44 | `LOGICAL_OR` / `BooleanLogicalKind.OR` | `NoOperationAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 45 | `LOGICAL_NOT` / `BooleanLogicalKind.NOT` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D66-BOOL` |
| 46 | `SCALAR_ADD` / `ScalarElementwiseKind.ADD` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D58-SCALAR` |
| 47 | `SCALAR_SUB` / `ScalarElementwiseKind.SUB` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D58-SCALAR` |
| 48 | `SCALAR_MUL` / `ScalarElementwiseKind.MUL` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D58-SCALAR` |
| 49 | `SCALAR_DIV` / `ScalarElementwiseKind.DIV` | `ScalarValueAttrs; 1 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D58-SCALAR` |
| 50 | `SCALAR_POW` / `ScalarElementwiseKind.POW` | `ScalarValueAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-POWER` |
| 51 | `WHERE` / `WhereSelectionKind.WHERE` | `NoOperationAttrs; 3 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-WHERE` |
| 52 | `RECIPROCAL` / `UnaryElementwiseKind.RECIPROCAL` | `NoOperationAttrs; 1 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D58-RECIP` |
| 53 | `LOG` / `UnaryElementwiseKind.LOG` | `NoOperationAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-ELEM` |
| 54 | `LOG1P` / `UnaryElementwiseKind.LOG1P` | `NoOperationAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-ELEM` |
| 55 | `EXP` / `UnaryElementwiseKind.EXP` | `NoOperationAttrs; 1 -> 1` | N | F | D/P | `BLOCKED` | `D` | `B-EXP` |
| 56 | `EXPM1` / `UnaryElementwiseKind.EXPM1` | `NoOperationAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-ELEM` |
| 57 | `ERF` / `UnaryElementwiseKind.ERF` | `NoOperationAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-ELEM` |
| 58 | `SQRT` / `UnaryElementwiseKind.SQRT` | `NoOperationAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-ELEM` |
| 59 | `RSQRT` / `UnaryElementwiseKind.RSQRT` | `NoOperationAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-ELEM` |
| 60 | `FLOOR` / `UnaryElementwiseKind.FLOOR` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D58-RAW` |
| 61 | `CEIL` / `UnaryElementwiseKind.CEIL` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D58-RAW` |
| 62 | `SIGN` / `UnaryElementwiseKind.SIGN` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D58-RAW` |
| 63 | `RELU` / `UnaryElementwiseKind.RELU` | `NoOperationAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D58-RAW` |
| 64 | `SIGMOID` / `UnaryElementwiseKind.SIGMOID` | `NoOperationAttrs; 1 -> 1` | N | F | D/P | `BLOCKED` | `D` | `B-EXP` |
| 65 | `TANH` / `UnaryElementwiseKind.TANH` | `NoOperationAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-RECURSIVE` |
| 66 | `GELU` / `UnaryElementwiseKind.GELU` | `NoOperationAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-RECURSIVE` |
| 67 | `GELU_TANH_APPROXIMATION` / `UnaryElementwiseKind.GELU_TANH_APPROXIMATION` | `NoOperationAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-RECURSIVE` |
| 68 | `SILU` / `UnaryElementwiseKind.SILU` | `NoOperationAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-RECURSIVE` |
| 69 | `GATHER_ELEMENTS` / `AxisGatherKind.GATHER_ELEMENTS` | `IndexAxisAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-IDX` |
| 70 | `SCATTER_ADD` / `AxisScatterKind.SCATTER_ADD` | `IndexAxisAttrs; 3 -> 1` | Y | F | D/P | `BLOCKED` | `D/ND/FC` | `B-ADDITIVE` |
| 71 | `GATHER_ND` / `GatherNdKind.GATHER_ND` | `GatherNdAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-MOVE` |
| 72 | `SCATTER_ND` / `ScatterNdKind.SCATTER_ND` | `ScatterNdAttrs; 3 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D66-MOVE` |
| 73 | `SELECT` / `SelectKind.SELECT` | `SelectAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 74 | `PAD` / `PadKind.PAD` | `PadAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 75 | `SLICE` / `SliceKind.SLICE` | `SliceAttrs/CropToShapeAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 76 | `SLICE_UPDATE` / `SliceKind.SLICE_UPDATE` | `SliceAttrs/CropToShapeAttrs; 2 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 77 | `CONCAT` / `TensorCompositionKind.CONCAT` | `CompositionAxisAttrs; 1..Integer.MAX_VALUE -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 78 | `STACK` / `TensorCompositionKind.STACK` | `CompositionAxisAttrs; 1..Integer.MAX_VALUE -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 79 | `TILE` / `TileKind.TILE` | `TileAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-MOVE` |
| 80 | `FOLD_AXIS` / `WindowTransformKind.FOLD_AXIS` | `FoldAxisAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-WINDOW` |
| 81 | `UNFOLD2D` / `WindowTransformKind.UNFOLD2D` | `Window2dAttrs/Unfold2dAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-WINDOW` |
| 82 | `FOLD2D` / `WindowTransformKind.FOLD2D` | `Fold2dAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D66-WINDOW` |
| 83 | `UNFOLD3D` / `WindowTransformKind.UNFOLD3D` | `Window3dAttrs/Unfold3dAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-WINDOW` |
| 84 | `FOLD3D` / `WindowTransformKind.FOLD3D` | `Fold3dAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D66-WINDOW` |
| 85 | `MEAN_SQUARED_ERROR` / `LossKind.MEAN_SQUARED_ERROR` | `MeanSquaredErrorAttrs; 2 -> 1` | Y | T | C/P | `MPSGRAPH` | `D` | `D62-MSE` |
| 86 | `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` / `LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | `DenseCategoricalCrossEntropyWithLogitsAttrs; 2 -> 1` | N | F | D/P | `BLOCKED` | `D` | `B-LOSS` |
| 87 | `INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` / `LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` | `IndexCategoricalCrossEntropyWithLogitsAttrs; 2 -> 1` | N | F | C/P | `BLOCKED` | `D/ND/FC` | `B-LOSS` |
| 88 | `BATCH_NORM_INFERENCE` / `BatchNormKind.BATCH_NORM_INFERENCE` | `BatchNormInferenceAttrs; 5 -> 1` | N | F | D/P | `BLOCKED` | `D` | `B-NORM` |
| 89 | `BATCH_NORM_TRAINING` / `BatchNormKind.BATCH_NORM_TRAINING` | `BatchNormTrainingAttrs; 5 -> 5` | N | F | C/P | `BLOCKED` | `D/ND/FC` | `B-NORM` |
| 90 | `LAYER_NORM` / `LayerNormKind.LAYER_NORM` | `LayerNormAttrs; 1 -> 1 / AffineLayerNormAttrs; 3 -> 1` | N | F | C/P | `BLOCKED` | `D` | `B-NORM` |
| 91 | `RMS_NORM` / `RmsNormKind.RMS_NORM` | `RmsNormAttrs; 1..2 -> 1` | N | F | C/P | `BLOCKED` | `D` | `B-NORM` |
| 92 | `SOFTMAX` / `SoftmaxKind.SOFTMAX` | `SoftmaxAttrs; 1 -> 1` | N | F | D/P | `BLOCKED` | `D` | `B-NORM` |
| 93 | `LOG_SOFTMAX` / `SoftmaxKind.LOG_SOFTMAX` | `SoftmaxAttrs; 1 -> 1` | N | F | C/P | `BLOCKED` | `D` | `B-NORM` |
| 94 | `SORT` / `OrderingKind.SORT` | `SortAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D63-ORDER` |
| 95 | `ARGSORT` / `OrderingKind.ARGSORT` | `SortAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `ND` | `D63-ORDER` |
| 96 | `TOP_K` / `TopKKind.TOP_K` | `TopKAttrs; 1 -> 2` | Y | T | C/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D63-ORDER` |
| 97 | `MAX_POOL2D` / `Pool2dKind.MAX_POOL2D` | `MaxPool2dAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D64-POOL` |
| 98 | `AVERAGE_POOL2D` / `Pool2dKind.AVERAGE_POOL2D` | `AveragePool2dAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D64-POOL` |
| 99 | `MAX_POOL3D` / `Pool3dKind.MAX_POOL3D` | `MaxPool3dAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D64-POOL` |
| 100 | `AVERAGE_POOL3D` / `Pool3dKind.AVERAGE_POOL3D` | `AveragePool3dAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `D` | `D64-POOL` |
| 101 | `DROPOUT` / `DropoutKind.DROPOUT` | `DropoutAttrs; 2 -> 3` | Y | T | U/A | `CUSTOM_PROGRAM` | `D/ND/FC` | `D65-RNG` |
| 102 | `INITIAL_STATE` / `GraphRngKind.INITIAL_STATE` | `GraphRngStateAttrs; 0 -> 1` | Y | T | U/A | `CUSTOM_PROGRAM` | `ND` | `D65-RNG` |
| 103 | `RNN_TANH` / `RecurrentScanKind.RNN_TANH` | `RecurrentDirection; 5..6 -> 2` | N | F | U/P | `BLOCKED` | `FC-B` | `B-RNN` |
| 104 | `GRU_RESET_AFTER` / `RecurrentScanKind.GRU_RESET_AFTER` | `RecurrentDirection; 5..6 -> 2` | N | F | U/P | `BLOCKED` | `FC-B` | `B-RNN` |
| 105 | `LSTM` / `RecurrentScanKind.LSTM` | `RecurrentDirection; 6..7 -> 3` | N | F | U/P | `BLOCKED` | `FC-B` | `B-RNN` |
| 106 | `PROD` / `AggregateReductionKind.PROD` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `D` | `D60-EXACT-RED` |
| 107 | `ALL` / `AggregateReductionKind.ALL` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D60-EXACT-RED` |
| 108 | `ANY` / `AggregateReductionKind.ANY` | `NoOperationAttrs/AxisReductionAttrs/MultiAxisReductionAttrs; 1 -> 1` | Y | T | D/A | `CUSTOM_PROGRAM` | `ND` | `D60-EXACT-RED` |
| 109 | `ARG_MAX` / `AggregateReductionKind.ARG_MAX` | `ArgExtremaAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `ND` | `D63-ARG` |
| 110 | `ARG_MIN` / `AggregateReductionKind.ARG_MIN` | `ArgExtremaAttrs; 1 -> 1` | Y | T | C/A | `CUSTOM_PROGRAM` | `ND` | `D63-ARG` |
| 111 | `LOG_SUM_EXP` / `AggregateReductionKind.LOG_SUM_EXP` | `MultiAxisReductionAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-AGG` |
| 112 | `VARIANCE` / `AggregateReductionKind.VARIANCE` | `StatisticalReductionAttrs; 1 -> 1` | Y | F | D/P | `BLOCKED` | `D` | `B-AGG` |
| 113 | `STANDARD_DEVIATION` / `AggregateReductionKind.STANDARD_DEVIATION` | `StatisticalReductionAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-AGG` |
| 114 | `L1_NORM` / `AggregateReductionKind.L1_NORM` | `MultiAxisReductionAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-AGG` |
| 115 | `L2_NORM` / `AggregateReductionKind.L2_NORM` | `MultiAxisReductionAttrs; 1 -> 1` | Y | F | C/P | `BLOCKED` | `D` | `B-AGG` |

## Production-domain evidence codes

| Code | Exact current bounded domain and primary evidence |
|---|---|
| `D15` | Wires 2–5: ACCELERATOR FLOAT32 static canonical/broadcast arithmetic under current gradient-role constraints; `supportsBinary`, schema/native arithmetic validation, MPSGraph execution, Task 0015 tests. |
| `D20-RED` | Wires 13–14: ACCELERATOR FLOAT32 full/axis/multi-axis and admitted SUM-to-shape reductions with exact output geometry; `supportsReduction`, reduction encoding/native validation, Tasks 0020/0062/0066 generated-graph tests. |
| `D22` | Wires 1 and 12: profile-common exact FLOAT32 canonical NEG/ABS domain; bounded NEG dual route and direct ABS, Task 0022 route/native/conformance evidence. |
| `D52` | Wires 20–34: ACCELERATOR canonical positive-static FLOAT32 comparison/extrema/scan domain with exact BOOL or FLOAT32 outputs and no unsupported gradient roles; fixed Task-0052 custom route and retained proof/native tests. |
| `D58-RAW` | Wires 60–63: profile-common exact canonical FLOAT32 FLOOR/CEIL/SIGN/RELU raw-word domain; fixed Task-0058 custom route and native/public tests. |
| `D58-RECIP` | Wire 52: ACCELERATOR canonical positive-static FLOAT32 no-gradient reciprocal composition; `supportsNoGradReciprocal`, MPSGraph composition, Task-0058 tests. |
| `D58-SCALAR` | Wires 46–49: ACCELERATOR canonical positive-static FLOAT32 no-gradient scalar ADD/SUB/MUL/DIV with FLOAT32 scalar attributes; `supportsScalar`, MPSGraph composition, Task-0058 tests. |
| `D60-EXACT-RED` | Wires 106–108: profile-common exact integral PROD and BOOL ALL/ANY over admitted axes; fixed custom route, Task-0060 capability/native/public evidence. |
| `D61-MM` | Wire 15: exact promoted integral profile-common and ACCELERATOR floating vector/matrix/batched/broadcast MATMUL domains; occurrence-selective fixed MPSGraph/custom route, Task-0061 type/shape/native/generated-gradient evidence. |
| `D62-MSE` | Wire 85: ACCELERATOR same-type positive-static FLOAT32 MSE NONE/SUM/MEAN with exact gradient-flag relation; fixed MPSGraph composition (including nested execution in a custom whole program), Task-0062 tests. |
| `D63-ARG` | Wires 109–110: profile-common numeric canonical positive-static single-axis ARG_MAX/ARG_MIN with INT64 output and exact tie/order policy; fixed Task-0063 custom route and native/public tests. |
| `D63-ORDER` | Wires 94–96: profile-common all-six-carrier SORT/ARGSORT/TOP_K positive-static domains with exact stable order, INT64 indices, and TOP_K two-output roles; fixed Task-0063 custom route and saved-role tests. |
| `D64-CONV` | Wires 36–37: ACCELERATOR static grouped Conv2d/Conv3d FLOAT32-result domains over FLOAT32/BFLOAT16 roles; BFLOAT16-containing forms are no-grad and Conv3d generated backward remains explicitly fail-closed; fixed custom route and Task-0064 tests. |
| `D64-POOL` | Wires 97–100: profile-common three-floating-carrier exact maximum Pool2d/3d and ACCELERATOR FLOAT32 average Pool2d/3d; fixed custom route, Task-0064 winner/divisor/native/public evidence. |
| `D65-RNG` | Wires 101–102: profile-common raw INITIAL_STATE and ACCELERATOR FLOAT32 explicit-state DROPOUT with output/mask/next-state roles; fixed custom route, Task-0065 replay/branch/session/backward evidence. |
| `D66-AFF` | Wires 6–11: profile-common all-six-carrier static rank-0..16 RESHAPE/EXPAND/PERMUTE/EXPAND_DIMS/SQUEEZE/CONTIGUOUS over authenticated logical/physical layouts; fixed custom route and Task-0066 raw/public/generated-gradient evidence. |
| `D66-BOOL` | Wires 40–45: profile-common three-floating-carrier classification and BOOL logical operations over admitted scalar/broadcast/affine occurrences; fixed custom route and Task-0057/0066 tests. |
| `D66-CAST` | Wire 39: all 36 ordered casts over the six Model carriers, with floating-to-floating differentiation only and exact integer-defined conversion; fixed custom route and Task-0059/0066 exhaustive tests. |
| `D66-IDX` | Wires 16, 18–19, and 69: profile-common admitted all-carrier GATHER/SCATTER replacement/UNFOLD_AXIS movement with INT32/INT64 index parity where applicable, scalar support, bounds/uniqueness before mutation, and no additive semantics. Wire 17 ONE_HOT instead admits selected affine/materialized INT32/INT64 scalar-or-tensor indices (including authenticated positive- and zero-stride views), positive depth, and canonical BOOL output whose final dimension is depth. All use fixed custom routes and Task-0066 tests. |
| `D66-MOVE` | Wires 71–79: profile-common exact all-carrier ND/replacement/select/pad/slice/composition/tile movement over authenticated static layouts; fixed custom route and Task-0059/0060/0066 tests. |
| `D66-WHERE` | Wire 51: profile-common BOOL-conditioned admitted floating promotion/broadcast selection with saved condition role and exact output carrier; fixed custom route and Task-0057/0066 tests. |
| `D66-WINDOW` | Wires 80–84: admitted non-overlap folds plus 2D/3D window movement; FOLD_AXIS excludes BOOL, image windows/folds use FLOAT64/FLOAT32/BFLOAT16, and overlap/additive cases remain false; fixed custom route and Task-0059/0060/0066 tests. |

## Production-false blocker codes

| Code | Wires | Current exact blocker |
|---|---|---|
| `B-ATTN` | 35 | No complete exact attention contraction/scale/mask/softmax/output/gradient proof; historical opaque selector evidence remains blocked. |
| `B-POWER` | 38, 50 | Tensor/scalar power lacks complete exact domain and exponent/special-class proof. |
| `B-ELEM` | 53–54, 56–59 | LOG/LOG1P/EXPM1/ERF/SQRT/RSQRT lack complete profile-qualified exact primitive/composition proof. |
| `B-EXP` | 55, 64 | Task 0053 remains blocked on its pinned constructive-real EXP bridge; no candidate is production-authorized. |
| `B-RECURSIVE` | 65–68 | TANH/GELU/GELU-tanh/SILU depend on blocked elementary/EXP/SIGMOID sites or lack complete exact formula proof. |
| `B-ADDITIVE` | 70 | SCATTER_ADD requires collision/order/additive arithmetic and generated-gradient proof; replacement infrastructure is not authorization. |
| `B-LOSS` | 86–87 | Categorical losses retain stable softmax/log/guard/target/ignore/reduction and generated-gradient blockers. |
| `B-NORM` | 88–93 | Batch/layer/RMS normalization and softmax/log-softmax retain aggregate/elementary/state/multi-output and gradient blockers. |
| `B-RNN` | 103–105 | Recurrent rows lack complete TANH/EXP/SIGMOID proof, runtime valid-length/state contracts, custom recurrence, and Compiler BPTT. |
| `B-AGG` | 111–115 | LOG_SUM_EXP/VARIANCE/STANDARD_DEVIATION/L1/L2 need complete contributor/order/correction/elementary/generated-gradient proof. Existing primitives are dependencies, not authorization. |

## Production-false structural partition

Production false does not mean structural machinery is absent. The 32 `P=F` rows split exactly and
disjointly into:

- `S=Y/P=F` (18 retained non-authorizing recipes): wires
  `38,50,53,54,56,57,58,59,65,66,67,68,70,111,112,113,114,115`;
- `S=N/P=F` (14 structurally unavailable rows): wires
  `35,55,64,86,87,88,89,90,91,92,93,103,104,105`.

Catalog and native structural recipes for the first set are evidence about encodability only. They
must remain unreachable from production unless an exact Model occurrence first passes the
capability gate and authenticated candidate/preparation path; neither the catalog nor forceable
test-only structure authorizes capability.

## Mechanical reconciliation

The implementation phase must add one permanent source-backed audit test/artifact that fails unless:

1. current Model declarations total 40 families, 115 constants, and 137 exact signatures;
2. every wire `1..115` maps to exactly one Model kind and every Model kind maps back once;
3. structural `101/14`, capability-kind `83/32`, MPSGraph `75/35/5`, and custom `70/45/0`
   partitions are disjoint and exhaustive, including the exact `18 S=Y/P=F` and
   `14 S=N/P=F` split;
4. the 83 true rows each have an actual accepted representative occurrence plus exact preparation,
   route, Java/native schema, execution, and behavioral evidence rather than membership arithmetic;
5. no production-false Model occurrence is accepted by `MetalCapabilityProvider` or reaches an
   ordinary or test-forced production preparation route without first passing the exact capability
   and candidate-authentication gates; retained structural lowering/native recipes are inventoried
   as non-authorizing, and active docs make no production-capability claim for them;
6. generated backward claims are checked from the actual Compiler-emitted graph, including saved
   condition/index/mask/state roles and forward-only Conv3d/recurrent boundaries; the supported
   formula registry remains exactly 38 families / 111 constants / 133 fingerprints and its explicit
   deferred set is exactly Conv3d plus the three recurrent signatures, so the combined partition is
   40 / 115 / 137 without claiming formulas for deferred rows;
7. schema 15, ABI 5, thirteen exports, 42 attribute wires, three route wires, identity 22, and
   identity-21 rejection remain exact; and
8. active docs agree while historical tasks/evidence remain byte-unchanged.

Representative positives prove reachability only for the exact coded domain. Boundary negatives must
cover profile, carrier, attrs, cardinality, rank/shape/layout, gradient flags, route forcing, schema,
identity, and native preflight where applicable. Route-specific resource evidence is mandatory:

| Route | Required valid topology | Required rejection and fail-before-mutation evidence |
|---|---|---|
| `MPSGRAPH` | Inputs may intentionally repeat; outputs are pairwise distinct and distinct from every input. | Prove a repeated-input positive case. Reject every input/output and output/output alias in Java before downcall and at raw native run; targets retain sentinels. |
| `CUSTOM_PROGRAM` | Complete value table is pairwise distinct; each `output[i]` equals exactly `values[targetValueIndices[i]]`. | Reject duplicate value handles, wrong/swapped/duplicate target handles, and extra aliases in Java and raw native paths before execution; inputs and target sentinels remain unchanged. |
| `CUSTOM_SINGLE_NEG` | Distinct input and output. | Reject input/output alias before dispatch and retain sentinels. |

## Audit findings and dispositions

- `C-001` — `FirstOrderGradientCoverage` intentionally owns only the 38-family / 111-kind /
  133-fingerprint supported-formula inventory. `AutogradPreflight` rejects Conv3d plus RNN/GRU/LSTM
  before Tensor construction, and `FirstOrderGradientCoverageTest` proves the supported and deferred
  partitions are disjoint and combine to the full 40 / 115 / 137 Model inventory. This forward-only
  boundary remains unchanged; no nonexistent formula or owner was added.
- `A-001` — the former `83/32` test derived production truth as `executable - 18` from a manually
  repeated set and did not prove that all 83 rows had a constructible capability-positive
  occurrence or exact downstream route. Commit `7f9601e1` replaces it with the canonical
  wire/Model mapping, one real capability query per row, and exact structural/catalog/route totals.
- `A-002` — root `README.md` said backend-local route/workload identities were version 20 although
  current source and other active documents use version 22 and reject version 21 and older.
  Commit `7f9601e1` corrects the active-document mismatch without rewriting historical evidence.
- `A-003` — Java and raw-native alias evidence did not close the opposite MPSGraph/custom-program
  matrices. Commit `7f9601e1` adds repeated-input MPSGraph success plus MPSGraph input/output
  rejection, and exact custom target-alias success plus duplicate-value, wrong, swapped, and
  duplicate-target rejection with unchanged sentinels.
- `A-004` — the public Engine test still expected profile-common INT32 `FOLD_AXIS` to be unowned,
  contrary to Task-0066 capability, route, and native execution. Commit `2541c6f3` executes the
  exact integral result through Metal under both profiles while retaining Model-level BOOL
  `FOLD_AXIS` and integral `FOLD2D`/`FOLD3D` rejection.
- `A-005` — the public Engine test and active Metal guide still said jointly requested all-FLOAT32
  Conv2d input/weight gradients were unowned even though the Compiler-generated partition is
  primitive-closed. Commit `2541c6f3` executes the joint forward/input/weight publications and
  corrects the active boundary while retaining mixed-carrier and Conv3d gradient rejection.
- `A-006` — the ordering integration test and active Metal documentation still expected generated
  floating `SORT` and `TOP_K` values-output backward ownership to fail despite Task-0066 closure.
  Commit `2541c6f3` executes both generated gradients through Metal with exact values; the review
  remediation preserves the pre-existing public Engine rejection for a Model-valid ordering extent
  above `UINT32_MAX`, and ordering index roles remain non-differentiable.

Other findings remain audit output, not assumptions. In particular `L1_NORM`, variance, and every
other false row must stay false until complete Model/profile/formula/order/gradient/native evidence
is reviewed; the presence of some primitive dependencies is not enough.

## Serial workflow

1. Commit this plan/ledger from clean base and obtain independent plan/evidence review.
2. Mechanically generate and verify the canonical mapping and representative 83/32 reachability.
3. Audit Compiler inference/autograd/saved roles and direct/generated backward for all 115 rows.
4. Audit preparation/schema/catalog/route/identity and Java/native validation/execution for all rows.
5. Audit behavioral/public Engine evidence and active docs without editing historical evidence.
6. Record every mismatch with a wire, severity, owner, exact evidence, and disposition.
7. Remediate every in-scope source/test/doc mismatch end to end. A false-to-true finding first needs
   a concrete reviewed amendment; otherwise it remains a separately reported blocker.
8. Run focused behavioral smoke, complete relevant suites, native package verification when native
   source changes, full test/build, documentation/diff checks, and independent cumulative
   code/security/evidence review with zero P0/P1/P2.

## Validation contract

No benchmark or timing command. Final validation includes the permanent 115-row audit check,
focused Compiler gradient coverage and Metal capability/catalog/route tests, any affected real
native/public Engine smoke, complete Metal and focused Compiler/conformance/integration suites,
Javadoc/architecture, `./gradlew test`, `./gradlew build`, and `git diff --check`. The final tree must
be clean and current docs must record exact findings, remediations, remaining blockers, commits, and
independent approvals.
