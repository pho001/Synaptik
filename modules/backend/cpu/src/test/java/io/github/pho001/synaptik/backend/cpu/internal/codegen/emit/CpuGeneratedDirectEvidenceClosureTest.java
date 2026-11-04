package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Shared generated fixtures and consumer-visible invalid-role boundaries. */
class CpuGeneratedDirectEvidenceClosureTest {

    /** Records actual selected pointwise facts for the existing diagnostic inventory. */
    void pointwiseCandidateUniverseClosesRequestedCarrierLayoutAndStrategyAxes() throws Exception {
        for (PointwiseCandidate candidate : pointwiseCandidates()) {
            PointwiseFixture fixture = candidate.fixture();
            var context = fixture.context(candidate.inputLayout(), candidate.inputs(), candidate.execution(),
                    candidate.materializationPolicy());
            io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertCurrentSupport(
                    new CpuCapabilityProvider(), new OperationCapabilityQuery(fixture.operation(),
                            descriptors(context, context.nodes().getFirst().inputs()),
                            descriptors(context, context.nodes().getFirst().outputs())), candidate.id());
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            CpuGeneratedCoverageEvidenceRegistry.generated("pointwise-matrix:" + candidate.id(), context, plan);
        }
    }


    @Test void pointwiseInvalidTypeRolesAndShapesRemainRejected() {
        // Mixed same-category operands now execute; these are still-invalid descriptor roles.
        var candidates = rejectedPointwiseCandidates();
        assertEquals(candidates.size(), candidates.stream().map(PointwiseRejectedCandidate::id).distinct().count());
        assertTrue(candidates.stream().allMatch(candidate -> Set.of("INVALID_RESULT_ROLE",
                "INVALID_BOOLEAN_NUMERIC_ROLE", "SHAPE_INCOMPATIBLE").contains(candidate.reason())));
        var provider = new CpuCapabilityProvider();
        for (PointwiseRejectedCandidate candidate : candidates) {
            assertFalse(provider.supports(new OperationCapabilityQuery(candidate.operation(), candidate.inputs(), List.of(candidate.output()))), candidate.id() + ": " + candidate.reason());
        }
    }


    static List<TensorDescriptor> descriptors(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<ValueId> ids) {
        return ids.stream().map(id -> context.values().stream().filter(value -> value.id().equals(id))
                .findFirst().orElseThrow().descriptor()).toList();
    }


    /**
     * Evidence-local attention occurrence.  It stays here because only the generated-coverage
     * audit needs to open this duplicate-role boundary; lowering tests retain package visibility.
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> attentionContext(boolean duplicateQueryKey) {
        Shape q = Shape.of(2, 2);
        TensorDescriptor query = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor key = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor value = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor output = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        return CpuScatterLoweringTest.context(new Operation(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                duplicateQueryKey ? List.of(0, 0, 1) : List.of(0, 1, 2),
                duplicateQueryKey ? List.of(query, value) : List.of(query, key, value), output);
    }


    static List<NonPointwiseFixture> nonPointwiseFixtures() {
        return List.of(
                nonPointwise("pad", CpuNonAffineMovementLoweringTest.context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-7))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("tile", CpuNonAffineMovementLoweringTest.context(new Operation(TileKind.TILE,
                        new TileAttrs(List.of(3L))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(6)))),
                nonPointwise("concat", CpuNonAffineMovementLoweringTest.context(new Operation(
                        TensorCompositionKind.CONCAT, new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(1))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("stack", CpuNonAffineMovementLoweringTest.context(new Operation(
                        TensorCompositionKind.STACK, new CompositionAxisAttrs(1)), List.of(0, 1),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2)))),
                nonPointwise("unfold-axis", CpuNonAffineMovementLoweringTest.context(new Operation(
                        WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 2, 1)), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 2, 2)))),
                nonPointwise("unfold2d", CpuNonAffineMovementLoweringTest.context(new Operation(
                        WindowTransformKind.UNFOLD2D, new Unfold2dAttrs(CpuFoldLoweringTest.window(false),
                                ScalarValue.float64(0))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.FLOAT64, Shape.of(1, 1, 3, 3))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.FLOAT64, Shape.of(1, 4, 4)))),
                nonPointwise("slice-update", CpuNonAffineMovementLoweringTest.context(new Operation(
                        SliceKind.SLICE_UPDATE, new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-2L))),
                        List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("gather", CpuIndexingLoweringTest.context(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                        CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, Shape.of(2, 2)))),
                nonPointwise("gather-elements", CpuIndexingLoweringTest.context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                        new IndexAxisAttrs(1)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2))),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2)))),
                nonPointwise("gather-nd", CpuIndexingLoweringTest.context(new Operation(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(0)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 1))),
                        CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)))),
                nonPointwise("one-hot", CpuIndexingLoweringTest.context(new Operation(OneHotKind.ONE_HOT,
                        new OneHotAttrs(3)), List.of(0), List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                        CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)))),
                nonPointwise("scatter-elements-min", scatter(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MIN))),
                nonPointwise("scatter-add", CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ADD,
                        new IndexAxisAttrs(1)), List.of(0, 1, 2), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(5, 2)), CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 5, 2, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)))),
                nonPointwise("scatter-nd", CpuScatterLoweringTest.context(new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(1, ScatterReduction.MAX)), List.of(0, 1, 2), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 5, 1)), CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 5, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)))),
                nonPointwise("fold-axis", CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.INT64, Shape.of(3, 3), Shape.of(5))),
                nonPointwise("fold2d", CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD2D,
                        new io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs(Shape.of(1, 1, 3, 3), CpuFoldLoweringTest.window(false))),
                        DataType.FLOAT32, Shape.of(1, 4, 4), Shape.of(1, 1, 3, 3))),
                nonPointwise("sort", CpuOrderingLoweringTest.context(new Operation(OrderingKind.SORT,
                        new SortAttrs(1, false)), DataType.FLOAT64, Shape.of(2, 4), Shape.of(2, 4), false)),
                nonPointwise("argsort", CpuOrderingLoweringTest.context(new Operation(OrderingKind.ARGSORT,
                        new SortAttrs(1, true)), DataType.INT64, Shape.of(4, 8), Shape.of(4, 8), false)),
                nonPointwise("top-k", CpuOrderingLoweringTest.context(new Operation(TopKKind.TOP_K,
                        new TopKAttrs(1, 2, true, false)), DataType.FLOAT32, Shape.of(3, 5), Shape.of(3, 2), true)),
                nonPointwise("dropout-f64", CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(2, 3), .25d)),
                nonPointwise("dropout-f32", CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32, Shape.of(2, 3), .25d)),
                nonPointwise("cum-sum-inclusive-forward", CpuScanLoweringTest.context(CumulativeScanKind.CUM_SUM,
                        DataType.FLOAT32, Shape.of(2, 3), 1, false, false)),
                nonPointwise("cum-sum-exclusive-reverse", CpuScanLoweringTest.context(CumulativeScanKind.CUM_SUM,
                        DataType.INT64, Shape.of(2, 3), 1, true, true)),
                nonPointwise("cum-prod-inclusive-forward", CpuScanLoweringTest.context(CumulativeScanKind.CUM_PROD,
                        DataType.BFLOAT16, Shape.of(2, 3), 1, false, false)),
                nonPointwise("cum-prod-exclusive-reverse", CpuScanLoweringTest.context(CumulativeScanKind.CUM_PROD,
                        DataType.INT64, Shape.of(4, 8), 1, true, true)),
                nonPointwise("aggregate-min", CpuAggregateLoweringTest.context(AggregateReductionKind.MIN,
                        DataType.BFLOAT16, Shape.of(4, 8), new AxisReductionAttrs(1, false), Shape.of(4))),
                nonPointwise("aggregate-mean", CpuAggregateLoweringTest.context(AggregateReductionKind.MEAN,
                        DataType.FLOAT32, Shape.of(4, 8), new AxisReductionAttrs(1, false), Shape.of(4))),
                nonPointwise("log-sum-exp", advancedContext(AggregateReductionKind.LOG_SUM_EXP)),
                nonPointwise("variance", advancedContext(AggregateReductionKind.VARIANCE)),
                nonPointwise("l2-norm", advancedContext(AggregateReductionKind.L2_NORM)));
    }

    /* Ordinary non-pointwise closure moved to CpuOrdinaryNonPointwiseGeneratedMatrixTest. */
    static List<NonPointwiseFixture> ordinaryFixtureSeeds() {
        var fixtures = new ArrayList<NonPointwiseFixture>();
        // Keep only the families owned by this ordinary increment; advanced reductions have
        // their own closure rows and must not be accidentally counted here.
        fixtures.addAll(nonPointwiseFixtures().stream().filter(fixture -> !Set.of(
                "scatter-elements-min", "scatter-add", "scatter-nd",
                "cum-sum-inclusive-forward", "cum-sum-exclusive-reverse",
                "cum-prod-inclusive-forward", "cum-prod-exclusive-reverse",
                "aggregate-min", "aggregate-mean", "log-sum-exp", "variance", "l2-norm")
                .contains(fixture.id())).toList());
        for (ScatterReduction reduction : ScatterReduction.values()) {
            fixtures.add(nonPointwise("scatter-elements-" + reduction.name().toLowerCase(),
                    scatter(AxisScatterKind.SCATTER_ELEMENTS,
                            new ScatterElementsAttrs(0, reduction))));
            fixtures.add(nonPointwise("scatter-nd-" + reduction.name().toLowerCase(),
                    CpuScatterLoweringTest.context(new Operation(ScatterNdKind.SCATTER_ND,
                            new ScatterNdAttrs(1, reduction)), List.of(0, 1, 2),
                            List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)),
                                    CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 5, 1)),
                                    CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 5, 4))),
                            CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)))));
        }
        fixtures.add(nonPointwise("scatter-add", CpuScatterLoweringTest.context(new Operation(
                AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1)), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(5, 2)),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 5, 2, 4))),
                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)))));
        List<DataType> scanTypes = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.INT64, DataType.INT32);
        int scanType = 0;
        for (CumulativeScanKind kind : CumulativeScanKind.values()) for (boolean exclusive : List.of(false, true))
            for (boolean reverse : List.of(false, true)) {
                DataType type = scanTypes.get(scanType++ % scanTypes.size());
                fixtures.add(nonPointwise("scan-" + kind.name().toLowerCase() + '-' + exclusive + '-' + reverse,
                        CpuScanLoweringTest.context(kind, type, Shape.of(2, 3), 1, exclusive, reverse)));
            }
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN,
                AggregateReductionKind.MAX, AggregateReductionKind.ALL, AggregateReductionKind.ANY,
                AggregateReductionKind.SUM, AggregateReductionKind.MEAN, AggregateReductionKind.PROD)) {
            DataType type = kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY
                    ? DataType.BOOL : kind == AggregateReductionKind.MEAN ? DataType.FLOAT32 : DataType.INT64;
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-full",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8),
                            NoOperationAttrs.INSTANCE, Shape.scalar())));
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-axis",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8),
                            new AxisReductionAttrs(1, false), Shape.of(4))));
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-multi",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(2, 4, 8),
                            new MultiAxisReductionAttrs(List.of(0, 2), false), Shape.of(4))));
        }
        fixtures.add(nonPointwise("sum-to-shape", CpuAggregateLoweringTest.context(
                AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(2, 3, 4),
                new SumToShapeAttrs(Shape.of(3, 1)), Shape.of(3, 1))));
        fixtures.add(nonPointwise("initial-state", CpuRandomLoweringTest.initialContext(0x1234L, -7L)));
        assertEquals(60, fixtures.size());
        return List.copyOf(fixtures);
    }

    static List<NonPointwiseFixture> specializedGeneratedFixtures() {
        return List.of(
                // Arg extrema is a generated reduction family with a numeric input and an
                // INT64 result role; it must not be silently covered by ordinary aggregate
                // witnesses, whose output role is the input type.
                nonPointwise("arg-min", CpuArgExtremaLoweringTest.context(
                        AggregateReductionKind.ARG_MIN, DataType.FLOAT32, Shape.of(2, 3, 4),
                        1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                nonPointwise("arg-max", CpuArgExtremaLoweringTest.context(
                        AggregateReductionKind.ARG_MAX, DataType.INT64, Shape.of(2, 3, 4),
                        1, true, ArgExtremaTiePolicy.LAST_INDEX)),
                nonPointwise("masked-sum", CpuMaskedReductionLoweringTest.context(
                        AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2, 3, 4), Shape.of(3, 1), 1)),
                nonPointwise("masked-mean", CpuMaskedReductionLoweringTest.context(
                        AggregateReductionKind.MEAN, DataType.FLOAT64, Shape.of(2, 3, 4), Shape.of(2, 1, 4), 1)),
                nonPointwise("advanced-log-sum-exp", advancedContext(AggregateReductionKind.LOG_SUM_EXP)),
                nonPointwise("advanced-variance", advancedContext(AggregateReductionKind.VARIANCE)),
                nonPointwise("advanced-standard-deviation", advancedContext(AggregateReductionKind.STANDARD_DEVIATION)),
                nonPointwise("advanced-l1-norm", advancedContext(AggregateReductionKind.L1_NORM)),
                nonPointwise("advanced-l2-norm", advancedContext(AggregateReductionKind.L2_NORM)),
                nonPointwise("softmax", CpuSoftmaxLoweringTest.context(SoftmaxKind.SOFTMAX,
                        DataType.FLOAT32, Shape.of(2, 3, 4), 1)),
                nonPointwise("log-softmax", CpuSoftmaxLoweringTest.context(SoftmaxKind.LOG_SOFTMAX,
                        DataType.FLOAT64, Shape.of(2, 3, 4), 1)),
                nonPointwise("layer", CpuTrailingNormalizationLoweringTest.context(true, false,
                        DataType.FLOAT32, Shape.of(2, 3), Shape.of(3), List.of(0))),
                nonPointwise("layer-affine", CpuTrailingNormalizationLoweringTest.context(true, true,
                        DataType.FLOAT64, Shape.of(2, 3), Shape.of(3), List.of(0, 1, 2))),
                nonPointwise("rms", CpuTrailingNormalizationLoweringTest.context(false, false,
                        DataType.BFLOAT16, Shape.of(2, 3), Shape.of(3), List.of(0))),
                nonPointwise("rms-scaled", CpuTrailingNormalizationLoweringTest.context(false, true,
                        DataType.FLOAT32, Shape.of(2, 3), Shape.of(3), List.of(0, 1))),
                nonPointwise("batch-norm-inference", CpuBatchNormInferenceLoweringTest.context(
                        List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
                                DataType.FLOAT32), Shape.of(2, 3, 4), 1, List.of(0, 1, 2, 3, 4))),
                nonPointwise("batch-norm-training", CpuBatchNormTrainingLoweringTest.context(
                        Shape.of(2, 3, 4), 1)),
                nonPointwise("conv2d", CpuConv2dLoweringTest.context(List.of(DataType.FLOAT32,
                        DataType.FLOAT32, DataType.FLOAT32), Shape.of(1, 2, 4, 4), Shape.of(2, 2, 3, 3),
                        Shape.of(1, 2, 2, 2), Conv2dAttrs.defaults(), null)),
                nonPointwise("conv3d", CpuConv3dLoweringTest.context(List.of(DataType.FLOAT32,
                        DataType.FLOAT32), Shape.of(1, 2, 4, 4, 4), Shape.of(2, 2, 3, 3, 3),
                        Shape.of(1, 2, 2, 2, 2), Conv3dAttrs.defaults(), null)),
                nonPointwise("max-pool2d", CpuPool2dLoweringTest.context(Pool2dKind.MAX_POOL2D,
                        new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false), DataType.FLOAT32,
                        Shape.of(1, 2, 3, 3), Shape.of(1, 2, 2, 2))),
                nonPointwise("average-pool2d", CpuPool2dLoweringTest.context(Pool2dKind.AVERAGE_POOL2D,
                        new AveragePool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false), DataType.FLOAT64,
                        Shape.of(1, 2, 3, 3), Shape.of(1, 2, 2, 2))),
                nonPointwise("max-pool3d", CpuPool3dLoweringTest.context(Pool3dKind.MAX_POOL3D,
                        new MaxPool3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false), DataType.FLOAT32,
                        Shape.of(1, 2, 3, 3, 3), Shape.of(1, 2, 2, 2, 2))),
                nonPointwise("average-pool3d", CpuPool3dLoweringTest.context(Pool3dKind.AVERAGE_POOL3D,
                        new AveragePool3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false), DataType.FLOAT64,
                        Shape.of(1, 2, 3, 3, 3), Shape.of(1, 2, 2, 2, 2))),
                nonPointwise("matmul", CpuMatmulLoweringTest.context(DataType.FLOAT32, Shape.of(2, 3),
                        Shape.of(3, 4), Shape.of(2, 4))),
                nonPointwise("attention", attentionContext(true)),
                lossFixture("mse", new Operation(LossKind.MEAN_SQUARED_ERROR,
                        new MeanSquaredErrorAttrs(LossReduction.MEAN)), List.of(0, 1),
                        List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3)),
                                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())),
                lossFixture("dense-cross-entropy", new Operation(
                        LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN)), List.of(0, 1),
                        List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())),
                lossFixture("index-cross-entropy", new Operation(
                        LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN, Optional.empty())),
                        List.of(0, 1), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())));
    }

    private static NonPointwiseFixture lossFixture(String id, Operation operation, List<Integer> roles,
            List<TensorDescriptor> inputs, TensorDescriptor output) {
        return nonPointwise(id, CpuScatterLoweringTest.context(operation, roles, inputs, output));
    }

    static PrepareContext<CpuPartitionAnalysisInputs> scatter(AxisScatterKind kind,
            ScatterElementsAttrs attrs) {
        return CpuScatterLoweringTest.context(new Operation(kind, attrs), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(4))),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> advancedContext(AggregateReductionKind kind) {
        var attrs = kind == AggregateReductionKind.VARIANCE || kind == AggregateReductionKind.STANDARD_DEVIATION
                ? new StatisticalReductionAttrs(List.of(1), false, 1)
                : new MultiAxisReductionAttrs(List.of(1), false);
        return CpuAggregateLoweringTest.context(kind, DataType.FLOAT64, Shape.of(4, 8), attrs, Shape.of(4));
    }

    static NonPointwiseFixture nonPointwise(String id,
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new NonPointwiseFixture(id, context);
    }

    record NonPointwiseFixture(String id, PrepareContext<CpuPartitionAnalysisInputs> context) { }

    static List<PrepareContext<CpuPartitionAnalysisInputs>> coverageFixtureContexts() {
        var contexts = new ArrayList<PrepareContext<CpuPartitionAnalysisInputs>>();
        for (PointwiseFixture fixture : generalPointwiseFixtures()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : nonPointwiseFixtures()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : ordinaryFixtureSeeds()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : specializedGeneratedFixtures()) contexts.add(fixture.context());
        contexts.add(CpuRandomLoweringTest.initialContext(0x1234L, -7L));
        return List.copyOf(contexts);
    }

    static List<PointwiseFixture> generalPointwiseFixtures() {
        Shape shape = Shape.of(8);
        var forms = new ArrayList<PointwiseFixture>();
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) for (DataType type : numericTypes())
            if (type.isFloating() || (kind != BinaryArithmeticKind.DIV && kind != BinaryArithmeticKind.POW))
                forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(type, type), type, CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) for (DataType type : floatingTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type), type,
                    kind == UnaryElementwiseKind.GELU ? CpuPointwiseOpcode.GELU_EXACT
                            : CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (FloatingClassificationKind kind : FloatingClassificationKind.values()) for (DataType type : floatingTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type), DataType.BOOL,
                    CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) for (DataType type : numericTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type, type), DataType.BOOL,
                    CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) forms.add(fixture(kind.name(),
                new Operation(kind, NoOperationAttrs.INSTANCE), java.util.Collections.nCopies(
                        kind == BooleanLogicalKind.NOT ? 1 : 2, DataType.BOOL), DataType.BOOL,
                kind == BooleanLogicalKind.NOT ? CpuPointwiseOpcode.LOGICAL_NOT
                        : CpuPointwiseOpcode.valueOf("LOGICAL_" + kind.name()), shape));
        for (DataType type : floatingTypes()) forms.add(fixture("WHERE_" + type,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE), List.of(DataType.BOOL, type, type), type,
                CpuPointwiseOpcode.WHERE, shape));
        for (DataType source : io.github.pho001.synaptik.backend.cpu.internal.CpuTestDtypes.legacyOracleTypes()) for (DataType target : io.github.pho001.synaptik.backend.cpu.internal.CpuTestDtypes.legacyOracleTypes()) forms.add(fixture(
                "CAST_" + source + "_" + target, new Operation(CastKind.CAST, new CastAttrs(target)),
                List.of(source), target, CpuPointwiseOpcode.CAST, shape));
        return List.copyOf(forms);
    }

    private static List<DataType> floatingTypes() { return List.of(
            DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16); }
    private static List<DataType> numericTypes() { return List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.FLOAT16, DataType.INT64, DataType.INT32); }

    static List<PointwiseRejectedCandidate> rejectedPointwiseCandidates() {
        Shape shape = Shape.of(8); Shape incompatible = Shape.of(7);
        var result = new ArrayList<PointwiseRejectedCandidate>();
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            result.add(rejected(kind + "/bool-role", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.BOOL, shape), descriptor(DataType.BOOL, shape)), descriptor(DataType.BOOL, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
            result.add(rejected(kind + "/shape", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, incompatible)), descriptor(DataType.FLOAT32, shape), "SHAPE_INCOMPATIBLE"));
        }
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) {
            result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape)), descriptor(DataType.FLOAT32, shape), "INVALID_RESULT_ROLE"));
        }
        result.add(rejected("WHERE/condition-role", new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape)),
                descriptor(DataType.FLOAT32, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape)), descriptor(DataType.BOOL, shape), "INVALID_RESULT_ROLE"));
        for (FloatingClassificationKind kind : FloatingClassificationKind.values()) result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape)), descriptor(DataType.FLOAT32, shape), "INVALID_RESULT_ROLE"));
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) result.add(rejected(kind + "/numeric-role", new Operation(kind, NoOperationAttrs.INSTANCE),
                java.util.Collections.nCopies(kind == BooleanLogicalKind.NOT ? 1 : 2, descriptor(DataType.INT32, shape)), descriptor(DataType.BOOL, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
        return List.copyOf(result);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) { return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private static PointwiseRejectedCandidate rejected(String id, Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output, String reason) { return new PointwiseRejectedCandidate(id, operation, inputs, output, reason); }
    record PointwiseRejectedCandidate(String id, Operation operation, List<TensorDescriptor> inputs,
            TensorDescriptor output, String reason) {
        CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture coverageFixture() {
            return new CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture(id, operation.kind().name(),
                    operation, inputs, List.of(output), reason);
        }
    }

    private static PointwiseFixture fixture(String id, Operation operation, List<DataType> inputs,
            DataType output, CpuPointwiseOpcode opcode, Shape shape) {
        return new PointwiseFixture(id, operation, inputs, output, opcode, shape);
    }

    

    record PointwiseFixture(String id, Operation operation, List<DataType> inputTypes,
            DataType outputType, CpuPointwiseOpcode opcode, Shape shape) {
        List<DataType> boundaryTypes() {
            var types = new ArrayList<>(inputTypes); types.add(outputType); return List.copyOf(types);
        }
        List<TensorDescriptor> inputDescriptors() { return inputTypes.stream()
                .map(type -> descriptor(type, shape)).toList(); }
        TensorDescriptor outputDescriptor() { return descriptor(outputType, shape); }
        PrepareContext<CpuPartitionAnalysisInputs> context() {
            return context(LayoutDescriptor.contiguous(shape), List.of(),
                    CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
        }
        PrepareContext<CpuPartitionAnalysisInputs> context(LayoutDescriptor inputLayout,
                List<CpuKernelSpecialization.CarrierAccess> carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
                CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) {
            var inputs = java.util.stream.IntStream.range(0, inputTypes.size()).mapToObj(ValueId::new).toList();
            ValueId output = new ValueId(inputTypes.size());
            var node = new CompiledNode(new NodeId(0), operation, inputs, List.of(output));
            var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
            var descriptors = inputTypes.stream().map(type -> new TensorDescriptor(type, shape,
                    Optional.of(inputLayout), false)).collect(java.util.stream.Collectors.toCollection(ArrayList::new));
            descriptors.add(outputDescriptor());
            var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
            for (int index = 0; index < descriptors.size(); index++) {
                ValueId id = new ValueId(index); TensorDescriptor descriptor = descriptors.get(index);
                boolean result = index == inputTypes.size(); values.add(new GraphValue(id, descriptor));
                memory.add(new LogicalMemoryRequirement(id, descriptor, result ? Optional.of(partition)
                        : Optional.empty(), result ? List.of() : List.of(partition), result));
            }
            return new PrepareContext<>(partition, List.of(node), values, memory, java.util.Map.of(), new CpuPartitionAnalysisInputs(false, carriers, execution, materializationPolicy));
        }
        private static TensorDescriptor descriptor(DataType type, Shape shape) {
            return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
        }
    }

    static List<PointwiseCandidate> pointwiseCandidates() {
        var result = new ArrayList<PointwiseCandidate>();
        var general = LayoutDescriptor.of(Shape.of(8), new long[] {2}, 0, true);
        var materialization = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 1, 20, 1, 3, 1_000_000, 1, 1);
        for (PointwiseFixture fixture : generalPointwiseFixtures()) {
            result.add(candidate(fixture, "heap-contiguous-scalar", "HEAP", "CONTIGUOUS", "DIRECT", "SCALAR",
                    carrierPattern(fixture, false, false), LayoutDescriptor.contiguous(fixture.shape()), scalar(1),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "segment-contiguous-vector", "SEGMENT", "CONTIGUOUS", "DIRECT", "VECTOR",
                    carrierPattern(fixture, true, true), LayoutDescriptor.contiguous(fixture.shape()), vector(1),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "mixed-contiguous-parallel-vector", "MIXED", "CONTIGUOUS", "DIRECT", "PARALLEL_VECTOR",
                    carrierPattern(fixture, false, true), LayoutDescriptor.contiguous(fixture.shape()), vector(4),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "heap-general-parallel-scalar", "HEAP", "GENERAL", "DIRECT", "PARALLEL_SCALAR",
                    carrierPattern(fixture, false, false), general, scalar(4),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "heap-general-materialization-candidate", "HEAP", "GENERAL", "MATERIALIZATION_CANDIDATE", "SCALAR",
                    carrierPattern(fixture, false, false), general, scalar(1), materialization));
        }
        return List.copyOf(result);
    }

    /**
     * Exhaustive carrier-role witnesses derived from the same Model occurrences as the 1025-row
     * selected matrix.  They are execution witnesses, not additional inventory owners: each
     * ordered input/output position may use its typed heap carrier or a MemorySegment.  Keeping
     * them separate preserves the selected-matrix denominator while making unsupported role
     * combinations fail at real provider/preparer/generator boundaries.
     */
    static List<PointwiseCandidate> pointwiseCarrierRoleWitnesses() {
        var witnesses = new ArrayList<PointwiseCandidate>();
        for (PointwiseFixture fixture : generalPointwiseFixtures()) {
            int positions = fixture.inputTypes().size() + 1;
            for (int mask = 0; mask < (1 << positions); mask++) {
                var carriers = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
                for (int position = 0; position < positions; position++) {
                    boolean segment = (mask & (1 << position)) != 0;
                    carriers.add(segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                            : heapCarrier(position == fixture.inputTypes().size() ? fixture.outputType()
                                    : fixture.inputTypes().get(position)));
                }
                String roles = Integer.toBinaryString(mask | (1 << positions)).substring(1);
                witnesses.add(candidate(fixture, "carrier-roles-" + roles, "ROLE_" + roles,
                        "CONTIGUOUS", "DIRECT", "SCALAR", List.copyOf(carriers),
                        LayoutDescriptor.contiguous(fixture.shape()), scalar(1),
                        CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            }
        }
        return List.copyOf(witnesses);
    }

    private static PointwiseCandidate candidate(PointwiseFixture fixture, String suffix, String carrier,
            String layout, String materialization, String strategy,
            List<CpuKernelSpecialization.CarrierAccess> carriers, LayoutDescriptor inputLayout,
            CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        return new PointwiseCandidate(fixture.id() + '/' + suffix, fixture.id(), fixture,
                PointwiseDisposition.SUPPORTED, carrier, layout, materialization, strategy,
                carriers, inputLayout, execution, policy, materialization.equals("MATERIALIZATION_CANDIDATE")
                        && (fixture.operation().kind() == WhereSelectionKind.WHERE || fixture.inputTypes().stream()
                                .noneMatch(type -> type == DataType.BFLOAT16 || type == DataType.FLOAT16)));
    }

    private static List<CpuKernelSpecialization.CarrierAccess> carrierPattern(PointwiseFixture fixture,
            boolean segments, boolean outputSegments) {
        var pattern = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
        for (DataType type : fixture.inputTypes()) pattern.add(segments ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                : heapCarrier(type));
        pattern.add(outputSegments ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : heapCarrier(fixture.outputType()));
        return List.copyOf(pattern);
    }

    static CpuKernelSpecialization.CarrierAccess heapCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16, FLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
        };
    }

    static CpuPartitionAnalysisInputs.PortableExecutionConfig scalar(int parallelism) {
        return new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                parallelism, parallelism, 1);
    }

    static CpuPartitionAnalysisInputs.PortableExecutionConfig vector(int parallelism) {
        return new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                parallelism, parallelism, 1);
    }

    private static CpuPartitionPreparationPlan.ExecutionStrategy kernelStrategy(
            CpuPartitionPreparationPlan.ExecutionStrategy selected) {
        return selected.orchestration()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL
                ? new CpuPartitionPreparationPlan.ExecutionStrategy(selected.compute(),
                        CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.SINGLE_THREAD)
                : selected;
    }

    private enum PointwiseDisposition { SUPPORTED }

    record PointwiseCandidate(String id, String form, PointwiseFixture fixture,
            PointwiseDisposition disposition, String carrierAxis, String layoutAxis,
            String materializationAxis, String requestedStrategy,
            List<CpuKernelSpecialization.CarrierAccess> inputs, LayoutDescriptor inputLayout,
            CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy,
            boolean materializationSelected) { }

    static void assertClosedSpecializedClass(byte[] bytes, CpuKernelSpecialization specialization) {
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertTrue(model.methods().stream().anyMatch(method -> method.methodName()
                        .stringValue().equals("invoke") && method.flags().has(AccessFlag.STATIC)
                        && method.methodTypeSymbol().descriptorString().equals(
                                specialization.entryType().descriptorString()))),
                () -> assertTrue(model.methods().stream().allMatch(method -> method.flags()
                        .has(AccessFlag.STATIC)), "every generated helper method must be static"),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(members.stream().noneMatch(member -> member.type().stringValue()
                        .contains("Ljava/lang/Object;") || member.owner().asInternalName()
                        .startsWith("java/lang/reflect/") || member.owner().asInternalName()
                        .startsWith("java/lang/invoke/") || member.owner().asInternalName()
                        .startsWith("java/util/"))), () -> assertTrue(members.stream()
                        .filter(member -> member.owner().asInternalName()
                                .startsWith("io/github/pho001/synaptik/"))
                        .allMatch(member -> member.owner().asInternalName().equals(
                                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath")
                                || member.owner().asInternalName().startsWith(
                                        "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/Generated_")),
                        () -> "unexpected generated helper ownership: " + members.stream()
                                .map(member -> member.owner().asInternalName()).distinct().toList()));
    }

}
