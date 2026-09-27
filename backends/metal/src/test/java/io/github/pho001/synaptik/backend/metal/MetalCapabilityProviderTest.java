package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalCapabilityProviderTest {
    private final MetalCapabilityProvider provider = new MetalCapabilityProvider();
    @Test
    void registeredWireCapabilityLedgerClosesAtEightyOneTrueAndThirtyFourFalse() {
        java.util.Set<MetalMpsGraphProgram.NodeKind> structuralOnly = java.util.Set.of(
                MetalMpsGraphProgram.NodeKind.TENSOR_POW,
                MetalMpsGraphProgram.NodeKind.SCALAR_POW,
                MetalMpsGraphProgram.NodeKind.LOG,
                MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalMpsGraphProgram.NodeKind.EXPM1,
                MetalMpsGraphProgram.NodeKind.ERF,
                MetalMpsGraphProgram.NodeKind.SQRT,
                MetalMpsGraphProgram.NodeKind.RSQRT,
                MetalMpsGraphProgram.NodeKind.TANH,
                MetalMpsGraphProgram.NodeKind.GELU,
                MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION,
                MetalMpsGraphProgram.NodeKind.SILU,
                MetalMpsGraphProgram.NodeKind.SCATTER_ADD,
                MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                MetalMpsGraphProgram.NodeKind.VARIANCE,
                MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION,
                MetalMpsGraphProgram.NodeKind.L1_NORM,
                MetalMpsGraphProgram.NodeKind.L2_NORM);
        assertEquals(18, structuralOnly.size());
        long trueRows = java.util.Arrays.stream(MetalMpsGraphProgram.NodeKind.values())
                .filter(MetalMpsGraphProgram.NodeKind::executable)
                .filter(kind -> !structuralOnly.contains(kind))
                .count();
        assertEquals(81L, trueRows);
        assertEquals(34L, MetalMpsGraphProgram.NodeKind.values().length - trueRows);
        structuralOnly.forEach(kind -> assertTrue(kind.executable(), kind.name()));
        assertFalse(MetalMpsGraphProgram.NodeKind.EXP.executable());
        assertFalse(MetalMpsGraphProgram.NodeKind.SIGMOID.executable());
    }

    @Test
    void acceleratorMseIsTheOnlyNormalizationOrLossCapability() {
        Shape shape = Shape.of(2, 3);
        for (LossReduction reduction : LossReduction.values()) {
            Operation mse = new Operation(
                    LossKind.MEAN_SQUARED_ERROR, new MeanSquaredErrorAttrs(reduction));
            for (boolean predictionGrad : List.of(false, true)) {
                for (boolean targetGrad : List.of(false, true)) {
                    TensorDescriptor prediction = descriptor(shape, predictionGrad);
                    TensorDescriptor target = descriptor(shape, targetGrad);
                    TensorDescriptor output = descriptor(
                            reduction == LossReduction.NONE ? shape : Shape.scalar(),
                            predictionGrad || targetGrad);
                    OperationCapabilityQuery query = new OperationCapabilityQuery(
                            NumericalProfile.ACCELERATOR,
                            mse,
                            List.of(prediction, target),
                            List.of(output));
                    assertTrue(provider.supports(query),
                            reduction + " gradients " + predictionGrad + "/" + targetGrad);
                    assertFalse(provider.supports(new OperationCapabilityQuery(
                            NumericalProfile.STRICT_IEEE,
                            mse,
                            query.inputs(),
                            query.outputs())));
                }
            }
        }

        Operation none = new Operation(
                LossKind.MEAN_SQUARED_ERROR,
                new MeanSquaredErrorAttrs(LossReduction.NONE));
        TensorDescriptor value = descriptor(shape, false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(typed(DataType.FLOAT64, shape, false), typed(DataType.FLOAT64, shape, false)),
                List.of(typed(DataType.FLOAT64, shape, false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(
                        typed(DataType.BFLOAT16, shape, false),
                        typed(DataType.FLOAT32, shape, false)),
                List.of(typed(DataType.FLOAT32, shape, false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(descriptor(Shape.scalar(), false), descriptor(Shape.scalar(), false)),
                List.of(descriptor(Shape.scalar(), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(value, descriptor(Shape.of(3, 2), false)),
                List.of(value))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(value, value),
                List.of(descriptor(shape, true)))));
        TensorDescriptor dynamic = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(3)),
                Optional.empty(),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(dynamic, dynamic),
                List.of(dynamic))));
        TensorDescriptor empty = descriptor(Shape.of(2, 0), false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(empty, empty),
                List.of(empty))));
        TensorDescriptor viewed = view(shape, 3, 1);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                none,
                List.of(viewed, value),
                List.of(value))));

        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        TensorDescriptor channel = descriptor(Shape.of(3), false);
        TensorDescriptor sample = descriptor(Shape.of(2), false);
        TensorDescriptor indexTarget = typed(DataType.INT32, Shape.of(2), false);
        List<TensorDescriptor> batchInputs = List.of(value, channel, channel, channel, channel);
        List<OperationCapabilityQuery> blockedQueries = List.of(
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                                new DenseCategoricalCrossEntropyWithLogitsAttrs(
                                        1, LossReduction.NONE)),
                        List.of(value, value),
                        List.of(sample)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                                new IndexCategoricalCrossEntropyWithLogitsAttrs(
                                        1, LossReduction.NONE, Optional.empty())),
                        List.of(value, indexTarget),
                        List.of(sample)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                BatchNormKind.BATCH_NORM_INFERENCE,
                                new BatchNormInferenceAttrs(1, epsilon)),
                        batchInputs,
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                BatchNormKind.BATCH_NORM_TRAINING,
                                new BatchNormTrainingAttrs(
                                        1, ScalarValue.float32(0.5f), epsilon)),
                        batchInputs,
                        List.of(value, channel, channel, channel, channel)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                LayerNormKind.LAYER_NORM,
                                new LayerNormAttrs(Shape.of(3), epsilon)),
                        List.of(value),
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                LayerNormKind.LAYER_NORM,
                                new AffineLayerNormAttrs(Shape.of(3), epsilon)),
                        List.of(value, channel, channel),
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                RmsNormKind.RMS_NORM,
                                new RmsNormAttrs(Shape.of(3), epsilon)),
                        List.of(value),
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(
                                RmsNormKind.RMS_NORM,
                                new RmsNormAttrs(Shape.of(3), epsilon)),
                        List.of(value, channel),
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(1)),
                        List.of(value),
                        List.of(value)),
                new OperationCapabilityQuery(
                        NumericalProfile.ACCELERATOR,
                        new Operation(SoftmaxKind.LOG_SOFTMAX, new SoftmaxAttrs(1)),
                        List.of(value),
                        List.of(value)));
        blockedQueries.forEach(query -> assertFalse(
                provider.supports(query), query.operation().kind().toString()));
    }

    @Test
    void task0059AdvertisesCastSelectAndSliceWithoutOpeningUnsafeLayouts() {
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType source : DataType.values()) {
                for (DataType target : DataType.values()) {
                    boolean expected = source == target
                            || source == DataType.BOOL
                            || target == DataType.BOOL
                            || source == DataType.INT32 && target == DataType.INT64
                            || source == DataType.INT64 && target == DataType.INT32
                            || source == DataType.BFLOAT16 && target == DataType.FLOAT32;
                    assertEquals(
                            expected,
                            provider.supports(new OperationCapabilityQuery(
                                    profile,
                                    new Operation(CastKind.CAST, new CastAttrs(target)),
                                    List.of(typed(source, Shape.of(2), false)),
                                    List.of(typed(target, Shape.of(2), false)))),
                            source + " -> " + target + " " + profile);
                }
            }
        }
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType type : DataType.values()) {
                Shape dataShape = Shape.of(2, 3);
                TensorDescriptor data = typed(type, dataShape, false);
                Shape selectShape = Shape.of(3);
                TensorDescriptor selected = new TensorDescriptor(
                        type,
                        selectShape,
                        Optional.of(LayoutDescriptor.of(
                                selectShape, new long[] {1}, 3L, true)),
                        false);
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)),
                        List.of(data),
                        List.of(selected))), type + " SELECT " + profile);
                Shape sliceShape = Shape.of(2, 2);
                TensorDescriptor sliced = new TensorDescriptor(
                        type,
                        sliceShape,
                        Optional.of(LayoutDescriptor.of(
                                sliceShape, new long[] {3, 1}, 1L, true)),
                        false);
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(
                                SliceKind.SLICE,
                                new SliceAttrs(
                                        List.of(1L), List.of(2L), List.of(1), List.of(1L))),
                        List.of(data),
                        List.of(sliced))), type + " SLICE " + profile);
            }
        }
        Shape unsafeShape = Shape.of(2, 3);
        TensorDescriptor unsafe = new TensorDescriptor(
                DataType.FLOAT32,
                unsafeShape,
                Optional.of(LayoutDescriptor.of(
                        unsafeShape, new long[] {1, 1}, 0L, true)),
                false);
        TensorDescriptor unresolved = new TensorDescriptor(
                DataType.FLOAT32, unsafeShape, Optional.empty(), false);
        TensorDescriptor selected = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(3),
                Optional.of(LayoutDescriptor.of(
                        Shape.of(3), new long[] {1}, 0L, true)),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(SelectKind.SELECT, new SelectAttrs(0, 0)),
                List.of(unsafe),
                List.of(selected))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(SelectKind.SELECT, new SelectAttrs(0, 0)),
                List.of(unresolved),
                List.of(selected))));
        TensorDescriptor broadcast = new TensorDescriptor(
                DataType.FLOAT32,
                unsafeShape,
                Optional.of(LayoutDescriptor.of(
                        unsafeShape, new long[] {0, 1}, 0L, true)),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(SelectKind.SELECT, new SelectAttrs(0, 0)),
                List.of(broadcast),
                List.of(selected))));
        TensorDescriptor negativeSliceOutput = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 2),
                Optional.of(LayoutDescriptor.of(
                        Shape.of(2, 2), new long[] {3, 1}, 1L, true)),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(
                        SliceKind.SLICE,
                        new SliceAttrs(
                                List.of(2L), List.of(2L), List.of(1), List.of(-1L))),
                List.of(typed(DataType.FLOAT32, unsafeShape, false)),
                List.of(negativeSliceOutput))));
        assertThrows(IllegalArgumentException.class, () -> new SliceAttrs(
                List.of(0L), List.of(1L), List.of(0), List.of(0L)));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(CastKind.CAST, new CastAttrs(DataType.BOOL)),
                List.of(typed(DataType.FLOAT32, Shape.of(2), true)),
                List.of(typed(DataType.BOOL, Shape.of(2), false)))));
    }

    @Test
    void task0060AdvertisesOnlyExactReplacementNonoverlapAndLogicalAggregateDomains() {
        var scatterNone = new Operation(
                ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(0, ScatterReduction.NONE));
        var scatterAdd = new Operation(
                ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(0, ScatterReduction.ADD));
        var signedSlice = new Operation(
                SliceKind.SLICE_UPDATE,
                new SliceAttrs(List.of(3L), List.of(2L), List.of(0), List.of(-2L)));
        var foldAxis = new Operation(
                WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(1, 4, 2));
        var overlappingAxis = new Operation(
                WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(1, 4, 1));
        var paddedCeil2d = new Window2dAttrs(2, 2, 2, 2, 1, 1, 1, 1, true);
        var paddedCeil3d =
                new Window3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, true);
        var fold2d = new Operation(
                WindowTransformKind.FOLD2D,
                new Fold2dAttrs(Shape.of(1, 1, 3, 3), paddedCeil2d));
        var fold3d = new Operation(
                WindowTransformKind.FOLD3D,
                new Fold3dAttrs(Shape.of(1, 1, 3, 3, 3), paddedCeil3d));
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType type : DataType.values()) {
                for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                    assertTrue(provider.supports(new OperationCapabilityQuery(
                            profile,
                            scatterNone,
                            List.of(
                                    typed(type, Shape.of(3), false),
                                    typed(indexType, Shape.of(2, 1), false),
                                    typed(type, Shape.of(2), false)),
                            List.of(typed(type, Shape.of(3), false)))));
                }
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        scatterAdd,
                        List.of(
                                typed(type, Shape.of(3), false),
                                typed(DataType.INT32, Shape.of(2, 1), false),
                                typed(type, Shape.of(2), false)),
                        List.of(typed(type, Shape.of(3), false)))));
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        signedSlice,
                        List.of(
                                typed(type, Shape.of(4), false),
                                typed(type, Shape.of(2), false)),
                        List.of(typed(type, Shape.of(4), false)))));
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(
                                SliceKind.SLICE_UPDATE,
                                new SliceAttrs(
                                        List.of(0L), List.of(0L), List.of(0), List.of(1L))),
                        List.of(
                                typed(type, Shape.of(4), false),
                                typed(type, Shape.of(0), false)),
                        List.of(typed(type, Shape.of(4), false)))));
            }
            for (DataType type :
                    List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        foldAxis,
                        List.of(typed(type, Shape.of(2, 2, 2), false)),
                        List.of(typed(type, Shape.of(2, 4), false)))));
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        overlappingAxis,
                        List.of(typed(type, Shape.of(2, 3, 2), false)),
                        List.of(typed(type, Shape.of(2, 4), false)))));
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        fold2d,
                        List.of(typed(type, Shape.of(1, 4, 9), false)),
                        List.of(typed(type, Shape.of(1, 1, 3, 3), false)))));
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        fold3d,
                        List.of(typed(type, Shape.of(1, 8, 27), false)),
                        List.of(typed(type, Shape.of(1, 1, 3, 3, 3), false)))));
            }
            for (DataType type : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        foldAxis,
                        List.of(typed(type, Shape.of(2, 2, 2), false)),
                        List.of(typed(type, Shape.of(2, 4), false)))),
                        type + " FOLD_AXIS");
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        fold2d,
                        List.of(typed(type, Shape.of(1, 4, 9), false)),
                        List.of(typed(type, Shape.of(1, 1, 3, 3), false)))),
                        type + " FOLD2D");
                assertFalse(provider.supports(new OperationCapabilityQuery(
                        profile,
                        fold3d,
                        List.of(typed(type, Shape.of(1, 8, 27), false)),
                        List.of(typed(type, Shape.of(1, 1, 3, 3, 3), false)))),
                        type + " FOLD3D");
            }
            for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(AggregateReductionKind.PROD, NoOperationAttrs.INSTANCE),
                        List.of(typed(type, Shape.of(2, 3), false)),
                        List.of(typed(type, Shape.scalar(), false)))));
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(
                                AggregateReductionKind.PROD,
                                new MultiAxisReductionAttrs(List.of(), false)),
                        List.of(typed(type, Shape.scalar(), false)),
                        List.of(typed(type, Shape.scalar(), false)))));
            }
            for (AggregateReductionKind kind :
                    List.of(AggregateReductionKind.ALL, AggregateReductionKind.ANY)) {
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(kind, new MultiAxisReductionAttrs(List.of(), false)),
                        List.of(typed(DataType.BOOL, Shape.of(2, 3), false)),
                        List.of(typed(DataType.BOOL, Shape.of(2, 3), false)))));
            }
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(AggregateReductionKind.PROD, NoOperationAttrs.INSTANCE),
                    List.of(typed(DataType.FLOAT32, Shape.of(2), false)),
                    List.of(typed(DataType.FLOAT32, Shape.scalar(), false)))));
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(AggregateReductionKind.ALL, NoOperationAttrs.INSTANCE),
                    List.of(typed(DataType.INT32, Shape.of(2), false)),
                    List.of(typed(DataType.INT32, Shape.scalar(), false)))));
        }
    }

    @Test
    void task0059BothImageUnfoldKindsAcceptOnlyModelLegalFloatingCarriers() {
        var window2d = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        var window3d = new Window3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType type : DataType.values()) {
                boolean expected = type == DataType.FLOAT64
                        || type == DataType.FLOAT32
                        || type == DataType.BFLOAT16;
                assertEquals(
                        expected,
                        provider.supports(new OperationCapabilityQuery(
                                profile,
                                new Operation(WindowTransformKind.UNFOLD2D, window2d),
                                List.of(typed(type, Shape.of(1, 1, 2, 2), false)),
                                List.of(typed(type, Shape.of(1, 4, 1), false)))),
                        "UNFOLD2D " + type + " " + profile);
                assertEquals(
                        expected,
                        provider.supports(new OperationCapabilityQuery(
                                profile,
                                new Operation(WindowTransformKind.UNFOLD3D, window3d),
                                List.of(typed(type, Shape.of(1, 1, 2, 2, 2), false)),
                                List.of(typed(type, Shape.of(1, 8, 1), false)))),
                        "UNFOLD3D " + type + " " + profile);
            }
        }
    }

    @Test
    void enforcesClosedStrictAndAcceleratorCapabilityMatrices() {
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, provider.backendId());
        assertEquals("metal", provider.backendId().value());
        TensorDescriptor matrix = descriptor(Shape.of(2, 3), false);
        TensorDescriptor row = descriptor(Shape.of(3), false);
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.NEG, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.ABS, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR, UnaryElementwiseKind.ABS, matrix, matrix)));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR, UnaryElementwiseKind.NEG, matrix, matrix)));
        Shape rank16 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE,
                UnaryElementwiseKind.ABS,
                descriptor(rank16, true),
                descriptor(rank16, true))));

        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            boolean supported = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL
                    || kind == BinaryArithmeticKind.DIV
                    || kind == BinaryArithmeticKind.MIN
                    || kind == BinaryArithmeticKind.MAX;
            assertFalse(provider.supports(binaryQuery(
                    NumericalProfile.STRICT_IEEE, kind, matrix, row, matrix)));
            assertEquals(supported, provider.supports(binaryQuery(
                    NumericalProfile.ACCELERATOR, kind, matrix, row, matrix)));
            assertEquals(supported, provider.supports(binaryQuery(
                    NumericalProfile.ACCELERATOR, kind, row, matrix, matrix)));
        }
        TensorDescriptor scalarSeed = descriptor(Shape.scalar(), false);
        TensorDescriptor gradientVector = descriptor(Shape.of(3), true);
        assertTrue(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.MUL,
                scalarSeed,
                gradientVector,
                gradientVector)));
        assertTrue(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.MUL,
                gradientVector,
                scalarSeed,
                gradientVector)));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.STRICT_IEEE,
                BinaryArithmeticKind.MUL,
                scalarSeed,
                gradientVector,
                gradientVector)));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                scalarSeed,
                gradientVector,
                gradientVector)));
        TensorDescriptor gradientMatrix = descriptor(Shape.of(2, 3), true);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.MUL,
                scalarSeed,
                gradientMatrix,
                gradientMatrix)));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.MUL,
                gradientMatrix,
                row,
                gradientMatrix)));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) {
            assertEquals(exactRawUnary(kind),
                    provider.supports(unaryQuery(
                            NumericalProfile.STRICT_IEEE, kind, matrix, matrix)),
                    "strict " + kind);
            assertEquals(
                    exactRawUnary(kind) || kind == UnaryElementwiseKind.RECIPROCAL,
                    provider.supports(unaryQuery(
                            NumericalProfile.ACCELERATOR, kind, matrix, matrix)),
                    "accelerator " + kind);
        }
        TensorDescriptor reshapedView = new TensorDescriptor(
                DataType.FLOAT32,
                matrix.shape(),
                Optional.of(LayoutDescriptor.of(
                        matrix.shape(), new long[] {3, 1}, 0L, true)),
                false);
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE,
                new TargetShapeAttrs(matrix.shape()));
        var strictReshape = new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                reshape,
                List.of(matrix),
                List.of(reshapedView));
        var acceleratorReshape = new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                reshape,
                List.of(matrix),
                List.of(reshapedView));
        assertTrue(provider.supports(strictReshape));
        assertTrue(provider.supports(acceleratorReshape));
    }

    @Test
    void everyStrictExactRepresentativeIsAlsoAcceleratorSupported() {
        TensorDescriptor matrix = descriptor(Shape.of(2, 3), false);
        TensorDescriptor flat = descriptor(Shape.of(6), false);
        TensorDescriptor singletonRow = descriptor(Shape.of(1, 3), false);
        TensorDescriptor singletonMatrix = descriptor(Shape.of(2, 1, 3), false);
        TensorDescriptor reshaped = view(Shape.of(2, 3), 3, 1);
        TensorDescriptor expanded = view(Shape.of(2, 3), 0, 1);
        TensorDescriptor permuted = view(Shape.of(3, 2), 1, 3);
        TensorDescriptor rankExpanded = view(Shape.of(2, 1, 3), 3, 3, 1);
        TensorDescriptor squeezed = view(Shape.of(2, 3), 3, 1);
        Shape rank16 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        TensorDescriptor rank16Value = descriptor(rank16, false);
        List<Occurrence> common = List.of(
                new Occurrence(
                        "NEG",
                        new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "ABS",
                        new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "FLOOR",
                        new Operation(UnaryElementwiseKind.FLOOR, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "CEIL",
                        new Operation(UnaryElementwiseKind.CEIL, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "SIGN",
                        new Operation(UnaryElementwiseKind.SIGN, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "RELU",
                        new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                        matrix,
                        matrix),
                new Occurrence(
                        "RESHAPE",
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(2, 3))),
                        flat,
                        reshaped),
                new Occurrence(
                        "EXPAND",
                        new Operation(
                                ShapeTransformKind.EXPAND,
                                new TargetShapeAttrs(Shape.of(2, 3))),
                        singletonRow,
                        expanded),
                new Occurrence(
                        "PERMUTE",
                        new Operation(
                                AxisTransformKind.PERMUTE,
                                new PermutationAttrs(List.of(1, 0))),
                        matrix,
                        permuted),
                new Occurrence(
                        "EXPAND_DIMS",
                        new Operation(
                                AxisTransformKind.EXPAND_DIMS,
                                new AxisTransformAttrs(1)),
                        matrix,
                        rankExpanded),
                new Occurrence(
                        "SQUEEZE",
                        new Operation(
                                AxisTransformKind.SQUEEZE,
                                new AxisTransformAttrs(1)),
                        singletonMatrix,
                        squeezed),
                new Occurrence(
                        "CONTIGUOUS",
                        new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                        expanded,
                        matrix),
                new Occurrence(
                        "rank-16 NEG boundary",
                        new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                        rank16Value,
                        rank16Value));

        for (Occurrence occurrence : common) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.STRICT_IEEE,
                    occurrence.operation(),
                    List.of(occurrence.input()),
                    List.of(occurrence.output()))), occurrence.name() + " strict");
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    occurrence.operation(),
                    List.of(occurrence.input()),
                    List.of(occurrence.output()))), occurrence.name() + " accelerator");
        }
    }
    @Test
    void scalarArithmeticAndReciprocalAdmitOnlyAcceleratorNoGradOccurrences() {
        TensorDescriptor noGrad = descriptor(Shape.of(2, 3), false);
        TensorDescriptor grad = descriptor(Shape.of(2, 3), true);
        for (ScalarElementwiseKind kind : List.of(
                ScalarElementwiseKind.ADD,
                ScalarElementwiseKind.SUB,
                ScalarElementwiseKind.MUL,
                ScalarElementwiseKind.DIV)) {
            Operation operation = new Operation(
                    kind,
                    new ScalarValueAttrs(ScalarValue.float32(
                            Float.intBitsToFloat(0x8000_0001))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    operation,
                    List.of(noGrad),
                    List.of(noGrad))), kind.name());
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.STRICT_IEEE,
                    operation,
                    List.of(noGrad),
                    List.of(noGrad))), "strict " + kind);
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    operation,
                    List.of(grad),
                    List.of(grad))), "grad " + kind);
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    operation,
                    List.of(noGrad),
                    List.of(grad))), "mismatched grad " + kind);
        }
        Operation scalarPow = new Operation(
                ScalarElementwiseKind.POW,
                new ScalarValueAttrs(ScalarValue.float32(2.0f)));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                scalarPow,
                List.of(noGrad),
                List.of(noGrad))));
        assertTrue(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR,
                UnaryElementwiseKind.RECIPROCAL,
                noGrad,
                noGrad)));
        assertFalse(provider.supports(unaryQuery(
                NumericalProfile.STRICT_IEEE,
                UnaryElementwiseKind.RECIPROCAL,
                noGrad,
                noGrad)));
        assertFalse(provider.supports(unaryQuery(
                NumericalProfile.ACCELERATOR,
                UnaryElementwiseKind.RECIPROCAL,
                grad,
                grad)));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.float64(1.0))),
                List.of(typed(DataType.FLOAT64)),
                List.of(typed(DataType.FLOAT64)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.float32(1.0f))),
                List.of(descriptor(Shape.scalar(), false)),
                List.of(descriptor(Shape.scalar(), false)))));
    }


    @Test
    void admitsOnlyBoundedCanonicalFloat32UnfoldAxisInBothProfiles() {
        TensorDescriptor input = descriptor(Shape.of(2, 6), true);
        TensorDescriptor output = descriptor(Shape.of(2, 2, 3), true);
        Operation unfold = new Operation(
                WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 3, 2));
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile, unfold, List.of(input), List.of(output))),
                    profile + " admits the same exact occurrence");
        }

        Operation fold = new Operation(
                WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(1, 6, 2));
        TensorDescriptor foldInput = descriptor(Shape.of(2, 2, 3), true);
        TensorDescriptor foldOutput = descriptor(Shape.of(2, 6), true);
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    profile, fold, List.of(foldInput), List.of(foldOutput))),
                    profile + " rejects an otherwise valid FOLD_AXIS solely by kind");
        }

        TensorDescriptor capInput = descriptor(Shape.of(2, 17), true);
        TensorDescriptor capOutput = descriptor(Shape.of(2, 1, 17), true);
        Operation capExceeded = new Operation(
                WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 17, 1));
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    profile, capExceeded, List.of(capInput), List.of(capOutput))),
                    profile + " rejects size 17 solely at the selector-expansion cap");
        }

        assertThrows(IllegalArgumentException.class, () ->
                new Operation(WindowTransformKind.UNFOLD_AXIS, NoOperationAttrs.INSTANCE),
                "the Model signature excludes wrong attributes before capability dispatch");
        assertThrows(IllegalArgumentException.class, () ->
                new OperationCapabilityQuery(
                        NumericalProfile.STRICT_IEEE, unfold, List.of(input), List.of()),
                "the occurrence signature excludes malformed output cardinality");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(typed(DataType.FLOAT64, Shape.of(2, 6), true)),
                List.of(output))),
                "input type must be FLOAT32");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(input),
                List.of(typed(DataType.FLOAT64, Shape.of(2, 2, 3), true)))),
                "output type must be FLOAT32");

        Operation scalarUnfold = new Operation(
                WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(0, 1, 1));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                scalarUnfold,
                List.of(descriptor(Shape.scalar(), true)),
                List.of(descriptor(Shape.of(1), true)))),
                "scalar input is outside the non-scalar window contract");
        long[] rank16Dimensions = new long[16];
        java.util.Arrays.fill(rank16Dimensions, 1L);
        rank16Dimensions[15] = 6L;
        long[] rank17Dimensions = java.util.Arrays.copyOf(rank16Dimensions, 17);
        rank17Dimensions[15] = 2L;
        rank17Dimensions[16] = 3L;
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(
                        WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(15, 3, 2)),
                List.of(descriptor(Shape.of(rank16Dimensions), true)),
                List.of(descriptor(Shape.of(rank17Dimensions), true)))),
                "rank-sixteen input cannot produce the required rank-seventeen output");

        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(input),
                List.of(descriptor(Shape.of(2, 3, 3), true)))),
                "output Shape must match exact window geometry");
        TensorDescriptor inputView = new TensorDescriptor(
                DataType.FLOAT32,
                input.shape(),
                Optional.of(LayoutDescriptor.of(
                        input.shape(), new long[] {7, 1}, 0L, true)),
                true);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(inputView),
                List.of(output))),
                "input layout must be canonical independently of gradient state");
        TensorDescriptor outputView = new TensorDescriptor(
                DataType.FLOAT32,
                output.shape(),
                Optional.of(LayoutDescriptor.of(
                        output.shape(), new long[] {12, 3, 1}, 0L, true)),
                true);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(input),
                List.of(outputView))),
                "output layout must be canonical independently of gradient state");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                unfold,
                List.of(input),
                List.of(descriptor(output.shape(), false)))),
                "requiresGrad must agree independently of layout");

        assertThrows(IllegalArgumentException.class,
                () -> new UnfoldAxisAttrs(1, 0, 1),
                "zero window size is excluded by the typed attributes");
        assertThrows(IllegalArgumentException.class,
                () -> new UnfoldAxisAttrs(1, 3, 0),
                "zero step is excluded by the typed attributes");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 7, 1)),
                List.of(input),
                List.of(descriptor(Shape.of(2, 1, 7), true)))),
                "window size cannot exceed the selected extent");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 3, 3)),
                List.of(descriptor(Shape.of(2, 7), true)),
                List.of(descriptor(Shape.of(2, 3, 3), true)))),
                "step participates in the exact output-position count");
    }

    @Test
    void admitsOnlyExactInt32GatherOneHotAndReplacementScatterInBothProfiles() {
        TensorDescriptor data = typed(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        TensorDescriptor indices = typed(DataType.INT32, Shape.of(5, 6), false);
        TensorDescriptor gathered = typed(DataType.FLOAT32, Shape.of(2, 5, 6, 4), true);
        Operation gather = new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1));
        TensorDescriptor oneHotIndices = typed(DataType.INT32, Shape.of(2, 3), false);
        TensorDescriptor oneHot = typed(DataType.BOOL, Shape.of(2, 3, 4), false);
        Operation encode = new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4));

        TensorDescriptor scatterData = typed(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor scatterIndices = typed(DataType.INT32, Shape.of(2, 2), false);
        TensorDescriptor scatterUpdates = typed(DataType.FLOAT32, Shape.of(2, 2), false);
        TensorDescriptor scattered = typed(DataType.FLOAT32, Shape.of(2, 3), true);
        Operation scatter = new Operation(
                AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.NONE));
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile, gather, List.of(data, indices), List.of(gathered))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile, encode, List.of(oneHotIndices), List.of(oneHot))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile,
                    scatter,
                    List.of(scatterData, scatterIndices, scatterUpdates),
                    List.of(scattered))));
        }

        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                List.of(data, indices),
                List.of(gathered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, typed(DataType.INT64, Shape.of(5, 6), false)),
                List.of(gathered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, indices),
                List.of(typed(DataType.FLOAT32, Shape.of(2, 5, 4), true)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                gather,
                List.of(data, indices),
                List.of(typed(DataType.FLOAT32, gathered.shape(), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                encode,
                List.of(typed(DataType.INT64, Shape.of(2, 3), false)),
                List.of(oneHot))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                encode,
                List.of(oneHotIndices),
                List.of(typed(DataType.BOOL, Shape.of(2, 3, 5), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.ADD)),
                List.of(scatterData, scatterIndices, scatterUpdates),
                List.of(scattered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                scatter,
                List.of(
                        scatterData,
                        typed(DataType.INT64, scatterIndices.shape(), false),
                        scatterUpdates),
                List.of(scattered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                scatter,
                List.of(
                        scatterData,
                        scatterIndices,
                        typed(DataType.FLOAT32, Shape.of(1, 2), false)),
                List.of(scattered))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                scatter,
                List.of(scatterData, scatterIndices, scatterUpdates),
                List.of(typed(DataType.FLOAT32, scattered.shape(), false)))));
    }

    @Test
    void rejectsEveryBoundaryOutsideTheExactDomain() {
        TensorDescriptor valid = descriptor(Shape.of(2, 3), false);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        Shape rank17 = Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        var noLayout = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        var offset = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 1, true)),
                false);
        var view = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 0, true)),
                false);
        var strided = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 0, false)),
                false);

        List<UnaryElementwiseKind> exactUnaryKinds = List.of(
                UnaryElementwiseKind.NEG,
                UnaryElementwiseKind.ABS,
                UnaryElementwiseKind.FLOOR,
                UnaryElementwiseKind.CEIL,
                UnaryElementwiseKind.SIGN,
                UnaryElementwiseKind.RELU);
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (UnaryElementwiseKind kind : exactUnaryKinds) {
                assertTrue(provider.supports(unaryQuery(profile, kind, valid, valid)));
                TensorDescriptor grad = descriptor(Shape.of(2, 3), true);
                assertTrue(provider.supports(unaryQuery(profile, kind, grad, grad)));
                assertFalse(provider.supports(unaryQuery(
                        profile, kind, typed(DataType.FLOAT64), typed(DataType.FLOAT64))));
                assertFalse(provider.supports(unaryQuery(
                        profile, kind, valid, descriptor(Shape.of(3, 2), false))));
                assertFalse(provider.supports(unaryQuery(
                        profile, kind, valid, descriptor(Shape.of(2, 3), true))));
                assertFalse(provider.supports(unaryQuery(profile, kind, noLayout, noLayout)));
                assertFalse(provider.supports(unaryQuery(profile, kind, offset, offset)));
                assertFalse(provider.supports(unaryQuery(profile, kind, view, valid)));
                assertFalse(provider.supports(unaryQuery(profile, kind, strided, strided)));
                assertFalse(provider.supports(unaryQuery(
                        profile,
                        kind,
                        descriptor(Shape.of(), false),
                        descriptor(Shape.of(), false))));
                assertFalse(provider.supports(unaryQuery(
                        profile, kind, descriptor(rank17, false), descriptor(rank17, false))));
            }
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(NumericalProfile.ACCELERATOR, new Operation(BinaryComparisonKind.GREATER_THAN, NoOperationAttrs.INSTANCE), List.of(valid, valid), List.of(valid))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, typed(DataType.FLOAT64), typed(DataType.FLOAT64))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, valid, descriptor(Shape.of(3, 2), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE, valid, descriptor(Shape.of(2, 3), true))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false),
                new TensorDescriptor(DataType.FLOAT32, dynamic, Optional.empty(), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(Shape.of(), false), descriptor(Shape.of(), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(Shape.of(2, 0), false), descriptor(Shape.of(2, 0), false))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                descriptor(rank17, false), descriptor(rank17, false))));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, noLayout, noLayout)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, offset, offset)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, view, view)));
        assertFalse(provider.supports(query(NumericalProfile.STRICT_IEEE, strided, strided)));

        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, valid, valid,
                descriptor(Shape.of(3, 2), false))));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, valid, valid,
                descriptor(Shape.of(2, 3), true))));
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR, BinaryArithmeticKind.ADD, view, valid, valid)));
        TensorDescriptor float64 = typed(DataType.FLOAT64);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                float64,
                float64,
                float64)));
        TensorDescriptor scalarRank = descriptor(Shape.of(), false);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                scalarRank,
                scalarRank,
                scalarRank)));
        TensorDescriptor overRank = descriptor(rank17, false);
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.ADD,
                overRank,
                overRank,
                overRank)));

    }

    @Test
    void acceleratorReductionMatrixIsExactAndStrictRemainsClosed() {
        TensorDescriptor input = descriptor(Shape.of(2, 3, 4), false);
        TensorDescriptor scalar = descriptor(Shape.scalar(), false);
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                input,
                scalar)));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.STRICT_IEEE,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                input,
                scalar)));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.MEAN,
                new AxisReductionAttrs(1, true),
                input,
                descriptor(Shape.of(2, 1, 4), false))));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new MultiAxisReductionAttrs(List.of(2, 0), false),
                input,
                descriptor(Shape.of(3), false))));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.MEAN,
                new MultiAxisReductionAttrs(List.of(), true),
                input,
                input)));
        assertTrue(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.of(1, 4)),
                input,
                descriptor(Shape.of(1, 4), false))));

        TensorDescriptor view = new TensorDescriptor(
                DataType.FLOAT32,
                input.shape(),
                Optional.of(LayoutDescriptor.of(
                        input.shape(), new long[] {12, 4, 1}, 0L, true)),
                false);
        TensorDescriptor mask = new TensorDescriptor(
                DataType.BOOL,
                Shape.of(3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),
                false);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(AggregateReductionKind.SUM, new MaskedReductionAttrs(1)),
                List.of(input, mask),
                List.of(descriptor(Shape.of(2, 4), false)))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new AxisReductionAttrs(3, false),
                input,
                descriptor(Shape.of(2, 3), false))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.of(2, 2)),
                input,
                descriptor(Shape.of(2, 2), false))));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                view,
                scalar)));
        assertFalse(provider.supports(reductionQuery(
                NumericalProfile.ACCELERATOR,
                AggregateReductionKind.SUM,
                NoOperationAttrs.INSTANCE,
                scalar,
                scalar)));
    }

    @Test
    void matmulCapabilityMatchesGeneralStaticTypeProfileGradientAndLayoutContract() {
        Operation matmul = new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
        TensorDescriptor left = descriptor(Shape.of(2, 3), true);
        TensorDescriptor right = descriptor(Shape.of(3, 4), false);
        TensorDescriptor output = descriptor(Shape.of(2, 4), true);
        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR, matmul, left, right, output));
        assertFalse(supportsMatmul(
                NumericalProfile.STRICT_IEEE, matmul, left, right, output));

        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(3), true),
                descriptor(Shape.of(3), false),
                descriptor(Shape.of(), true)));
        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(3), false),
                descriptor(Shape.of(3, 4), false),
                descriptor(Shape.of(4), false)));
        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(2, 3), false),
                descriptor(Shape.of(3), false),
                descriptor(Shape.of(2), false)));
        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(2, 5, 3), false),
                descriptor(Shape.of(1, 3, 4), false),
                descriptor(Shape.of(2, 5, 4), false)));

        TensorDescriptor transposed =
                transposeCandidate(DataType.FLOAT32, Shape.of(2, 5, 3), true);
        assertTrue(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                transposed,
                descriptor(Shape.of(3, 4), false),
                descriptor(Shape.of(2, 5, 4), true)));
        assertTrue(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(0, 2, 1))),
                List.of(descriptor(Shape.of(2, 3, 5), true)),
                List.of(transposed))));
        for (DataType carrier :
                List.of(DataType.BFLOAT16, DataType.INT32, DataType.INT64)) {
            TensorDescriptor carrierTranspose =
                    transposeCandidate(carrier, Shape.of(2, 3), false);
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    new Operation(
                            AxisTransformKind.PERMUTE,
                            new PermutationAttrs(List.of(1, 0))),
                    List.of(typed(carrier, Shape.of(3, 2), false)),
                    List.of(carrierTranspose))),
                    carrier + " local MATMUL transpose carrier");
        }

        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType leftType : List.of(DataType.INT32, DataType.INT64)) {
                for (DataType rightType : List.of(DataType.INT32, DataType.INT64)) {
                    DataType result = leftType == DataType.INT64 || rightType == DataType.INT64
                            ? DataType.INT64 : DataType.INT32;
                    assertTrue(supportsMatmul(
                            profile,
                            matmul,
                            typed(leftType, Shape.of(2, 3), false),
                            typed(rightType, Shape.of(3, 4), false),
                            typed(result, Shape.of(2, 4), false)),
                            profile + " " + leftType + " x " + rightType);
                    if (result != leftType) {
                        assertFalse(supportsMatmul(
                                profile,
                                matmul,
                                typed(leftType, Shape.of(2, 3), false),
                                typed(rightType, Shape.of(3, 4), false),
                                typed(leftType, Shape.of(2, 4), false)));
                    }
                }
            }
        }

        for (List<DataType> pair : List.of(
                List.of(DataType.BFLOAT16, DataType.FLOAT32),
                List.of(DataType.FLOAT32, DataType.BFLOAT16))) {
            TensorDescriptor mixedLeft = typed(pair.get(0), Shape.of(2, 3), false);
            TensorDescriptor mixedRight = typed(pair.get(1), Shape.of(3, 4), false);
            TensorDescriptor mixedOutput = typed(DataType.FLOAT32, Shape.of(2, 4), false);
            assertTrue(supportsMatmul(
                    NumericalProfile.ACCELERATOR,
                    matmul,
                    mixedLeft,
                    mixedRight,
                    mixedOutput));
            assertFalse(supportsMatmul(
                    NumericalProfile.STRICT_IEEE,
                    matmul,
                    mixedLeft,
                    mixedRight,
                    mixedOutput));
        }
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (DataType leftType : DataType.values()) {
                for (DataType rightType : DataType.values()) {
                    boolean integral = (leftType == DataType.INT32 || leftType == DataType.INT64)
                            && (rightType == DataType.INT32 || rightType == DataType.INT64);
                    boolean acceleratorFloating = profile == NumericalProfile.ACCELERATOR
                            && ((leftType == DataType.FLOAT32
                                            && rightType == DataType.FLOAT32)
                                    || (leftType == DataType.BFLOAT16
                                            && rightType == DataType.FLOAT32)
                                    || (leftType == DataType.FLOAT32
                                            && rightType == DataType.BFLOAT16));
                    DataType resultType;
                    if (integral) {
                        resultType = leftType == DataType.INT64 || rightType == DataType.INT64
                                ? DataType.INT64 : DataType.INT32;
                    } else if (leftType == DataType.FLOAT64 || rightType == DataType.FLOAT64) {
                        resultType = DataType.FLOAT64;
                    } else if (leftType == DataType.FLOAT32 || rightType == DataType.FLOAT32) {
                        resultType = DataType.FLOAT32;
                    } else if (leftType == DataType.BFLOAT16
                            && rightType == DataType.BFLOAT16) {
                        resultType = DataType.BFLOAT16;
                    } else {
                        resultType = DataType.FLOAT32;
                    }
                    assertEquals(
                            integral || acceleratorFloating,
                            supportsMatmul(
                                    profile,
                                    matmul,
                                    typed(leftType, Shape.of(2, 3), false),
                                    typed(rightType, Shape.of(3, 4), false),
                                    typed(resultType, Shape.of(2, 4), false)),
                            profile + " " + leftType + " x " + rightType);
                }
            }
        }
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.BFLOAT16, Shape.of(2, 3), true),
                typed(DataType.FLOAT32, Shape.of(3, 4), false),
                typed(DataType.FLOAT32, Shape.of(2, 4), true)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.BFLOAT16, Shape.of(2, 3), false),
                typed(DataType.BFLOAT16, Shape.of(3, 4), false),
                typed(DataType.BFLOAT16, Shape.of(2, 4), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.BFLOAT16, Shape.of(2, 3), true),
                typed(DataType.BFLOAT16, Shape.of(3, 4), false),
                typed(DataType.BFLOAT16, Shape.of(2, 4), true)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.FLOAT64, Shape.of(2, 3), false),
                typed(DataType.FLOAT32, Shape.of(3, 4), false),
                typed(DataType.FLOAT64, Shape.of(2, 4), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.FLOAT64, Shape.of(2, 3), true),
                typed(DataType.FLOAT32, Shape.of(3, 4), false),
                typed(DataType.FLOAT64, Shape.of(2, 4), true)));

        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                left,
                right,
                descriptor(Shape.of(2, 4), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.BFLOAT16, Shape.of(2, 3), false),
                typed(DataType.BFLOAT16, Shape.of(3, 4), false),
                typed(DataType.FLOAT32, Shape.of(2, 4), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                typed(DataType.FLOAT64, Shape.of(2, 3), false),
                typed(DataType.FLOAT32, Shape.of(3, 4), false),
                typed(DataType.FLOAT64, Shape.of(2, 4), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                left,
                descriptor(Shape.of(2, 4), false),
                output));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(2, 0), true),
                descriptor(Shape.of(0, 4), false),
                output));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.scalar(), false),
                descriptor(Shape.of(1), false),
                descriptor(Shape.scalar(), false)));
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                descriptor(Shape.of(2, 2, 3), false),
                descriptor(Shape.of(3, 3, 4), false),
                descriptor(Shape.of(3, 2, 4), false)));
        DynamicDimension dynamic = new DynamicDimension("N");
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(dynamic, new StaticDimension(3)),
                        Optional.empty(),
                        false),
                descriptor(Shape.of(3, 4), false),
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(dynamic, new StaticDimension(4)),
                        Optional.empty(),
                        false)));
        TensorDescriptor unauthenticated = new TensorDescriptor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.of(
                        Shape.of(2, 3), new long[] {2, 1}, 0L, true)),
                true);
        assertFalse(supportsMatmul(
                NumericalProfile.ACCELERATOR,
                matmul,
                unauthenticated,
                right,
                output));
    }

    @Test
    void admitsExactTask0052AcceleratorDomainAndKeepsEveryExclusionClosed() {
        TensorDescriptor left = descriptor(Shape.of(2, 1, 3), false);
        TensorDescriptor right = descriptor(Shape.of(1, 4, 3), false);
        TensorDescriptor broadcast = descriptor(Shape.of(2, 4, 3), false);
        TensorDescriptor boolBroadcast = typed(DataType.BOOL, Shape.of(2, 4, 3), false);
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) {
            var query = new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(left, right),
                    List.of(boolBroadcast));
            assertTrue(provider.supports(query), kind.name());
            assertFalse(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.STRICT_IEEE,
                    query.operation(),
                    query.inputs(),
                    query.outputs())));
        }
        for (BinaryArithmeticKind kind : List.of(
                BinaryArithmeticKind.MIN, BinaryArithmeticKind.MAX)) {
            assertTrue(provider.supports(binaryQuery(
                    NumericalProfile.ACCELERATOR, kind, left, right, broadcast)));
        }
        TensorDescriptor matrix = descriptor(Shape.of(2, 3), false);
        for (ScalarElementwiseKind kind : List.of(
                ScalarElementwiseKind.MIN, ScalarElementwiseKind.MAX)) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    new Operation(kind, new ScalarValueAttrs(ScalarValue.float32(-0.0f))),
                    List.of(matrix),
                    List.of(matrix))));
        }
        assertTrue(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(ScalarElementwiseKind.CLAMP, new ClampRangeAttrs(
                        ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f))),
                List.of(matrix),
                List.of(matrix))));
        for (AggregateReductionKind kind : List.of(
                AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
            assertTrue(provider.supports(reductionQuery(
                    NumericalProfile.ACCELERATOR,
                    kind,
                    NoOperationAttrs.INSTANCE,
                    matrix,
                    descriptor(Shape.scalar(), false))));
            assertTrue(provider.supports(reductionQuery(
                    NumericalProfile.ACCELERATOR,
                    kind,
                    new AxisReductionAttrs(1, false),
                    matrix,
                    descriptor(Shape.of(2), false))));
            assertTrue(provider.supports(reductionQuery(
                    NumericalProfile.ACCELERATOR,
                    kind,
                    new MultiAxisReductionAttrs(List.of(1, 0), false),
                    matrix,
                    descriptor(Shape.scalar(), false))));
            assertTrue(provider.supports(reductionQuery(
                    NumericalProfile.ACCELERATOR,
                    kind,
                    new MultiAxisReductionAttrs(List.of(), false),
                    matrix,
                    matrix)));
        }
        for (CumulativeScanKind kind : CumulativeScanKind.values()) {
            for (boolean exclusive : List.of(false, true)) {
                for (boolean reverse : List.of(false, true)) {
                    assertTrue(provider.supports(new OperationCapabilityQuery(
                            NumericalProfile.ACCELERATOR,
                            new Operation(kind, new CumulativeScanAttrs(
                                    1, exclusive, reverse)),
                            List.of(matrix),
                            List.of(matrix))));
                }
            }
        }
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                List.of(left, right),
                List.of(broadcast))), "comparison output must be BOOL");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(CumulativeScanKind.CUM_SUM,
                        new CumulativeScanAttrs(0, false, false)),
                List.of(descriptor(Shape.scalar(), false)),
                List.of(descriptor(Shape.scalar(), false)))), "scan input must have positive rank");
        assertFalse(provider.supports(binaryQuery(
                NumericalProfile.ACCELERATOR,
                BinaryArithmeticKind.MIN,
                descriptor(Shape.of(2, 3), true),
                matrix,
                matrix)), "requiresGrad stays excluded");
    }

    @Test
    void admitsExactCanonicalBoolClassificationLogicAndWhereInBothProfiles() {
        TensorDescriptor classifiedInput = typed(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor boolMatrix = typed(DataType.BOOL, Shape.of(2, 3), false);
        TensorDescriptor boolColumn = typed(DataType.BOOL, Shape.of(2, 1), false);
        TensorDescriptor boolRow = typed(DataType.BOOL, Shape.of(3), false);
        TensorDescriptor trueBranch = typed(DataType.FLOAT32, Shape.of(2, 1), true);
        TensorDescriptor falseBranch = typed(DataType.FLOAT32, Shape.of(1, 3), false);
        TensorDescriptor selected = typed(DataType.FLOAT32, Shape.of(2, 3), true);

        for (NumericalProfile profile : NumericalProfile.values()) {
            for (FloatingClassificationKind kind : FloatingClassificationKind.values()) {
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(classifiedInput),
                        List.of(boolMatrix))), profile + " " + kind);
            }
            for (BooleanLogicalKind kind : List.of(
                    BooleanLogicalKind.AND, BooleanLogicalKind.OR)) {
                assertTrue(provider.supports(new OperationCapabilityQuery(
                        profile,
                        new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(boolColumn, boolRow),
                        List.of(boolMatrix))), profile + " " + kind);
            }
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(BooleanLogicalKind.NOT, NoOperationAttrs.INSTANCE),
                    List.of(boolMatrix),
                    List.of(boolMatrix))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(boolRow, trueBranch, falseBranch),
                    List.of(selected))));
        }

        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(FloatingClassificationKind.IS_FINITE, NoOperationAttrs.INSTANCE),
                List.of(classifiedInput),
                List.of(typed(DataType.FLOAT32, Shape.of(2, 3), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(BooleanLogicalKind.AND, NoOperationAttrs.INSTANCE),
                List.of(typed(DataType.FLOAT32, Shape.of(2, 1), false), boolRow),
                List.of(boolMatrix))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(
                        typed(DataType.BOOL, Shape.of(2), false),
                        trueBranch,
                        falseBranch),
                List.of(selected))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(boolRow, trueBranch, falseBranch),
                List.of(typed(DataType.FLOAT32, Shape.of(2, 3), false)))));
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(
                        typed(DataType.BOOL, Shape.scalar(), false),
                        trueBranch,
                        falseBranch),
                List.of(selected))),
                "rank-zero BOOL constants stay outside the advertised Metal WHERE domain");
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.STRICT_IEEE,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(
                        boolRow,
                        typed(DataType.FLOAT32, Shape.scalar(), false),
                        typed(DataType.FLOAT32, Shape.scalar(), false)),
                List.of(typed(DataType.FLOAT32, Shape.scalar(), false)))),
                "rank-zero FLOAT32 branches stay outside the advertised Metal WHERE domain");
        Shape rankSeventeen =
                Shape.of(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assertFalse(provider.supports(new OperationCapabilityQuery(
                NumericalProfile.ACCELERATOR,
                new Operation(FloatingClassificationKind.IS_FINITE, NoOperationAttrs.INSTANCE),
                List.of(typed(DataType.FLOAT32, rankSeventeen, false)),
                List.of(typed(DataType.BOOL, rankSeventeen, false)))),
                "rank seventeen stays outside the exact classification domain");
    }

    @Test
    void rejectsNullWithTheContractMessage() {
        var failure = assertThrows(NullPointerException.class, () -> provider.supports(null));
        assertEquals("query", failure.getMessage());
    }

    private static OperationCapabilityQuery query(
            NumericalProfile profile, TensorDescriptor input, TensorDescriptor output) {
        return unaryQuery(profile, UnaryElementwiseKind.NEG, input, output);
    }

    private static OperationCapabilityQuery unaryQuery(
            NumericalProfile profile,
            UnaryElementwiseKind kind,
            TensorDescriptor input,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(input),
                List.of(output));
    }

    private static OperationCapabilityQuery binaryQuery(
            NumericalProfile profile,
            BinaryArithmeticKind kind,
            TensorDescriptor left,
            TensorDescriptor right,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(left, right),
                List.of(output));
    }

    private static OperationCapabilityQuery reductionQuery(
            NumericalProfile profile,
            AggregateReductionKind kind,
            io.github.pho001.synaptik.model.operation.OperationAttrs attrs,
            TensorDescriptor input,
            TensorDescriptor output) {
        return new OperationCapabilityQuery(
                profile,
                new Operation(kind, attrs),
                List.of(input),
                List.of(output));
    }

    private static boolean exactRawUnary(UnaryElementwiseKind kind) {
        return kind == UnaryElementwiseKind.NEG
                || kind == UnaryElementwiseKind.ABS
                || kind == UnaryElementwiseKind.FLOOR
                || kind == UnaryElementwiseKind.CEIL
                || kind == UnaryElementwiseKind.SIGN
                || kind == UnaryElementwiseKind.RELU;
    }

    private static Operation neg() {
        return new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor transposeCandidate(
            DataType type, Shape shape, boolean requiresGrad) {
        int rank = shape.rank();
        long[] sourceDimensions = shape.toLongArray();
        long swap = sourceDimensions[rank - 2];
        sourceDimensions[rank - 2] = sourceDimensions[rank - 1];
        sourceDimensions[rank - 1] = swap;
        long[] sourceStrides =
                LayoutDescriptor.contiguous(Shape.of(sourceDimensions)).strides();
        long[] outputStrides = sourceStrides.clone();
        outputStrides[rank - 2] = sourceStrides[rank - 1];
        outputStrides[rank - 1] = sourceStrides[rank - 2];
        return new TensorDescriptor(
                type,
                shape,
                Optional.of(LayoutDescriptor.of(shape, outputStrides, 0L, true)),
                requiresGrad);
    }

    private boolean supportsMatmul(
            NumericalProfile profile,
            Operation operation,
            TensorDescriptor left,
            TensorDescriptor right,
            TensorDescriptor output) {
        return provider.supports(new OperationCapabilityQuery(
                profile, operation, List.of(left, right), List.of(output)));
    }

    private static TensorDescriptor view(Shape shape, long... strides) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false);
    }

    private record Occurrence(
            String name,
            Operation operation,
            TensorDescriptor input,
            TensorDescriptor output) { }

    private static TensorDescriptor typed(
            DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor typed(DataType type) {
        Shape shape = Shape.of(2, 3);
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                type.isDifferentiable());
    }
}
