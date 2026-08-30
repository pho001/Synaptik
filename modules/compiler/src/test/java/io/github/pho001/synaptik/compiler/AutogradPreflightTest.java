package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.BatchNormTrainingResult;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

final class AutogradPreflightTest {
    @Test
    void locksTheExactSourceBacked0005AInventory() {
        assertArrayEquals(new BinaryArithmeticKind[] {
            BinaryArithmeticKind.ADD,
            BinaryArithmeticKind.SUB,
            BinaryArithmeticKind.MUL,
            BinaryArithmeticKind.DIV,
            BinaryArithmeticKind.MIN,
            BinaryArithmeticKind.MAX,
            BinaryArithmeticKind.POW
        }, BinaryArithmeticKind.values());
        assertArrayEquals(new ScalarElementwiseKind[] {
            ScalarElementwiseKind.ADD,
            ScalarElementwiseKind.SUB,
            ScalarElementwiseKind.MUL,
            ScalarElementwiseKind.DIV,
            ScalarElementwiseKind.MIN,
            ScalarElementwiseKind.MAX,
            ScalarElementwiseKind.POW,
            ScalarElementwiseKind.CLAMP
        }, ScalarElementwiseKind.values());
        assertArrayEquals(new UnaryElementwiseKind[] {
            UnaryElementwiseKind.ABS,
            UnaryElementwiseKind.NEG,
            UnaryElementwiseKind.RECIPROCAL,
            UnaryElementwiseKind.LOG,
            UnaryElementwiseKind.LOG1P,
            UnaryElementwiseKind.EXP,
            UnaryElementwiseKind.EXPM1,
            UnaryElementwiseKind.ERF,
            UnaryElementwiseKind.SQRT,
            UnaryElementwiseKind.RSQRT,
            UnaryElementwiseKind.FLOOR,
            UnaryElementwiseKind.CEIL,
            UnaryElementwiseKind.SIGN,
            UnaryElementwiseKind.RELU,
            UnaryElementwiseKind.SIGMOID,
            UnaryElementwiseKind.TANH,
            UnaryElementwiseKind.GELU,
            UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
            UnaryElementwiseKind.SILU
        }, UnaryElementwiseKind.values());
        assertArrayEquals(
                new WhereSelectionKind[] {WhereSelectionKind.WHERE},
                WhereSelectionKind.values());
        assertArrayEquals(new CastKind[] {CastKind.CAST}, CastKind.values());
        assertArrayEquals(new BinaryComparisonKind[] {
            BinaryComparisonKind.GREATER_THAN,
            BinaryComparisonKind.GREATER_OR_EQUAL,
            BinaryComparisonKind.LESS_THAN,
            BinaryComparisonKind.LESS_OR_EQUAL,
            BinaryComparisonKind.EQUAL,
            BinaryComparisonKind.NOT_EQUAL
        }, BinaryComparisonKind.values());
        assertArrayEquals(new BooleanLogicalKind[] {
            BooleanLogicalKind.AND,
            BooleanLogicalKind.OR,
            BooleanLogicalKind.NOT
        }, BooleanLogicalKind.values());
        assertArrayEquals(new FloatingClassificationKind[] {
            FloatingClassificationKind.IS_FINITE,
            FloatingClassificationKind.IS_NAN,
            FloatingClassificationKind.IS_INF
        }, FloatingClassificationKind.values());
        assertEquals(
                48,
                BinaryArithmeticKind.values().length
                        + ScalarElementwiseKind.values().length
                        + UnaryElementwiseKind.values().length
                        + WhereSelectionKind.values().length
                        + CastKind.values().length
                        + BinaryComparisonKind.values().length
                        + BooleanLogicalKind.values().length
                        + FloatingClassificationKind.values().length);
    }

    @Test
    void locksTheExactSourceBacked0005DStructuredKindInventory() {
        assertArrayEquals(
                new ScaledDotProductAttentionKind[] {
                    ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION
                },
                ScaledDotProductAttentionKind.values());
        assertArrayEquals(new Conv2dKind[] {Conv2dKind.CONV2D}, Conv2dKind.values());
        assertArrayEquals(
                new Pool2dKind[] {
                    Pool2dKind.MAX_POOL2D, Pool2dKind.AVERAGE_POOL2D
                },
                Pool2dKind.values());
        assertArrayEquals(
                new LossKind[] {
                    LossKind.MEAN_SQUARED_ERROR,
                    LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                },
                LossKind.values());
    }

    @Test
    void selectsOnlyObjectiveToTargetOccurrencesInDeterministicPostorder() {
        Tensor target = tensor(Shape.of(2), true);
        Tensor unrelated = tensor(Shape.of(2), true);
        Tensor selected = target.mul(unrelated);
        Tensor objective = selected.sum();

        AutogradPreflight.StagePlan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        assertEquals(List.of(
                selected.provenance().orElseThrow().producer(),
                objective.provenance().orElseThrow().producer()), plan.producerPostorder());
        assertEquals(2, plan.selectedOccurrences().size());
        assertTrue(plan.selectedOccurrences().get(0).selectedInput(0));
        assertTrue(!plan.selectedOccurrences().get(0).selectedInput(1));
    }

    @Test
    void rejectsUnsupportedAndNonDifferentiableSelectedRoutes() {
        Tensor branch = tensor(Shape.of(2), true);
        Tensor conditionSource = tensor(Shape.of(2), true);
        Tensor condition = conditionSource.greaterThan(branch);
        Tensor objective = Tensor.where(condition, branch, branch).sum();
        IllegalArgumentException conditionFailure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective),
                        FunctionalGradientTestSupport.stage(
                                objective, List.of(conditionSource)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(conditionFailure.getMessage().contains("non-differentiable")
                || conditionFailure.getMessage().contains("SUPPORTED_0004")
                || conditionFailure.getMessage().contains("GREATER_THAN"));
    }

    @Test
    void requiresExactScalarForwardObjectiveAndIdentityUniqueTargets() {
        Tensor vector = tensor(Shape.of(2), true);
        assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.TRAINING_STEP,
                        List.of(vector),
                        FunctionalGradientTestSupport.stage(vector, List.of(vector)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertThrows(
                IllegalArgumentException.class,
                () -> FunctionalGradientTestSupport.stage(vector, List.of(vector, vector)));
    }

    @Test
    void admitsExactAndBindingDependentSumToShapeInversion() {
        DynamicDimension sourceExtent = new DynamicDimension("N");
        Tensor exactTarget = tensor(Shape.ofDimensions(sourceExtent), true);
        Tensor exactObjective =
                exactTarget.sumToShape(Shape.ofDimensions(sourceExtent)).sum();
        AutogradPreflight.StagePlan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(exactObjective),
                FunctionalGradientTestSupport.stage(exactObjective, List.of(exactTarget)),
                CompileTimeConstantGraph.Ingress.empty());
        assertEquals(2, plan.selectedOccurrences().size());

        Tensor bindingDependent = tensor(Shape.ofDimensions(sourceExtent), true);
        Tensor bindingObjective = bindingDependent
                .sumToShape(Shape.ofDimensions(new DynamicDimension("M")))
                .sum();
        AutogradPreflight.StagePlan bindingPlan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(bindingObjective),
                FunctionalGradientTestSupport.stage(
                        bindingObjective, List.of(bindingDependent)),
                CompileTimeConstantGraph.Ingress.empty());
        assertEquals(2, bindingPlan.selectedOccurrences().size());
    }

    @Test
    void sliceUpdateSupportsLengthDefinedExtractionFromDynamicBase() {
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor base = tensor(Shape.ofDimensions(dynamic), true);
        Tensor update = tensor(Shape.ofDimensions(new StaticDimension(2)), true);
        Tensor objective = base.sliceUpdate(
                        update,
                        new long[] {0},
                        new int[] {0},
                        new long[] {1})
                .sum();

        AutogradPreflight.StagePlan basePlan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(base)),
                CompileTimeConstantGraph.Ingress.empty());
        assertTrue(basePlan.selectedOccurrences().stream()
                .anyMatch(occurrence -> occurrence.selectedInput(0)));

        AutogradPreflight.StagePlan updatePlan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(update)),
                CompileTimeConstantGraph.Ingress.empty());
        assertTrue(updatePlan.selectedOccurrences().stream()
                .anyMatch(occurrence -> occurrence.selectedInput(1)));
    }

    @Test
    void keepsMaskAndRepresentativePolicyDependentRoutesNonDifferentiable() {
        Tensor data = tensor(Shape.of(3), true);
        Tensor maskSource = tensor(Shape.of(3), true);
        Tensor mask = maskSource.greaterThan(data);
        Tensor maskedObjective = data.sum(0, mask);
        assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(maskedObjective),
                        FunctionalGradientTestSupport.stage(
                                maskedObjective, List.of(maskSource)),
                        CompileTimeConstantGraph.Ingress.empty()));

        Tensor cropObjective = data.cropToShape(Shape.of(2), Shape.of(0)).sum();
        AutogradPreflight.StagePlan cropPlan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(cropObjective),
                FunctionalGradientTestSupport.stage(cropObjective, List.of(data)),
                CompileTimeConstantGraph.Ingress.empty());
        assertTrue(cropPlan.selectedOccurrences().stream()
                .anyMatch(occurrence -> occurrence.producer().operation().kind()
                        == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE));
    }

    @Test
    void locksTheExactSourceBacked0005BKindInventory() {
        assertArrayEquals(new AggregateReductionKind[] {
            AggregateReductionKind.SUM,
            AggregateReductionKind.MEAN,
            AggregateReductionKind.PROD,
            AggregateReductionKind.MIN,
            AggregateReductionKind.MAX,
            AggregateReductionKind.ALL,
            AggregateReductionKind.ANY,
            AggregateReductionKind.ARG_MAX,
            AggregateReductionKind.ARG_MIN,
            AggregateReductionKind.LOG_SUM_EXP,
            AggregateReductionKind.VARIANCE,
            AggregateReductionKind.STANDARD_DEVIATION,
            AggregateReductionKind.L1_NORM,
            AggregateReductionKind.L2_NORM
        }, AggregateReductionKind.values());
        assertArrayEquals(
                new CumulativeScanKind[] {
                    CumulativeScanKind.CUM_SUM, CumulativeScanKind.CUM_PROD
                },
                CumulativeScanKind.values());
        assertArrayEquals(
                new SoftmaxKind[] {SoftmaxKind.SOFTMAX, SoftmaxKind.LOG_SOFTMAX},
                SoftmaxKind.values());
        assertArrayEquals(
                new LayerNormKind[] {LayerNormKind.LAYER_NORM},
                LayerNormKind.values());
        assertArrayEquals(
                new RmsNormKind[] {RmsNormKind.RMS_NORM},
                RmsNormKind.values());
        assertArrayEquals(
                new BatchNormKind[] {
                    BatchNormKind.BATCH_NORM_INFERENCE,
                    BatchNormKind.BATCH_NORM_TRAINING
                },
                BatchNormKind.values());
    }

    @Test
    void selectsBatchTrainingRolesByAscendingPublicOutputSlot() {
        Tensor input = tensor(Shape.of(2, 3, 4), true);
        Tensor scale = tensor(Shape.of(3), true);
        Tensor bias = tensor(Shape.of(3), true);
        Tensor runningMean = tensor(Shape.of(3), true);
        Tensor runningVariance = tensor(Shape.of(3), true);
        BatchNormTrainingResult result = input.batchNormTraining(
                1,
                scale,
                bias,
                runningMean,
                runningVariance,
                ScalarValue.float32(0.1f),
                ScalarValue.float32(1.0e-5f));
        Tensor objective = result.output().sum()
                .add(result.nextRunningMean().sum())
                .add(result.nextRunningVariance().sum());

        AutogradPreflight.StagePlan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(
                        objective,
                        List.of(input, scale, bias, runningMean, runningVariance)),
                CompileTimeConstantGraph.Ingress.empty());
        List<AutogradPreflight.SelectedOccurrence> batch = plan.selectedOccurrences().stream()
                .filter(occurrence -> occurrence.producer().operation().kind()
                        == BatchNormKind.BATCH_NORM_TRAINING)
                .toList();

        assertEquals(List.of(0, 1, 2),
                batch.stream().map(AutogradPreflight.SelectedOccurrence::outputIndex).toList());
        assertArrayEquals(new boolean[] {true, true, true, false, false},
                batch.get(0).selectedInputs());
        assertArrayEquals(new boolean[] {true, false, false, true, false},
                batch.get(1).selectedInputs());
        assertArrayEquals(new boolean[] {true, false, false, false, true},
                batch.get(2).selectedInputs());
    }

    @Test
    void rejectsBatchTrainingSavedAuxiliarySlotsAsSelectedRoutes() {
        Tensor input = tensor(Shape.of(2, 3, 4), true);
        Tensor vector = tensor(Shape.of(3), true);
        BatchNormTrainingResult result = input.batchNormTraining(
                1,
                vector,
                vector,
                vector,
                vector,
                ScalarValue.float32(0.1f),
                ScalarValue.float32(1.0e-5f));
        Tensor savedMean = result.output().provenance().orElseThrow().producer().output(3);
        Tensor objective = savedMean.sum();

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective),
                        FunctionalGradientTestSupport.stage(objective, List.of(savedMean)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(failure.getMessage().contains("saved auxiliary"));
        IllegalArgumentException routeFailure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective),
                        FunctionalGradientTestSupport.stage(objective, List.of(input)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(routeFailure.getMessage().contains("non-differentiable"));
    }

    @Test
    void admitsMixedFloatingNormalizationDivisionDirectZeroAndMeanRows() {
        Tensor narrow = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
        Tensor wide = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
        for (Tensor objective : List.of(
                narrow.add(wide).mean(),
                narrow.div(wide).mean(),
                narrow.cast(DataType.FLOAT32).mean(),
                narrow.floor().mean(),
                narrow.ceil().mean(),
                narrow.sign().mean())) {
            AutogradPreflight.StagePlan plan = AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    FunctionalGradientTestSupport.stage(objective, List.of(narrow)),
                    CompileTimeConstantGraph.Ingress.empty());
            assertTrue(!plan.selectedOccurrences().isEmpty());
        }
    }

    @Test
    void rejectsConv3dFromCompleteForwardInventoryBeforeDerivativeAllocation()
            throws Exception {
        for (CompileMode mode : List.of(
                CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            Tensor input = tensor(Shape.of(1, 2, 5, 5, 5), true);
            Tensor weight = tensor(Shape.of(4, 2, 3, 3, 3), true);
            Tensor bias = tensor(Shape.of(4), true);
            Tensor conv3d = mode == CompileMode.FORWARD_AND_BACKWARD
                    ? input.conv3d(weight, Conv3dAttrs.defaults())
                    : input.conv3d(weight, bias, Conv3dAttrs.defaults());
            Tensor target = tensor(Shape.of(2), true);
            Tensor objective = target.mul(target).sum();
            FunctionalGradientRequest.Stage request =
                    FunctionalGradientTestSupport.stage(objective, List.of(target));
            long before = nextTensorId();

            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> AutogradPreflight.preflight(
                            mode,
                            List.of(conv3d, objective),
                            request,
                            CompileTimeConstantGraph.Ingress.empty()));

            assertTrue(failure.getMessage().startsWith("producerPostorder[0] "),
                    failure.getMessage());
            assertTrue(failure.getMessage().contains(
                    Conv3dKind.class.getName() + ".CONV3D attrs="
                            + Conv3dAttrs.class.getName()), failure.getMessage());
            assertTrue(failure.getMessage().endsWith(
                    "Conv3d is forward-only until Compiler task 0006C closes its gradients"),
                    failure.getMessage());
            assertEquals(before, nextTensorId());
        }
    }

    @Test
    void rejectsEveryNewVolumetricSignatureFromCompleteInventoryBeforeAllocation()
            throws Exception {
        Window3dAttrs window = new Window3dAttrs(
                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor input = tensor(Shape.of(1, 2, 4, 4, 4), true);
        Tensor directColumns = input.unfold3d(window);
        List<Tensor> deferred = List.of(
                input.maxPool3d(new MaxPool3dAttrs(
                        2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false)),
                input.averagePool3d(new AveragePool3dAttrs(
                        2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false)),
                directColumns,
                input.unfold3d(window, ScalarValue.float32(0)),
                tensor(Shape.of(1, 16, 27), true)
                        .fold3d(input.descriptor().shape(), window));

        for (CompileMode mode : List.of(
                CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            for (Tensor occurrence : deferred) {
                Tensor target = tensor(Shape.of(2), true);
                Tensor objective = target.mul(target).sum();
                Tensor invalidSeed = TensorFactory.create(new TensorDescriptor(
                        DataType.INT32, Shape.of(2), Optional.empty(), false));
                FunctionalGradientRequest.Stage invalidSeedRequest =
                        new FunctionalGradientRequest.Stage(
                                List.of(new FunctionalGradientRequest.ForwardTensorReference(
                                        objective)),
                                List.of(Optional.of(invalidSeed)),
                                List.of(target),
                                false,
                                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                long before = nextTensorId();
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class,
                        () -> AutogradPreflight.preflight(
                                mode,
                                List.of(occurrence, objective),
                                invalidSeedRequest,
                                CompileTimeConstantGraph.Ingress.empty()));
                assertTrue(failure.getMessage().startsWith("producerPostorder[0] "),
                        failure.getMessage());
                if (occurrence.provenance().orElseThrow().operation().kind()
                        instanceof Pool3dKind) {
                    assertTrue(failure.getMessage().contains(Pool3dKind.class.getName()),
                            failure.getMessage());
                    assertTrue(failure.getMessage().endsWith(
                            "Pool3d is forward-only until Compiler task 0006B2 closes its gradients"),
                            failure.getMessage());
                } else {
                    assertTrue(failure.getMessage().contains(
                            WindowTransformKind.class.getName()), failure.getMessage());
                    assertTrue(failure.getMessage().endsWith(
                            "three-dimensional window transforms are forward-only until Compiler task 0006B2 closes their gradients"),
                            failure.getMessage());
                }
                assertEquals(before, nextTensorId());
            }
        }
    }

    private static Tensor tensor(Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), requiresGrad));
    }

    private static long nextTensorId() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
    }
}
