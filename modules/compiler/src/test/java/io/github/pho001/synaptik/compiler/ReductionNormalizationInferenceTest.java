package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class ReductionNormalizationInferenceTest {
    @Test
    void acceptsReductionsScansNormalizationAndOrdering() {
        Tensor x = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor scale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(3), true);
        List<Tensor> outputs = List.of(
                x.sum(), x.mean(1), x.sum(new int[] {0, 1}, true),
                x.variance(new int[] {1}, false, 1), x.argMax(1), x.cumSum(1),
                x.softmax(1), x.logSoftmax(1),
                x.layerNorm(Shape.of(3), scale, bias, ScalarValue.float32(1e-5f)),
                x.rmsNorm(Shape.of(3), ScalarValue.float32(1e-5f)),
                x.sort(1), x.argsort(1), x.topK(2, 1).values());
        for (Tensor output : outputs) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
    }

    @Test
    void coversEveryKindAndAttributesVariant() {
        TensorDescriptor floating = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor bool = descriptor(DataType.BOOL, Shape.of(2, 3), false);
        for (var kind : AggregateReductionKind.values()) {
            OperationAttrs attrs;
            List<TensorDescriptor> inputs = List.of(
                    kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY
                            ? bool : floating);
            if (kind == AggregateReductionKind.ARG_MAX
                    || kind == AggregateReductionKind.ARG_MIN) {
                attrs = new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX);
            } else if (kind == AggregateReductionKind.VARIANCE
                    || kind == AggregateReductionKind.STANDARD_DEVIATION) {
                attrs = new StatisticalReductionAttrs(List.of(1), false, 0);
            } else if (kind == AggregateReductionKind.LOG_SUM_EXP
                    || kind == AggregateReductionKind.L1_NORM
                    || kind == AggregateReductionKind.L2_NORM) {
                attrs = new MultiAxisReductionAttrs(List.of(1), false);
            } else {
                attrs = NoOperationAttrs.INSTANCE;
            }
            Operation operation = new Operation(kind, attrs);
            assertDoesNotThrow(() -> ReductionNormalizationInference.infer(operation, inputs));
        }
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(AggregateReductionKind.SUM, new AxisReductionAttrs(1, true)),
                List.of(floating)));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(AggregateReductionKind.SUM,
                        new MultiAxisReductionAttrs(List.of(0, 1), false)),
                List.of(floating)));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(AggregateReductionKind.SUM, new MaskedReductionAttrs(1)),
                List.of(floating, bool)));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(AggregateReductionKind.SUM,
                        new SumToShapeAttrs(Shape.of(1, 3))),
                List.of(floating)));
        for (var kind : CumulativeScanKind.values()) {
            assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                    new Operation(kind, new CumulativeScanAttrs(1, false, false)),
                    List.of(floating)));
        }
        for (var kind : SoftmaxKind.values()) {
            assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                    new Operation(kind, new SoftmaxAttrs(1)), List.of(floating)));
        }
        TensorDescriptor vector = descriptor(DataType.FLOAT32, Shape.of(3), true);
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(LayerNormKind.LAYER_NORM,
                        new LayerNormAttrs(Shape.of(3), ScalarValue.float32(1e-5f))),
                List.of(floating)));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(LayerNormKind.LAYER_NORM,
                        new AffineLayerNormAttrs(
                                Shape.of(3), ScalarValue.float32(1e-5f))),
                List.of(floating, vector, vector)));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(RmsNormKind.RMS_NORM,
                        new RmsNormAttrs(Shape.of(3), ScalarValue.float32(1e-5f))),
                List.of(floating, vector)));
        List<TensorDescriptor> batchInputs =
                List.of(floating, vector, vector, vector, vector);
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                        new BatchNormInferenceAttrs(1, ScalarValue.float32(1e-5f))),
                batchInputs));
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(BatchNormKind.BATCH_NORM_TRAINING,
                        new BatchNormTrainingAttrs(1,
                                ScalarValue.float32(.1f), ScalarValue.float32(1e-5f))),
                batchInputs));
        for (var kind : OrderingKind.values()) {
            assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                    new Operation(kind, new SortAttrs(1, false)), List.of(floating)));
        }
        assertDoesNotThrow(() -> ReductionNormalizationInference.infer(
                new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, true)),
                List.of(floating)));
    }

    @Test
    void rejectsManuallyConstructedInvalidReductionRuleCategories() {
        TensorDescriptor bool = descriptor(DataType.BOOL, Shape.of(2, 3), false);
        TensorDescriptor floating = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);

        assertInvalid(
                new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE),
                List.of(bool), List.of(descriptor(DataType.BOOL, Shape.scalar(), false)),
                "input must be numeric");
        assertInvalid(
                new Operation(AggregateReductionKind.MEAN, new MaskedReductionAttrs(1)),
                List.of(floating, floating),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), true)),
                "mask must be BOOL");
        assertInvalid(
                new Operation(AggregateReductionKind.VARIANCE,
                        new StatisticalReductionAttrs(List.of(1), false, 1)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 1), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), true)),
                "constraint statistical reduction domain failed");
        assertInvalid(
                new Operation(AggregateReductionKind.ARG_MAX,
                        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 0), true)),
                List.of(descriptor(DataType.INT64, Shape.of(2), false)),
                "constraint arg-extrema selected extent failed");
    }

    @Test
    void rejectsManuallyConstructedInvalidScanAndNormalizationRuleCategories() {
        assertInvalid(
                new Operation(CumulativeScanKind.CUM_SUM,
                        new CumulativeScanAttrs(0, false, false)),
                List.of(descriptor(DataType.BOOL, Shape.of(2), false)),
                List.of(descriptor(DataType.BOOL, Shape.of(2), false)),
                "input must be numeric");
        assertInvalid(
                new Operation(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(0)),
                List.of(descriptor(DataType.INT32, Shape.of(2), false)),
                List.of(descriptor(DataType.INT32, Shape.of(2), false)),
                "input must be floating");
        assertInvalid(
                new Operation(LayerNormKind.LAYER_NORM,
                        new LayerNormAttrs(Shape.of(4), ScalarValue.float32(1e-5f))),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                "constraint normalized axis 0 failed");
        assertInvalid(
                new Operation(RmsNormKind.RMS_NORM,
                        new RmsNormAttrs(Shape.of(3), ScalarValue.float32(1e-5f))),
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(2, 3), true),
                        descriptor(DataType.FLOAT32, Shape.of(4), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                "weight shape mismatch");
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(2, 3), false);
        TensorDescriptor channel = descriptor(DataType.FLOAT32, Shape.of(3), false);
        assertInvalid(
                new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                        new BatchNormInferenceAttrs(1, ScalarValue.float32(1e-5f))),
                List.of(input,
                        descriptor(DataType.FLOAT32, Shape.of(4), false),
                        channel, channel, channel),
                List.of(input), "constraint channel input[1] failed");

        TensorDescriptor smallDomainInput =
                descriptor(DataType.FLOAT32, Shape.of(1, 3), false);
        TensorDescriptor statistic = descriptor(DataType.FLOAT32, Shape.of(3), false);
        assertInvalid(
                new Operation(BatchNormKind.BATCH_NORM_TRAINING,
                        new BatchNormTrainingAttrs(1,
                                ScalarValue.float32(.1f), ScalarValue.float32(1e-5f))),
                List.of(smallDomainInput, statistic, statistic, statistic, statistic),
                List.of(smallDomainInput, statistic, statistic, statistic, statistic),
                "constraint batch-normalization training domain failed");
    }

    @Test
    void rejectsManuallyConstructedInvalidOrderingRuleCategories() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(2), true);
        assertInvalid(
                new Operation(OrderingKind.SORT, new SortAttrs(1, false)),
                List.of(input), List.of(input),
                "axis is not normalized for input rank");
        assertInvalid(
                new Operation(TopKKind.TOP_K, new TopKAttrs(0, 3, true, true)),
                List.of(input),
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(3), true),
                        descriptor(DataType.INT64, Shape.of(3), false)),
                "constraint top-K selected extent failed");
    }

    private static void assertInvalid(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs,
            String expectedDetail) {
        CompiledGraphModel graph = graph(operation, inputs, outputs);
        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();
        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains(expectedDetail), message));
    }

    private static CompiledGraphModel graph(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputIds = new ArrayList<>();
        List<ValueId> outputIds = new ArrayList<>();
        long id = 0;
        for (TensorDescriptor input : inputs) {
            ValueId valueId = new ValueId(id++);
            values.add(new GraphValue(valueId, input));
            inputIds.add(valueId);
        }
        for (TensorDescriptor output : outputs) {
            ValueId valueId = new ValueId(id++);
            values.add(new GraphValue(valueId, output));
            outputIds.add(valueId);
        }
        NodeId nodeId = new NodeId(0);
        return new CompiledGraphModel(
                values,
                List.of(new CompiledNode(nodeId, operation, inputIds, outputIds)),
                inputIds,
                outputIds,
                Map.of(nodeId, GraphPhase.FORWARD));
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(descriptor(dataType, shape, requiresGrad));
    }
}
