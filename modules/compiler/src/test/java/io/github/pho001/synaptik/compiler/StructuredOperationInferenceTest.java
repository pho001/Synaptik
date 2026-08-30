package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class StructuredOperationInferenceTest {
    @Test
    void acceptsStructuredLossAndStateFamilies() {
        Tensor a = tensor(Shape.of(2, 3));
        Tensor b = tensor(Shape.of(3, 4));
        Tensor image = tensor(Shape.of(1, 2, 5, 5));
        Tensor weight = tensor(Shape.of(4, 2, 3, 3));
        Tensor bias = tensor(Shape.of(4));
        Tensor query = tensor(Shape.of(2, 3, 4));
        Tensor key = tensor(Shape.of(2, 5, 4));
        Tensor value = tensor(Shape.of(2, 5, 6));
        Tensor mask = typed(DataType.BOOL, Shape.of(2, 3, 5), false);
        Tensor indices = typed(DataType.INT64, Shape.of(2), false);
        List<Tensor> outputs = List.of(
                a.matmul(b),
                image.conv2d(weight, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1)),
                image.conv2d(weight, bias, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1)),
                image.maxPool2d(new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false)),
                image.averagePool2d(
                        new AveragePool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, true)),
                a.meanSquaredError(tensor(Shape.of(2, 3)), LossReduction.MEAN),
                a.categoricalCrossEntropyWithLogits(
                        tensor(Shape.of(2, 3)), 1, LossReduction.NONE),
                a.categoricalCrossEntropyWithLogits(indices, 1, LossReduction.NONE),
                query.scaledDotProductAttention(key, value),
                query.scaledDotProductAttention(key, value, mask));
        for (Tensor output : outputs) {
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                    GraphCapture.capture(List.of(output))));
        }
        ScaledDotProductAttentionResult attention =
                query.scaledDotProductAttentionWithWeights(key, value);
        assertEquals(Shape.of(2, 3, 6), attention.output().descriptor().shape());
        assertEquals(Shape.of(2, 3, 5), attention.weights().descriptor().shape());
        assertSame(
                attention.output().provenance().orElseThrow().producer(),
                attention.weights().provenance().orElseThrow().producer());
        assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(GraphCapture.capture(
                List.of(attention.output(), attention.weights()))));
        DropoutResult dropout = a.dropout(.2, GraphRngState.initial(1, 2));
        assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(
                GraphCapture.capture(List.of(dropout.output()))));
    }

    @Test
    void rejectsManuallyConstructedInvalidMatrixAndAttentionRuleCategories() {
        TensorDescriptor left = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        TensorDescriptor right = descriptor(DataType.FLOAT32, Shape.of(4, 5), true);
        assertInvalid(
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(left, right),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 5), true)),
                "constraint matmul contraction failed");

        TensorDescriptor query = descriptor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        TensorDescriptor key = descriptor(DataType.FLOAT32, Shape.of(2, 5, 4), true);
        TensorDescriptor value = descriptor(DataType.FLOAT32, Shape.of(2, 6, 7), true);
        assertInvalid(
                new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                List.of(query, key, value),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3, 7), true)),
                "constraint attention key/value sequence failed");
        assertInvalid(
                new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                List.of(
                        query,
                        descriptor(DataType.FLOAT32, Shape.of(2, 5, 4), true),
                        descriptor(DataType.FLOAT32, Shape.of(2, 5, 7), true),
                        descriptor(DataType.FLOAT32, Shape.of(2, 3, 5), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3, 7), true)),
                "mask must be BOOL");
    }

    @Test
    void rejectsManuallyConstructedInvalidConvolutionAndPoolingRuleCategories() {
        TensorDescriptor image = descriptor(DataType.FLOAT32, Shape.of(1, 3, 5, 5), true);
        TensorDescriptor weight = descriptor(DataType.FLOAT32, Shape.of(4, 2, 3, 3), true);
        assertInvalid(
                new Operation(Conv2dKind.CONV2D,
                        new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2)),
                List.of(image, weight),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 4, 3, 3), true)),
                "constraint input channels divisible by groups failed");

        assertInvalid(
                new Operation(Pool2dKind.MAX_POOL2D,
                        new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), true)),
                "effective kernel does not fit");
    }

    @Test
    void independentlyRejectsConv3dRoleKernelConstraintAndGeometryCategories() {
        Operation operation = new Operation(Conv3dKind.CONV3D, Conv3dAttrs.defaults());
        TensorDescriptor validInput = descriptor(
                DataType.FLOAT32, Shape.of(1, 4, 5, 5, 5), true);
        TensorDescriptor validWeight = descriptor(
                DataType.FLOAT32, Shape.of(6, 4, 3, 3, 3), true);
        TensorDescriptor validOutput = descriptor(
                DataType.FLOAT32, Shape.of(1, 6, 3, 3, 3), true);

        assertInvalid(
                operation,
                List.of(descriptor(DataType.INT32, Shape.of(1, 4, 5, 5, 5), false), validWeight),
                List.of(validOutput),
                "conv3d input must be floating");
        assertInvalid(
                operation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 4, 5, 5), true), validWeight),
                List.of(validOutput),
                "conv3d input rank must be 5: 4");
        assertInvalid(
                operation,
                List.of(validInput, descriptor(DataType.FLOAT32,
                        Shape.ofDimensions(
                                new StaticDimension(6), new StaticDimension(4),
                                new DynamicDimension("Kd"), new StaticDimension(3),
                                new StaticDimension(3)), true)),
                List.of(validOutput),
                "conv3d kernel depth must be static");

        Conv3dAttrs groupsTwo = new Conv3dAttrs(
                1, 1, 1, 0, 0, 0, 1, 1, 1, 2);
        assertInvalid(
                new Operation(Conv3dKind.CONV3D, groupsTwo),
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 5, 5, 5, 5), true),
                        descriptor(DataType.FLOAT32, Shape.of(6, 2, 3, 3, 3), true)),
                List.of(validOutput),
                "constraint conv3d input channels divisible by groups failed");
        assertInvalid(
                new Operation(Conv3dKind.CONV3D, groupsTwo),
                List.of(
                        validInput,
                        descriptor(DataType.FLOAT32, Shape.of(5, 2, 3, 3, 3), true)),
                List.of(descriptor(
                        DataType.FLOAT32, Shape.of(1, 5, 3, 3, 3), true)),
                "constraint conv3d output channels divisible by groups failed");
        assertInvalid(
                new Operation(Conv3dKind.CONV3D, groupsTwo),
                List.of(
                        validInput,
                        descriptor(DataType.FLOAT32, Shape.of(6, 3, 3, 3, 3), true)),
                List.of(validOutput),
                "constraint conv3d weight channels per group failed");
        assertInvalid(
                operation,
                List.of(
                        validInput,
                        validWeight,
                        descriptor(DataType.FLOAT32, Shape.of(5), true)),
                List.of(validOutput),
                "constraint conv3d bias channels failed");
        assertInvalid(
                operation,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 4, 2, 5, 5), true),
                        validWeight),
                List.of(validOutput),
                "conv3d effective kernel does not fit padded depth");
        assertInvalid(
                operation,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 4, 3, 2, 5), true),
                        validWeight),
                List.of(validOutput),
                "conv3d effective kernel does not fit padded height");
        assertInvalid(
                operation,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 4, 3, 3, 2), true),
                        validWeight),
                List.of(validOutput),
                "conv3d effective kernel does not fit padded width");
    }

    @Test
    void reportsConv3dCheckedOverflowAndCompleteStoredDescriptorMismatchContext() {
        Operation overflow = new Operation(
                Conv3dKind.CONV3D,
                new Conv3dAttrs(
                        1, 1, 1, 0, 0, 0, Long.MAX_VALUE, 1, 1, 1));
        assertInvalid(
                overflow,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 1, 5, 5, 5), false),
                        descriptor(DataType.FLOAT32, Shape.of(1, 1, 2, 1, 1), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 5, 5), false)),
                "descriptor derivation failed: long overflow");
        assertInvalid(
                new Operation(
                        Conv3dKind.CONV3D,
                        new Conv3dAttrs(
                                1, 1, 1, Long.MAX_VALUE, 0, 0, 1, 1, 1, 1)),
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 1, 5, 5, 5), false),
                        descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 5, 5), false)),
                "descriptor derivation failed: long overflow");
        assertInvalid(
                new Operation(
                        Conv3dKind.CONV3D,
                        new Conv3dAttrs(1, 1, 1, 1, 0, 0, 1, 1, 1, 1)),
                List.of(
                        descriptor(DataType.FLOAT32,
                                Shape.of(1, 1, Long.MAX_VALUE, 5, 5), false),
                        descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 1, 1, 5, 5), false)),
                "descriptor derivation failed: long overflow");
        assertInvalid(
                new Operation(
                        Conv3dKind.CONV3D,
                        new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2)),
                List.of(
                        descriptor(DataType.FLOAT32,
                                Shape.ofDimensions(
                                        new StaticDimension(1), new DynamicDimension("Cin"),
                                        new StaticDimension(5), new StaticDimension(5),
                                        new StaticDimension(5)), false),
                        descriptor(DataType.FLOAT32,
                                Shape.of(6, Long.MAX_VALUE, 1, 1, 1), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 6, 5, 5, 5), false)),
                "descriptor derivation failed: long overflow");

        Operation operation = new Operation(Conv3dKind.CONV3D, Conv3dAttrs.defaults());
        CompiledGraphModel graph = graph(
                operation,
                List.of(
                        descriptor(DataType.FLOAT32, Shape.of(1, 4, 5, 5, 5), true),
                        descriptor(DataType.FLOAT32, Shape.of(6, 4, 3, 3, 3), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 6, 4, 3, 3), true)));
        String message = assertThrows(
                IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();
        assertAll(
                () -> assertTrue(message.startsWith(
                        "nodes[0] NodeId[value=0] " + Conv3dKind.class.getName()
                                + ".CONV3D: output[0] ValueId[value=2]"), message),
                () -> assertTrue(message.contains("expected=TensorDescriptor"), message),
                () -> assertTrue(message.contains("stored=TensorDescriptor"), message));
    }

    @Test
    void rejectsManuallyConstructedInvalidLossRuleCategories() {
        TensorDescriptor logits = descriptor(DataType.FLOAT32, Shape.of(2, 3), true);
        assertInvalid(
                new Operation(LossKind.MEAN_SQUARED_ERROR,
                        new MeanSquaredErrorAttrs(LossReduction.NONE)),
                List.of(logits, descriptor(DataType.FLOAT32, Shape.of(2, 4), false)),
                List.of(logits), "constraint MSE axis 1 failed");
        assertInvalid(
                new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new DenseCategoricalCrossEntropyWithLogitsAttrs(
                                1, LossReduction.NONE)),
                List.of(logits, descriptor(DataType.BOOL, Shape.of(2, 3), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), true)),
                "target must be floating");
        assertInvalid(
                new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new IndexCategoricalCrossEntropyWithLogitsAttrs(
                                1, LossReduction.NONE, Optional.empty())),
                List.of(logits, descriptor(DataType.FLOAT32, Shape.of(2), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), true)),
                "target must be INT32 or INT64");
    }

    @Test
    void rejectsManuallyConstructedInvalidGraphStateRuleCategories() {
        assertInvalid(
                new Operation(GraphRngKind.INITIAL_STATE,
                        new GraphRngStateAttrs(1, 2)),
                List.of(),
                List.of(descriptor(DataType.INT32, Shape.of(2), false)),
                "output[0]");

        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(2), true);
        assertInvalid(
                new Operation(DropoutKind.DROPOUT, new DropoutAttrs(.2)),
                List.of(input, descriptor(DataType.INT32, Shape.of(2), false)),
                List.of(
                        input,
                        descriptor(DataType.BOOL, Shape.of(2), false),
                        descriptor(DataType.INT64, Shape.of(2), false)),
                "invalid RNG state descriptor");
    }

    @Test
    void independentlyValidatesPool3dRankTypeGeometryAndStoredDescriptor() {
        MaxPool3dAttrs attrs = new MaxPool3dAttrs(
                3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        assertInvalid(
                new Operation(Pool3dKind.MAX_POOL3D, attrs),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 5, 5), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 3, 3, 3), true)),
                "pool3d input rank must be five");
        assertInvalid(
                new Operation(Pool3dKind.MAX_POOL3D, attrs),
                List.of(descriptor(DataType.INT32, Shape.of(1, 2, 5, 5, 5), false)),
                List.of(descriptor(DataType.INT32, Shape.of(1, 2, 3, 3, 3), false)),
                "pool3d input must be floating");
        assertInvalid(
                new Operation(Pool3dKind.MAX_POOL3D, attrs),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 2, 5, 5), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 0, 3, 3), true)),
                "effective kernel does not fit padded depth");
        assertInvalid(
                new Operation(
                        Pool3dKind.AVERAGE_POOL3D,
                        new AveragePool3dAttrs(
                                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 4, 4, 4), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(1, 2, 2, 3, 3), true)),
                "output[0]");
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

    private static Tensor tensor(Shape shape) {
        return typed(DataType.FLOAT32, shape, true);
    }

    private static Tensor typed(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(descriptor(dataType, shape, requiresGrad));
    }
}
