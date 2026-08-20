package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.RecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class RecurrentScanInferenceTest {
    @Test
    void exposesOnePackagePrivateStatelessInferenceContract() throws Exception {
        var constructor = RecurrentScanInference.class.getDeclaredConstructor();
        var method = RecurrentScanInference.class.getDeclaredMethod(
                "infer", Operation.class, List.class, int.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(RecurrentScanInference.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(RecurrentScanInference.class.getModifiers())),
                () -> assertEquals(0, RecurrentScanInference.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())));
    }

    @Test
    void derivesEveryKindDirectionAndBiasCardinalityFromOrderedDescriptors() {
        Fixtures f = fixtures(DataType.FLOAT64, 5, 2, 4, 3, true);
        for (RecurrentDirection direction : RecurrentDirection.values()) {
            assertInference(RecurrentScan.rnn(
                    f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                    direction));
            assertInference(RecurrentScan.rnn(
                    f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                    f.rnnBias, direction));
            assertInference(RecurrentScan.gru(
                    f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                    direction));
            assertInference(RecurrentScan.gru(
                    f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                    f.gruBias, direction));
            assertInference(RecurrentScan.lstm(
                    f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                    f.lstmHiddenWeight, direction));
            assertInference(RecurrentScan.lstm(
                    f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                    f.lstmHiddenWeight, f.lstmBias, direction));
        }
    }

    @Test
    void acceptsZeroTimeAndBatchAndExcludesValidLengthsFromGradientEligibility() {
        Fixtures zeroTime = fixtures(DataType.FLOAT32, 0, 2, 3, 4, false);
        var time = infer(producer(RecurrentScan.rnn(
                zeroTime.input, zeroTime.lengths, zeroTime.hidden,
                zeroTime.rnnInputWeight, zeroTime.rnnHiddenWeight,
                RecurrentDirection.REVERSE)));
        Fixtures zeroBatch = fixtures(DataType.FLOAT32, 4, 0, 3, 2, false);
        var batch = infer(producer(RecurrentScan.lstm(
                zeroBatch.input, zeroBatch.lengths, zeroBatch.hidden, zeroBatch.cell,
                zeroBatch.lstmInputWeight, zeroBatch.lstmHiddenWeight,
                RecurrentDirection.FORWARD)));

        assertAll(
                () -> assertEquals(Shape.of(0, 2, 4), time.outputs().get(0).shape()),
                () -> assertEquals(Shape.of(2, 4), time.outputs().get(1).shape()),
                () -> assertFalse(time.outputs().get(0).requiresGrad()),
                () -> assertEquals(Shape.of(4, 0, 2), batch.outputs().get(0).shape()),
                () -> assertEquals(Shape.of(0, 2), batch.outputs().get(2).shape()),
                () -> assertTrue(batch.constraints().isEmpty()));
    }

    @Test
    void rejectsCardinalityAndDescriptorsInSemanticRoleOrder() {
        List<TensorDescriptor> rnn = descriptors(fixtures(
                DataType.FLOAT32, 2, 2, 3, 4, false), RecurrentScanKind.RNN_TANH, false);
        Operation operation = new Operation(
                RecurrentScanKind.RNN_TANH, RecurrentDirection.FORWARD);

        assertFailure("input count must be 5 or 6", operation, rnn.subList(0, 4), 2);
        assertFailure("output count must be 2", operation, rnn, 3);
        assertFailure("input must have a floating data type", operation,
                replaced(rnn, 0, descriptor(DataType.INT32, Shape.of(2, 2, 3), false)), 2);
        assertFailure("input rank must be 3", operation,
                replaced(rnn, 0, descriptor(DataType.FLOAT32, Shape.of(2, 3), false)), 2);
        assertFailure("input must have a fully static shape", operation,
                replaced(rnn, 0, descriptor(DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("T"),
                                rnn.get(0).shape().dimension(1),
                                rnn.get(0).shape().dimension(2)), false)), 2);
        assertFailure("inputSize must be positive", operation,
                replaced(rnn, 0, descriptor(DataType.FLOAT32, Shape.of(2, 2, 0), false)), 2);
        assertFailure("validLengths data type must be INT64", operation,
                replaced(rnn, 1, descriptor(DataType.INT32, Shape.of(2), false)), 2);
        assertFalse(descriptor(DataType.INT64, Shape.of(2), false).requiresGrad());
        assertFailure("validLengths batch extent mismatch", operation,
                replaced(rnn, 1, descriptor(DataType.INT64, Shape.of(3), false)), 2);
        assertFailure("initialHidden data type must match input", operation,
                replaced(rnn, 2, descriptor(DataType.FLOAT64, Shape.of(2, 4), false)), 2);
        assertFailure("hiddenSize must be positive", operation,
                replaced(rnn, 2, descriptor(DataType.FLOAT32, Shape.of(2, 0), false)), 2);
        assertFailure("inputWeight packedHiddenSize extent mismatch", operation,
                replaced(rnn, 3, descriptor(DataType.FLOAT32, Shape.of(5, 3), false)), 2);
        assertFailure("hiddenWeight hiddenSize extent mismatch", operation,
                replaced(rnn, 4, descriptor(DataType.FLOAT32, Shape.of(4, 5), false)), 2);
    }

    @Test
    void rejectsLstmCellBiasAndPackedExtentFailures() {
        Fixtures fixture = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        List<TensorDescriptor> lstm = descriptors(fixture, RecurrentScanKind.LSTM, true);
        Operation operation = new Operation(RecurrentScanKind.LSTM, RecurrentDirection.REVERSE);

        assertFailure("initialCell hiddenSize extent mismatch", operation,
                replaced(lstm, 3, descriptor(DataType.FLOAT32, Shape.of(2, 5), false)), 3);
        assertFailure("bias packedHiddenSize extent mismatch", operation,
                replaced(lstm, 6, descriptor(DataType.FLOAT32, Shape.of(15), false)), 3);
        List<TensorDescriptor> overflow = new ArrayList<>(lstm);
        overflow.set(2, descriptor(DataType.FLOAT32, Shape.of(2, Long.MAX_VALUE), false));
        overflow.set(3, descriptor(DataType.FLOAT32, Shape.of(2, Long.MAX_VALUE), false));
        assertFailure("packed gate extent overflow", operation, overflow, 3);
    }

    @Test
    void outerValidationReportsExactSecondSlotMismatchContext() {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 2, 3, 4, false);
        List<TensorDescriptor> inputs = descriptors(f, RecurrentScanKind.GRU_RESET_AFTER, false);
        Operation operation = new Operation(
                RecurrentScanKind.GRU_RESET_AFTER, RecurrentDirection.REVERSE);
        var inferred = RecurrentScanInference.infer(operation, inputs, 2);
        CompiledGraphModel graph = singleNodeGraph(
                operation,
                inputs,
                List.of(inferred.outputs().get(0),
                        descriptor(DataType.FLOAT64, Shape.of(2, 4), true)));

        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();

        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains(RecurrentScanKind.class.getName()
                        + ".GRU_RESET_AFTER")),
                () -> assertTrue(message.contains("output[1] ValueId[value=6] expected=")),
                () -> assertTrue(message.contains("stored=TensorDescriptor[dataType=FLOAT64")));
    }

    @Test
    void failsClosedWhenCalledForAnotherFamily() {
        Operation operation = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
        assertFailure("unsupported recurrent kind", operation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), false)), 1);
    }

    private static void assertInference(RecurrentScanResult result) {
        assertInference(producer(result));
    }

    private static void assertInference(LstmRecurrentScanResult result) {
        assertInference(producer(result));
    }

    private static void assertInference(TensorProducer producer) {
        var inferred = infer(producer);
        assertAll(
                () -> assertEquals(producer.outputDescriptors(), inferred.outputs()),
                () -> assertTrue(inferred.constraints().isEmpty()),
                () -> assertTrue(inferred.outputs().stream()
                        .allMatch(output -> output.layout().isEmpty())));
    }

    private static CapturedGraphInference.InferenceResult infer(TensorProducer producer) {
        return RecurrentScanInference.infer(
                producer.operation(),
                producer.inputs().stream().map(Tensor::descriptor).toList(),
                producer.outputCount());
    }

    private static TensorProducer producer(RecurrentScanResult result) {
        return result.outputs().provenance().orElseThrow().producer();
    }

    private static TensorProducer producer(LstmRecurrentScanResult result) {
        return result.outputs().provenance().orElseThrow().producer();
    }

    private static void assertFailure(
            String evidence,
            Operation operation,
            List<TensorDescriptor> inputs,
            int outputCount) {
        String message = assertThrows(IllegalArgumentException.class,
                () -> RecurrentScanInference.infer(operation, inputs, outputCount)).getMessage();
        assertTrue(message.contains(evidence), message);
    }

    private static List<TensorDescriptor> replaced(
            List<TensorDescriptor> source, int index, TensorDescriptor replacement) {
        List<TensorDescriptor> result = new ArrayList<>(source);
        result.set(index, replacement);
        return result;
    }

    private static List<TensorDescriptor> descriptors(
            Fixtures f, RecurrentScanKind kind, boolean bias) {
        List<TensorDescriptor> result = new ArrayList<>();
        result.add(f.input.descriptor());
        result.add(f.lengths.descriptor());
        result.add(f.hidden.descriptor());
        if (kind == RecurrentScanKind.LSTM) {
            result.add(f.cell.descriptor());
            result.add(f.lstmInputWeight.descriptor());
            result.add(f.lstmHiddenWeight.descriptor());
            if (bias) result.add(f.lstmBias.descriptor());
        } else if (kind == RecurrentScanKind.GRU_RESET_AFTER) {
            result.add(f.gruInputWeight.descriptor());
            result.add(f.gruHiddenWeight.descriptor());
            if (bias) result.add(f.gruBias.descriptor());
        } else {
            result.add(f.rnnInputWeight.descriptor());
            result.add(f.rnnHiddenWeight.descriptor());
            if (bias) result.add(f.rnnBias.descriptor());
        }
        return List.copyOf(result);
    }

    private static CompiledGraphModel singleNodeGraph(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputIds = new ArrayList<>();
        List<ValueId> outputIds = new ArrayList<>();
        long nextId = 0;
        for (TensorDescriptor input : inputs) {
            ValueId id = new ValueId(nextId++);
            values.add(new GraphValue(id, input));
            inputIds.add(id);
        }
        for (TensorDescriptor output : outputs) {
            ValueId id = new ValueId(nextId++);
            values.add(new GraphValue(id, output));
            outputIds.add(id);
        }
        NodeId nodeId = new NodeId(0);
        return new CompiledGraphModel(
                values,
                List.of(new CompiledNode(nodeId, operation, inputIds, outputIds)),
                inputIds,
                outputIds,
                Map.of(nodeId, GraphPhase.FORWARD));
    }

    private static Fixtures fixtures(
            DataType type,
            long time,
            long batch,
            long inputSize,
            long hiddenSize,
            boolean requiresGrad) {
        return new Fixtures(
                tensor(type, Shape.of(time, batch, inputSize), requiresGrad),
                tensor(DataType.INT64, Shape.of(batch), false),
                tensor(type, Shape.of(batch, hiddenSize), requiresGrad),
                tensor(type, Shape.of(batch, hiddenSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(hiddenSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(3 * hiddenSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize, inputSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize, hiddenSize), requiresGrad),
                tensor(type, Shape.of(4 * hiddenSize), requiresGrad));
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(descriptor(type, shape, requiresGrad));
    }

    private static TensorDescriptor descriptor(
            DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(type, shape, Optional.empty(), requiresGrad);
    }

    private record Fixtures(
            Tensor input,
            Tensor lengths,
            Tensor hidden,
            Tensor cell,
            Tensor rnnInputWeight,
            Tensor rnnHiddenWeight,
            Tensor rnnBias,
            Tensor gruInputWeight,
            Tensor gruHiddenWeight,
            Tensor gruBias,
            Tensor lstmInputWeight,
            Tensor lstmHiddenWeight,
            Tensor lstmBias) {}
}
