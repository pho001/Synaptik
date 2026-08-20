package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.LstmRecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.RecurrentScanResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

final class RecurrentScanCompilerTest {
    @Test
    void capturesOneFlatNodeForCanonicalSiblingWrappersWithExactOrder() {
        Fixtures f = fixtures(DataType.FLOAT32, 7, 3, 5, 4, true);
        LstmRecurrentScanResult result = RecurrentScan.lstm(
                f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                f.lstmHiddenWeight, f.lstmBias, RecurrentDirection.REVERSE);
        TensorProducer producer = producer(result);

        CompiledGraphModel graph = GraphCapture.capture(List.of(
                result.finalCell(), result.outputs(), result.finalHidden()));
        CompiledNode node = graph.nodes().getFirst();

        assertAll(
                () -> assertEquals(1, graph.nodes().size()),
                () -> assertSame(producer.operation(), node.operation()),
                () -> assertSame(RecurrentScanKind.LSTM, node.operation().kind()),
                () -> assertSame(RecurrentDirection.REVERSE, node.operation().attrs()),
                () -> assertEquals(7, node.inputs().size()),
                () -> assertEquals(3, node.outputs().size()),
                () -> assertEquals(List.of(
                        node.outputs().get(2), node.outputs().get(0), node.outputs().get(1)),
                        graph.outputs()),
                () -> assertEquals(
                        producer.inputs().stream().map(Tensor::descriptor).toList(),
                        node.inputs().stream().map(id -> descriptor(graph, id)).toList()),
                () -> assertEquals(
                        producer.outputDescriptors(),
                        node.outputs().stream().map(id -> descriptor(graph, id)).toList()));
    }

    @Test
    void keepsStructurallyEqualIdentityDistinctOccurrencesThroughCse() {
        Fixtures f = fixtures(DataType.FLOAT32, 4, 2, 3, 5, false);
        RecurrentScanResult first = RecurrentScan.gru(
                f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                RecurrentDirection.FORWARD);
        RecurrentScanResult second = RecurrentScan.gru(
                f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                RecurrentDirection.FORWARD);
        assertNotSame(producer(first), producer(second));

        GraphCompilation compilation = compileForward(
                List.of(first.outputs().neg(), second.outputs().neg()),
                GraphOptimizationConfig.standard());
        List<CompiledNode> recurrent = compilation.validatedGraph().graph().nodes().stream()
                .filter(node -> node.operation().kind() instanceof RecurrentScanKind)
                .toList();

        assertAll(
                () -> assertEquals(2, recurrent.size()),
                () -> assertNotEquals(recurrent.get(0).id(), recurrent.get(1).id()),
                () -> assertEquals(recurrent.get(0).operation(), recurrent.get(1).operation()),
                () -> assertEquals(recurrent.get(0).inputs(), recurrent.get(1).inputs()),
                () -> assertEquals(2, compilation.forwardOutputs().size()));
    }

    @Test
    void preservesExistingCseForUnrelatedOperations() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor first = input.neg();
        Tensor second = input.neg();

        CompiledGraphModel graph = compileForward(
                List.of(first.abs(), second.abs()), GraphOptimizationConfig.standard())
                .validatedGraph().graph();

        assertEquals(1, graph.nodes().stream()
                .filter(node -> node.operation().kind() == UnaryElementwiseKind.NEG)
                .count());
    }

    @Test
    void dceRetainsEverySiblingSlotWhenAnySlotIsLiveAndRemovesWholeDeadNode() {
        Fixtures f = fixtures(DataType.FLOAT64, 3, 2, 4, 5, false);
        LstmRecurrentScanResult result = RecurrentScan.lstm(
                f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                f.lstmHiddenWeight, RecurrentDirection.FORWARD);

        CompiledGraphModel live = compileForward(
                List.of(result.finalHidden()), GraphOptimizationConfig.standard())
                .validatedGraph().graph();
        CompiledNode retained = live.nodes().getFirst();
        assertAll(
                () -> assertEquals(1, live.nodes().size()),
                () -> assertEquals(3, retained.outputs().size()),
                () -> assertEquals(retained.outputs().get(1), live.outputs().getFirst()),
                () -> assertEquals(List.of(
                                Shape.of(3, 2, 5), Shape.of(2, 5), Shape.of(2, 5)),
                        retained.outputs().stream()
                                .map(id -> descriptor(live, id).shape()).toList()));

        Tensor separateRoot = tensor(DataType.FLOAT64, Shape.of(1), false);
        CompiledGraphModel captured = GraphCapture.capture(List.of(result.outputs(), separateRoot));
        CompiledGraphModel withDeadOccurrence = new CompiledGraphModel(
                captured.values(),
                captured.nodes(),
                captured.inputs(),
                List.of(captured.outputs().get(1)),
                captured.nodePhases());
        CompiledGraphModel removed = ForwardDeadCodeElimination.eliminate(withDeadOccurrence);
        assertAll(
                () -> assertTrue(removed.nodes().isEmpty()),
                () -> assertEquals(1, removed.outputs().size()),
                () -> assertEquals(1, removed.values().stream()
                        .filter(value -> value.id().equals(removed.outputs().getFirst()))
                        .count()));
    }

    @Test
    void compilesEveryKindDirectionAndBiasShapeIncludingEmptyAxes() {
        for (RecurrentDirection direction : RecurrentDirection.values()) {
            Fixtures f = fixtures(DataType.BFLOAT16, 0, 0, 3, 2, false);
            List<Tensor> roots = List.of(
                    RecurrentScan.rnn(
                            f.input, f.lengths, f.hidden, f.rnnInputWeight,
                            f.rnnHiddenWeight, direction).outputs(),
                    RecurrentScan.rnn(
                            f.input, f.lengths, f.hidden, f.rnnInputWeight,
                            f.rnnHiddenWeight, f.rnnBias, direction).finalHidden(),
                    RecurrentScan.gru(
                            f.input, f.lengths, f.hidden, f.gruInputWeight,
                            f.gruHiddenWeight, direction).outputs(),
                    RecurrentScan.gru(
                            f.input, f.lengths, f.hidden, f.gruInputWeight,
                            f.gruHiddenWeight, f.gruBias, direction).finalHidden(),
                    RecurrentScan.lstm(
                            f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight, direction).outputs(),
                    RecurrentScan.lstm(
                            f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                            f.lstmHiddenWeight, f.lstmBias, direction).finalCell());

            CompiledGraphModel graph = compileForward(
                    roots, GraphOptimizationConfig.standard()).validatedGraph().graph();
            assertAll(
                    () -> assertEquals(6, graph.nodes().size()),
                    () -> assertEquals(6, graph.outputs().size()),
                    () -> assertTrue(graph.nodes().stream()
                            .allMatch(node -> node.operation().attrs() == direction)),
                    () -> assertTrue(graph.nodes().stream()
                            .allMatch(node -> node.operation().kind()
                                    instanceof RecurrentScanKind)));
        }
    }

    @Test
    void passesOrdinaryQueryPublicationPartitionAndMemoryHandoffWithoutCapabilityClaim() {
        Fixtures f = fixtures(DataType.FLOAT32, 3, 2, 4, 5, true);
        RecurrentScanResult result = RecurrentScan.rnn(
                f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                f.rnnBias, RecurrentDirection.REVERSE);
        TensorProducer producer = producer(result);
        BackendId backendId = new BackendId("recording-test-backend");
        List<OperationCapabilityQuery> queries = new ArrayList<>();

        CompileArtifacts artifacts = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(result.finalHidden(), result.outputs()),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(recordingProvider(backendId, queries)),
                List.of(snapshot(backendId)));

        OperationCapabilityQuery query = queries.getFirst();
        CompiledNode node = artifacts.graph().nodes().getFirst();
        assertAll(
                () -> assertEquals(1, queries.size()),
                () -> assertSame(producer.operation(), query.operation()),
                () -> assertEquals(
                        producer.inputs().stream().map(Tensor::descriptor).toList(),
                        query.inputs()),
                () -> assertEquals(producer.outputDescriptors(), query.outputs()),
                () -> assertEquals(2, artifacts.publication().forwardBindings().size()),
                () -> assertEquals(List.of(node.outputs().get(1), node.outputs().get(0)),
                        artifacts.graph().outputs()),
                () -> assertEquals(1, artifacts.partitions().size()),
                () -> assertEquals(artifacts.graph().values().size(),
                        artifacts.memory().requirements().size()),
                () -> assertTrue(artifacts.diagnostics().constraints().isEmpty()));
    }

    @Test
    void rejectsEveryBackwardModeFromCompleteInventoryBeforeAnyTensorAllocation()
            throws Exception {
        for (CompileMode mode : List.of(
                CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            Fixtures f = fixtures(DataType.FLOAT32, 2, 1, 3, 4, true);
            RecurrentScanResult recurrent = RecurrentScan.rnn(
                    f.input, f.lengths, f.hidden, f.rnnInputWeight, f.rnnHiddenWeight,
                    RecurrentDirection.FORWARD);
            Tensor target = tensor(DataType.FLOAT32, Shape.of(2), true);
            Tensor objective = target.mul(target).sum();
            long before = nextTensorId();

            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> GraphCompiler.compile(
                            mode,
                            List.of(recurrent.finalHidden(), objective),
                            Optional.of(FunctionalGradientTestSupport.request(
                                    objective, List.of(target))),
                            CompileTimeConstantGraph.Ingress.empty(),
                            GraphOptimizationConfig.standard()));

            assertAll(
                    () -> assertTrue(failure.getMessage().startsWith(
                            "producerPostorder[0] "), failure.getMessage()),
                    () -> assertTrue(failure.getMessage().contains(
                            RecurrentScanKind.class.getName() + ".RNN_TANH")),
                    () -> assertTrue(failure.getMessage().contains(
                            "forward-only until backpropagation through time is implemented")),
                    () -> assertEquals(before, nextTensorId()));
        }
    }

    @Test
    void rejectsFirstRecurrentOccurrenceInDeterministicProducerPostorder() throws Exception {
        Fixtures f = fixtures(DataType.FLOAT32, 2, 1, 3, 4, true);
        RecurrentScanResult first = RecurrentScan.gru(
                f.input, f.lengths, f.hidden, f.gruInputWeight, f.gruHiddenWeight,
                RecurrentDirection.REVERSE);
        LstmRecurrentScanResult second = RecurrentScan.lstm(
                f.input, f.lengths, f.hidden, f.cell, f.lstmInputWeight,
                f.lstmHiddenWeight, RecurrentDirection.FORWARD);
        Tensor target = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor objective = target.sum();
        long before = nextTensorId();

        String message = assertThrows(IllegalArgumentException.class,
                () -> GraphCompiler.compile(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(first.outputs(), second.outputs(), objective),
                        Optional.of(FunctionalGradientTestSupport.request(
                                objective, List.of(target))),
                        CompileTimeConstantGraph.Ingress.empty(),
                        GraphOptimizationConfig.disabled())).getMessage();

        assertAll(
                () -> assertTrue(message.contains("producerPostorder[0]")),
                () -> assertTrue(message.contains(".GRU_RESET_AFTER")),
                () -> assertFalse(message.contains(".LSTM")),
                () -> assertEquals(before, nextTensorId()));
    }

    private static GraphCompilation compileForward(
            List<Tensor> outputs, GraphOptimizationConfig optimization) {
        return GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                outputs,
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                optimization);
    }

    private static BackendCapabilityProvider recordingProvider(
            BackendId backendId, List<OperationCapabilityQuery> queries) {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                queries.add(query);
                return true;
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot(BackendId backendId) {
        BackendDeviceId deviceId = new BackendDeviceId(backendId, "0");
        return new BackendAvailabilitySnapshot(
                backendId, Map.of(deviceId, DeviceClass.CPU));
    }

    private static TensorDescriptor descriptor(
            CompiledGraphModel graph, io.github.pho001.synaptik.model.graph.ValueId id) {
        return graph.values().stream()
                .filter(value -> value.id().equals(id))
                .findFirst()
                .orElseThrow()
                .descriptor();
    }

    private static TensorProducer producer(RecurrentScanResult result) {
        return result.outputs().provenance().orElseThrow().producer();
    }

    private static TensorProducer producer(LstmRecurrentScanResult result) {
        return result.outputs().provenance().orElseThrow().producer();
    }

    private static long nextTensorId() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
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
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
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
