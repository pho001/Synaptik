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
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
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

        CompileArtifacts artifacts = GraphCompiler.compile(CompileMode.FORWARD_ONLY, List.of(result.finalHidden(), result.outputs()), Optional.empty(), CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(recordingProvider(backendId, queries)), List.of(snapshot(backendId)));

        OperationCapabilityQuery query = queries.getFirst();
        CompiledNode node = artifacts.graph().nodes().getFirst();
        assertAll(
                () -> assertEquals(1, queries.size()),
                () -> assertSame(producer.operation(), query.operation()),
                () -> assertEquals(
                        producer.inputs().stream().map(Tensor::descriptor).toList(),
                        query.inputs()),
                () -> assertEquals(
                        producer.outputDescriptors().stream()
                                .map(descriptor -> new TensorDescriptor(
                                        descriptor.dataType(),
                                        descriptor.shape(),
                                        Optional.of(LayoutDescriptor.contiguous(descriptor.shape())),
                                        descriptor.requiresGrad()))
                                .toList(),
                        query.outputs()),
                () -> assertEquals(2, artifacts.publication().forwardBindings().size()),
                () -> assertEquals(List.of(node.outputs().get(1), node.outputs().get(0)),
                        artifacts.graph().outputs()),
                () -> assertEquals(1, artifacts.partitions().size()),
                () -> assertEquals(artifacts.graph().values().size(),
                        artifacts.memory().requirements().size()),
                () -> assertTrue(artifacts.diagnostics().constraints().isEmpty()));
    }

    @Test
    void publishesFirstOrderGradientsForEveryOutputStateAndFloatingRole() {
        for (DataType type : floatingTypes()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                for (RecurrentScanKind kind : RecurrentScanKind.values()) {
                    for (boolean bias : List.of(false, true)) {
                        Fixtures f = fixtures(type, 3, 2, 3, 2, true);
                        RecurrentCase recurrent = recurrentCase(f, kind, direction, bias);
                        List<List<Tensor>> selections = new ArrayList<>();
                        recurrent.outputs().forEach(output -> selections.add(List.of(output)));
                        selections.add(recurrent.outputs());
                        for (List<Tensor> selected : selections) {
                            List<Tensor> seeds = seeds(selected);
                            var request = ScanGradientRulesTest.request(
                                    selected, seeds, recurrent.targets());
                            var expansion = FirstOrderAutograd.expand(
                                    AutogradPreflight.preflight(
                                            CompileMode.FORWARD_AND_BACKWARD,
                                            recurrent.outputs(), request,
                                            CompileTimeConstantGraph.Ingress.empty()).stageOne(),
                                    CompileTimeConstantGraph.Ingress.empty());
                            assertExpandedTargets(recurrent.targets(), expansion);

                            BackendId backendId = new BackendId("recurrent-gradient-query-test");
                            List<OperationCapabilityQuery> queries = new ArrayList<>();
                            CompileArtifacts artifacts = GraphCompiler.compile(
                                    CompileMode.FORWARD_AND_BACKWARD,
                                    recurrent.outputs(), Optional.of(request),
                                    CompileTimeConstantGraph.Ingress.empty(),
                                    GraphOptimizationConfig.standard(),
                                    BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                                    List.of(recordingProvider(backendId, queries)),
                                    List.of(snapshot(backendId)));
                            assertPublications(artifacts, recurrent);
                            CompiledNode fixed = artifacts.graph().nodes().stream()
                                    .filter(node -> node.operation().kind() == kind)
                                    .findFirst().orElseThrow();
                            List<CompiledNode> scans = artifacts.graph().nodes().stream()
                                    .filter(node -> node.scanBody().isPresent()).toList();
                            assertFalse(scans.isEmpty(), "BPTT must use real shared Scan bodies");
                            assertTrue(scans.stream().anyMatch(node ->
                                    ((ScanAttrs) node.operation().attrs()).reverse()
                                            == (direction == RecurrentDirection.FORWARD)),
                                    "the derivative traverses the same valid prefix in the opposite direction");
                            for (CompiledNode scan : scans) {
                                ScanAttrs attrs = (ScanAttrs) scan.operation().attrs();
                                assertTrue(attrs.lengthAware());
                                assertEquals(3, attrs.timeCount());
                                assertEquals(fixed.inputs().get(1), scan.inputs().getLast(),
                                        "every generated Scan must retain the exact runtime lengths binding");
                                assertFalse(scan.scanBody().orElseThrow()
                                        .demandedOutputIndices().isEmpty());
                            }
                            assertTrue(queries.stream().anyMatch(query -> query.scanBody().isPresent()));
                            for (OperationCapabilityQuery query : queries) {
                                if (query.scanBody().isEmpty()) {
                                    continue;
                                }
                                assertEquals(DataType.INT64, query.inputs().getLast().dataType());
                                assertFalse(query.inputs().getLast().requiresGrad());
                                assertTrue(CapturedGraphInference.inferAndValidate(
                                        query.scanBody().orElseThrow().graph()).constraints().isEmpty());
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void nonlinearConsumersAcceptGeneratedCotangentsForEveryFamilyAndJointOutputs() {
        for (DataType type : floatingTypes()) for (RecurrentScanKind kind : RecurrentScanKind.values()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                RecurrentCase recurrent = recurrentCase(fixtures(type, 3, 2, 3, 2, true),
                        kind, direction, true);
                var objectives = new ArrayList<Tensor>();
                Tensor joint = null;
                for (Tensor output : recurrent.outputs()) {
                    Tensor objective = output.mul(output).sum();
                    objectives.add(objective);
                    joint = joint == null ? objective : joint.add(objective);
                }
                objectives.add(joint);
                for (Tensor objective : objectives) {
                    var request = FunctionalGradientTestSupport.request(objective, recurrent.targets());
                    var plan = AutogradPreflight.preflight(CompileMode.FORWARD_AND_BACKWARD,
                            List.of(objective), request, CompileTimeConstantGraph.Ingress.empty());
                    var expansion = FirstOrderAutograd.expand(plan.stageOne(),
                            CompileTimeConstantGraph.Ingress.empty());
                    assertExpandedTargets(recurrent.targets(), expansion);
                    var compiled = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                            List.of(objective), Optional.of(request),
                            CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled());
                    assertEquals(recurrent.targets().size(), compiled.gradientResults().size());
                }
            }
        }
    }

    @Test
    void publishesEmptyTimeAndBatchGradientsWithoutInventingPositiveGeometry() {
        for (DataType type : floatingTypes()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                for (RecurrentScanKind kind : RecurrentScanKind.values()) {
                    for (long[] geometry : List.of(
                            new long[] {0, 2}, new long[] {3, 0}, new long[] {0, 0})) {
                        Fixtures f = fixtures(type, geometry[0], geometry[1], 3, 2, true);
                        RecurrentCase recurrent = recurrentCase(f, kind, direction, true);
                        var request = ScanGradientRulesTest.request(
                                recurrent.outputs(), seeds(recurrent.outputs()), recurrent.targets());
                        BackendId backendId = new BackendId("empty-recurrent-gradient-test");
                        CompileArtifacts artifacts = GraphCompiler.compile(
                                CompileMode.FORWARD_AND_BACKWARD,
                                recurrent.outputs(), Optional.of(request),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled(),
                                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                                List.of(recordingProvider(backendId, new ArrayList<>())),
                                List.of(snapshot(backendId)));
                        assertPublications(artifacts, recurrent);
                        assertEquals(Shape.of(geometry[0], geometry[1], 2),
                                recurrent.outputs().getFirst().descriptor().shape());
                        for (CompiledNode node : artifacts.graph().nodes()) {
                            if (node.scanBody().isPresent()) {
                                assertEquals(geometry[0],
                                        ((ScanAttrs) node.operation().attrs()).timeCount());
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void zeroTimeFinalStatesPublishTheExactCallerSeedsAsInitialStateGradients() {
        for (DataType type : floatingTypes()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                for (RecurrentScanKind kind : RecurrentScanKind.values()) {
                    Fixtures f = fixtures(type, 0, 2, 3, 2, true);
                    RecurrentCase recurrent = recurrentCase(f, kind, direction, false);
                    List<Tensor> states = recurrent.outputs().subList(1, recurrent.outputs().size());
                    List<Tensor> stateSeeds = seeds(states);
                    List<Tensor> targets = kind == RecurrentScanKind.LSTM
                            ? List.of(f.hidden, f.cell) : List.of(f.hidden);
                    var request = ScanGradientRulesTest.request(states, stateSeeds, targets);
                    var expansion = FirstOrderAutograd.expand(AutogradPreflight.preflight(
                            CompileMode.FORWARD_AND_BACKWARD, recurrent.outputs(), request,
                            CompileTimeConstantGraph.Ingress.empty()).stageOne(),
                            CompileTimeConstantGraph.Ingress.empty());
                    assertExpandedTargets(targets, expansion);
                    for (int i = 0; i < targets.size(); i++) {
                        assertSame(stateSeeds.get(i), expansion.targetGradients().get(i).gradient(),
                                "T=0 is the identity, not a dummy reverse Scan or a copied seed");
                    }
                }
            }
        }
    }

    @Test
    void admitsUnrelatedRecurrentBranchesInBackwardAndTrainingModes() {
        for (CompileMode mode : List.of(
                CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            for (RecurrentScanKind kind : RecurrentScanKind.values()) {
                Fixtures f = fixtures(DataType.FLOAT32, 2, 1, 3, 4, true);
                RecurrentCase recurrent = recurrentCase(f, kind, RecurrentDirection.REVERSE, false);
                Tensor target = tensor(DataType.FLOAT32, Shape.of(2), true);
                Tensor objective = target.mul(target).sum();
                List<Tensor> roots = new ArrayList<>(recurrent.outputs());
                roots.add(objective);
                var compilation = GraphCompiler.compile(mode, roots,
                        Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                        CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard());
                assertEquals(roots.size(), compilation.forwardOutputs().size());
                assertTrue(compilation.validatedGraph().graph().nodes().stream()
                        .anyMatch(node -> node.operation().kind() == kind));
            }
        }
    }

    @Test
    void rejectsRecurrentCreateGraphAndSecondOrderBeforeDerivativeAllocation() throws Exception {
        for (RecurrentScanKind kind : RecurrentScanKind.values()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                Fixtures f = fixtures(DataType.FLOAT32, 3, 2, 3, 2, true);
                RecurrentCase recurrent = recurrentCase(f, kind, direction, true);
                Tensor output = recurrent.outputs().getFirst();
                Tensor seed = seeds(List.of(output)).getFirst();
                var first = new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                        List.of(Optional.of(seed)), List.of(f.input), true,
                        FunctionalGradientRequest.DisconnectedPolicy.ZERO);
                var second = new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                        List.of(Optional.of(tensor(
                                DataType.FLOAT32, f.input.descriptor().shape(), false))),
                        List.of(f.input), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
                long before = nextTensorId();
                assertThrows(IllegalArgumentException.class,
                        () -> new FunctionalGradientRequest(List.of(first)));
                assertEquals(before, nextTensorId(),
                        "standalone createGraph is rejected by the public request contract");
                var request = new FunctionalGradientRequest(List.of(first, second));
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class,
                        () -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                                recurrent.outputs(), Optional.of(request),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled()));
                assertFalse(failure.getMessage().isBlank());
                assertEquals(before, nextTensorId(),
                        "unsupported recurrent higher-order requests fail before derivative allocation");
            }
        }
    }

    private static List<DataType> floatingTypes() {
        return List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16);
    }

    private static List<Tensor> seeds(List<Tensor> outputs) {
        return outputs.stream().map(output -> tensor(
                output.descriptor().dataType(), output.descriptor().shape(), false)).toList();
    }

    private static void assertExpandedTargets(
            List<Tensor> targets, FirstOrderAutograd.Expansion expansion) {
        assertEquals(targets.size(), expansion.targetGradients().size());
        for (int i = 0; i < targets.size(); i++) {
            var gradient = expansion.targetGradients().get(i);
            assertSame(targets.get(i), gradient.target());
            assertEquals(targets.get(i).descriptor().shape(), gradient.gradient().descriptor().shape());
            assertEquals(targets.get(i).descriptor().dataType(), gradient.gradient().descriptor().dataType());
            assertTrue(gradient.gradient().descriptor().layout().isEmpty());
        }
    }

    private static void assertPublications(CompileArtifacts artifacts, RecurrentCase recurrent) {
        for (Tensor output : recurrent.outputs()) {
            var binding = artifacts.publication().forwardBindings().stream()
                    .filter(candidate -> candidate.tensorId().equals(output.id()))
                    .findFirst().orElseThrow();
            assertEquals(output.descriptor().shape(),
                    descriptor(artifacts.graph(), binding.valueId()).shape());
            assertEquals(output.descriptor().dataType(),
                    descriptor(artifacts.graph(), binding.valueId()).dataType());
        }
        assertEquals(recurrent.targets().size(), artifacts.publication().gradientBindings().size());
        for (Tensor target : recurrent.targets()) {
            var binding = artifacts.publication().gradientBindings().stream()
                    .filter(candidate -> candidate.target().equals(target.id()))
                    .findFirst().orElseThrow();
            TensorDescriptor gradient = descriptor(artifacts.graph(), binding.valueId());
            assertEquals(target.descriptor().shape(), gradient.shape());
            assertEquals(target.descriptor().dataType(), gradient.dataType());
        }
        assertTrue(artifacts.publication().gradientBindings().stream()
                .noneMatch(binding -> binding.target().equals(recurrent.lengths().id())));
        assertEquals(artifacts.graph().values().size(), artifacts.memory().requirements().size());
        assertTrue(artifacts.diagnostics().constraints().isEmpty());
    }

    private static RecurrentCase recurrentCase(
            Fixtures f, RecurrentScanKind kind, RecurrentDirection direction, boolean bias) {
        if (kind == RecurrentScanKind.LSTM) {
            LstmRecurrentScanResult result = bias
                    ? RecurrentScan.lstm(f.input, f.lengths, f.hidden, f.cell,
                            f.lstmInputWeight, f.lstmHiddenWeight, f.lstmBias, direction)
                    : RecurrentScan.lstm(f.input, f.lengths, f.hidden, f.cell,
                            f.lstmInputWeight, f.lstmHiddenWeight, direction);
            List<Tensor> targets = new ArrayList<>(List.of(
                    f.input, f.hidden, f.cell, f.lstmInputWeight, f.lstmHiddenWeight));
            if (bias) targets.add(f.lstmBias);
            return new RecurrentCase(
                    List.of(result.outputs(), result.finalHidden(), result.finalCell()),
                    List.copyOf(targets), f.lengths);
        }
        Tensor inputWeight = kind == RecurrentScanKind.RNN_TANH
                ? f.rnnInputWeight : f.gruInputWeight;
        Tensor hiddenWeight = kind == RecurrentScanKind.RNN_TANH
                ? f.rnnHiddenWeight : f.gruHiddenWeight;
        Tensor packedBias = kind == RecurrentScanKind.RNN_TANH ? f.rnnBias : f.gruBias;
        RecurrentScanResult result;
        if (kind == RecurrentScanKind.RNN_TANH) {
            result = bias
                    ? RecurrentScan.rnn(f.input, f.lengths, f.hidden,
                            inputWeight, hiddenWeight, packedBias, direction)
                    : RecurrentScan.rnn(f.input, f.lengths, f.hidden,
                            inputWeight, hiddenWeight, direction);
        } else {
            result = bias
                    ? RecurrentScan.gru(f.input, f.lengths, f.hidden,
                            inputWeight, hiddenWeight, packedBias, direction)
                    : RecurrentScan.gru(f.input, f.lengths, f.hidden,
                            inputWeight, hiddenWeight, direction);
        }
        List<Tensor> targets = new ArrayList<>(List.of(
                f.input, f.hidden, inputWeight, hiddenWeight));
        if (bias) targets.add(packedBias);
        return new RecurrentCase(
                List.of(result.outputs(), result.finalHidden()), List.copyOf(targets), f.lengths);
    }

    private record RecurrentCase(List<Tensor> outputs, List<Tensor> targets, Tensor lengths) {}

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
