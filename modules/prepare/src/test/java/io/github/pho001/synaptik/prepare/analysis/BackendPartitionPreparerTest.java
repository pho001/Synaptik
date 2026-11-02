package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class BackendPartitionPreparerTest {
    @Test
    void completeSymbolicLengthProjectionReachesExistingAnalyzerWithoutReadingValues() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                var artifacts = symbolicLengthArtifacts(type, time, reverse);
                var calls = new java.util.concurrent.atomic.AtomicInteger();
                BackendPartitionPreparer<FakeInputs, FakePlan> analyzer = context -> {
                    calls.incrementAndGet();
                    throw new AssertionError("validated projection reached analyzer");
                };
                var preparations = artifacts.partitions().stream().map(partition -> new PartitionPreparation<>(
                        new FakeInputs("guard", 4), analyzer, new BackendPartitionFinalizer<FakePlan>() {
                            @Override public BackendId backendId() { return partition.owner(); }
                            @Override public BackendPartitionFinalizationResult finalizePartition(BackendPartitionFinalization<FakePlan> finalization) {
                                throw new AssertionError("no finalizer may run");
                            }
                        })).toList();
                var failure = assertThrows(AssertionError.class, () -> GraphPreparation.prepare(artifacts,
                        preparations, context -> { throw new AssertionError("no assembly may run"); }));
                assertEquals("validated projection reached analyzer",failure.getMessage());
                assertEquals(1, calls.get());
            }
    }

    /**
     * Bypasses Compiler inference with raw descriptors for an independent Prepare guard witness.
     * @param type homogeneous FLOAT32 or FLOAT64 body type
     * @param time non-negative fixed T
     * @param reverse whether traversal decreases original time
     * @return non-null raw artifacts containing an ordinary first partition and symbolic-length last
     *     partition, with deterministic derivative-metadata ordering and no analyzer invocation
     * @throws NullPointerException if type is null
     * @throws IllegalArgumentException if time is negative
     * @throws AssertionError if the test cannot access an existing package-owned metadata constructor
     */
    static CompileArtifacts symbolicLengthArtifacts(DataType type, long time, boolean reverse) {
        var batch = new DynamicDimension("Customers"); var width = new StaticDimension(2);
        var rowShape = Shape.ofDimensions(batch, width);
        var sequenceShape = Shape.ofDimensions(new StaticDimension(time), batch, width);
        var row = new TensorDescriptor(type, rowShape, Optional.empty(), false);
        var sequence = new TensorDescriptor(type, sequenceShape, Optional.empty(), false);
        var lengths = new TensorDescriptor(DataType.INT64, Shape.ofDimensions(batch), Optional.empty(), false);
        var scalar = new TensorDescriptor(type, Shape.scalar(), Optional.empty(), false);
        var local = new CompiledGraphModel(List.of(new GraphValue(new ValueId(0), row), new GraphValue(new ValueId(1), row)),
                List.of(), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(0)), Map.of());
        var body = new CompiledScanBody(local, List.of(new ValueId(0)), List.of(), List.of(new ValueId(0)), List.of(0));
        var early = new CompiledNode(new NodeId(0), new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var scan = new CompiledNode(new NodeId(1), new Operation(ScanKind.SCAN,
                new ScanAttrs(time, 1, 1, 0, 0, 1, reverse, true, 1, List.of(OptionalInt.of(0)))),
                List.of(new ValueId(2), new ValueId(3), new ValueId(4)), List.of(new ValueId(5), new ValueId(6)), Optional.of(body));
        var values = List.of(new GraphValue(new ValueId(0), scalar), new GraphValue(new ValueId(1), scalar),
                new GraphValue(new ValueId(2), row), new GraphValue(new ValueId(3), sequence), new GraphValue(new ValueId(4), lengths),
                new GraphValue(new ValueId(5), row), new GraphValue(new ValueId(6), sequence));
        var graph = new CompiledGraphModel(values, List.of(early, scan),
                List.of(new ValueId(0), new ValueId(2), new ValueId(3), new ValueId(4)), List.of(new ValueId(1), new ValueId(5)),
                Map.of(early.id(), GraphPhase.FORWARD, scan.id(), GraphPhase.FORWARD));
        var partitions = List.of(new PlannedPartition(new BackendId("first"), List.of(early.id())),
                new PlannedPartition(new BackendId("late-length"), List.of(scan.id())));
        var constants = construct(CompileConstantPlan.class, new Class<?>[] {List.class, List.class},
                graph.inputs().stream().map(id -> new CompileConstantPlan.BindableInput(new TensorId(100 + id.value()), id)).toList(), List.of());
        var publication = construct(PublicationPlan.class, new Class<?>[] {CompiledGraphModel.class, List.class, List.class}, graph,
                graph.outputs().stream().map(id -> new ForwardPublicationBinding(new TensorId(200 + id.value()), id)).toList(), List.of());
        var orders = new java.util.LinkedHashMap<NodeId, Integer>();
        orders.put(early.id(), 0); orders.put(scan.id(), 0);
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions, LogicalMemoryPlanning.plan(graph, partitions), publication,
                constants, construct(CompileDiagnostics.class, new Class<?>[] {List.class}, List.of()),
                new DerivativeGraphMetadata(graph, orders));
    }

    /**
     * Constructs existing package-owned compile metadata for raw projection tests only.
     * @param <T> existing metadata type
     * @param type class whose package-owned constructor is exercised
     * @param parameters exact constructor parameter classes
     * @param arguments positional constructor arguments
     * @return fresh metadata instance of the requested type
     * @throws AssertionError if the test's reflective constructor request fails
     */
    private static <T> T construct(Class<T> type, Class<?>[] parameters, Object... arguments) {
        try {
            var constructor = type.getDeclaredConstructor(parameters); constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    @Test
    void fakeAnalyzerIsDeterministicAndRetainsTheExactContextPartition() {
        PrepareContext<FakeInputs> context = context();
        FakePreparer preparer = new FakePreparer();

        BackendPartitionAnalysis<FakePlan> first = preparer.analyze(context);
        BackendPartitionAnalysis<FakePlan> second = preparer.analyze(context);

        assertAll(
                () -> assertSame(context.partition(), first.partition()),
                () -> assertSame(context.partition(), second.partition()),
                () -> assertEquals(first, second),
                () -> assertNotSame(first, second),
                () -> assertEquals(new FakePlan("vector", 4), first.plan()),
                () -> assertEquals(
                        List.of(
                                new PreparationResourceRequirement.Buffer(
                                        context.values().getFirst().id(), 24, 4),
                                new PreparationResourceRequirement.Workspace(0, 64, 16)),
                        first.requirements()),
                () -> assertThrows(NullPointerException.class, () -> preparer.analyze(null)));
    }

    private static PrepareContext<FakeInputs> context() {
        ValueId input = new ValueId(0);
        ValueId output = new ValueId(1);
        CompiledNode node = new CompiledNode(
                new NodeId(3),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                List.of(input),
                List.of(output));
        PlannedPartition partition =
                new PlannedPartition(new BackendId("cpu"), List.of(node.id()));
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        GraphValue inputValue = new GraphValue(input, descriptor);
        GraphValue outputValue = new GraphValue(output, descriptor);
        return new PrepareContext<>(partition, List.of(node), List.of(inputValue, outputValue), List.of(
                new LogicalMemoryRequirement(
                        input,
                        descriptor,
                        Optional.empty(),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        output,
                        descriptor,
                        Optional.of(partition),
                        List.of(),
                        true)), Map.of(), new FakeInputs("cpu-v1", 4));
    }

    private record FakeInputs(String target, long alignment)
            implements BackendAnalysisInputs {}

    private record FakePlan(String route, long alignment)
            implements BackendPreparationPlan {}

    private static final class FakePreparer
            implements BackendPartitionPreparer<FakeInputs, FakePlan> {
        @Override
        public BackendPartitionAnalysis<FakePlan> analyze(PrepareContext<FakeInputs> context) {
            Objects.requireNonNull(context, "context");
            FakePlan plan = new FakePlan("vector", context.backendInputs().alignment());
            return new BackendPartitionAnalysis<>(
                    context.partition(),
                    plan,
                    List.of(
                            new PreparationResourceRequirement.Buffer(
                                    context.values().getFirst().id(),
                                    24,
                                    context.backendInputs().alignment()),
                            new PreparationResourceRequirement.Workspace(0, 64, 16)));
        }
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                new OperationSignature(
                        NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
