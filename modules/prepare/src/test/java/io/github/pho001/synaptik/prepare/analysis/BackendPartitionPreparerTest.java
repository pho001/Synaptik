package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class BackendPartitionPreparerTest {
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
        return new PrepareContext<>(
                partition,
                List.of(node),
                List.of(inputValue, outputValue),
                List.of(
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
                                true)),
                Map.of(),
                new FakeInputs("cpu-v1", 4));
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
