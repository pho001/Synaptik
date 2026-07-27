package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.PublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CompileArtifactsTest {
    @Test
    void exposesTheExactSevenComponentImmutableRecipeAndRetainsExactReferences() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, false);
        ValueId input = new ValueId(1);
        CompiledGraphModel graph = passThroughGraph(input, descriptor);
        PublicationPlan publication = new PublicationPlan(
                graph,
                List.of(new PublicationBinding(new TensorId(7), input)),
                List.of());
        CompileConstantPlan constants =
                new CompileConstantPlan(List.of(input), List.of());
        CompileDiagnostics diagnostics = new CompileDiagnostics(List.of());
        LogicalMemoryPlan memory = LogicalMemoryPlanning.plan(graph, List.of());
        List<io.github.pho001.synaptik.planning.partition.PlannedPartition> partitions =
                new ArrayList<>();

        CompileArtifacts artifacts = new CompileArtifacts(
                CompileMode.FORWARD_ONLY,
                graph,
                partitions,
                memory,
                publication,
                constants,
                diagnostics);
        partitions.clear();

        assertEquals(
                List.of(
                        "mode",
                        "graph",
                        "partitions",
                        "memory",
                        "publication",
                        "constants",
                        "diagnostics"),
                java.util.Arrays.stream(CompileArtifacts.class.getRecordComponents())
                        .map(component -> component.getName())
                        .toList());
        assertSame(graph, artifacts.graph());
        assertSame(memory, artifacts.memory());
        assertSame(publication, artifacts.publication());
        assertSame(constants, artifacts.constants());
        assertSame(diagnostics, artifacts.diagnostics());
        assertTrue(artifacts.partitions().isEmpty());
        assertThrows(
                UnsupportedOperationException.class,
                () -> artifacts.partitions().clear());
    }

    @Test
    void rejectsMismatchedGraphMemoryAndSourceClassifications() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, false);
        ValueId input = new ValueId(1);
        CompiledGraphModel graph = passThroughGraph(input, descriptor);
        CompiledGraphModel equalButDistinctGraph = passThroughGraph(input, descriptor);
        PublicationPlan wrongPublication = new PublicationPlan(
                equalButDistinctGraph,
                List.of(new PublicationBinding(new TensorId(7), input)),
                List.of());
        CompileDiagnostics diagnostics = new CompileDiagnostics(List.of());
        LogicalMemoryPlan memory = LogicalMemoryPlanning.plan(graph, List.of());

        assertEquals(
                "publication graph must be the exact graph reference",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileArtifacts(
                                CompileMode.FORWARD_ONLY,
                                graph,
                                List.of(),
                                memory,
                                wrongPublication,
                                new CompileConstantPlan(List.of(input), List.of()),
                                diagnostics))
                        .getMessage());

        PublicationPlan publication = new PublicationPlan(
                graph,
                List.of(new PublicationBinding(new TensorId(7), input)),
                List.of());
        assertEquals(
                "memory does not match graph and partitions",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileArtifacts(
                                CompileMode.FORWARD_ONLY,
                                graph,
                                List.of(),
                                new LogicalMemoryPlan(List.of()),
                                publication,
                                new CompileConstantPlan(List.of(input), List.of()),
                                diagnostics))
                        .getMessage());

        assertEquals(
                "constants do not classify graph.inputs[0] ValueId[value=1] in graph-input order",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileArtifacts(
                                CompileMode.FORWARD_ONLY,
                                graph,
                                List.of(),
                                memory,
                                publication,
                                new CompileConstantPlan(List.of(), List.of()),
                                diagnostics))
                        .getMessage());
    }

    @Test
    void rejectsConstantTypeMismatchAndGradientEligibleConstant() {
        ValueId input = new ValueId(1);
        CompileDiagnostics diagnostics = new CompileDiagnostics(List.of());

        CompiledGraphModel floatGraph =
                passThroughGraph(input, descriptor(DataType.FLOAT32, false));
        PublicationPlan floatPublication = new PublicationPlan(
                floatGraph,
                List.of(new PublicationBinding(new TensorId(7), input)),
                List.of());
        assertEquals(
                "constantSources[0] data type INT32 does not match graph input descriptor FLOAT32",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileArtifacts(
                                CompileMode.FORWARD_ONLY,
                                floatGraph,
                                List.of(),
                                LogicalMemoryPlanning.plan(floatGraph, List.of()),
                                floatPublication,
                                new CompileConstantPlan(
                                        List.of(),
                                        List.of(new CompileConstantPlan.ConstantSource(
                                                input, ScalarValue.int32(1)))),
                                diagnostics))
                        .getMessage());

        CompiledGraphModel gradientGraph =
                passThroughGraph(input, descriptor(DataType.FLOAT32, true));
        PublicationPlan gradientPublication = new PublicationPlan(
                gradientGraph,
                List.of(new PublicationBinding(new TensorId(7), input)),
                List.of());
        assertEquals(
                "constantSources[0] fixes a gradient-eligible graph input",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileArtifacts(
                                CompileMode.FORWARD_ONLY,
                                gradientGraph,
                                List.of(),
                                LogicalMemoryPlanning.plan(gradientGraph, List.of()),
                                gradientPublication,
                                new CompileConstantPlan(
                                        List.of(),
                                        List.of(new CompileConstantPlan.ConstantSource(
                                                input, ScalarValue.float32(1.0f)))),
                                diagnostics))
                        .getMessage());
    }

    private static CompiledGraphModel passThroughGraph(
            ValueId input,
            TensorDescriptor descriptor) {
        return new CompiledGraphModel(
                List.of(new GraphValue(input, descriptor)),
                List.of(),
                List.of(input),
                List.of(input),
                Map.of());
    }

    private static TensorDescriptor descriptor(
            DataType dataType,
            boolean requiresGrad) {
        return new TensorDescriptor(
                dataType, Shape.of(2), Optional.empty(), requiresGrad);
    }
}
