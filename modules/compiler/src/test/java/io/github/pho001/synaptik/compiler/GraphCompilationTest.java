package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GraphCompilationTest {
    @Test
    void derivativeMetadataRetainsTheExactGraphAndSnapshotsEncounterOrder() {
        Tensor output = scalar().neg();
        CompiledGraphModel graph = GraphCapture.capture(List.of(output));
        LinkedHashMap<NodeId, Integer> orders = new LinkedHashMap<>();
        graph.nodes().forEach(node -> orders.put(node.id(), 0));

        DerivativeGraphMetadata metadata = new DerivativeGraphMetadata(graph, orders);
        orders.clear();

        assertSame(graph, metadata.graph());
        assertEquals(
                graph.nodes().stream().map(node -> node.id()).toList(),
                metadata.derivativeOrderByNode().keySet().stream().toList());
        assertTrue(metadata.derivativeOrderByNode().values().stream()
                .allMatch(order -> order == 0));
        assertThrows(
                UnsupportedOperationException.class,
                () -> metadata.derivativeOrderByNode().clear());
    }

    @Test
    void derivativeMetadataRejectsMissingOutOfOrderAndPhaseInconsistentEntries() {
        Tensor output = scalar().neg().abs();
        CompiledGraphModel graph = GraphCapture.capture(List.of(output));
        LinkedHashMap<NodeId, Integer> reversed = new LinkedHashMap<>();
        for (int index = graph.nodes().size() - 1; index >= 0; index--) {
            reversed.put(graph.nodes().get(index).id(), 0);
        }

        assertThrows(
                IllegalArgumentException.class,
                () -> new DerivativeGraphMetadata(graph, new LinkedHashMap<>()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DerivativeGraphMetadata(graph, reversed));
        LinkedHashMap<NodeId, Integer> wrongPhase = new LinkedHashMap<>();
        graph.nodes().forEach(node -> wrongPhase.put(node.id(), 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DerivativeGraphMetadata(graph, wrongPhase));
    }

    @Test
    void graphCompilationCarriesOneExactRemappedSidecarForBothDerivativeOrders() {
        Tensor target = scalar();
        Tensor objective = target.mul(target);
        FunctionalGradientRequest.Stage first = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
                List.of(Optional.empty()),
                List.of(target),
                true,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        FunctionalGradientRequest.Stage second = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.empty()),
                List.of(target),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);

        GraphCompilation compilation = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.standard());

        assertSame(compilation.validatedGraph().graph(), compilation.derivatives().graph());
        assertSame(compilation.validatedGraph().derivatives(), compilation.derivatives());
        assertEquals(
                List.of(1, 2),
                compilation.gradientResults().stream()
                        .map(GradientPublicationBinding::derivativeOrder)
                        .toList());
        compilation.derivatives().derivativeOrderByNode().forEach((nodeId, order) ->
                assertEquals(
                        order == 0 ? GraphPhase.FORWARD : GraphPhase.BACKWARD,
                        compilation.validatedGraph().graph().nodePhases().get(nodeId)));
    }

    private static Tensor scalar() {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), true));
    }
}
