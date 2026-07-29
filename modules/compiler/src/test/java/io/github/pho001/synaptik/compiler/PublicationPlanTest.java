package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class PublicationPlanTest {
    @Test
    void retainsExactGraphAndBindingsWithSeparateForwardAndGradientRoles() {
        ValueId forwardValue = new ValueId(1);
        ValueId gradientValue = new ValueId(2);
        CompiledGraphModel graph = passThroughGraph(
                List.of(forwardValue, gradientValue),
                List.of(forwardValue, gradientValue));
        ForwardPublicationBinding forward =
                new ForwardPublicationBinding(new TensorId(10), forwardValue);
        GradientPublicationBinding gradient =
                new GradientPublicationBinding(1, 0, new TensorId(20), gradientValue);
        List<ForwardPublicationBinding> forwardSource = new ArrayList<>(List.of(forward));

        PublicationPlan plan =
                new PublicationPlan(graph, forwardSource, List.of(gradient));
        forwardSource.clear();

        assertSame(graph, plan.graph());
        assertSame(forward, plan.forwardBindings().getFirst());
        assertSame(gradient, plan.gradientBindings().getFirst());
        assertThrows(
                UnsupportedOperationException.class,
                () -> plan.forwardBindings().add(forward));
    }

    @Test
    void acceptsSharedGradientValuesAndRejectsRoleDuplicatesAndBoundaryMismatch() {
        ValueId forwardValue = new ValueId(1);
        ValueId gradientValue = new ValueId(2);
        CompiledGraphModel graph = passThroughGraph(
                List.of(forwardValue, gradientValue),
                List.of(forwardValue, gradientValue));
        ForwardPublicationBinding forward =
                new ForwardPublicationBinding(new TensorId(10), forwardValue);
        GradientPublicationBinding firstGradient =
                new GradientPublicationBinding(1, 0, new TensorId(20), gradientValue);
        GradientPublicationBinding secondGradient =
                new GradientPublicationBinding(1, 1, new TensorId(21), gradientValue);

        PublicationPlan plan = new PublicationPlan(
                graph,
                List.of(forward),
                List.of(firstGradient, secondGradient));
        assertEquals(2, plan.gradientBindings().size());

        IllegalArgumentException duplicateTarget = assertThrows(
                IllegalArgumentException.class,
                () -> new PublicationPlan(
                        graph,
                        List.of(forward),
                        List.of(
                                firstGradient,
                                new GradientPublicationBinding(
                                        1, 1, firstGradient.target(), forwardValue))));
        assertEquals(
                "gradientBindings[1] duplicates an earlier target in this derivative order",
                duplicateTarget.getMessage());

        CompiledGraphModel extraBoundary = passThroughGraph(
                List.of(forwardValue, gradientValue, new ValueId(3)),
                List.of(forwardValue, gradientValue, new ValueId(3)));
        assertEquals(
                "graph output boundary does not match publication bindings",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new PublicationPlan(
                                extraBoundary,
                                List.of(forward),
                                List.of(firstGradient)))
                        .getMessage());
    }

    private static CompiledGraphModel passThroughGraph(
            List<ValueId> values,
            List<ValueId> outputs) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
        return new CompiledGraphModel(
                values.stream().map(value -> new GraphValue(value, descriptor)).toList(),
                List.of(),
                values,
                outputs,
                Map.of());
    }
}
