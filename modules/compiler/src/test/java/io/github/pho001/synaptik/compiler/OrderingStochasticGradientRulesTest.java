package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TopKResult;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class OrderingStochasticGradientRulesTest {
    @Test
    void sortCreatesExactlyOneMatchingArgsortAndTopKUsesCanonicalIndices() {
        Tensor input = tensor();
        Tensor sorted = input.sort(0, true);
        Tensor sortGradient = gradient(sorted.sum(), input);
        assertEquals(AxisScatterKind.SCATTER_ELEMENTS,
                sortGradient.provenance().orElseThrow().operation().kind());
        Tensor argsort = sortGradient.provenance().orElseThrow().inputs().get(1);
        assertEquals(OrderingKind.ARGSORT,
                argsort.provenance().orElseThrow().operation().kind());
        assertEquals(
                new SortAttrs(0, true),
                argsort.provenance().orElseThrow().operation().attrs());
        assertSame(input, argsort.provenance().orElseThrow().inputs().getFirst());

        TopKResult topK = input.topK(2, 0, true, false);
        Tensor topKGradient = gradient(topK.values().sum(), input);
        Tensor canonicalIndices =
                topK.values().provenance().orElseThrow().producer().output(1);
        assertSame(canonicalIndices,
                topKGradient.provenance().orElseThrow().inputs().get(1));
        assertSame(topK.indices(), canonicalIndices);
        assertCompiles(sorted.sum(), input);
        assertCompiles(topK.values().sum(), input);
    }

    @Test
    void dropoutReusesCanonicalMaskAndDoesNotConstructAnotherDropout() {
        Tensor input = tensor();
        DropoutResult dropout =
                input.dropout(0.25d, GraphRngState.initial(7L, 11L));
        Tensor gradient = gradient(dropout.output().sum(), input);
        assertEquals(WhereSelectionKind.WHERE,
                gradient.provenance().orElseThrow().operation().kind());
        Tensor canonicalMask =
                dropout.output().provenance().orElseThrow().producer().output(1);
        assertSame(canonicalMask,
                gradient.provenance().orElseThrow().inputs().getFirst());
        assertEquals(
                io.github.pho001.synaptik.model.operation.elementwise.scalar
                        .ScalarElementwiseKind.DIV,
                gradient.provenance().orElseThrow().inputs().get(1)
                        .provenance().orElseThrow().operation().kind());
        assertEquals(
                DropoutKind.DROPOUT,
                dropout.output().provenance().orElseThrow().operation().kind());
        assertCompiles(dropout.output().sum(), input);
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        var plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
        return FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
    }

    private static void assertCompiles(Tensor objective, Tensor target) {
        GraphCompilation compilation = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), compilation.gradientResults().getFirst().target());
    }

    private static Tensor tensor() {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(3), Optional.empty(), true));
    }
}
