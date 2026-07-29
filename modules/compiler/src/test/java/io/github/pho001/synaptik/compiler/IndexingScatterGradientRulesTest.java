package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class IndexingScatterGradientRulesTest {
    @Test
    void gatherFamiliesRouteThroughMatchingAdditiveScatter() {
        Tensor data = floating(Shape.of(3));
        Tensor axisIndices = indices(Shape.of(2));
        Tensor gatherGradient = gradient(data.gather(axisIndices, 0).sum(), data);
        assertEquals(AxisScatterKind.SCATTER_ADD,
                gatherGradient.provenance().orElseThrow().operation().kind());

        Tensor elementsGradient = gradient(data.gatherElements(axisIndices, 0).sum(), data);
        assertEquals(AxisScatterKind.SCATTER_ELEMENTS,
                elementsGradient.provenance().orElseThrow().operation().kind());

        Tensor tupleIndices = indices(Shape.of(2, 1));
        Tensor ndGradient = gradient(data.gatherNd(tupleIndices).sum(), data);
        assertEquals(ScatterNdKind.SCATTER_ND,
                ndGradient.provenance().orElseThrow().operation().kind());
    }

    @Test
    void everyScatterReductionSupportsBothFloatingRolesForElementsAndNd() {
        Tensor data = floating(Shape.of(3));
        Tensor axisIndices = indices(Shape.of(2));
        Tensor axisUpdates = floating(Shape.of(2));
        Tensor tupleIndices = indices(Shape.of(2, 1));
        Tensor ndUpdates = floating(Shape.of(2));

        for (ScatterReduction reduction : ScatterReduction.values()) {
            Tensor elements =
                    data.scatterElements(axisIndices, axisUpdates, 0, reduction);
            Tensor nd = data.scatterNd(tupleIndices, ndUpdates, reduction);
            Tensor elementsBase = gradient(elements.sum(), data);
            Tensor elementsUpdates = gradient(elements.sum(), axisUpdates);
            Tensor ndBase = gradient(nd.sum(), data);
            Tensor ndUpdateGradient = gradient(nd.sum(), ndUpdates);

            assertEquals(data.descriptor().shape(), elementsBase.descriptor().shape());
            assertEquals(axisUpdates.descriptor().shape(), elementsUpdates.descriptor().shape());
            assertEquals(data.descriptor().shape(), ndBase.descriptor().shape());
            assertEquals(ndUpdates.descriptor().shape(), ndUpdateGradient.descriptor().shape());
            if (reduction == ScatterReduction.MUL
                    || reduction == ScatterReduction.MIN
                    || reduction == ScatterReduction.MAX) {
                assertTrue(contains(elementsBase, AxisScatterKind.SCATTER_ELEMENTS));
                assertTrue(contains(ndUpdateGradient, ScatterNdKind.SCATTER_ND)
                        || contains(ndUpdateGradient, WhereSelectionKind.WHERE));
            }
            assertCompiles(elements.sum(), axisUpdates);
            assertCompiles(nd.sum(), data);
        }
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        var plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
        return FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
    }

    private static void assertCompiles(Tensor objective, Tensor target) {
        GraphCompilation compilation = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(FunctionalGradientTestSupport.request(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), compilation.gradientResults().getFirst().target());
    }

    private static boolean contains(Tensor root, Object kind) {
        Set<Tensor> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor current = pending.removeLast();
            if (!seen.add(current)) continue;
            var provenance = current.provenance().orElse(null);
            if (provenance == null) continue;
            if (provenance.operation().kind() == kind) return true;
            pending.addAll(provenance.inputs());
        }
        return false;
    }

    private static Tensor floating(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }

    private static Tensor indices(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.INT64, shape, Optional.empty(), false));
    }
}
