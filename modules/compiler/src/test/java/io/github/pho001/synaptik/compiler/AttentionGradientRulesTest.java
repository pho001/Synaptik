package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayDeque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class AttentionGradientRulesTest {
    @Test
    void bothOutputSlotsReuseTheCanonicalWeightsAndRouteExactRoles() {
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), true);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(2, 5, 6), true);
        ScaledDotProductAttentionResult attention =
                query.scaledDotProductAttentionWithWeights(key, value);
        Tensor canonicalWeights =
                attention.output().provenance().orElseThrow().producer().output(1);
        assertSame(attention.weights(), canonicalWeights);

        Tensor outputObjective = attention.output().sum();
        Tensor queryGradient = gradient(outputObjective, query);
        Tensor keyGradient = gradient(outputObjective, key);
        Tensor valueGradient = gradient(outputObjective, value);
        assertTrue(reaches(queryGradient, canonicalWeights));
        assertTrue(reaches(keyGradient, canonicalWeights));
        assertTrue(reaches(valueGradient, canonicalWeights));

        Tensor weightsObjective = attention.weights().sum();
        assertTrue(reaches(gradient(weightsObjective, query), canonicalWeights));
        assertTrue(reaches(gradient(weightsObjective, key), canonicalWeights));
        assertCompiles(outputObjective, query);
        assertCompiles(outputObjective, key);
        assertCompiles(outputObjective, value);
        assertCompiles(weightsObjective, query);
    }

    @Test
    void oneOutputAttentionFailsClosedBeforeFormulaConstruction() {
        Tensor query = tensor(DataType.FLOAT32, Shape.of(3, 4), true);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(5, 4), true);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(5, 6), true);
        Tensor output = query.scaledDotProductAttention(key, value);
        Tensor objective = output.sum();

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> preflight(objective, query));
        assertTrue(failure.getMessage().contains(
                "attention gradients require output and canonical weights slots"));
        assertEquals(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                output.provenance().orElseThrow().operation().kind());
    }

    @Test
    void mixedFloatingBroadcastMaskAndExplicitScaleRestoreEveryInputContract() {
        Tensor query = tensor(DataType.BFLOAT16, Shape.of(2, 3, 4), true);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(1, 5, 4), true);
        Tensor value = tensor(DataType.FLOAT64, Shape.of(5, 6), true);
        Tensor mask = tensor(DataType.BOOL, Shape.of(1, 3, 5), false);
        ScaledDotProductAttentionResult attention =
                query.scaledDotProductAttentionWithWeights(
                        key,
                        value,
                        mask,
                        new ScaledDotProductAttentionAttrs(
                                Optional.of(ScalarValue.float64(0.5d)), true));
        Tensor objective = attention.output().sum().add(attention.weights().sum());

        Tensor queryGradient = gradient(objective, query);
        Tensor keyGradient = gradient(objective, key);
        Tensor valueGradient = gradient(objective, value);
        assertEquals(query.descriptor().shape(), queryGradient.descriptor().shape());
        assertEquals(query.descriptor().dataType(), queryGradient.descriptor().dataType());
        assertEquals(key.descriptor().shape(), keyGradient.descriptor().shape());
        assertEquals(key.descriptor().dataType(), keyGradient.descriptor().dataType());
        assertEquals(value.descriptor().shape(), valueGradient.descriptor().shape());
        assertEquals(value.descriptor().dataType(), valueGradient.descriptor().dataType());
        assertCompiles(objective, query);
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        return FirstOrderAutograd.expand(
                        preflight(objective, target),
                        CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
    }

    private static AutogradPreflight.StagePlan preflight(Tensor objective, Tensor target) {
        return AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
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

    private static boolean reaches(Tensor root, Tensor expected) {
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (tensor == expected) {
                return true;
            }
            if (seen.put(tensor, Boolean.TRUE) == null) {
                tensor.provenance().ifPresent(
                        provenance -> provenance.inputs().forEach(pending::addLast));
            }
        }
        return false;
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }
}
