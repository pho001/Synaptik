package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class LossGradientRulesTest {
    @Test
    void meanSquaredErrorAndDenseCategoricalRestoreEveryReduction() {
        for (LossReduction reduction : LossReduction.values()) {
            Tensor prediction = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
            Tensor target = tensor(DataType.BFLOAT16, Shape.of(2, 3), true);
            Tensor mse = prediction.meanSquaredError(target, reduction);
            assertEquals(prediction.descriptor().shape(),
                    gradient(mseObjective(mse, reduction), prediction).descriptor().shape());
            assertEquals(target.descriptor().shape(),
                    gradient(mseObjective(mse, reduction), target).descriptor().shape());

            Tensor logits = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
            Tensor dense = tensor(DataType.BFLOAT16, Shape.of(2, 3, 4), true);
            Tensor loss =
                    logits.categoricalCrossEntropyWithLogits(dense, 1, reduction);
            Tensor objective = mseObjective(loss, reduction);
            assertEquals(logits.descriptor().shape(),
                    gradient(objective, logits).descriptor().shape());
            assertEquals(dense.descriptor().shape(),
                    gradient(objective, dense).descriptor().shape());
            assertCompiles(objective, logits);
            assertCompiles(objective, dense);
        }
    }

    @Test
    void indexCategoricalSupportsStaticDepthAndFailsClosedForDynamicOrZeroDepth() {
        Tensor logits = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor target = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor ignoredMean = logits.categoricalCrossEntropyWithLogits(
                target, 1, LossReduction.MEAN, ScalarValue.int64(-1));
        Tensor logitsGradient = gradient(ignoredMean, logits);
        assertEquals(logits.descriptor().shape(), logitsGradient.descriptor().shape());
        assertCompiles(ignoredMean, logits);
        for (LossReduction reduction : LossReduction.values()) {
            Tensor loss = logits.categoricalCrossEntropyWithLogits(target, 1, reduction);
            Tensor objective = reduction == LossReduction.NONE ? loss.sum() : loss;
            assertEquals(
                    logits.descriptor().shape(),
                    gradient(objective, logits).descriptor().shape());
        }

        Tensor dynamicLogits = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(
                        new DynamicDimension("N"),
                        new DynamicDimension("C")),
                true);
        Tensor dynamicTarget = tensor(
                DataType.INT32,
                Shape.ofDimensions(new DynamicDimension("N")),
                false);
        Tensor dynamicLoss = dynamicLogits.categoricalCrossEntropyWithLogits(
                dynamicTarget, 1, LossReduction.SUM, ScalarValue.int32(-1));
        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> preflight(dynamicLoss, dynamicLogits));
        assertTrue(dynamicFailure.getMessage().contains(
                "statically positive class extent"));

        Tensor zeroLogits = tensor(DataType.FLOAT32, Shape.of(0, 0), true);
        Tensor zeroTarget = tensor(DataType.INT64, Shape.of(0), false);
        Tensor zeroLoss = zeroLogits.categoricalCrossEntropyWithLogits(
                zeroTarget, 1, LossReduction.SUM, ScalarValue.int64(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> preflight(zeroLoss, zeroLogits));
    }

    private static Tensor mseObjective(Tensor loss, LossReduction reduction) {
        return reduction == LossReduction.NONE ? loss.sum() : loss;
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

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }
}
