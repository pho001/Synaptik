package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GradientRulesTest {
    @Test
    void multiplyUsesExactForwardOperandAndUnbroadcastsOnlySelectedRole() {
        Tensor target = tensor(Shape.of(2, 1));
        Tensor right = tensor(Shape.of(2, 3));
        Tensor objective = target.mul(right).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        FirstOrderAutograd.Expansion expansion =
                FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty());
        Tensor gradient = expansion.targetGradients().getFirst().gradient();
        var sumToShape = gradient.provenance().orElseThrow();
        Tensor multiplied = sumToShape.inputs().getFirst();

        assertTrue(sumToShape.operation().attrs() instanceof SumToShapeAttrs);
        assertSame(right, multiplied.provenance().orElseThrow().inputs().get(1));
        assertEquals(
                BinaryArithmeticKind.MUL,
                multiplied.provenance().orElseThrow().operation().kind());
    }

    @Test
    void selectedReductionAndLayoutRulesRestoreTheExactInputShape() {
        Tensor target = tensor(Shape.of(2, 3));
        Tensor objective = target.permute(1, 0).sum(0).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();

        assertEquals(target.descriptor().shape(), gradient.descriptor().shape());
    }

    @Test
    void compilesEverySupported0004VariantForEveryFloatingType() {
        for (DataType dataType :
                List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor target = tensor(dataType, Shape.of(2, 3));
            Tensor other = tensor(dataType, Shape.of(2, 3));
            ScalarValue scalar = one(dataType);
            Tensor condition = TensorFactory.create(new TensorDescriptor(
                    DataType.BOOL, Shape.of(2, 3), Optional.empty(), false));

            assertGradientCompiles(target.add(other).sum(), target);
            assertGradientCompiles(target.sub(other).sum(), target);
            assertGradientCompiles(target.mul(other).sum(), target);
            assertGradientCompiles(target.add(scalar).sum(), target);
            assertGradientCompiles(target.sub(scalar).sum(), target);
            assertGradientCompiles(target.mul(scalar).sum(), target);
            assertGradientCompiles(Tensor.where(condition, target, other).sum(), target);
            assertGradientCompiles(target.cast(dataType).sum(), target);
            assertGradientCompiles(target.neg().sum(), target);
            assertGradientCompiles(target.exp().sum(), target);
            assertGradientCompiles(target.expm1().sum(), target);
            assertGradientCompiles(target.sigmoid().sum(), target);
            assertGradientCompiles(target.tanh().sum(), target);
            assertGradientCompiles(target.sum(), target);
            assertGradientCompiles(target.sum(1).sum(), target);
            assertGradientCompiles(target.sum(1, true).sum(), target);
            assertGradientCompiles(target.sum(new int[] {1, 0}, false).sum(), target);
            assertGradientCompiles(target.sum(new int[] {}, false).sum(), target);
            assertGradientCompiles(target.cumSum(1, true, false).sum(), target);
            assertGradientCompiles(target.contiguous().sum(), target);
            assertGradientCompiles(target.reshape(3, 2).sum(), target);
            assertGradientCompiles(target.expandDims(1).sum(), target);
            assertGradientCompiles(target.permute(1, 0).sum(), target);

            Tensor expandable = tensor(dataType, Shape.of(2, 1));
            assertGradientCompiles(expandable.expand(2, 3).sum(), expandable);
            Tensor squeezable = tensor(dataType, Shape.of(2, 1, 3));
            assertGradientCompiles(squeezable.squeeze(1).sum(), squeezable);
        }
    }

    private static void assertGradientCompiles(Tensor objective, Tensor target) {
        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), result.gradientResults().getFirst().target());
    }

    private static ScalarValue one(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3F80);
            case FLOAT32 -> ScalarValue.float32(1.0f);
            case FLOAT64 -> ScalarValue.float64(1.0d);
            case INT32, INT64, BOOL -> throw new AssertionError(dataType);
        };
    }

    private static Tensor tensor(Shape shape) {
        return tensor(DataType.FLOAT32, shape);
    }

    private static Tensor tensor(DataType dataType, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), true));
    }
}
