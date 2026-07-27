package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class FirstOrderAutogradTest {
    @Test
    void accumulatesRepeatedInputPositionsLeftAssociatedAndPreservesTargetOrder() {
        Tensor target = tensor();
        Tensor objective = target.mul(target).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.TRAINING_STEP,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target, objective)),
                CompileTimeConstantGraph.Ingress.empty());

        FirstOrderAutograd.Expansion expansion =
                FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty());
        Tensor targetGradient = expansion.targetGradients().get(0).gradient();

        assertEquals(BinaryArithmeticKind.ADD,
                targetGradient.provenance().orElseThrow().operation().kind());
        assertSame(
                expansion.targetGradients().get(1).gradient(),
                expansion.ingress().bindings().getFirst().tensor());
        assertEquals(2, expansion.targetGradients().size());
    }

    @Test
    void reusesAtMostOneBaseZeroAndOnePerTypeAndAppendsBindingsAfterForwardIngress() {
        Tensor target = tensor();
        Tensor condition = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, Shape.of(2), Optional.empty(), false));
        Tensor objective = Tensor.where(condition, target, target).sum();
        Tensor callerConstant = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false));
        var callerBinding = new CompileTimeConstantGraph.Binding(
                callerConstant,
                new CompileTimeConstantGraph.Splat(
                        io.github.pho001.synaptik.model.datatype.ScalarValue.float32(7.0f)));
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        FirstOrderAutograd.Expansion expansion = FirstOrderAutograd.expand(
                plan, new CompileTimeConstantGraph.Ingress(List.of(callerBinding)));

        assertSame(callerBinding, expansion.ingress().bindings().getFirst());
        assertEquals(3, expansion.ingress().bindings().size());
    }

    @Test
    void createsExactStorageFreeScalarZeroAndOneLeavesForEveryFloatingType() {
        for (DataType dataType :
                List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            var constants = new FirstOrderAutograd.DerivativeConstants();
            Tensor zero = constants.zeroBase(dataType);
            Tensor one = constants.oneBase(dataType);

            assertSame(zero, constants.zeroBase(dataType));
            assertSame(one, constants.oneBase(dataType));
            assertEquals(2, constants.bindings().size());
            assertEquals(Shape.scalar(), zero.descriptor().shape());
            assertFalse(zero.descriptor().requiresGrad());
            assertTrueLeaf(zero);
            assertTrueLeaf(one);
            if (dataType == DataType.BFLOAT16) {
                assertEquals(
                        (short) 0x0000,
                        constants.bindings().get(0).splat().value().bfloat16Bits());
                assertEquals(
                        (short) 0x3F80,
                        constants.bindings().get(1).splat().value().bfloat16Bits());
            }
        }
    }

    private static void assertTrueLeaf(Tensor tensor) {
        assertFalse(tensor.provenance().isPresent());
        assertFalse(tensor.label().isPresent());
        assertFalse(tensor.hostStorage().isPresent());
    }

    private static Tensor tensor() {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), true));
    }
}
