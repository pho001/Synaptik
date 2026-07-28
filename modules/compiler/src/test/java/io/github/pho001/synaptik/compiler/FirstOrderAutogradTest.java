package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.BatchNormTrainingResult;
import io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult;
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
    void accumulatesAttentionOutputSlotsInStableCanonicalOrder() {
        Tensor query = tensor(Shape.of(2, 3, 4));
        Tensor key = tensor(Shape.of(2, 5, 4));
        Tensor value = tensor(Shape.of(2, 5, 6));
        ScaledDotProductAttentionResult attention =
                query.scaledDotProductAttentionWithWeights(key, value);
        Tensor objective = attention.output().sum().add(attention.weights().sum());
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(query)),
                CompileTimeConstantGraph.Ingress.empty());

        List<AutogradPreflight.SelectedOccurrence> selected = plan.selectedOccurrences().stream()
                .filter(occurrence -> occurrence.producer()
                        == attention.output().provenance().orElseThrow().producer())
                .toList();
        assertEquals(
                List.of(0, 1),
                selected.stream()
                        .map(AutogradPreflight.SelectedOccurrence::outputIndex)
                        .toList());
        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
        assertEquals(
                BinaryArithmeticKind.ADD,
                gradient.provenance().orElseThrow().operation().kind());
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

    @Test
    void cachesEveryExactTypedSplatByBitsAndExpandsOnlyThroughPublicExpressions() {
        var constants = new FirstOrderAutograd.DerivativeConstants();
        ScalarValue positiveZero = ScalarValue.float32(0.0f);
        ScalarValue negativeZero = ScalarValue.float32(-0.0f);
        Tensor positive = constants.base(positiveZero);
        Tensor negative = constants.base(negativeZero);

        assertSame(positive, constants.base(ScalarValue.float32(0.0f)));
        assertSame(negative, constants.base(ScalarValue.float32(-0.0f)));
        assertEquals(2, constants.bindings().size());
        assertTrueLeaf(positive);
        assertTrueLeaf(negative);

        Tensor shaped = constants.valueLike(negativeZero, tensor());
        assertEquals(
                io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND,
                shaped.provenance().orElseThrow().operation().kind());
        assertSame(negative, shaped.provenance().orElseThrow().inputs().getFirst());
    }

    @Test
    void preservesEveryRepeatedConcatPositionInDeterministicInputOrder() {
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor target = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(dynamic),
                Optional.empty(),
                true));
        Tensor objective = Tensor.concat(0, target, target).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients()
                .getFirst()
                .gradient();
        var addition = gradient.provenance().orElseThrow();
        CropToShapeAttrs first = (CropToShapeAttrs) addition.inputs()
                .get(0).provenance().orElseThrow().operation().attrs();
        CropToShapeAttrs second = (CropToShapeAttrs) addition.inputs()
                .get(1).provenance().orElseThrow().operation().attrs();

        assertEquals(BinaryArithmeticKind.ADD, addition.operation().kind());
        assertEquals(Shape.of(0), first.prefixShape());
        assertEquals(dynamic, second.prefixShape().dimension(0));
    }

    @Test
    void routesMeanToTheReductionOwnerAndRetainsOrdinaryDivision() {
        Tensor target = tensor();
        Tensor objective = target.mean();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients()
                .getFirst()
                .gradient();

        assertEquals(BinaryArithmeticKind.DIV,
                gradient.provenance().orElseThrow().operation().kind());
        Tensor denominator = gradient.provenance().orElseThrow().inputs().get(1);
        assertEquals(AggregateReductionKind.SUM,
                denominator.provenance().orElseThrow().inputs().getFirst()
                        .provenance().orElseThrow().operation().kind());
    }

    @Test
    void exposesExactTwoAndNegativeHalfCoefficientsForEveryFloatingType() {
        var constants = new FirstOrderAutograd.DerivativeConstants();
        assertEquals((short) 0x4000, constants.two(DataType.BFLOAT16).bfloat16Bits());
        assertEquals((short) 0xC000, constants.negativeTwo(DataType.BFLOAT16).bfloat16Bits());
        assertEquals((short) 0xBF00, constants.negativeHalf(DataType.BFLOAT16).bfloat16Bits());
        assertEquals(
                0x40000000,
                Float.floatToRawIntBits(constants.two(DataType.FLOAT32).float32Value()));
        assertEquals(
                0xC0000000,
                Float.floatToRawIntBits(constants.negativeTwo(DataType.FLOAT32).float32Value()));
        assertEquals(
                0xBF000000,
                Float.floatToRawIntBits(constants.negativeHalf(DataType.FLOAT32).float32Value()));
        assertEquals(
                0x4000000000000000L,
                Double.doubleToRawLongBits(constants.two(DataType.FLOAT64).float64Value()));
        assertEquals(
                0xC000000000000000L,
                Double.doubleToRawLongBits(constants.negativeTwo(DataType.FLOAT64).float64Value()));
        assertEquals(
                0xBFE0000000000000L,
                Double.doubleToRawLongBits(
                        constants.negativeHalf(DataType.FLOAT64).float64Value()));
    }

    @Test
    void accumulatesBatchTrainingSlotsAscendingAndLeftAssociated() {
        Tensor input = tensor(Shape.of(2, 3, 4));
        Tensor vector = tensor(Shape.of(3));
        BatchNormTrainingResult result = input.batchNormTraining(
                1,
                vector,
                vector,
                vector,
                vector,
                ScalarValue.float32(0.1f),
                ScalarValue.float32(1.0e-5f));
        Tensor objective = result.output().sum()
                .add(result.nextRunningMean().sum())
                .add(result.nextRunningVariance().sum());
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(input)),
                CompileTimeConstantGraph.Ingress.empty());

        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients()
                .getFirst()
                .gradient();

        assertEquals(BinaryArithmeticKind.ADD,
                gradient.provenance().orElseThrow().operation().kind());
        assertEquals(BinaryArithmeticKind.ADD,
                gradient.provenance().orElseThrow().inputs().getFirst()
                        .provenance().orElseThrow().operation().kind());
    }

    private static void assertTrueLeaf(Tensor tensor) {
        assertFalse(tensor.provenance().isPresent());
        assertFalse(tensor.label().isPresent());
        assertFalse(tensor.hostStorage().isPresent());
    }

    private static Tensor tensor() {
        return tensor(Shape.of(2));
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }
}
