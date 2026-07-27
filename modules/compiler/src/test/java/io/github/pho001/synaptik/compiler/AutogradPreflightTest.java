package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class AutogradPreflightTest {
    @Test
    void selectsOnlyObjectiveToTargetOccurrencesInDeterministicPostorder() {
        Tensor target = tensor(Shape.of(2), true);
        Tensor unrelated = tensor(Shape.of(2), true);
        Tensor selected = target.mul(unrelated);
        Tensor objective = selected.sum();

        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        assertEquals(List.of(
                selected.provenance().orElseThrow().producer(),
                objective.provenance().orElseThrow().producer()), plan.producerPostorder());
        assertEquals(2, plan.selectedOccurrences().size());
        assertTrue(plan.selectedOccurrences().get(0).selectedInput(0));
        assertTrue(!plan.selectedOccurrences().get(0).selectedInput(1));
    }

    @Test
    void rejectsUnsupportedAndNonDifferentiableSelectedRoutes() {
        Tensor target = tensor(Shape.of(2), true);
        Tensor unsupportedObjective = target.abs().sum();
        IllegalArgumentException unsupported = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(unsupportedObjective),
                        new AutogradPreflight.FirstOrderRequest(
                                unsupportedObjective, List.of(target)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(unsupported.getMessage().contains("ABS"));

        Tensor branch = tensor(Shape.of(2), true);
        Tensor conditionSource = tensor(Shape.of(2), true);
        Tensor condition = conditionSource.greaterThan(branch);
        Tensor objective = Tensor.where(condition, branch, branch).sum();
        IllegalArgumentException conditionFailure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective),
                        new AutogradPreflight.FirstOrderRequest(
                                objective, List.of(conditionSource)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(conditionFailure.getMessage().contains("non-differentiable")
                || conditionFailure.getMessage().contains("SUPPORTED_0004")
                || conditionFailure.getMessage().contains("GREATER_THAN"));
    }

    @Test
    void requiresExactScalarForwardObjectiveAndIdentityUniqueTargets() {
        Tensor vector = tensor(Shape.of(2), true);
        assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.TRAINING_STEP,
                        List.of(vector),
                        new AutogradPreflight.FirstOrderRequest(vector, List.of(vector)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertThrows(
                IllegalArgumentException.class,
                () -> new AutogradPreflight.FirstOrderRequest(vector, List.of(vector, vector)));
    }

    @Test
    void admitsOnlyLocallyProvableSumToShapeInversion() {
        DynamicDimension sourceExtent = new DynamicDimension("N");
        Tensor exactTarget = tensor(Shape.ofDimensions(sourceExtent), true);
        Tensor exactObjective =
                exactTarget.sumToShape(Shape.ofDimensions(sourceExtent)).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(exactObjective),
                new AutogradPreflight.FirstOrderRequest(exactObjective, List.of(exactTarget)),
                CompileTimeConstantGraph.Ingress.empty());
        assertEquals(2, plan.selectedOccurrences().size());

        Tensor bindingDependent = tensor(Shape.ofDimensions(sourceExtent), true);
        Tensor unsupportedObjective = bindingDependent
                .sumToShape(Shape.ofDimensions(new DynamicDimension("M")))
                .sum();
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(unsupportedObjective),
                        new AutogradPreflight.FirstOrderRequest(
                                unsupportedObjective, List.of(bindingDependent)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(failure.getMessage().contains("SUM_TO_SHAPE"));
    }

    @Test
    void sliceUpdateAppliesTheStaticBaseGuardOnlyToTheUpdateRole() {
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor base = tensor(Shape.ofDimensions(dynamic), true);
        Tensor update = tensor(Shape.ofDimensions(new StaticDimension(2)), true);
        Tensor objective = base.sliceUpdate(
                        update,
                        new long[] {0},
                        new int[] {0},
                        new long[] {1})
                .sum();

        AutogradPreflight.Plan basePlan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(base)),
                CompileTimeConstantGraph.Ingress.empty());
        assertTrue(basePlan.selectedOccurrences().stream()
                .anyMatch(occurrence -> occurrence.selectedInput(0)));

        IllegalArgumentException updateFailure = assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective),
                        new AutogradPreflight.FirstOrderRequest(objective, List.of(update)),
                        CompileTimeConstantGraph.Ingress.empty()));
        assertTrue(updateFailure.getMessage().contains("static selected base Dimensions"));
    }

    @Test
    void keepsMaskAndRepresentativePolicyDependentRoutesNonDifferentiable() {
        Tensor data = tensor(Shape.of(3), true);
        Tensor maskSource = tensor(Shape.of(3), true);
        Tensor mask = maskSource.greaterThan(data);
        Tensor maskedObjective = data.sum(0, mask);
        assertThrows(
                IllegalArgumentException.class,
                () -> AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD,
                        List.of(maskedObjective),
                        new AutogradPreflight.FirstOrderRequest(
                                maskedObjective, List.of(maskSource)),
                        CompileTimeConstantGraph.Ingress.empty()));

        for (Tensor objective : List.of(
                data.softmax(0).sum(),
                data.cropToShape(Shape.of(2), Shape.of(0)).sum())) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> AutogradPreflight.preflight(
                            CompileMode.FORWARD_AND_BACKWARD,
                            List.of(objective),
                            new AutogradPreflight.FirstOrderRequest(
                                    objective, List.of(data)),
                            CompileTimeConstantGraph.Ingress.empty()));
            assertTrue(failure.getMessage().contains("producerPostorder["));
        }
    }

    @Test
    void admitsMixedFloatingNormalizationDivisionDirectZeroAndMeanRows() {
        Tensor narrow = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
        Tensor wide = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
        for (Tensor objective : List.of(
                narrow.add(wide).mean(),
                narrow.div(wide).mean(),
                narrow.cast(DataType.FLOAT32).mean(),
                narrow.floor().mean(),
                narrow.ceil().mean(),
                narrow.sign().mean())) {
            AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    new AutogradPreflight.FirstOrderRequest(objective, List.of(narrow)),
                    CompileTimeConstantGraph.Ingress.empty());
            assertTrue(!plan.selectedOccurrences().isEmpty());
        }
    }

    private static Tensor tensor(Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), requiresGrad));
    }
}
