package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
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

    private static Tensor tensor(Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), requiresGrad));
    }
}
