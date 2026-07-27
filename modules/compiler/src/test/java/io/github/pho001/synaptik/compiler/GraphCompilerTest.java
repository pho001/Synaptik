package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

final class GraphCompilerTest {
    @Test
    void exposesTheExactPackagePrivateDirectCompileAndResultShapes() throws Exception {
        var compile = GraphCompiler.class.getDeclaredMethod(
                "compile",
                CompileMode.class,
                List.class,
                Optional.class,
                CompileTimeConstantGraph.Ingress.class,
                GraphOptimizationConfig.class);

        assertTrue(Modifier.isStatic(compile.getModifiers()));
        assertFalse(Modifier.isPublic(compile.getModifiers()));
        assertSame(GraphCompilation.class, compile.getReturnType());
        assertEquals(
                List.of("mode", "validatedGraph", "forwardOutputs", "gradientResults"),
                java.util.Arrays.stream(GraphCompilation.class.getRecordComponents())
                        .map(component -> component.getName())
                        .toList());
        assertTrue(GraphCompilation.class.isRecord());
        assertTrue(GraphCompilation.GradientResultRole.class.isRecord());
        assertFalse(Modifier.isPublic(GraphCompilation.class.getModifiers()));
    }

    @Test
    void compilesForwardOnlyWithoutBackwardState() {
        Tensor input = tensor();
        Tensor output = input.neg();

        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());

        assertEquals(result.validatedGraph().graph().outputs(), result.forwardOutputs());
        assertTrue(result.gradientResults().isEmpty());
        assertFalse(result.validatedGraph().graph().nodePhases()
                .containsValue(GraphPhase.BACKWARD));
    }

    @Test
    void compilesBackwardAndTrainingModesAsOnePhaseAwareGraph() {
        for (CompileMode mode :
                List.of(CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            Tensor target = tensor();
            Tensor objective = target.mul(target).sum();
            GraphCompilation result = GraphCompiler.compile(
                    mode,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    GraphOptimizationConfig.standard());

            assertEquals(1, result.forwardOutputs().size());
            assertEquals(1, result.gradientResults().size());
            assertEquals(target.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
            assertTrue(result.validatedGraph().constants().size() >= 1);
        }
    }

    @Test
    void enforcesModeRequestPresenceBeforeExpansion() {
        Tensor output = tensor().sum();
        var request = new AutogradPreflight.FirstOrderRequest(output, List.of(output));
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.TRAINING_STEP,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
    }

    @Test
    void stableDeduplicatesEqualGradientBoundaryValuesButPreservesEveryTargetRole() {
        Tensor target = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), true));
        Tensor objective = target.add(ScalarValue.float32(2.0f));

        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target, objective))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());

        assertEquals(2, result.gradientResults().size());
        assertEquals(
                result.gradientResults().get(0).gradient(),
                result.gradientResults().get(1).gradient());
        assertEquals(2, result.validatedGraph().graph().outputs().size());
    }

    @Test
    void knownUnsupportedPreflightFailureConsumesNoTensorIdentity() throws Exception {
        Tensor target = tensor();
        Tensor other = tensor();
        Tensor objective = target.pow(other).sum();
        var request = new AutogradPreflight.FirstOrderRequest(objective, List.of(target));
        long before = nextTensorId();

        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));

        assertEquals(before, nextTensorId());
    }

    @Test
    void compilesSharedAlgebraDivisionMeanAndDirectZeroInBothOptimizationModes() {
        for (GraphOptimizationConfig optimization :
                List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            Tensor narrow = TensorFactory.create(new TensorDescriptor(
                    DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
            Tensor wide = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
            Tensor objective = narrow.div(wide).floor().mean();

            GraphCompilation result = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(narrow))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    optimization);

            assertEquals(narrow.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
        }
    }

    @Test
    void preflightOnly0004AFailuresConsumeNoTensorIdentity() throws Exception {
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor source = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(dynamic),
                Optional.empty(),
                true));
        Tensor bindingDependent = source
                .sumToShape(Shape.ofDimensions(new DynamicDimension("M")))
                .sum();
        long beforeShapeFailure = nextTensorId();
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(bindingDependent),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        bindingDependent, List.of(source))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertEquals(beforeShapeFailure, nextTensorId());

        Tensor base = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(dynamic),
                Optional.empty(),
                true));
        Tensor update = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2)),
                Optional.empty(),
                true));
        Tensor replacement = base.sliceUpdate(
                        update,
                        new long[] {0},
                        new int[] {0},
                        new long[] {1})
                .sum();
        long beforeRoleFailure = nextTensorId();
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(replacement),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        replacement, List.of(update))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertEquals(beforeRoleFailure, nextTensorId());
    }

    @Test
    void capturesRepresentative0004AFormulasOnceWithPhasesRolesAndConstants() {
        for (GraphOptimizationConfig optimization :
                List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            Tensor target = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT32, Shape.of(2, 2), Optional.empty(), true));
            Tensor objective = target.erf()
                    .matmul(target)
                    .slice(
                            new long[] {0},
                            new long[] {2},
                            new int[] {0},
                            new long[] {1})
                    .pad(
                            new long[] {0, 1},
                            new long[] {0, 0},
                            ScalarValue.float32(0.0f))
                    .sum();

            GraphCompilation result = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    optimization);

            assertEquals(1, result.forwardOutputs().size());
            assertEquals(target.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.FORWARD));
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
            assertTrue(result.validatedGraph().constants().size() >= 2);
        }
    }

    private static Tensor tensor() {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), true));
    }

    private static long nextTensorId() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
    }
}
