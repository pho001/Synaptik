package io.github.pho001.synaptik.engine.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.engine.ScalarObjectiveBackwardResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

/** Locks the exact metadata-only ordinary Engine surface from a distinct package. */
final class EngineTypedPublicShapeTest {
    @Test
    void exposesOnlyTheSpecifiedOrdinaryLifecycleAndMetadata() throws Exception {
        for (Class<?> type : List.of(Engine.class, CompiledGraph.class, PreparedExecution.class,
                RunResult.class, HostTensorValue.class, ScalarObjectiveBackwardResult.class,
                RunResult.Publication.class)) {
            assertTrue(Modifier.isPublic(type.getModifiers()));
            assertTrue(Modifier.isFinal(type.getModifiers()));
        }
        assertTrue(CompiledGraph.Input.class.isRecord());
        assertTrue(Modifier.isPublic(CompiledGraph.Input.class.getModifiers()));
        assertTrue(Modifier.isStatic(CompiledGraph.Input.class.getModifiers()));
        assertEquals(List.of(TensorId.class, TensorDescriptor.class), Arrays.stream(
                CompiledGraph.Input.class.getRecordComponents()).map(component -> component.getType())
                .toList());

        assertEquals(List.of("backward", "close", "compile", "compile", "compute", "compute",
                "compute", "compute", "isClosed", "prepare", "prepareTuned", "run", "standard"),
                methodNames(Engine.class));
        assertEquals(List.of("inputs"), methodNames(CompiledGraph.class));
        assertEquals(List.of("compiledGraph"), methodNames(PreparedExecution.class));
        assertEquals(List.of("close", "isClosed", "materialize", "publications", "resultCount"),
                methodNames(RunResult.class));
        assertEquals(List.of("byteSize", "bytes", "dataType", "elementCount", "shape"),
                methodNames(HostTensorValue.class));
        assertEquals(List.of("gradients", "objective"),
                methodNames(ScalarObjectiveBackwardResult.class));
        assertEquals(List.of("derivativeOrder", "descriptor", "index", "isClosed", "role",
                "targetIndex", "tensorId"), methodNames(RunResult.Publication.class));
        assertEquals(List.of("FORWARD", "GRADIENT"),
                Arrays.stream(RunResult.Role.values()).map(Enum::name).toList());

        assertEquals(CompiledGraph.class,
                Engine.class.getMethod("compile", List.class).getReturnType());
        assertEquals(CompiledGraph.class,
                Engine.class.getMethod("compile", List.class, List.class, List.class).getReturnType());
        assertEquals(PreparedExecution.class,
                Engine.class.getMethod("prepare", CompiledGraph.class).getReturnType());
        assertEquals(ModelAutotuningPreparation.class,
                Engine.class.getMethod("prepareTuned", CompiledGraph.class,
                        ModelAutotuningRequest.class).getReturnType());
        assertEquals(RunResult.class,
                Engine.class.getMethod("run", PreparedExecution.class, List.class).getReturnType());
        assertEquals(ScalarObjectiveBackwardResult.class,
                Engine.class.getMethod("backward", Tensor.class, List.class, long.class)
                        .getReturnType());
        assertEquals(HostTensorValue.class,
                Engine.class.getMethod("compute", Tensor.class).getReturnType());
        assertEquals(HostTensorValue.class,
                Engine.class.getMethod("compute", Tensor.class, long.class).getReturnType());
        var orderedCompute = Engine.class.getMethod("compute", List.class);
        assertEquals(List.class, orderedCompute.getReturnType());
        assertEquals("java.util.List<io.github.pho001.synaptik.engine.HostTensorValue>",
                orderedCompute.getGenericReturnType().getTypeName());
        var boundedOrderedCompute = Engine.class.getMethod("compute", List.class, long.class);
        assertEquals(List.class, boundedOrderedCompute.getReturnType());
        assertEquals("java.util.List<io.github.pho001.synaptik.engine.HostTensorValue>",
                boundedOrderedCompute.getGenericReturnType().getTypeName());
        org.junit.jupiter.api.Assertions.assertThrows(NoSuchMethodException.class,
                () -> Engine.class.getMethod("forward", Tensor.class, List.class, long.class));
        org.junit.jupiter.api.Assertions.assertThrows(NoSuchMethodException.class,
                () -> Engine.class.getMethod("forward", List.class, List.class, long.class));
        assertEquals(HostTensorValue.class,
                RunResult.class.getMethod("materialize", RunResult.Publication.class, long.class)
                        .getReturnType());
        assertEquals(OptionalInt.class,
                RunResult.Publication.class.getMethod("derivativeOrder").getReturnType());
        assertEquals(OptionalInt.class,
                RunResult.Publication.class.getMethod("targetIndex").getReturnType());

        for (Class<?> type : List.of(CompiledGraph.class, PreparedExecution.class, RunResult.class,
                HostTensorValue.class, ScalarObjectiveBackwardResult.class,
                RunResult.Publication.class)) {
            assertEquals(0, Arrays.stream(type.getDeclaredConstructors())
                    .filter(constructor -> Modifier.isPublic(constructor.getModifiers())
                            || Modifier.isProtected(constructor.getModifiers())).count());
        }
        for (Class<?> type : List.of(Engine.class, CompiledGraph.class, PreparedExecution.class,
                RunResult.class, HostTensorValue.class, ScalarObjectiveBackwardResult.class,
                RunResult.Publication.class, ModelAutotuningRequest.class,
                ModelAutotuningRequest.ModelIdentity.class,
                ModelAutotuningPreparation.class,
                ModelAutotuningPreparation.Evidence.class,
                ModelAutotuningPreparation.WorkloadEvidence.class,
                ModelAutotuningPreparation.OccurrenceEvidence.class,
                ModelAutotuningPreparation.CandidateEvidence.class,
                ModelAutotuningPreparation.SampleSummary.class,
                ModelAutotuningPreparation.CompatibilityIdentity.class,
                ModelAutotuningPreparation.ContextIdentity.class,
                ModelAutotuningPreparation.CandidateIdentity.class,
                CompiledGraph.Input.class)) {
            Arrays.stream(type.getDeclaredMethods()).filter(method -> Modifier.isPublic(
                    method.getModifiers())).forEach(method -> assertOrdinary(method.toGenericString()));
            Arrays.stream(type.getDeclaredFields()).filter(field -> Modifier.isPublic(
                    field.getModifiers()) || Modifier.isProtected(field.getModifiers()))
                    .forEach(field -> assertOrdinary(field.toGenericString()));
        }
        assertFalse(AutoCloseable.class.isAssignableFrom(CompiledGraph.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(PreparedExecution.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(HostTensorValue.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(ScalarObjectiveBackwardResult.class));
        assertEquals(List.of("config", "modelIdentity", "representativeInputs"),
                methodNames(ModelAutotuningRequest.class));
        assertEquals(List.of("bytes", "equals", "hashCode", "schemaVersion", "toString"),
                methodNames(ModelAutotuningRequest.ModelIdentity.class));
        assertEquals(List.of("evidence", "outcome", "preparedExecution"),
                methodNames(ModelAutotuningPreparation.class));
        assertEquals(List.of("TUNED", "SAFE_HEURISTIC_FALLBACK"),
                Arrays.stream(ModelAutotuningPreparation.Outcome.values()).map(Enum::name).toList());
        assertEquals(List.of("CACHE_HIT", "MEASURED"),
                Arrays.stream(ModelAutotuningPreparation.Source.values()).map(Enum::name).toList());
        assertEquals(List.of("SESSION", "PERSISTENT"),
                Arrays.stream(ModelAutotuningPreparation.ReuseScope.values()).map(Enum::name).toList());
        assertEquals(0, Arrays.stream(ModelAutotuningPreparation.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers())
                        || Modifier.isProtected(constructor.getModifiers())).count());
        assertEquals(1, Arrays.stream(ModelAutotuningRequest.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers())).count());
        assertEquals(1, HostTensorValue.class.getDeclaredConstructors().length);
        var hostConstructor = HostTensorValue.class.getDeclaredConstructors()[0];
        assertEquals(0, hostConstructor.getModifiers());
        assertEquals(List.of(io.github.pho001.synaptik.model.datatype.DataType.class,
                        io.github.pho001.synaptik.model.shape.Shape.class, byte[].class),
                Arrays.asList(hostConstructor.getParameterTypes()));
        assertEquals(1, ScalarObjectiveBackwardResult.class.getDeclaredConstructors().length);
        var backwardConstructor =
                ScalarObjectiveBackwardResult.class.getDeclaredConstructors()[0];
        assertEquals(0, backwardConstructor.getModifiers());
        assertEquals(List.of(HostTensorValue.class, List.class),
                Arrays.asList(backwardConstructor.getParameterTypes()));
        assertEquals(HostTensorValue.class,
                ScalarObjectiveBackwardResult.class.getMethod("objective").getReturnType());
        var gradients = ScalarObjectiveBackwardResult.class.getMethod("gradients");
        assertEquals(List.class, gradients.getReturnType());
        assertEquals("java.util.List<io.github.pho001.synaptik.engine.HostTensorValue>",
                gradients.getGenericReturnType().getTypeName());
    }

    @Test
    void inputRecordChecksComponentsInOrder() {
        var descriptor = descriptor();
        assertEquals("tensorId", org.junit.jupiter.api.Assertions.assertThrows(
                NullPointerException.class, () -> new CompiledGraph.Input(null, descriptor))
                .getMessage());
        assertEquals("descriptor", org.junit.jupiter.api.Assertions.assertThrows(
                NullPointerException.class, () -> new CompiledGraph.Input(new TensorId(1), null))
                .getMessage());
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                io.github.pho001.synaptik.model.shape.Shape.of(1), java.util.Optional.empty(), false);
    }

    private static List<String> methodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods()).filter(method -> Modifier.isPublic(
                method.getModifiers())).map(method -> method.getName()).sorted().toList();
    }

    private static void assertOrdinary(String signature) {
        for (String forbidden : List.of("Advanced", ".compiler.", ".prepare.", ".runtime.",
                ".planning.", ".backend.", ".internal.", "ValueId", "Representation", "Slot")) {
            assertFalse(signature.contains(forbidden), () -> "forbidden ordinary signature " + signature);
        }
    }
}
