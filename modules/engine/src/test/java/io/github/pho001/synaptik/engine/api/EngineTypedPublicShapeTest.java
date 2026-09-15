package io.github.pho001.synaptik.engine.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
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
                RunResult.class, RunResult.Publication.class)) {
            assertTrue(Modifier.isPublic(type.getModifiers()));
            assertTrue(Modifier.isFinal(type.getModifiers()));
        }
        assertTrue(CompiledGraph.Input.class.isRecord());
        assertTrue(Modifier.isPublic(CompiledGraph.Input.class.getModifiers()));
        assertTrue(Modifier.isStatic(CompiledGraph.Input.class.getModifiers()));
        assertEquals(List.of(TensorId.class, TensorDescriptor.class), Arrays.stream(
                CompiledGraph.Input.class.getRecordComponents()).map(component -> component.getType())
                .toList());

        assertEquals(List.of("close", "compile", "compile", "isClosed", "prepare", "run",
                "standard"), methodNames(Engine.class));
        assertEquals(List.of("inputs"), methodNames(CompiledGraph.class));
        assertEquals(List.of("compiledGraph"), methodNames(PreparedExecution.class));
        assertEquals(List.of("close", "isClosed", "publications", "resultCount"),
                methodNames(RunResult.class));
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
        assertEquals(RunResult.class,
                Engine.class.getMethod("run", PreparedExecution.class, List.class).getReturnType());
        assertEquals(OptionalInt.class,
                RunResult.Publication.class.getMethod("derivativeOrder").getReturnType());
        assertEquals(OptionalInt.class,
                RunResult.Publication.class.getMethod("targetIndex").getReturnType());

        for (Class<?> type : List.of(CompiledGraph.class, PreparedExecution.class, RunResult.class,
                RunResult.Publication.class)) {
            assertEquals(0, Arrays.stream(type.getDeclaredConstructors())
                    .filter(constructor -> Modifier.isPublic(constructor.getModifiers())
                            || Modifier.isProtected(constructor.getModifiers())).count());
        }
        for (Class<?> type : List.of(Engine.class, CompiledGraph.class, PreparedExecution.class,
                RunResult.class, RunResult.Publication.class, CompiledGraph.Input.class)) {
            Arrays.stream(type.getDeclaredMethods()).filter(method -> Modifier.isPublic(
                    method.getModifiers())).forEach(method -> assertOrdinary(method.toGenericString()));
            Arrays.stream(type.getDeclaredFields()).filter(field -> Modifier.isPublic(
                    field.getModifiers()) || Modifier.isProtected(field.getModifiers()))
                    .forEach(field -> assertOrdinary(field.toGenericString()));
        }
        assertFalse(AutoCloseable.class.isAssignableFrom(CompiledGraph.class));
        assertFalse(AutoCloseable.class.isAssignableFrom(PreparedExecution.class));
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
