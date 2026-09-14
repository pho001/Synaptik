package io.github.pho001.synaptik.engine.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.engine.AdvancedCompiledGraph;
import io.github.pho001.synaptik.engine.AdvancedEngine;
import io.github.pho001.synaptik.engine.AdvancedPreparedExecution;
import io.github.pho001.synaptik.engine.AdvancedRunResult;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Locks the intentionally small advanced Engine API from a distinct package. */
final class AdvancedEnginePublicShapeTest {
    @Test
    void exposesOnlyTheSpecifiedOpaqueLifecycleSurface() {
        assertPublicFinal(AdvancedEngine.class);
        assertPublicFinal(AdvancedCompiledGraph.class);
        assertPublicFinal(AdvancedPreparedExecution.class);
        assertPublicFinal(AdvancedRunResult.class);
        assertTrue(AutoCloseable.class.isAssignableFrom(AdvancedEngine.class));
        assertTrue(AutoCloseable.class.isAssignableFrom(AdvancedRunResult.class));

        assertEquals(List.of("borrow", "close", "compile", "isClosed", "prepare", "run",
                        "takeOwnership"), publicMethodNames(AdvancedEngine.class));
        assertEquals(List.of(), publicMethodNames(AdvancedCompiledGraph.class));
        assertEquals(List.of(), publicMethodNames(AdvancedPreparedExecution.class));
        assertEquals(List.of("close", "isClosed", "resultCount"),
                publicMethodNames(AdvancedRunResult.class));

        for (Class<?> type : List.of(AdvancedEngine.class, AdvancedCompiledGraph.class,
                AdvancedPreparedExecution.class, AdvancedRunResult.class)) {
            assertEquals(0, Arrays.stream(type.getDeclaredConstructors())
                    .filter(constructor -> Modifier.isPublic(constructor.getModifiers())
                            || Modifier.isProtected(constructor.getModifiers())).count());
            Arrays.stream(type.getDeclaredMethods())
                    .filter(method -> Modifier.isPublic(method.getModifiers()))
                    .forEach(method -> {
                        String signature = method.toGenericString();
                        assertFalse(signature.contains("CompileArtifacts"));
                        assertFalse(signature.contains("synaptik.runtime.execution.PreparedExecution"));
                        assertFalse(signature.contains("synaptik.runtime.run.RunResult"));
                        assertFalse(signature.contains("synaptik.runtime.run.RunState"));
                        assertFalse(signature.contains(".internal."));
                    });
        }
    }

    private static void assertPublicFinal(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(Modifier.isFinal(type.getModifiers()));
    }

    private static List<String> publicMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName()).sorted().toList();
    }
}
