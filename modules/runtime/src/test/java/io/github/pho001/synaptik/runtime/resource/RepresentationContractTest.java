package io.github.pho001.synaptik.runtime.resource;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class RepresentationContractTest {
    @Test
    void representationRolesHaveExactDistinctLifecycleOnlySurfaces() {
        assertRepresentationRole(BufferRepresentation.class);
        assertRepresentationRole(WorkspaceRepresentation.class);

        assertAll(
                () -> assertFalse(
                        BufferRepresentation.class.isAssignableFrom(
                                WorkspaceRepresentation.class)),
                () -> assertFalse(
                        WorkspaceRepresentation.class.isAssignableFrom(
                                BufferRepresentation.class)));
    }

    @Test
    void closeDeclaresNoCheckedFailure() {
        var bufferClosed = new AtomicBoolean();
        var workspaceClosed = new AtomicBoolean();
        BufferRepresentation buffer = () -> bufferClosed.set(true);
        WorkspaceRepresentation workspace = () -> workspaceClosed.set(true);

        buffer.close();
        workspace.close();

        assertAll(
                () -> assertTrue(bufferClosed.get()),
                () -> assertTrue(workspaceClosed.get()));
    }

    private static void assertRepresentationRole(Class<?> type) {
        Method[] methods = type.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(type.getModifiers())),
                () -> assertTrue(type.isInterface()),
                () -> assertArrayEquals(
                        new Class<?>[] {AutoCloseable.class}, type.getInterfaces()),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(1, methods.length),
                () -> assertEquals("close", methods[0].getName()),
                () -> assertEquals(void.class, methods[0].getReturnType()),
                () -> assertEquals(0, methods[0].getParameterCount()),
                () -> assertEquals(0, methods[0].getExceptionTypes().length),
                () -> assertTrue(Modifier.isPublic(methods[0].getModifiers())),
                () -> assertTrue(Modifier.isAbstract(methods[0].getModifiers())));
    }
}
