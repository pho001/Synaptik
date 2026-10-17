package io.github.pho001.synaptik.backend.provider.openblas;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Verifies the exact public loading and lifetime surface. */
final class OpenBlasLibraryPublicShapeTest {
    /** Locks the two public final types and their deliberately narrow construction boundary. */
    @Test
    void publicTypesAreFinalAndExposeNoPublicConstructors() {
        assertTrue(Modifier.isPublic(OpenBlasLibrary.class.getModifiers()));
        assertTrue(Modifier.isFinal(OpenBlasLibrary.class.getModifiers()));
        assertEquals(Set.of(AutoCloseable.class), Set.of(OpenBlasLibrary.class.getInterfaces()));
        assertEquals(0, publicConstructors(OpenBlasLibrary.class));

        assertTrue(Modifier.isPublic(OpenBlasLoadException.class.getModifiers()));
        assertTrue(Modifier.isFinal(OpenBlasLoadException.class.getModifiers()));
        assertEquals(IllegalStateException.class, OpenBlasLoadException.class.getSuperclass());
        assertEquals(0, publicConstructors(OpenBlasLoadException.class));
    }

    /** Locks the exact declared public member surface of the library handle and failure type. */
    @Test
    void publicDeclaredMembersAreExact() {
        Set<String> libraryMethods = Arrays.stream(OpenBlasLibrary.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(OpenBlasLibraryPublicShapeTest::signature)
                .collect(Collectors.toUnmodifiableSet());
        assertEquals(
                Set.of(
                        "static open(java.lang.String):OpenBlasLibrary",
                        "static open(java.nio.file.Path):OpenBlasLibrary",
                        "isOpen():boolean",
                        "sgemm(int,int,int,float,java.lang.foreign.MemorySegment,java.lang.foreign.MemorySegment,float,java.lang.foreign.MemorySegment):void",
                        "dgemm(int,int,int,double,java.lang.foreign.MemorySegment,java.lang.foreign.MemorySegment,double,java.lang.foreign.MemorySegment):void",
                        "threadCount():int",
                        "setThreadCount(int):void",
                        "close():void"),
                libraryMethods);

        assertEquals(
                Set.of(),
                Arrays.stream(OpenBlasLoadException.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .collect(Collectors.toUnmodifiableSet()));
    }

    /** Locks the exact direct thread-control surface with no overload or result abstraction. */
    @Test
    void threadControlMethodsHaveExactPublicShape() throws ReflectiveOperationException {
        Method getter = OpenBlasLibrary.class.getDeclaredMethod("threadCount");
        Method setter = OpenBlasLibrary.class.getDeclaredMethod("setThreadCount", int.class);
        assertTrue(Modifier.isPublic(getter.getModifiers()));
        assertFalse(Modifier.isStatic(getter.getModifiers()));
        assertEquals(int.class, getter.getReturnType());
        assertTrue(Modifier.isPublic(setter.getModifiers()));
        assertFalse(Modifier.isStatic(setter.getModifiers()));
        assertEquals(void.class, setter.getReturnType());
    }

    /** Locks the exact GEMM carriers and confirms no overload broadens the public call surface. */
    @Test
    void gemmMethodsHaveExactPublicShape() throws ReflectiveOperationException {
        Method sgemm = OpenBlasLibrary.class.getDeclaredMethod(
                "sgemm", int.class, int.class, int.class, float.class,
                MemorySegment.class, MemorySegment.class, float.class, MemorySegment.class);
        Method dgemm = OpenBlasLibrary.class.getDeclaredMethod(
                "dgemm", int.class, int.class, int.class, double.class,
                MemorySegment.class, MemorySegment.class, double.class, MemorySegment.class);

        assertTrue(Modifier.isPublic(sgemm.getModifiers()));
        assertFalse(Modifier.isStatic(sgemm.getModifiers()));
        assertEquals(void.class, sgemm.getReturnType());
        assertTrue(Modifier.isPublic(dgemm.getModifiers()));
        assertFalse(Modifier.isStatic(dgemm.getModifiers()));
        assertEquals(void.class, dgemm.getReturnType());
        assertEquals(2, Arrays.stream(OpenBlasLibrary.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("sgemm") || method.getName().equals("dgemm"))
                .count());
    }

    /** Locks the two package-private overloads used by deterministic native-boundary tests. */
    @Test
    void testSeamOverloadsArePackagePrivateAndExact() throws ReflectiveOperationException {
        Method nameOpen = OpenBlasLibrary.class.getDeclaredMethod(
                "open", String.class, OpenBlasNativeAccess.class);
        Method pathOpen = OpenBlasLibrary.class.getDeclaredMethod(
                "open", Path.class, OpenBlasNativeAccess.class);
        assertFalse(Modifier.isPublic(nameOpen.getModifiers()));
        assertFalse(Modifier.isProtected(nameOpen.getModifiers()));
        assertFalse(Modifier.isPrivate(nameOpen.getModifiers()));
        assertTrue(Modifier.isStatic(nameOpen.getModifiers()));
        assertFalse(Modifier.isPublic(pathOpen.getModifiers()));
        assertFalse(Modifier.isProtected(pathOpen.getModifiers()));
        assertFalse(Modifier.isPrivate(pathOpen.getModifiers()));
        assertTrue(Modifier.isStatic(pathOpen.getModifiers()));
    }

    private static int publicConstructors(Class<?> type) {
        return (int) Arrays.stream(type.getDeclaredConstructors())
                .map(Constructor::getModifiers)
                .filter(Modifier::isPublic)
                .count();
    }

    private static String signature(Method method) {
        String prefix = Modifier.isStatic(method.getModifiers()) ? "static " : "";
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(Collectors.joining(","));
        return prefix + method.getName() + "(" + parameters + "):" + method.getReturnType().getSimpleName();
    }
}
