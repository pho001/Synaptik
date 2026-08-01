package io.github.pho001.synaptik.runtime.execution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class PreparedExecutionTest {
    @Test
    void exposesExactPublicFinalRecordSurface() {
        Class<PreparedExecution> type = PreparedExecution.class;
        var components = type.getRecordComponents();
        var constructor = type.getDeclaredConstructors()[0];
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.execution", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(Record.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(2, components.length),
                () -> assertEquals(
                        List.of("memoryPlan", "schedule"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, PreparedSchedule.class},
                        Arrays.stream(components).map(component -> component.getType())
                                .toArray(Class<?>[]::new)),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, PreparedSchedule.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(
                        List.of("memoryPlan", "schedule"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        List.of("equals", "hashCode", "memoryPlan", "schedule", "toString"),
                        declaredMethodNames(type)));
    }

    @Test
    void validatesTopLevelReferencesInComponentOrderWithExactMessages() {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedExecution(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "schedule",
                        () -> new PreparedExecution(plan, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedExecution(null, schedule)));
    }

    @Test
    void planAssociationUsesReferenceIdentityRatherThanStructuralEquality() {
        PreparedMemoryPlan plan = plan(1);
        PreparedMemoryPlan equalPlan = plan(1);
        PreparedSchedule foreignSchedule = new PreparedSchedule(equalPlan, List.of());

        assertAll(
                () -> assertEquals(plan, equalPlan),
                () -> assertNotSame(plan, equalPlan),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "schedule memory plan does not match prepared execution memory plan",
                        () -> new PreparedExecution(plan, foreignSchedule)));
    }

    @Test
    void retainsExactPlanAndEmptyOrNonEmptyScheduleReferences() {
        PreparedMemoryPlan emptyPlan = plan(0);
        PreparedSchedule emptySchedule = new PreparedSchedule(emptyPlan, List.of());
        PreparedExecution empty = new PreparedExecution(emptyPlan, emptySchedule);

        PreparedMemoryPlan nonEmptyPlan = plan(1);
        var executable = new TestExecutable(nonEmptyPlan);
        PreparedSchedule nonEmptySchedule = new PreparedSchedule(
                nonEmptyPlan,
                List.of(new PreparedSchedule.ExecutionStep(executable)));
        PreparedExecution nonEmpty = new PreparedExecution(nonEmptyPlan, nonEmptySchedule);

        assertAll(
                () -> assertSame(emptyPlan, empty.memoryPlan()),
                () -> assertSame(emptySchedule, empty.schedule()),
                () -> assertTrue(empty.schedule().steps().isEmpty()),
                () -> assertSame(nonEmptyPlan, nonEmpty.memoryPlan()),
                () -> assertSame(nonEmptySchedule, nonEmpty.schedule()),
                () -> assertSame(
                        executable,
                        ((PreparedSchedule.ExecutionStep) nonEmpty.schedule().steps().getFirst())
                                .executable()));
    }

    @Test
    void preservesOrdinaryRecordEqualityHashingAndDiagnosticText() {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        PreparedExecution first = new PreparedExecution(plan, schedule);
        PreparedExecution equal = new PreparedExecution(plan, schedule);
        PreparedExecution differentSchedule = new PreparedExecution(
                plan,
                new PreparedSchedule(
                        plan,
                        List.of(new PreparedSchedule.ExecutionStep(new TestExecutable(plan)))));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentSchedule),
                () -> assertEquals(
                        "PreparedExecution[memoryPlan=PreparedMemoryPlan[buffers=[], workspaces=[]], "
                                + "schedule=PreparedSchedule[memoryPlan=PreparedMemoryPlan[buffers=[], "
                                + "workspaces=[]], steps=[]]]",
                        first.toString()));
    }

    @Test
    void immutableRecipeSupportsConcurrentReadersWithoutPerRunMutation() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        PreparedExecution execution = new PreparedExecution(plan, schedule);

        try (var readers = Executors.newFixedThreadPool(2)) {
            var first = readers.submit(() -> read(execution, plan, schedule));
            var second = readers.submit(() -> read(execution, plan, schedule));
            assertAll(
                    () -> assertEquals(2_000, first.get()),
                    () -> assertEquals(2_000, second.get()));
        }
    }

    @Test
    void constructionPerformsNoLifecycleBindingExecutionOrResourceAction() {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());

        PreparedExecution first = new PreparedExecution(plan, schedule);
        PreparedExecution second = new PreparedExecution(plan, schedule);

        assertAll(
                () -> assertFalse(AutoCloseable.class.isAssignableFrom(PreparedExecution.class)),
                () -> assertSame(plan, first.memoryPlan()),
                () -> assertSame(schedule, first.schedule()),
                () -> assertSame(plan, second.memoryPlan()),
                () -> assertSame(schedule, second.schedule()));
    }

    @Test
    void compiledContractContainsNoForbiddenUpstreamOrDynamicMechanismReferences()
            throws Exception {
        String compiled = classBytes(PreparedExecution.class);

        assertAll(
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/config")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/util/List")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(compiled.contains("java/lang/AutoCloseable")),
                () -> assertFalse(Arrays.stream(PreparedExecution.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("run")
                                || method.getName().equals("close")
                                || method.getName().equals("bind")
                                || method.getName().equals("execute"))),
                () -> assertTrue(Arrays.stream(PreparedExecution.class.getDeclaredFields())
                        .noneMatch(field -> field.getType() == Object.class
                                || field.getType().isArray())));
    }

    private static int read(
            PreparedExecution execution,
            PreparedMemoryPlan plan,
            PreparedSchedule schedule) {
        int observed = 0;
        for (int iteration = 0; iteration < 1_000; iteration++) {
            assertSame(plan, execution.memoryPlan());
            assertSame(schedule, execution.schedule());
            observed += 2;
        }
        return observed;
    }

    private static List<String> declaredMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods()).map(Method::getName).sorted().toList();
    }

    private static String classBytes(Class<?> type) throws Exception {
        String resource = "/" + type.getName().replace('.', '/') + ".class";
        try (InputStream stream = type.getResourceAsStream(resource)) {
            return new String(stream.readAllBytes(), StandardCharsets.ISO_8859_1);
        }
    }

    private static PreparedMemoryPlan plan(int bufferCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(
                    new BufferSlot(100L + index), index, 1L));
        }
        return new PreparedMemoryPlan(buffers, List.of());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            throw new AssertionError("prepared execution construction must not inspect buffers");
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            throw new AssertionError("prepared execution construction must not inspect workspaces");
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new AssertionError("prepared execution construction must not bind executables");
        }
    }
}
