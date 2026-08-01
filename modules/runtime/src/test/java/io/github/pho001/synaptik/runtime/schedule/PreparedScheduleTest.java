package io.github.pho001.synaptik.runtime.schedule;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class PreparedScheduleTest {
    @Test
    void exposesExactTopLevelRecordSurface() throws ReflectiveOperationException {
        Class<PreparedSchedule> type = PreparedSchedule.class;
        var components = type.getRecordComponents();
        var constructor = type.getDeclaredConstructors()[0];
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        var stepsType = (ParameterizedType) components[1].getGenericType();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.schedule", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(Record.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            PreparedSchedule.Step.class, PreparedSchedule.ExecutionStep.class
                        },
                        type.getDeclaredClasses()),
                () -> assertEquals(2, components.length),
                () -> assertEquals(
                        List.of("memoryPlan", "steps"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, List.class},
                        Arrays.stream(components).map(component -> component.getType())
                                .toArray(Class<?>[]::new)),
                () -> assertEquals(List.class, stepsType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {PreparedSchedule.Step.class},
                        stepsType.getActualTypeArguments()),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, List.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(List.of("memoryPlan", "steps"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        List.of("equals", "hashCode", "memoryPlan", "steps", "toString"),
                        declaredMethodNames(type)));
    }

    @Test
    void exposesExactClosedStepAndExecutionStepSurface() throws ReflectiveOperationException {
        Class<PreparedSchedule.Step> step = PreparedSchedule.Step.class;
        Class<PreparedSchedule.ExecutionStep> execution = PreparedSchedule.ExecutionStep.class;
        Method stepMemoryPlan = step.getDeclaredMethod("memoryPlan");
        Method executionMemoryPlan = execution.getDeclaredMethod("memoryPlan");
        var component = execution.getRecordComponents()[0];

        assertAll(
                () -> assertTrue(Modifier.isPublic(step.getModifiers())),
                () -> assertTrue(Modifier.isStatic(step.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(step.getModifiers())),
                () -> assertTrue(step.isInterface()),
                () -> assertTrue(step.isSealed()),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedSchedule.ExecutionStep.class},
                        step.getPermittedSubclasses()),
                () -> assertEquals(1, step.getDeclaredMethods().length),
                () -> assertEquals(PreparedMemoryPlan.class, stepMemoryPlan.getReturnType()),
                () -> assertTrue(Modifier.isPublic(stepMemoryPlan.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(stepMemoryPlan.getModifiers())),
                () -> assertTrue(Modifier.isPublic(execution.getModifiers())),
                () -> assertTrue(Modifier.isStatic(execution.getModifiers())),
                () -> assertTrue(Modifier.isFinal(execution.getModifiers())),
                () -> assertTrue(execution.isRecord()),
                () -> assertArrayEquals(new Class<?>[] {step}, execution.getInterfaces()),
                () -> assertEquals(1, execution.getRecordComponents().length),
                () -> assertEquals("executable", component.getName()),
                () -> assertEquals(PreparedExecutable.class, component.getType()),
                () -> assertEquals(1, execution.getDeclaredConstructors().length),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedExecutable.class},
                        execution.getDeclaredConstructors()[0].getParameterTypes()),
                () -> assertEquals(
                        List.of("equals", "executable", "hashCode", "memoryPlan", "toString"),
                        declaredMethodNames(execution)),
                () -> assertEquals(
                        PreparedMemoryPlan.class, executionMemoryPlan.getReturnType()));
    }

    @Test
    void executionStepRequiresAndRetainsExactExecutableAndPlanReferences() {
        PreparedMemoryPlan plan = plan(1);
        TestExecutable executable = new TestExecutable(plan);
        PreparedSchedule.ExecutionStep step = new PreparedSchedule.ExecutionStep(executable);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "executable",
                        () -> new PreparedSchedule.ExecutionStep(null)),
                () -> assertSame(executable, step.executable()),
                () -> assertSame(plan, step.memoryPlan()),
                () -> assertSame(executable.memoryPlan(), step.memoryPlan()));
    }

    @Test
    void scheduleRequiresTopLevelInputsBeforeScanningSteps() {
        PreparedMemoryPlan plan = plan(0);
        var nullStep = new ArrayList<PreparedSchedule.Step>();
        nullStep.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedSchedule(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "steps",
                        () -> new PreparedSchedule(plan, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedSchedule(null, nullStep)));
    }

    @Test
    void rejectsFirstInvalidOccurrenceInEncounterOrderWithExactMessages() {
        PreparedMemoryPlan plan = plan(0);
        PreparedMemoryPlan foreign = plan(0);
        var valid = new PreparedSchedule.ExecutionStep(new TestExecutable(plan));
        var mismatch = new PreparedSchedule.ExecutionStep(new TestExecutable(foreign));
        var nullFirst = new ArrayList<PreparedSchedule.Step>();
        nullFirst.add(valid);
        nullFirst.add(null);
        nullFirst.add(mismatch);
        var mismatchFirst = new ArrayList<PreparedSchedule.Step>();
        mismatchFirst.add(valid);
        mismatchFirst.add(mismatch);
        mismatchFirst.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "steps[1]",
                        () -> new PreparedSchedule(plan, nullFirst)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "steps[1] memory plan does not match schedule memory plan",
                        () -> new PreparedSchedule(plan, mismatchFirst)));
    }

    @Test
    void planAssociationUsesReferenceIdentityRatherThanValueEquality() {
        PreparedMemoryPlan plan = plan(1);
        PreparedMemoryPlan equalPlan = plan(1);
        var foreignStep = new PreparedSchedule.ExecutionStep(new TestExecutable(equalPlan));

        assertAll(
                () -> assertEquals(plan, equalPlan),
                () -> assertNotSame(plan, equalPlan),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "steps[0] memory plan does not match schedule memory plan",
                        () -> new PreparedSchedule(plan, List.of(foreignStep))));
    }

    @Test
    void snapshotsListStructureAndRetainsExactStepsInDeterministicOrder() {
        PreparedMemoryPlan plan = plan(0);
        var first = new PreparedSchedule.ExecutionStep(new TestExecutable(plan));
        var second = new PreparedSchedule.ExecutionStep(new TestExecutable(plan));
        var supplied = new ArrayList<PreparedSchedule.Step>(List.of(second, first));

        PreparedSchedule schedule = new PreparedSchedule(plan, supplied);
        supplied.clear();

        assertAll(
                () -> assertSame(plan, schedule.memoryPlan()),
                () -> assertEquals(2, schedule.steps().size()),
                () -> assertSame(second, schedule.steps().get(0)),
                () -> assertSame(first, schedule.steps().get(1)),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> schedule.steps().add(first)));
    }

    @Test
    void acceptsEmptyAndRepeatedStepAndExecutableOccurrences() {
        PreparedMemoryPlan plan = plan(1);
        TestExecutable executable = new TestExecutable(plan);
        var repeated = new PreparedSchedule.ExecutionStep(executable);
        var equalOccurrence = new PreparedSchedule.ExecutionStep(executable);
        PreparedSchedule empty = new PreparedSchedule(plan, List.of());
        PreparedSchedule schedule =
                new PreparedSchedule(plan, List.of(repeated, repeated, equalOccurrence));

        assertAll(
                () -> assertTrue(empty.steps().isEmpty()),
                () -> assertEquals(3, schedule.steps().size()),
                () -> assertSame(repeated, schedule.steps().get(0)),
                () -> assertSame(repeated, schedule.steps().get(1)),
                () -> assertSame(equalOccurrence, schedule.steps().get(2)),
                () -> assertSame(
                        executable,
                        ((PreparedSchedule.ExecutionStep) schedule.steps().get(0)).executable()),
                () -> assertSame(
                        executable,
                        ((PreparedSchedule.ExecutionStep) schedule.steps().get(2)).executable()),
                () -> assertEquals(repeated, equalOccurrence));
    }

    @Test
    void constructionPerformsNoBindingExecutionResourceActionOrOwnershipTransfer() {
        PreparedMemoryPlan plan = plan(0);
        TestExecutable executable = new TestExecutable(plan);
        var step = new PreparedSchedule.ExecutionStep(executable);

        PreparedSchedule first = new PreparedSchedule(plan, List.of(step));
        PreparedSchedule second = new PreparedSchedule(plan, List.of(step, step));

        assertAll(
                () -> assertFalse(AutoCloseable.class.isAssignableFrom(PreparedSchedule.class)),
                () -> assertFalse(AutoCloseable.class.isAssignableFrom(
                        PreparedSchedule.ExecutionStep.class)),
                () -> assertSame(step, first.steps().getFirst()),
                () -> assertSame(step, second.steps().getFirst()),
                () -> assertSame(step, second.steps().getLast()));
    }

    @Test
    void immutableScheduleCanBeTraversedConcurrentlyWithoutMutation() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        var first = new PreparedSchedule.ExecutionStep(new TestExecutable(plan));
        var second = new PreparedSchedule.ExecutionStep(new TestExecutable(plan));
        PreparedSchedule schedule =
                new PreparedSchedule(plan, List.of(first, second, first));

        try (var executor = Executors.newFixedThreadPool(2)) {
            var firstRead = executor.submit(() -> traverse(schedule, first, second));
            var secondRead = executor.submit(() -> traverse(schedule, first, second));
            assertAll(
                    () -> assertEquals(3_000, firstRead.get()),
                    () -> assertEquals(3_000, secondRead.get()));
        }
    }

    @Test
    void compiledContractContainsNoForbiddenUpstreamOrDynamicMechanismReferences()
            throws Exception {
        String scheduleClass = classBytes(PreparedSchedule.class);
        String stepClass = classBytes(PreparedSchedule.Step.class);
        String executionClass = classBytes(PreparedSchedule.ExecutionStep.class);
        String compiled = scheduleClass + stepClass + executionClass;

        assertAll(
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(Arrays.stream(PreparedSchedule.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("bind")
                                || method.getName().equals("execute")
                                || method.getName().equals("close"))),
                () -> assertFalse(Arrays.stream(PreparedSchedule.Step.class.getDeclaredMethods())
                        .anyMatch(method -> method.getParameterCount() != 0)),
                () -> assertFalse(Arrays.stream(PreparedSchedule.ExecutionStep.class.getDeclaredFields())
                        .anyMatch(field -> field.getType() == Object.class)));
    }

    private static int traverse(
            PreparedSchedule schedule,
            PreparedSchedule.ExecutionStep first,
            PreparedSchedule.ExecutionStep second) {
        int observed = 0;
        for (int iteration = 0; iteration < 1_000; iteration++) {
            assertSame(first, schedule.steps().get(0));
            assertSame(second, schedule.steps().get(1));
            assertSame(first, schedule.steps().get(2));
            assertSame(schedule.memoryPlan(), first.memoryPlan());
            observed += schedule.steps().size();
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
            buffers.add(
                    new PreparedMemoryPlan.BufferEntry(
                            new BufferSlot(500L - index), index, 1L));
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
            throw new AssertionError("schedule construction must not inspect buffers");
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            throw new AssertionError("schedule construction must not inspect workspaces");
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new AssertionError("schedule construction must not bind executables");
        }
    }
}
