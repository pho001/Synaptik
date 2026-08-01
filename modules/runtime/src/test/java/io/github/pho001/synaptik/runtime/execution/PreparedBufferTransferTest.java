package io.github.pho001.synaptik.runtime.execution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class PreparedBufferTransferTest {
    @Test
    void exposesExactAbstractSurface() throws ReflectiveOperationException {
        Class<PreparedBufferTransfer> type = PreparedBufferTransfer.class;
        var fields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        var constructor = type.getDeclaredConstructors()[0];

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.execution", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(type.getModifiers())),
                () -> assertFalse(type.isInterface()),
                () -> assertEquals(Object.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isProtected(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, int.class, int.class, int.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(
                        List.of(
                                "memoryPlan",
                                "bufferIndex",
                                "sourceRepresentationIndex",
                                "destinationRepresentationIndex"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "acceptsDestinationBufferRepresentation",
                                "acceptsSourceBufferRepresentation",
                                "bind",
                                "bindCompatible",
                                "bufferIndex",
                                "destinationRepresentationIndex",
                                "memoryPlan",
                                "sourceRepresentationIndex"),
                        declaredMethodNames(type)));

        for (String methodName : List.of(
                "memoryPlan",
                "bufferIndex",
                "sourceRepresentationIndex",
                "destinationRepresentationIndex",
                "bind")) {
            Method method = Arrays.stream(type.getDeclaredMethods())
                    .filter(candidate -> candidate.getName().equals(methodName))
                    .findFirst()
                    .orElseThrow();
            assertTrue(Modifier.isPublic(method.getModifiers()));
            assertTrue(Modifier.isFinal(method.getModifiers()));
        }
        for (String methodName : List.of(
                "acceptsSourceBufferRepresentation",
                "acceptsDestinationBufferRepresentation",
                "bindCompatible")) {
            Method method = Arrays.stream(type.getDeclaredMethods())
                    .filter(candidate -> candidate.getName().equals(methodName))
                    .findFirst()
                    .orElseThrow();
            assertTrue(Modifier.isProtected(method.getModifiers()));
            assertTrue(Modifier.isAbstract(method.getModifiers()));
        }
    }

    @Test
    void constructionValidatesInOrderAndRetainsExactCoordinatesWithoutAction() {
        PreparedMemoryPlan empty = plan(0);
        PreparedMemoryPlan plan = plan(1);
        TestTransfer transfer = new TestTransfer(plan, 0, 2, 1);

        assertAll(
                () -> assertFailure(NullPointerException.class, "memoryPlan",
                        () -> new TestTransfer(null, -1, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bufferIndex must be non-negative",
                        () -> new TestTransfer(plan, -1, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bufferIndex out of prepared-plan range: 0",
                        () -> new TestTransfer(empty, 0, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "sourceRepresentationIndex must be non-negative",
                        () -> new TestTransfer(plan, 0, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "destinationRepresentationIndex must be non-negative",
                        () -> new TestTransfer(plan, 0, 0, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "sourceRepresentationIndex and destinationRepresentationIndex must be "
                                + "distinct",
                        () -> new TestTransfer(plan, 0, 1, 1)),
                () -> assertSame(plan, transfer.memoryPlan()),
                () -> assertEquals(0, transfer.bufferIndex()),
                () -> assertEquals(2, transfer.sourceRepresentationIndex()),
                () -> assertEquals(1, transfer.destinationRepresentationIndex()),
                () -> assertEquals(List.of(), transfer.events));
    }

    @Test
    void bindValidatesStateAndBothRangesBeforeCompatibility() {
        PreparedMemoryPlan plan = plan(1);
        PreparedMemoryPlan foreignPlan = plan(1);
        TestTransfer transfer = new TestTransfer(plan, 0, 0, 2);
        RunState state = state(plan, new CompatibleBuffer(), new CompatibleBuffer());
        RunState foreign = state(foreignPlan, new CompatibleBuffer(), new CompatibleBuffer());
        state.close();

        assertAll(
                () -> assertFailure(
                        NullPointerException.class, "runState", () -> transfer.bind(null)),
                () -> assertFailure(IllegalStateException.class, "run state is closed",
                        () -> transfer.bind(state)),
                () -> assertFailure(IllegalArgumentException.class,
                        "run state memory plan does not match prepared buffer transfer memory plan",
                        () -> transfer.bind(foreign)),
                () -> assertEquals(List.of(), transfer.events));

        TestTransfer sourceRange = new TestTransfer(plan, 0, 2, 1);
        TestTransfer destinationRange = new TestTransfer(plan, 0, 0, 2);
        RunState matching = state(plan, new CompatibleBuffer(), new CompatibleBuffer());
        assertAll(
                () -> assertFailure(IllegalArgumentException.class,
                        "sourceRepresentationIndex out of run-state range: 2",
                        () -> sourceRange.bind(matching)),
                () -> assertFailure(IllegalArgumentException.class,
                        "destinationRepresentationIndex out of run-state range: 2",
                        () -> destinationRange.bind(matching)),
                () -> assertEquals(List.of(), sourceRange.events),
                () -> assertEquals(List.of(), destinationRange.events));
    }

    @Test
    void bindingChecksSourceThenDestinationAndSuppliesExactReferencesOnce() {
        PreparedMemoryPlan plan = plan(1);
        CompatibleBuffer source = new CompatibleBuffer();
        CompatibleBuffer destination = new CompatibleBuffer();
        RunState state = state(plan, source, destination);
        TestTransfer transfer = new TestTransfer(plan, 0, 0, 1);

        TestBound bound = (TestBound) transfer.bind(state);

        assertAll(
                () -> assertEquals(List.of("source", "destination", "bind"), transfer.events),
                () -> assertSame(source, transfer.source),
                () -> assertSame(destination, transfer.destination),
                () -> assertSame(state, bound.runState()),
                () -> assertEquals(0, bound.bufferIndex()),
                () -> assertEquals(0, bound.sourceRepresentationIndex()),
                () -> assertEquals(1, bound.destinationRepresentationIndex()),
                () -> assertSame(source, bound.source),
                () -> assertSame(destination, bound.destination));
    }

    @Test
    void incompatibilityStopsAtFirstFailedHook() {
        PreparedMemoryPlan plan = plan(1);
        TestTransfer sourceFailure = new TestTransfer(plan, 0, 0, 1);
        RunState wrongSource = state(plan, new OtherBuffer(), new CompatibleBuffer());
        TestTransfer destinationFailure = new TestTransfer(plan, 0, 0, 1);
        RunState wrongDestination = state(plan, new CompatibleBuffer(), new OtherBuffer());

        assertAll(
                () -> assertFailure(IllegalArgumentException.class,
                        "source buffer representation is incompatible with prepared buffer "
                                + "transfer",
                        () -> sourceFailure.bind(wrongSource)),
                () -> assertEquals(List.of("source"), sourceFailure.events),
                () -> assertFailure(IllegalArgumentException.class,
                        "destination buffer representation is incompatible with prepared buffer "
                                + "transfer",
                        () -> destinationFailure.bind(wrongDestination)),
                () -> assertEquals(List.of("source", "destination"), destinationFailure.events));
    }

    @Test
    void rejectsNullForeignOrMismatchedBoundResult() {
        PreparedMemoryPlan plan = plan(1);
        RunState supplied = state(plan, new CompatibleBuffer(), new CompatibleBuffer());
        RunState foreign = state(plan, new CompatibleBuffer(), new CompatibleBuffer());

        ReturningTransfer nullTransfer = new ReturningTransfer(plan, null);
        ReturningTransfer foreignTransfer =
                new ReturningTransfer(plan, new EmptyBound(foreign, 0, 0, 1));
        ReturningTransfer positionTransfer =
                new ReturningTransfer(plan, new EmptyBound(supplied, 0, 1, 0));

        assertAll(
                () -> assertFailure(NullPointerException.class, "boundBufferTransfer",
                        () -> nullTransfer.bind(supplied)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bound buffer transfer does not belong to supplied run state",
                        () -> foreignTransfer.bind(supplied)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bound buffer transfer does not match prepared buffer transfer positions",
                        () -> positionTransfer.bind(supplied)));
    }

    @Test
    void immutableRecipeBindsConcurrentlyToDistinctStates() throws Exception {
        PreparedMemoryPlan plan = plan(1);
        ImmutableTransfer transfer = new ImmutableTransfer(plan);
        CompatibleBuffer firstSource = new CompatibleBuffer();
        CompatibleBuffer firstDestination = new CompatibleBuffer();
        CompatibleBuffer secondSource = new CompatibleBuffer();
        CompatibleBuffer secondDestination = new CompatibleBuffer();
        RunState firstState = state(plan, firstSource, firstDestination);
        RunState secondState = state(plan, secondSource, secondDestination);

        TestBound first;
        TestBound second;
        try (var executor = Executors.newFixedThreadPool(2)) {
            var firstBind = executor.submit(() -> (TestBound) transfer.bind(firstState));
            var secondBind = executor.submit(() -> (TestBound) transfer.bind(secondState));
            first = firstBind.get();
            second = secondBind.get();
        }

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertSame(firstState, first.runState()),
                () -> assertSame(secondState, second.runState()),
                () -> assertSame(firstSource, first.source),
                () -> assertSame(firstDestination, first.destination),
                () -> assertSame(secondSource, second.source),
                () -> assertSame(secondDestination, second.destination));
    }

    @Test
    void compiledContractContainsNoForbiddenMechanisms() throws Exception {
        String compiled = classBytes(PreparedBufferTransfer.class);
        assertAll(
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(
                        AutoCloseable.class.isAssignableFrom(PreparedBufferTransfer.class)));
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

    private static PreparedMemoryPlan plan(int count) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < count; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 4, 1));
        }
        return new PreparedMemoryPlan(buffers, List.of());
    }

    private static RunState state(
            PreparedMemoryPlan plan, BufferRepresentation... representations) {
        return new RunState(
                plan,
                List.of(Arrays.stream(representations)
                        .map(representation -> new BufferRepresentationBinding(
                                representation, RunResourceOwnership.RUN_OWNED))
                        .toList()),
                List.of());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static class TestTransfer extends PreparedBufferTransfer {
        private final List<String> events = new ArrayList<>();
        private BufferRepresentation source;
        private BufferRepresentation destination;

        private TestTransfer(PreparedMemoryPlan plan, int buffer, int source, int destination) {
            super(plan, buffer, source, destination);
        }

        @Override
        protected boolean acceptsSourceBufferRepresentation(BufferRepresentation representation) {
            events.add("source");
            source = representation;
            return representation instanceof CompatibleBuffer;
        }

        @Override
        protected boolean acceptsDestinationBufferRepresentation(
                BufferRepresentation representation) {
            events.add("destination");
            destination = representation;
            return representation instanceof CompatibleBuffer;
        }

        @Override
        protected BoundBufferTransfer bindCompatible(
                RunState runState,
                BufferRepresentation sourceRepresentation,
                BufferRepresentation destinationRepresentation) {
            events.add("bind");
            return new TestBound(
                    runState,
                    bufferIndex(),
                    sourceRepresentationIndex(),
                    destinationRepresentationIndex(),
                    (CompatibleBuffer) sourceRepresentation,
                    (CompatibleBuffer) destinationRepresentation);
        }
    }

    private static final class ImmutableTransfer extends PreparedBufferTransfer {
        private ImmutableTransfer(PreparedMemoryPlan plan) {
            super(plan, 0, 0, 1);
        }

        @Override
        protected boolean acceptsSourceBufferRepresentation(BufferRepresentation representation) {
            return representation instanceof CompatibleBuffer;
        }

        @Override
        protected boolean acceptsDestinationBufferRepresentation(
                BufferRepresentation representation) {
            return representation instanceof CompatibleBuffer;
        }

        @Override
        protected BoundBufferTransfer bindCompatible(
                RunState runState,
                BufferRepresentation sourceRepresentation,
                BufferRepresentation destinationRepresentation) {
            return new TestBound(
                    runState, 0, 0, 1,
                    (CompatibleBuffer) sourceRepresentation,
                    (CompatibleBuffer) destinationRepresentation);
        }
    }

    private static final class ReturningTransfer extends PreparedBufferTransfer {
        private final BoundBufferTransfer result;

        private ReturningTransfer(PreparedMemoryPlan plan, BoundBufferTransfer result) {
            super(plan, 0, 0, 1);
            this.result = result;
        }

        @Override
        protected boolean acceptsSourceBufferRepresentation(BufferRepresentation representation) {
            return true;
        }

        @Override
        protected boolean acceptsDestinationBufferRepresentation(
                BufferRepresentation representation) {
            return true;
        }

        @Override
        protected BoundBufferTransfer bindCompatible(
                RunState runState,
                BufferRepresentation sourceRepresentation,
                BufferRepresentation destinationRepresentation) {
            return result;
        }
    }

    private static class TestBound extends BoundBufferTransfer {
        private final CompatibleBuffer source;
        private final CompatibleBuffer destination;

        private TestBound(
                RunState state,
                int buffer,
                int sourceIndex,
                int destinationIndex,
                CompatibleBuffer source,
                CompatibleBuffer destination) {
            super(state, buffer, sourceIndex, destinationIndex);
            this.source = source;
            this.destination = destination;
        }

        @Override
        protected void executeTransfer() {}
    }

    private static final class EmptyBound extends BoundBufferTransfer {
        private EmptyBound(RunState state, int buffer, int source, int destination) {
            super(state, buffer, source, destination);
        }

        @Override
        protected void executeTransfer() {}
    }

    private static final class CompatibleBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }

    private static final class OtherBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }
}
