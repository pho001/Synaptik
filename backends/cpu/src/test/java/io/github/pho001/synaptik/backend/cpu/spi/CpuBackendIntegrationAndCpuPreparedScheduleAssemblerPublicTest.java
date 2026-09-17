package io.github.pho001.synaptik.backend.cpu.spi;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Uses the supported CPU SPI from a distinct package without importing CPU internals. */
final class CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest {
    @Test
    void exposesExactSurfaceAvailabilityAndReusableOnePartitionRecipe() throws Exception {
        assertTrue(Modifier.isPublic(CpuBackendIntegration.class.getModifiers()));
        assertTrue(Modifier.isFinal(CpuBackendIntegration.class.getModifiers()));
        assertEquals(0, Arrays.stream(CpuBackendIntegration.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers())
                        || Modifier.isProtected(constructor.getModifiers())).count());
        assertEquals(List.of("availabilitySnapshot", "borrow", "capabilityProvider", "close",
                        "copyToCanonicalHostBytes", "open", "preparations", "prepare",
                        "scheduleAssembler"),
                Arrays.stream(CpuBackendIntegration.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(method -> method.getName()).sorted().toList());
        Arrays.stream(CpuBackendIntegration.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .forEach(method -> assertFalse(method.toGenericString().contains(".internal.")));

        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            assertSame(CpuCapabilityProvider.CPU_BACKEND_ID,
                    integration.capabilityProvider().backendId());
            assertEquals(new BackendAvailabilitySnapshot(CpuCapabilityProvider.CPU_BACKEND_ID,
                            Map.of(new BackendDeviceId(CpuCapabilityProvider.CPU_BACKEND_ID, "host"),
                                    DeviceClass.CPU)), integration.availabilitySnapshot());

            CompileArtifacts artifacts = cpuArtifacts(integration);
            PreparedMemoryPlan memoryPlan = memoryPlan(artifacts);
            List<PreparedBufferAssignment> assignments = assignments(artifacts, memoryPlan);
            PreparedExecutable knownExecutable = new EmptyExecutable(memoryPlan);
            PreparedScheduleAssembler assembler = integration.scheduleAssembler();
            PreparedSchedule schedule = assembler.assemble(new PreparedScheduleContext(
                    artifacts, memoryPlan,
                    List.of(new PreparedPartition(
                            artifacts.partitions().getFirst(), knownExecutable)),
                    assignments));
            assertSame(memoryPlan, schedule.memoryPlan());
            assertSame(memoryPlan, assertInstanceOf(
                    PreparedSchedule.RepresentationCreationStep.class,
                    schedule.steps().getFirst()).representationPlan().memoryPlan());
            assertSame(knownExecutable, assertInstanceOf(PreparedSchedule.ExecutionStep.class,
                    schedule.steps().get(1)).executable());

            var publishedValues = new java.util.ArrayList<ValueId>();
            artifacts.publication().forwardBindings()
                    .forEach(binding -> publishedValues.add(binding.valueId()));
            artifacts.publication().gradientBindings()
                    .forEach(binding -> publishedValues.add(binding.valueId()));
            assertEquals(2 + publishedValues.size(), schedule.steps().size());
            for (int resultIndex = 0; resultIndex < publishedValues.size(); resultIndex++) {
                int expectedResultIndex = resultIndex;
                var publication = assertInstanceOf(PreparedSchedule.PublicationStep.class,
                        schedule.steps().get(resultIndex + 2)).publication();
                ValueId publishedValue = publishedValues.get(resultIndex);
                int expectedBufferIndex = assignments.stream()
                        .filter(assignment -> assignment.valueId().equals(publishedValue))
                        .findFirst().orElseThrow().planIndex();
                assertAll(
                        () -> assertSame(memoryPlan, publication.memoryPlan()),
                        () -> assertEquals(expectedBufferIndex, publication.bufferIndex()),
                        () -> assertEquals(0, publication.representationIndex()),
                        () -> assertEquals(expectedResultIndex, publication.resultIndex()));
            }

            var execution = integration.prepare(artifacts);
            assertSame(execution.memoryPlan(), execution.schedule().memoryPlan());
            assertInstanceOf(PreparedSchedule.RepresentationCreationStep.class,
                    execution.schedule().steps().getFirst());
            assertInstanceOf(PreparedSchedule.ExecutionStep.class,
                    execution.schedule().steps().get(1));
            assertInstanceOf(PreparedSchedule.PublicationStep.class,
                    execution.schedule().steps().getLast());

            try (Arena firstArena = Arena.ofShared(); Arena secondArena = Arena.ofShared();
                    var first = integration.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4,
                            firstArena.allocate(16, 4)));
                    var second = integration.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4,
                            secondArena.allocate(16, 4)));
                    var executor = Executors.newFixedThreadPool(2)) {
                var runner = new PreparedExecutionRunner();
                var firstRun = executor.submit(() -> runner.run(execution, List.of(first)));
                var secondRun = executor.submit(() -> runner.run(execution, List.of(second)));
                try (var firstResult = firstRun.get(); var secondResult = secondRun.get()) {
                    assertNotSame(firstResult, secondResult);
                }
            }
        }
    }

    @Test
    void rejectsUnsupportedArtifactShapesBeforePreparation() {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            CompileArtifacts zero = compile(List.of(leaf()), List.of(integration.capabilityProvider()),
                    List.of(integration.availabilitySnapshot()));
            assertTrue(zero.partitions().isEmpty());
            assertThrows(IllegalArgumentException.class, () -> integration.preparations(zero));
            assertThrows(IllegalArgumentException.class, () -> integration.prepare(zero));
            PreparedMemoryPlan empty = new PreparedMemoryPlan(List.of(), List.of());
            assertThrows(IllegalArgumentException.class, () -> integration.scheduleAssembler()
                    .assemble(new PreparedScheduleContext(zero, empty, List.of(), List.of())));

            BackendId other = new BackendId("other");
            BackendCapabilityProvider provider = provider(other);
            CompileArtifacts nonCpu = compile(List.of(leaf().add(leaf())), List.of(provider),
                    List.of(snapshot(other)));
            assertThrows(IllegalArgumentException.class, () -> integration.preparations(nonCpu));
            assertThrows(IllegalArgumentException.class, () -> integration.prepare(nonCpu));
            assertThrows(IllegalArgumentException.class, () -> integration.scheduleAssembler()
                    .assemble(context(nonCpu, empty)));

            Tensor input = leaf();
            CompileArtifacts mixed = compile(List.of(input.neg().abs()), List.of(
                            selectiveProvider(CpuCapabilityProvider.CPU_BACKEND_ID,
                                    query -> query.operation().kind()
                                            == io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.NEG),
                            selectiveProvider(other, query -> query.operation().kind()
                                    == io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.ABS)),
                    List.of(snapshot(CpuCapabilityProvider.CPU_BACKEND_ID), snapshot(other)));
            assertEquals(2, mixed.partitions().size());
            assertThrows(IllegalArgumentException.class, () -> integration.preparations(mixed));
            assertThrows(IllegalArgumentException.class, () -> integration.prepare(mixed));
            assertThrows(IllegalArgumentException.class, () -> integration.scheduleAssembler()
                    .assemble(context(mixed, empty)));
        }
    }

    @Test
    void borrowValidatesIntrinsicFactsButAcceptsReadOnlyAndWritableStorage() throws Exception {
        CpuBackendIntegration integration = CpuBackendIntegration.open();
        CompileArtifacts artifacts = cpuArtifacts(integration);
        try {
            float[] writable = {1, 2};
            try (var borrowed = integration.borrow(new MemorySegmentStorage(DataType.FLOAT32, 2,
                    MemorySegment.ofArray(writable)));
                    var readOnly = integration.borrow(new MemorySegmentStorage(DataType.FLOAT32, 2,
                            MemorySegment.ofArray(writable).asReadOnly()))) {
                assertNotNull(borrowed);
                assertNotNull(readOnly);
            }
            assertThrows(IllegalArgumentException.class, () -> integration.borrow(
                    new MemorySegmentStorage(DataType.FLOAT32, 2,
                            MemorySegment.ofArray(new int[] {1, 2}))));

            MemorySegmentStorage dead;
            Arena arena = Arena.ofConfined();
            dead = new MemorySegmentStorage(DataType.INT64, 1, arena.allocate(8, 8));
            arena.close();
            assertThrows(IllegalStateException.class, () -> integration.borrow(dead));

            try (Arena confined = Arena.ofConfined(); var executor = Executors.newSingleThreadExecutor()) {
                var storage = new MemorySegmentStorage(DataType.INT32, 1, confined.allocate(4, 4));
                assertInstanceOf(IllegalStateException.class,
                        assertThrows(java.util.concurrent.ExecutionException.class,
                                () -> executor.submit(() -> integration.borrow(storage)).get())
                                .getCause());
            }
        } finally {
            closeConcurrently(integration);
        }
        assertThrows(IllegalStateException.class, () -> integration.borrow(
                new MemorySegmentStorage(DataType.BOOL, 1, MemorySegment.ofArray(new byte[1]))));
        assertThrows(IllegalStateException.class, () -> integration.preparations(artifacts));
        assertThrows(IllegalStateException.class, () -> integration.prepare(null));
        assertThrows(IllegalStateException.class, integration::scheduleAssembler);
    }

    private static CompileArtifacts cpuArtifacts(CpuBackendIntegration integration) {
        Tensor input = leaf();
        return compile(List.of(input.contiguous()), List.of(integration.capabilityProvider()),
                List.of(integration.availabilitySnapshot()));
    }

    private static Tensor leaf() {
        Shape shape = Shape.of(4);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static CompileArtifacts compile(List<Tensor> outputs,
            List<BackendCapabilityProvider> providers,
            List<BackendAvailabilitySnapshot> availability) {
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, outputs, Optional.empty(),
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), providers, availability);
    }

    private static BackendCapabilityProvider provider(BackendId id) {
        return selectiveProvider(id, ignored -> true);
    }

    private static BackendCapabilityProvider selectiveProvider(BackendId id,
            java.util.function.Predicate<
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery> support) {
        return new BackendCapabilityProvider() {
            @Override public BackendId backendId() { return id; }
            @Override public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                return support.test(query);
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot(BackendId id) {
        return new BackendAvailabilitySnapshot(id,
                Map.of(new BackendDeviceId(id, "host"), DeviceClass.CPU));
    }

    private static PreparedScheduleContext context(CompileArtifacts artifacts,
            PreparedMemoryPlan memoryPlan) {
        return new PreparedScheduleContext(artifacts, memoryPlan,
                artifacts.partitions().stream()
                        .map(partition -> new PreparedPartition(partition,
                                new EmptyExecutable(memoryPlan)))
                        .toList(), List.of());
    }

    private static PreparedMemoryPlan memoryPlan(CompileArtifacts artifacts) {
        var entries = new java.util.ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < artifacts.graph().values().size(); index++) {
            var descriptor = artifacts.graph().values().get(index).descriptor();
            long byteSize = Math.multiplyExact(
                    descriptor.shape().knownElementCount().orElseThrow(),
                    descriptor.dataType().byteWidth());
            entries.add(new PreparedMemoryPlan.BufferEntry(
                    new BufferSlot(index), byteSize, descriptor.dataType().byteWidth()));
        }
        return new PreparedMemoryPlan(entries, List.of());
    }

    private static List<PreparedBufferAssignment> assignments(
            CompileArtifacts artifacts, PreparedMemoryPlan memoryPlan) {
        var assignments = new java.util.ArrayList<PreparedBufferAssignment>();
        for (int index = 0; index < artifacts.graph().values().size(); index++) {
            assignments.add(new PreparedBufferAssignment(
                    artifacts.graph().values().get(index).id(),
                    memoryPlan.buffers().get(index).slot(), index));
        }
        return List.copyOf(assignments);
    }

    private static void closeConcurrently(CpuBackendIntegration integration) throws Exception {
        var ready = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var firstClose = executor.submit(() -> {
                ready.countDown();
                release.await();
                integration.close();
                return null;
            });
            var secondClose = executor.submit(() -> {
                ready.countDown();
                release.await();
                integration.close();
                return null;
            });
            try {
                assertTrue(ready.await(10, TimeUnit.SECONDS));
            } finally {
                release.countDown();
            }
            firstClose.get(10, TimeUnit.SECONDS);
            secondClose.get(10, TimeUnit.SECONDS);
        }
    }

    private static final class EmptyExecutable extends PreparedExecutable {
        EmptyExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
        }
        @Override protected boolean acceptsBufferRepresentation(int selectionIndex,
                io.github.pho001.synaptik.runtime.resource.BufferRepresentation representation) {
            return false;
        }
        @Override protected boolean acceptsWorkspaceRepresentation(int selectionIndex,
                io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation representation) {
            return false;
        }
        @Override protected BoundInvocation bindCompatible(RunState runState,
                io.github.pho001.synaptik.runtime.resource.BufferRepresentation[] buffers,
                io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation[] workspaces) {
            return new BoundInvocation(runState) {
                @Override protected void executeBound() { }
            };
        }
    }
}
