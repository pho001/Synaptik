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
import io.github.pho001.synaptik.model.datatype.ScalarValue;
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
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.RunMemoryGeometry;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
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
        assertEquals(List.of("acceptsStorageLayoutTransfer", "availabilitySnapshot",
                        "bindStorageLayoutTransfer", "borrow", "capabilityProvider", "close",
                        "completePlanTuning", "copyToCanonicalHostBytes", "localWorkloadTuning",
                        "open", "partitionPreparation", "scheduleAssembler",
                        "scheduleContributor"),
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
            PreparedSchedule schedule = assembler.assemble(context(
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

            var execution = prepare(integration, artifacts);
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
            assertThrows(IllegalArgumentException.class, () -> prepare(integration, zero));
            PreparedMemoryPlan empty = new PreparedMemoryPlan(List.of(), List.of());
            assertThrows(IllegalArgumentException.class, () -> integration.scheduleAssembler()
                    .assemble(context(zero, empty, List.of(), List.of())));

            BackendId other = new BackendId("other");
            BackendCapabilityProvider provider = provider(other);
            CompileArtifacts nonCpu = compile(List.of(leaf().add(leaf())), List.of(provider),
                    List.of(snapshot(other)));
            assertThrows(IllegalArgumentException.class, () -> prepare(integration, nonCpu));
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
            assertThrows(IllegalArgumentException.class, () -> prepare(integration, mixed));
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
        assertThrows(IllegalStateException.class, integration::partitionPreparation);
        assertThrows(IllegalStateException.class, integration::scheduleAssembler);
    }

    @Test
    void storageLayoutTransferPredicateFailsClosedOnByteExtentOverflow() {
        Shape shape = Shape.of(1);
        TensorDescriptor overflow = new TensorDescriptor(
                DataType.FLOAT64,
                shape,
                Optional.of(LayoutDescriptor.of(
                        shape, new long[] {1L}, Long.MAX_VALUE - 1L, true)),
                false);
        try (CpuBackendIntegration integration = CpuBackendIntegration.open();
                Arena arena = Arena.ofConfined();
                var representation = integration.borrow(new MemorySegmentStorage(
                        DataType.FLOAT64, 1L, arena.allocate(Double.BYTES, Double.BYTES)))) {
            assertFalse(integration.acceptsStorageLayoutTransfer(
                    representation, overflow, false));
            assertThrows(IllegalArgumentException.class,
                    () -> integration.bindStorageLayoutTransfer(
                            representation, overflow, false));
        }
    }

    /**
     * Publishes exact low raw bits through real borrowed affine callers alongside ordinary work.
     * Empty views retain zero backing bytes despite their nonzero declared origin.
     */
    @Test
    void publishesBorrowedLowAffineAndEmptySourcesWithoutArithmetic() throws Exception {
        for (DataType type : List.of(DataType.FLOAT16, DataType.BFLOAT16)) {
            for (boolean empty : List.of(false, true)) {
                Shape shape = empty ? Shape.of(0, 2) : Shape.of(2, 2);
                var descriptor = new TensorDescriptor(type, shape, Optional.of(
                        LayoutDescriptor.of(shape, new long[] {5, 2}, 1, true)), false);
                Tensor source = TensorFactory.create(descriptor);
                Tensor work = leaf();
                try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
                    CompileArtifacts artifacts = compile(List.of(source, work.contiguous()),
                            List.of(integration.capabilityProvider()),
                            List.of(integration.availabilitySnapshot()));
                    var sourceBinding = artifacts.constants().bindableInputBindings().stream()
                            .filter(binding -> binding.tensorId().equals(source.id()))
                            .findFirst().orElseThrow();
                    var value = artifacts.graph().values().stream()
                            .filter(candidate -> candidate.id().equals(sourceBinding.valueId()))
                            .findFirst().orElseThrow();
                    var requirement = artifacts.memory().requirements().stream()
                            .filter(candidate -> candidate.valueId().equals(value.id()))
                            .findFirst().orElseThrow();
                    var contribution = integration.scheduleAssembler().producerlessPublishedValue(
                            value, requirement, Optional.empty());
                    assertSame(value, contribution.value());
                    assertSame(requirement, contribution.logicalRequirement());
                    assertEquals(empty ? 0 : 18,
                            contribution.size().fixedByteSize().orElseThrow());
                    ScalarValue scalar = type == DataType.FLOAT16
                            ? ScalarValue.float16Bits((short) 0x8000)
                            : ScalarValue.bfloat16Bits((short) 0x8000);
                    assertThrows(IllegalArgumentException.class,
                            () -> integration.scheduleAssembler().producerlessPublishedValue(
                                    value, requirement, Optional.of(scalar)));

                    var execution = prepare(integration, artifacts);
                    short[] backing = empty ? new short[0] : new short[] {
                        (short) 0x55aa, (short) 0x8000, (short) 0x55aa, (short) 0x0001,
                        (short) 0x55aa, (short) 0x55aa, (short) 0x7d55, (short) 0x55aa,
                        (short) 0xffff
                    };
                    var expected = ByteBuffer.allocate(empty ? 0 : 8)
                            .order(ByteOrder.BIG_ENDIAN);
                    if (!empty) {
                        expected.putShort((short) 0x8000).putShort((short) 0x0001)
                                .putShort((short) 0x7d55).putShort((short) 0xffff);
                    }
                    float[] ordinaryValues = {1.25f, -2f, .5f, 7f};
                    var ordinaryExpected = ByteBuffer.allocate(16).order(ByteOrder.BIG_ENDIAN);
                    for (float ordinaryValue : ordinaryValues) ordinaryExpected.putFloat(ordinaryValue);
                    MemorySegment storage = MemorySegment.ofArray(backing).asReadOnly();
                    try (var borrowed = integration.borrow(new MemorySegmentStorage(
                                type, backing.length, storage));
                            var ordinary = integration.borrow(new MemorySegmentStorage(
                                    DataType.FLOAT32, 4, MemorySegment.ofArray(ordinaryValues)))) {
                        var inputs = artifacts.constants().bindableInputBindings().stream()
                                .map(binding -> binding.tensorId().equals(source.id())
                                        ? borrowed : ordinary).toList();
                        var runner = new PreparedExecutionRunner();
                        for (int run = 0; run < 2; run++) {
                            try (var result = runner.run(execution, inputs)) {
                                assertArrayEquals(expected.array(),
                                        integration.copyToCanonicalHostBytes(
                                                result.publicationRepresentation(0),
                                                descriptor, Long.MAX_VALUE));
                                assertArrayEquals(ordinaryExpected.array(),
                                        integration.copyToCanonicalHostBytes(
                                                result.publicationRepresentation(1),
                                                work.descriptor(), Long.MAX_VALUE));
                            }
                            assertTrue(storage.scope().isAlive());
                            if (!empty) assertEquals((short) 0x8000, backing[1]);
                        }
                    }
                }
            }
        }
    }
    /**
     * Reuses a genuinely fixed CPU unit inside a complete source-driven named-batch plan.
     * A named permanent-empty source still requires explicit B and retains its dynamic marker.
     */
    @Test
    void runsSourceOnlyNamedBatchWithFixedWorkAcrossZeroAndReentry() throws Exception {
        for (int features : new int[] {0, 2}) {
            var namedShape = Shape.ofDimensions(
                    new io.github.pho001.synaptik.model.shape.DynamicDimension("B"),
                    new io.github.pho001.synaptik.model.shape.StaticDimension(features));
            Tensor source = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT16, namedShape, Optional.empty(), false));
            Tensor work = leaf();
            try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
                CompileArtifacts artifacts = compile(List.of(source, work.contiguous()),
                        List.of(integration.capabilityProvider()),
                        List.of(integration.availabilitySnapshot()));
                var execution = prepare(integration, artifacts);
                assertTrue(execution.memoryPlan().requiresBatchSize());
                var sourcePublication = execution.schedule().steps().stream()
                        .filter(PreparedSchedule.PublicationStep.class::isInstance)
                        .map(PreparedSchedule.PublicationStep.class::cast)
                        .map(PreparedSchedule.PublicationStep::publication)
                        .filter(publication -> publication.resultIndex() == 0)
                        .findFirst().orElseThrow();
                assertTrue(execution.memoryPlan().buffers().get(sourcePublication.bufferIndex())
                        .size().fixedByteSize().isEmpty());
                assertThrows(IllegalArgumentException.class, () -> RunMemoryGeometry.resolve(
                        execution.memoryPlan(), OptionalLong.empty()));
                assertThrows(ArithmeticException.class, () -> RunMemoryGeometry.resolve(
                        execution.memoryPlan(), OptionalLong.of((long) Integer.MAX_VALUE + 1)));

                float[] workData = {1.25f, -0.0f, -3.5f, 8.0f};
                var expectedWork = ByteBuffer.allocate(16).order(ByteOrder.BIG_ENDIAN);
                for (float number : workData) expectedWork.putFloat(number);
                try (var ordinary = integration.borrow(new MemorySegmentStorage(
                        DataType.FLOAT32, 4, MemorySegment.ofArray(workData)))) {
                    for (int batch : new int[] {3, 0, 1, 3}) {
                        var geometry = RunMemoryGeometry.resolve(
                                execution.memoryPlan(), OptionalLong.of(batch));
                        short[] sourceData = new short[batch * features];
                        short[] raw = {(short) 0x8000, 1, (short) 0x7d55, (short) 0xffff};
                        var expected = ByteBuffer.allocate(sourceData.length * 2)
                                .order(ByteOrder.BIG_ENDIAN);
                        for (int index = 0; index < sourceData.length; index++) {
                            sourceData[index] = raw[index % raw.length];
                            expected.putShort(sourceData[index]);
                        }
                        Shape actualShape = Shape.of(batch, features);
                        var actualDescriptor = new TensorDescriptor(DataType.FLOAT16, actualShape,
                                Optional.of(LayoutDescriptor.contiguous(actualShape)), false);
                        try (var borrowed = integration.borrow(new MemorySegmentStorage(
                                DataType.FLOAT16, sourceData.length,
                                MemorySegment.ofArray(sourceData).asReadOnly()))) {
                            var inputs = artifacts.constants().bindableInputBindings().stream()
                                    .map(binding -> binding.tensorId().equals(source.id())
                                            ? borrowed : ordinary).toList();
                            try (var result = new PreparedExecutionRunner().run(
                                    execution, inputs, geometry)) {
                                assertArrayEquals(expected.array(),
                                        integration.copyToCanonicalHostBytes(
                                                result.publicationRepresentation(0),
                                                actualDescriptor, Long.MAX_VALUE));
                                assertArrayEquals(expectedWork.array(),
                                        integration.copyToCanonicalHostBytes(
                                                result.publicationRepresentation(1),
                                                work.descriptor(), Long.MAX_VALUE));
                            }
                        }
                    }
                }
            }
        }
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
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, outputs, Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), providers, availability);
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
        return context(artifacts, memoryPlan,
                artifacts.partitions().stream()
                        .map(partition -> new PreparedPartition(partition,
                                new EmptyExecutable(memoryPlan)))
                        .toList(), List.of());
    }

    private static PreparedScheduleContext context(CompileArtifacts artifacts,
            PreparedMemoryPlan memoryPlan, List<PreparedPartition> partitions,
            List<PreparedBufferAssignment> assignments) {
        var constants = new java.util.LinkedHashMap<ValueId,
                io.github.pho001.synaptik.model.datatype.ScalarValue>();
        artifacts.constants().constantSources().forEach(source ->
                constants.put(source.valueId(), source.value()));
        var publications = new java.util.ArrayList<ValueId>();
        artifacts.publication().forwardBindings().forEach(binding ->
                publications.add(binding.valueId()));
        artifacts.publication().gradientBindings().forEach(binding ->
                publications.add(binding.valueId()));
        return new PreparedScheduleContext(
                artifacts.partitions(),
                artifacts.graph().values(),
                artifacts.memory().requirements(),
                artifacts.constants().bindableInputs(),
                constants,
                publications,
                memoryPlan,
                partitions,
                assignments,
                List.of());
    }

    private static io.github.pho001.synaptik.runtime.execution.PreparedExecution prepare(
            CpuBackendIntegration integration, CompileArtifacts artifacts) {
        return GraphPreparation.prepare(artifacts, List.of(integration.partitionPreparation()),
                integration.scheduleAssembler());
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
            assignments.add(new PreparedBufferAssignment(artifacts.graph().values().get(index).id(), memoryPlan.buffers().get(index).slot(), index, java.util.List.of(io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider.CPU_BACKEND_ID)));
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
