package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Constructor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Native-free ownership-transfer, fallback, and close checks for supported composition. */
final class CpuBackendCompositionTest {
    @Test
    void preparesAndMaterializesCanonicalSourceOnlyConstantsPerRun() throws Exception {
        try (CpuBackendComposition composition = CpuBackendComposition.openAutomatic()) {
            CompileArtifacts base = ordinaryArtifacts(composition);
            List<ConstantCase> constants = List.of(
                    constant(DataType.FLOAT64, Shape.scalar(), ScalarValue.float64(-0.0d)),
                    constant(DataType.FLOAT32, Shape.of(2), ScalarValue.float32(
                            Float.intBitsToFloat(0x7fc0_0042))),
                    constant(DataType.BFLOAT16, Shape.of(1),
                            ScalarValue.bfloat16Bits((short) 0xffc1)),
                    constant(DataType.INT64, Shape.of(1), ScalarValue.int64(Long.MIN_VALUE)),
                    constant(DataType.INT32, Shape.of(1), ScalarValue.int32(Integer.MIN_VALUE)),
                    constant(DataType.BOOL, Shape.of(1), ScalarValue.bool(true)),
                    constant(DataType.FLOAT32, Shape.of(0, Long.MAX_VALUE),
                            ScalarValue.float32(3.0f)));
            CompileArtifacts artifacts = withConstants(base, constants);

            var execution = composition.prepare(artifacts);
            var creation = assertInstanceOf(PreparedSchedule.RepresentationCreationStep.class,
                    execution.schedule().steps().getFirst());
            int ordinaryBufferCount = base.memory().requirements().size();
            assertAll(
                    () -> assertEquals(base.memory().requirements().size() + constants.size(),
                            execution.memoryPlan().buffers().size()),
                    () -> assertEquals(2 + artifacts.publication().forwardBindings().size(),
                            execution.schedule().steps().size()),
                    () -> assertInstanceOf(PreparedSchedule.ExecutionStep.class,
                            execution.schedule().steps().get(1)));
            for (int index = 0; index < constants.size(); index++) {
                assertInstanceOf(PreparedRepresentationPlan.InitializedBuffer.class,
                        creation.representationPlan().bufferPreparations()
                                .get(ordinaryBufferCount + index).getFirst());
                assertEquals(constants.get(index).descriptor.dataType().byteWidth(),
                        execution.memoryPlan().buffers().get(ordinaryBufferCount + index)
                                .byteAlignment());
                assertEquals(constants.get(index).descriptor.layout().orElseThrow()
                                .referencedElementSpan()
                                * constants.get(index).descriptor.dataType().byteWidth(),
                        execution.memoryPlan().buffers().get(ordinaryBufferCount + index)
                                .byteSize());
            }

            try (Arena arena = Arena.ofShared();
                    var input = composition.borrow(new MemorySegmentStorage(
                            DataType.FLOAT32, 4, arena.allocate(16, 4)))) {
                var runner = new PreparedExecutionRunner();
                try (var first = runner.run(execution, List.of(input));
                        var second = runner.run(execution, List.of(input))) {
                    for (int index = 0; index < constants.size(); index++) {
                        int resultIndex = index + 1;
                        assertNotSame(first.publicationRepresentation(resultIndex),
                                second.publicationRepresentation(resultIndex));
                        byte[] actual = composition.copyToCanonicalHostBytes(
                                first.publicationRepresentation(resultIndex),
                                constants.get(index).descriptor, Long.MAX_VALUE);
                        assertArrayEquals(expectedBytes(constants.get(index)), actual);
                    }
                }

                try (var executor = Executors.newFixedThreadPool(2)) {
                    var firstRun = executor.submit(() -> runner.run(execution, List.of(input)));
                    var secondRun = executor.submit(() -> runner.run(execution, List.of(input)));
                    try (var first = firstRun.get(); var second = secondRun.get()) {
                        assertNotSame(first.publicationRepresentation(1),
                                second.publicationRepresentation(1));
                        assertArrayEquals(expectedBytes(constants.getFirst()),
                                composition.copyToCanonicalHostBytes(
                                        first.publicationRepresentation(1),
                                        constants.getFirst().descriptor, Long.MAX_VALUE));
                        assertArrayEquals(expectedBytes(constants.getFirst()),
                                composition.copyToCanonicalHostBytes(
                                        second.publicationRepresentation(1),
                                        constants.getFirst().descriptor, Long.MAX_VALUE));
                    }
                }
            }
        }
    }

    @Test
    void rejectsSourceOnlyGeometryInDeterministicOrder() {
        try (CpuBackendComposition composition = CpuBackendComposition.openAutomatic()) {
            CompileArtifacts base = ordinaryArtifacts(composition);
            TensorDescriptor dynamic = new TensorDescriptor(DataType.FLOAT32,
                    Shape.ofDimensions(new DynamicDimension("N")), Optional.empty(), false);
            TensorDescriptor unresolved = new TensorDescriptor(
                    DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
            Shape nonCanonicalShape = Shape.of(2);
            TensorDescriptor nonCanonical = new TensorDescriptor(DataType.FLOAT32,
                    nonCanonicalShape,
                    Optional.of(LayoutDescriptor.of(nonCanonicalShape, new long[] {2}, 0, false)),
                    false);
            Shape overflowingShape = Shape.of(Long.MAX_VALUE);
            TensorDescriptor overflowing = new TensorDescriptor(DataType.FLOAT64,
                    overflowingShape,
                    Optional.of(LayoutDescriptor.contiguous(overflowingShape)), false);

            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> composition.prepare(withConstants(base, List.of(
                            new ConstantCase(dynamic, ScalarValue.float32(1.0f))))))
                    .getMessage().contains("dynamic shape"));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> composition.prepare(withConstants(base, List.of(
                            new ConstantCase(unresolved, ScalarValue.float32(1.0f))))))
                    .getMessage().contains("unresolved layout"));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> composition.prepare(withConstants(base, List.of(
                            new ConstantCase(nonCanonical, ScalarValue.float32(1.0f))))))
                    .getMessage().contains("non-canonical layout"));
            assertThrows(ArithmeticException.class,
                    () -> composition.prepare(withConstants(base, List.of(
                            new ConstantCase(overflowing, ScalarValue.float64(1.0d))))));
        }
    }

    @Test
    void qualificationFailureClosesPartialOwnershipAndReturnsPortableComposition() {
        AtomicInteger closes = new AtomicInteger();
        CpuOpenBlasInvocation failing = new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return true; }
            @Override public int threadCount() { throw new IllegalStateException("qualification"); }
            @Override public void setThreadCount(int count) { }
            @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                    MemorySegment b, float beta, MemorySegment c) { }
            @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                    MemorySegment b, double beta, MemorySegment c) { }
        };
        try (CpuBackendComposition composition = CpuBackendComposition.open(
                loadedSession(failing, closes))) {
            assertTrue(composition.availabilitySnapshot().devices().size() == 1);
        }
        assertEquals(1, closes.get());
    }

    @Test
    void fatalQualificationFailureClosesTransferredOwnershipAndPreservesFailureTree() {
        var primary = new AssertionError("qualification");
        var restoreFailure = new IllegalStateException("restore cleanup");
        var ownerCloseFailure = new AssertionError("owner cleanup");
        AtomicInteger restoreAttempts = new AtomicInteger();
        AtomicInteger closes = new AtomicInteger();
        int[] threads = {3};
        CpuOpenBlasInvocation failing = new CpuOpenBlasInvocation() {
            @Override public boolean isOpen() { return true; }
            @Override public int threadCount() { return threads[0]; }
            @Override public void setThreadCount(int count) {
                if (count == 3 && restoreAttempts.incrementAndGet() == 2) {
                    throw restoreFailure;
                }
                threads[0] = count;
            }
            @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                    MemorySegment b, float beta, MemorySegment c) {
                throw primary;
            }
            @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                    MemorySegment b, double beta, MemorySegment c) { }
        };
        CpuOpenBlasDiscoverySession session = loadedSession(failing, () -> {
            closes.incrementAndGet();
            throw ownerCloseFailure;
        });

        AssertionError thrown = assertThrows(
                AssertionError.class, () -> CpuBackendComposition.open(session));

        assertAll(
                () -> assertSame(primary, thrown),
                () -> assertArrayEquals(new Throwable[] {restoreFailure}, thrown.getSuppressed()),
                () -> assertArrayEquals(
                        new Throwable[] {ownerCloseFailure}, restoreFailure.getSuppressed()),
                () -> assertEquals(2, restoreAttempts.get()),
                () -> assertEquals(1, closes.get()));
        session.close();
        assertEquals(1, closes.get());
    }

    @Test
    void successfulQualificationTransfersAndClosesProviderOwnershipExactlyOnce() {
        AtomicInteger closes = new AtomicInteger();
        ComputingInvocation invocation = new ComputingInvocation();
        CpuBackendComposition composition = CpuBackendComposition.open(
                loadedSession(invocation, closes));
        assertEquals(1, invocation.threads);
        composition.close();
        composition.close();
        assertEquals(1, closes.get());
        assertEquals(3, invocation.threads);
    }

    private static CpuOpenBlasDiscoverySession loadedSession(CpuOpenBlasInvocation invocation,
            AtomicInteger closes) {
        return loadedSession(invocation, closes::incrementAndGet);
    }

    private static CompileArtifacts ordinaryArtifacts(CpuBackendComposition composition) {
        Shape shape = Shape.of(4);
        Tensor input = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY,
                List.of(input.contiguous()), Optional.empty(), GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(new CpuCapabilityProvider()),
                List.of(composition.availabilitySnapshot()));
    }

    private static ConstantCase constant(DataType type, Shape shape, ScalarValue value) {
        return new ConstantCase(new TensorDescriptor(
                type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false), value);
    }

    private static CompileArtifacts withConstants(
            CompileArtifacts base, List<ConstantCase> constants) {
        var values = new ArrayList<>(base.graph().values());
        var inputs = new ArrayList<>(base.graph().inputs());
        var outputs = new ArrayList<>(base.graph().outputs());
        var sources = new ArrayList<CompileConstantPlan.ConstantSource>();
        var publications = new ArrayList<ForwardPublicationBinding>();
        publications.add(new ForwardPublicationBinding(new TensorId(7_000), outputs.getFirst()));
        for (int index = 0; index < constants.size(); index++) {
            ConstantCase constant = constants.get(index);
            ValueId id = new ValueId(10_000 + index);
            values.add(new GraphValue(id, constant.descriptor));
            inputs.add(id);
            outputs.add(id);
            sources.add(new CompileConstantPlan.ConstantSource(id, constant.value));
            publications.add(new ForwardPublicationBinding(new TensorId(7_001 + index), id));
        }
        var graph = new CompiledGraphModel(values, base.graph().nodes(), inputs, outputs,
                base.graph().nodePhases());
        var constantPlan = construct(CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                base.constants().bindableInputBindings(), sources);
        var publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph, publications, List.of());
        return new CompileArtifacts(base.mode(), graph, base.partitions(),
                LogicalMemoryPlanning.plan(graph, base.partitions()), publication, constantPlan,
                base.diagnostics(), new DerivativeGraphMetadata(
                        graph, base.derivatives().derivativeOrderByNode()));
    }

    private static byte[] expectedBytes(ConstantCase constant) {
        long count = constant.descriptor.shape().knownElementCount().orElseThrow();
        ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(
                count * constant.descriptor.dataType().byteWidth())).order(ByteOrder.BIG_ENDIAN);
        for (long index = 0; index < count; index++) {
            switch (constant.descriptor.dataType()) {
                case FLOAT64 -> bytes.putDouble(constant.value.float64Value());
                case FLOAT32 -> bytes.putFloat(constant.value.float32Value());
                case BFLOAT16 -> bytes.putShort(constant.value.bfloat16Bits());
                case INT64 -> bytes.putLong(constant.value.int64Value());
                case INT32 -> bytes.putInt(constant.value.int32Value());
                case BOOL -> bytes.put((byte) (constant.value.booleanValue() ? 1 : 0));
            }
        }
        return bytes.array();
    }

    private static <T> T construct(
            Class<T> type, Class<?>[] parameterTypes, Object... arguments) {
        try {
            Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private record ConstantCase(TensorDescriptor descriptor, ScalarValue value) {}

    private static CpuOpenBlasDiscoverySession loadedSession(CpuOpenBlasInvocation invocation,
            CpuOpenBlasDiscoverySession.CloseAction closeAction) {
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("test-openblas");
        var result = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED);
        return new CpuOpenBlasDiscoverySession(result, Optional.of(
                new CpuOpenBlasDiscoverySession.OwnedResource(invocation, closeAction)));
    }

    private static final class ComputingInvocation implements CpuOpenBlasInvocation {
        int threads = 3;
        boolean open = true;
        @Override public boolean isOpen() { return open; }
        @Override public int threadCount() { return threads; }
        @Override public void setThreadCount(int count) { threads = count; }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                float sum = 0;
                for (int inner = 0; inner < k; inner++) sum +=
                        a.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, row * k + inner)
                                * b.getAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,
                                        inner * n + column);
                long index = (long) row * n + column;
                c.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, index,
                        alpha * sum + beta * c.getAtIndex(
                                java.lang.foreign.ValueLayout.JAVA_FLOAT, index));
            }
        }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                double sum = 0;
                for (int inner = 0; inner < k; inner++) sum +=
                        a.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE, row * k + inner)
                                * b.getAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE,
                                        inner * n + column);
                long index = (long) row * n + column;
                c.setAtIndex(java.lang.foreign.ValueLayout.JAVA_DOUBLE, index,
                        alpha * sum + beta * c.getAtIndex(
                                java.lang.foreign.ValueLayout.JAVA_DOUBLE, index));
            }
        }
    }
}
