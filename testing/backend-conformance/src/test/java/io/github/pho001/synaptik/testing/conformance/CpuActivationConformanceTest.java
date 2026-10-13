package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** CPU0022 through provider, analysis, finalization and borrowed sliced heap/native storage. */
final class CpuActivationConformanceTest {
    private static final Shape SHAPE = Shape.of(17);
    private static final List<UnaryElementwiseKind> LOW = List.of(UnaryElementwiseKind.ERF,
            UnaryElementwiseKind.RSQRT, UnaryElementwiseKind.SIGMOID, UnaryElementwiseKind.GELU,
            UnaryElementwiseKind.GELU_TANH_APPROXIMATION, UnaryElementwiseKind.SILU);

    @Test void lowCompositesConformOnEveryCarrierPairAndStridedLayout() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (UnaryElementwiseKind kind : LOW)
                for (boolean strided : List.of(false, true))
                    for (boolean nativeIn : List.of(false, true))
                        for (boolean nativeOut : List.of(false, true))
                            execute(type, kind, nativeIn, nativeOut, strided, false);
    }

    @Test void floatAndDoubleExtremesConformOnScalarAndVectorBodiesAndTails() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (UnaryElementwiseKind kind : List.of(UnaryElementwiseKind.ERF, UnaryElementwiseKind.GELU))
                for (boolean vector : List.of(false, true))
                    for (boolean nativeIn : List.of(false, true))
                        for (boolean nativeOut : List.of(false, true))
                            execute(type, kind, nativeIn, nativeOut, false, vector);
    }

    private static void execute(DataType type, UnaryElementwiseKind kind, boolean nativeIn,
            boolean nativeOut, boolean strided, boolean vector) {
        int stride = strided ? 2 : 1, count = 17;
        var descriptor = new TensorDescriptor(type, SHAPE,
                Optional.of(LayoutDescriptor.of(SHAPE, new long[] {stride}, 1, true)), false);
        int span = Math.toIntExact(descriptor.layout().orElseThrow().referencedElementSpan());
        var operation = new Operation(kind, NoOperationAttrs.INSTANCE);
        assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(operation,
                List.of(descriptor), List.of(descriptor))));
        var node = new CompiledNode(new NodeId(0), operation, List.of(new ValueId(0)), List.of(new ValueId(1)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        CarrierAccess heap = type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY;
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(vector
                ? CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                : CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR, 1, 1, 1);
        var context = new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(new ValueId(0), descriptor), new GraphValue(new ValueId(1), descriptor)),
                List.of(new LogicalMemoryRequirement(new ValueId(0), descriptor, Optional.empty(),
                                List.of(partition), false),
                        new LogicalMemoryRequirement(new ValueId(1), descriptor, Optional.of(partition),
                                List.of(), true)), Map.of(), new CpuPartitionAnalysisInputs(false,
                        List.of(nativeIn ? CarrierAccess.MEMORY_SEGMENT : heap,
                                nativeOut ? CarrierAccess.MEMORY_SEGMENT : heap), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        assertEquals(vector ? CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                : CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR,
                analysis.plan().units().getFirst().executionStrategy().compute());
        PreparedExecutable executable = finalizePortable(analysis);
        try (Arena arena = Arena.ofConfined(); CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            MemorySegment inputBacking = backing(arena, type, nativeIn, span + 2);
            MemorySegment outputBacking = backing(arena, type, nativeOut, span + 2);
            MemorySegment in = inputBacking.asSlice(type.byteWidth(), (long) span * type.byteWidth());
            MemorySegment out = outputBacking.asSlice(type.byteWidth(), (long) span * type.byteWidth());
            for (int i = 0; i < span + 2; i++) {
                put(type, inputBacking, i, -71); put(type, outputBacking, i, -71);
            }
            boolean low = type == DataType.BFLOAT16 || type == DataType.FLOAT16;
            double huge = low ? 6 : type == DataType.FLOAT64 ? 1e100 : 1e10;
            double[] values = {-huge, huge, -0d, 0d, Double.NaN,
                    Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY};
            for (int i = 0; i < count; i++) put(type, in, 1 + i * stride, values[i % values.length]);
            byte[] snapshot = inputBacking.toArray(ValueLayout.JAVA_BYTE);
            try (RunState state = new RunState(executable.memoryPlan(), List.of(
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            type, span, in)), RunResourceOwnership.BORROWED)),
                    List.of(new BufferRepresentationBinding(integration.borrow(new MemorySegmentStorage(
                            type, span, out)), RunResourceOwnership.BORROWED))), List.of())) {
                executable.bind(state).execute();
                for (int p = 0; p < span + 2; p++) {
                    int logical = (p - 2) / stride;
                    if (p < 2 || (p - 2) % stride != 0 || logical >= count) {
                        assertEquals(-71d, get(type, outputBacking, p), "sentinel");
                        continue;
                    }
                    double x = values[logical % values.length];
                    double expected = expected(kind, x, low);
                    MemorySegment rounded = backing(arena, type, false, 1);
                    put(type, rounded, 0, expected);
                    double actual = get(type, outputBacking, p);
                    String label = type + " " + kind + " x=" + x + " native=" + nativeIn + "/" + nativeOut;
                    if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual), label);
                    else assertEquals(Double.doubleToRawLongBits(get(type, rounded, 0)),
                            Double.doubleToRawLongBits(actual), label);
                }
                assertArrayEquals(snapshot, inputBacking.toArray(ValueLayout.JAVA_BYTE));
            }
        }
    }

    private static double expected(UnaryElementwiseKind kind, double x, boolean low) {
        if (Double.isNaN(x)) return Double.NaN;
        if (kind == UnaryElementwiseKind.ERF) return x == 0 ? x : Math.copySign(1, x);
        if (kind == UnaryElementwiseKind.GELU) return x <= 0 ? Math.copySign(0, x) : x;
        assertTrue(low);
        float f = (float) x;
        if (kind == UnaryElementwiseKind.RSQRT) return 1f / (float) StrictMath.sqrt(f);
        if (kind == UnaryElementwiseKind.GELU_TANH_APPROXIMATION) {
            if (f == Float.NEGATIVE_INFINITY) return -0f;
            float cube = f * f * f;
            float inner = (float) StrictMath.sqrt(2f / (float) Math.PI) * (f + 0.044715f * cube);
            return 0.5f * f * (1f + (float) StrictMath.tanh(inner));
        }
        if (kind == UnaryElementwiseKind.SILU && f == Float.NEGATIVE_INFINITY) return -0f;
        float exp = (float) StrictMath.exp(f >= 0 ? -f : f);
        float numerator = kind == UnaryElementwiseKind.SIGMOID ? (f >= 0 ? 1f : exp)
                : (f >= 0 ? f : f * exp);
        return numerator / (1f + exp);
    }

    private static MemorySegment backing(Arena arena, DataType type, boolean nativeStorage, int count) {
        if (nativeStorage) return arena.allocate((long) count * type.byteWidth(), type.byteWidth());
        return type == DataType.FLOAT64 ? MemorySegment.ofArray(new double[count])
                : type == DataType.FLOAT32 ? MemorySegment.ofArray(new float[count]) : MemorySegment.ofArray(new short[count]);
    }
    private static void put(DataType type, MemorySegment segment, int index, double x) {
        switch (type) {
            case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, x);
            case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) x);
            case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, Float.floatToFloat16((float) x));
            case BFLOAT16 -> {
                int bits = Float.floatToRawIntBits((float) x);
                segment.setAtIndex(ValueLayout.JAVA_SHORT, index, (short) (Float.isNaN((float) x) ? 0x7fc0
                        : (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16));
            }
            default -> throw new AssertionError(type);
        }
    }
    private static double get(DataType type, MemorySegment segment, int index) {
        return switch (type) {
            case FLOAT64 -> segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            case FLOAT32 -> segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case FLOAT16 -> Float.float16ToFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case BFLOAT16 -> Float.intBitsToFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index) << 16);
            default -> throw new AssertionError(type);
        };
    }

    private static PreparedExecutable finalizePortable(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        for (PreparationResourceRequirement requirement : analysis.requirements()) {
            var buffer = assertInstanceOf(PreparationResourceRequirement.Buffer.class, requirement);
            var slot = new BufferSlot(buffers.size());
            buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(), buffer.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot, buffers.size() - 1, 0));
        }
        return new CpuPartitionFinalizer().finalizePartition(new BackendPartitionFinalization<>(
                analysis, new PreparedMemoryPlan(buffers, List.of()), assignments)).executable();
    }
}
