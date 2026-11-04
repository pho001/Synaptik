package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exact storage-only geometry and per-run ownership independent of native arithmetic kernels. */
final class MetalProducerlessPublishedValuesTest {
    @Test void fp8SourceDeclarationsRejectBeforeAnyPhysicalAllocation() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            for (var type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
                var value = new GraphValue(new ValueId(1), descriptor(type, Shape.of(2)));
                assertThrows(IllegalArgumentException.class, () -> MetalProducerlessPublishedValues.geometry(
                        context, value, requirement(value), Optional.empty()));
                assertThrows(IllegalArgumentException.class,
                        () -> MetalProducerlessPublishedValues.byteSize(value.descriptor()));
            }
            assertEquals(0, api.creates);
        }
    }

    @Test
    void geometryRetainsExactSourceReferencesAndModelViewSpanForEveryCarrier() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            for (DataType type : new DataType[] {DataType.FLOAT32, DataType.INT32, DataType.BOOL, DataType.FLOAT64, DataType.BFLOAT16, DataType.INT64, DataType.FLOAT16}) {
                for (var descriptor : List.of(
                        new TensorDescriptor(type, Shape.of(2, 3),
                                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {5, 2}, 1, true)), false),
                        new TensorDescriptor(type, Shape.of(2, 3),
                                Optional.of(LayoutDescriptor.of(Shape.of(2, 3), new long[] {0, 1}, 2, true)), false),
                        new TensorDescriptor(type, Shape.of(0, 3),
                                Optional.of(LayoutDescriptor.of(Shape.of(0, 3), new long[] {5, 1}, 7, true)), false))) {
                    LayoutDescriptor layout = descriptor.layout().orElseThrow();
                    var value = new GraphValue(new ValueId(1), descriptor);
                    var requirement = requirement(value);
                    var geometry = MetalProducerlessPublishedValues.geometry(
                            context, value, requirement, Optional.empty());
                    assertSame(value, geometry.value());
                    assertSame(requirement, geometry.logicalRequirement());
                    assertEquals(MetalCapabilityProvider.METAL_BACKEND_ID, geometry.owner());
                    assertEquals(layout.referencedElementSpan() * type.byteWidth(),
                            geometry.size().fixedByteSize().orElseThrow());
                    assertEquals(type.byteWidth(), geometry.byteAlignment());
                }
            }
            assertEquals(0, api.creates);
        }
    }

    @Test
    void emptyOffsetsRemainZeroWhileNonemptyByteSpanOverflowRejectsWithoutAllocation() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            Shape empty = Shape.of(0, 3);
            var emptyValue = new GraphValue(new ValueId(1), new TensorDescriptor(DataType.FLOAT64, empty,
                    Optional.of(LayoutDescriptor.of(empty, new long[] {Long.MAX_VALUE, 1}, Long.MAX_VALUE, true)), false));
            assertEquals(0, MetalProducerlessPublishedValues.geometry(
                    context, emptyValue, requirement(emptyValue), Optional.empty())
                    .size().fixedByteSize().orElseThrow());
            var hugeValue = new GraphValue(new ValueId(2), new TensorDescriptor(DataType.FLOAT64, Shape.scalar(),
                    Optional.of(LayoutDescriptor.of(Shape.scalar(), new long[0], Long.MAX_VALUE - 1, true)), false));
            assertThrows(ArithmeticException.class, () -> MetalProducerlessPublishedValues.geometry(
                    context, hugeValue, requirement(hugeValue), Optional.empty()));
            assertEquals(0, api.creates);
        }
    }

    @Test
    void unsupportedDeclarationsAndConstantRolesRejectBeforeAnyPhysicalAllocation() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            Shape shape = Shape.of(2);
            TensorDescriptor canonical = descriptor(DataType.FLOAT32, shape);
            var declarations = List.of(
                    new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), false),
                    new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("rows")), Optional.empty(), false),
                    new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true),
                    new TensorDescriptor(DataType.FLOAT32, shape,
                            Optional.of(LayoutDescriptor.of(shape, new long[] {1}, 0, true)), false));
            for (TensorDescriptor descriptor : declarations) {
                var value = new GraphValue(new ValueId(1), descriptor);
                assertThrows(IllegalArgumentException.class, () -> MetalProducerlessPublishedValues.geometry(
                        context, value, requirement(value), Optional.of(ScalarValue.float32(1))));
            }
            var value = new GraphValue(new ValueId(1), canonical);
            assertThrows(IllegalArgumentException.class, () -> MetalProducerlessPublishedValues.geometry(
                    context, value, requirement(value), Optional.of(ScalarValue.int32(1))));
            var wrongSize = schedule(value, ScalarValue.float32(1), 12, 4);
            assertThrows(IllegalArgumentException.class, () -> MetalProducerlessPublishedValues.contribution(
                    context, wrongSize, wrongSize.bufferAssignments().getFirst()));
            var wrongAlignment = schedule(value, ScalarValue.float32(1), 8, 8);
            assertThrows(IllegalArgumentException.class, () -> MetalProducerlessPublishedValues.contribution(
                    context, wrongAlignment, wrongAlignment.bufferAssignments().getFirst()));
            assertEquals(0, api.creates);
        }
    }

    @Test
    void onePreparedRecipeInitializesFreshRunOwnedRawBuffersIncludingEmptyConstants() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            for (DataType type : new DataType[] {DataType.FLOAT32, DataType.INT32, DataType.BOOL, DataType.FLOAT64, DataType.BFLOAT16, DataType.INT64, DataType.FLOAT16}) for (long count : new long[] {3, 0, 1}) {
                ScalarValue scalar = scalar(type);
                var value = new GraphValue(new ValueId(1), descriptor(type, Shape.of(count)));
                long bytes = count * type.byteWidth();
                var schedule = schedule(value, scalar, bytes, type.byteWidth());
                var recipe = assertInstanceOf(PreparedRepresentationPlan.InitializedBuffer.class,
                        MetalProducerlessPublishedValues.contribution(
                                context, schedule, schedule.bufferAssignments().getFirst()).orElseThrow());
                int before = api.creates;
                assertThrows(IllegalArgumentException.class,
                        () -> recipe.creator().create(bytes + type.byteWidth(), type.byteWidth()));
                assertEquals(before, api.creates);
                try (var first = (MetalBufferRepresentation) recipe.creator().create(bytes, type.byteWidth());
                        var second = (MetalBufferRepresentation) recipe.creator().create(bytes, type.byteWidth());
                        Arena arena = Arena.ofConfined()) {
                    assertNotSame(first, second);
                    assertNotSame(first.executionHandle(), second.executionHandle());
                    assertEquals(bytes, first.byteSize());
                    assertEquals(before + 2, api.creates);
                    MemorySegment copied = arena.allocate(bytes, type.byteWidth());
                    first.download(0, copied, 0, bytes);
                    for (long index = 0; index < count; index++) {
                        long offset = index * type.byteWidth();
                        long actual = switch (type.byteWidth()) {
                            case 1 -> Byte.toUnsignedLong(copied.get(ValueLayout.JAVA_BYTE, offset));
                            case 2 -> Short.toUnsignedLong(copied.get(ValueLayout.JAVA_SHORT, offset));
                            case 4 -> Integer.toUnsignedLong(copied.get(ValueLayout.JAVA_INT, offset));
                            case 8 -> copied.get(ValueLayout.JAVA_LONG, offset);
                            default -> throw new AssertionError(type);
                        };
                        assertEquals(rawBits(type), actual, type + " count=" + count);
                    }
                }
                assertTrue(api.buffers.isEmpty());
                assertEquals(api.creates, api.releases);
            }
        }
    }

    @Test
    void uploadFailureReleasesExactlyOnceAndTheSamePreparedInitializerCanRetry() {
        var api = new StorageApi();
        try (var context = MetalDeviceContext.open(api)) {
            var value = new GraphValue(new ValueId(1), descriptor(DataType.INT64, Shape.of(3)));
            var schedule = schedule(value, ScalarValue.int64(Long.MIN_VALUE), 24, 8);
            var recipe = assertInstanceOf(PreparedRepresentationPlan.InitializedBuffer.class,
                    MetalProducerlessPublishedValues.contribution(
                            context, schedule, schedule.bufferAssignments().getFirst()).orElseThrow());
            RuntimeException failure = new IllegalStateException("upload failure");
            api.uploadFailure = failure;
            assertSame(failure, assertThrows(IllegalStateException.class,
                    () -> recipe.creator().create(24, 8)));
            assertEquals(1, api.creates);
            assertEquals(1, api.releases);
            assertTrue(api.buffers.isEmpty());
            api.uploadFailure = null;
            try (var retry = recipe.creator().create(24, 8)) {
                assertInstanceOf(MetalBufferRepresentation.class, retry);
                assertEquals(2, api.creates);
            }
            assertEquals(2, api.releases);
        }
    }

    @Test
    void callerRecipeAllocatesNothingAndContextClosureRejectsNewGeometry() {
        var api = new StorageApi();
        var context = MetalDeviceContext.open(api);
        var value = new GraphValue(new ValueId(1), descriptor(DataType.BOOL, Shape.of(0)));
        var schedule = schedule(value, null, 0, 1);
        assertInstanceOf(PreparedRepresentationPlan.CallerInput.class,
                MetalProducerlessPublishedValues.contribution(
                        context, schedule, schedule.bufferAssignments().getFirst()).orElseThrow());
        assertEquals(0, api.creates);
        context.close();
        assertThrows(IllegalStateException.class, () -> MetalProducerlessPublishedValues.geometry(
                context, value, requirement(value), Optional.empty()));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static LogicalMemoryRequirement requirement(GraphValue value) {
        return new LogicalMemoryRequirement(value.id(), value.descriptor(), Optional.empty(), List.of(), true);
    }

    private static PreparedScheduleContext schedule(GraphValue value, ScalarValue scalar, long bytes, long alignment) {
        var memory = new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), bytes, alignment)), List.of());
        var assignment = new PreparedBufferAssignment(value.id(), memory.buffers().getFirst().slot(), 0,
                List.of(MetalCapabilityProvider.METAL_BACKEND_ID));
        return new PreparedScheduleContext(List.of(), List.of(value), List.of(requirement(value)),
                scalar == null ? List.of(value.id()) : List.of(), scalar == null ? Map.of() : Map.of(value.id(), scalar),
                List.of(value.id()), memory, List.of(), List.of(assignment), List.of());
    }

    private static ScalarValue scalar(DataType type) {
        return switch (type) {  case BOOL -> ScalarValue.bool(true); case FLOAT8_E4M3, FLOAT8_E5M2 -> throw new IllegalArgumentException("Metal does not support FP8 carriers"); case FLOAT16 -> ScalarValue.float16Bits((short) rawBits(type));
        case BFLOAT16 -> ScalarValue.bfloat16Bits((short) rawBits(type));
        case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat((int) rawBits(type)));
        case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(rawBits(type)));
        case INT32 -> ScalarValue.int32((int) rawBits(type));
        case INT64 -> ScalarValue.int64(rawBits(type)); };
    }

    private static long rawBits(DataType type) {
        return switch (type) {  case BOOL -> 1; case FLOAT8_E4M3, FLOAT8_E5M2 -> throw new IllegalArgumentException("Metal does not support FP8 carriers"); case FLOAT16 -> 0x7e15;
        case BFLOAT16 -> 0x7fc1;
        case FLOAT32 -> 0x7fc12345L;
        case FLOAT64 -> 0x7ff8123456789abcL;
        case INT32 -> 0x80000001L;
        case INT64 -> Long.MIN_VALUE + 1; };
    }

    /** Storage-only seam recording exact allocation, byte transfer, and release ownership. */
    private static final class StorageApi extends MetalNativeApi {
        private final Map<Handle, byte[]> buffers = new HashMap<>();
        private long next = 1;
        private int creates;
        private int releases;
        private RuntimeException uploadFailure;

        @Override Handle createContext() { return handle(); }
        @Override void releaseContext(Handle context) {}
        @Override Handle createBuffer(Handle context, long bytes) {
            Handle result = handle();
            buffers.put(result, new byte[Math.toIntExact(bytes)]);
            creates++;
            return result;
        }
        @Override void releaseBuffer(Handle buffer) {
            assertNotNull(buffers.remove(buffer), "buffer must be released exactly once");
            releases++;
        }
        @Override void upload(Handle buffer, long offset, MemorySegment source, long count) {
            if (uploadFailure != null) throw uploadFailure;
            MemorySegment.ofArray(buffers.get(buffer)).asSlice(offset, count).copyFrom(source.asSlice(0, count));
        }
        @Override void download(Handle buffer, long offset, MemorySegment target, long count) {
            target.asSlice(0, count).copyFrom(MemorySegment.ofArray(buffers.get(buffer)).asSlice(offset, count));
        }
        @Override NativeCreateResult createProgramExecutableNative(Handle context, MemorySegment image) {
            throw new AssertionError("Storage-only publication must not compile an executable");
        }
        @Override int releaseExecutableNative(Handle executable) {
            throw new AssertionError("Storage-only publication must not own an executable");
        }
        @Override int runExecutableNative(Handle executable, int inputCount, MemorySegment inputs,
                int outputCount, MemorySegment outputs) {
            throw new AssertionError("Storage-only publication must not execute a graph");
        }
        @Override NativeCreateResult createNegKernelPipelineNative(Handle context, long count) {
            throw new AssertionError("Storage-only publication must not compile a NEG pipeline");
        }
        @Override int releaseNegKernelPipelineNative(Handle pipeline) {
            throw new AssertionError("Storage-only publication must not own a NEG pipeline");
        }
        @Override int runNegKernelPipelineNative(Handle pipeline, Handle input, Handle output) {
            throw new AssertionError("Storage-only publication must not execute arithmetic");
        }
        @Override public void close() { assertTrue(buffers.isEmpty()); }
        private Handle handle() { return new Handle(MemorySegment.ofAddress(next++)); }
    }
}
