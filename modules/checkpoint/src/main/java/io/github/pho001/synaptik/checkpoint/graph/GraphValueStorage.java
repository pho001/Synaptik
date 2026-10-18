package io.github.pho001.synaptik.checkpoint.graph;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Cold raw-lane copying for canonical Model host leaves. HostTensorStorage promises no byte order;
 * this boundary explicitly requires native-order logical lanes and uses unaligned integer access.
 * Float carriers are allocated by logical type but never undergo numeric conversion.
 */
final class GraphValueStorage {
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    private GraphValueStorage() { }

    /**
     * Checks static canonical strides even when layout is unresolved, then logical byte size.
     * @param descriptor non-null complete logical metadata
     * @return nonnegative checked logical byte count
     * @throws IllegalArgumentException if shape is not static
     * @throws ArithmeticException if stride, element or byte geometry overflows
     */
    static long bytes(TensorDescriptor descriptor) {
        LayoutDescriptor.contiguous(descriptor.shape());
        return Math.multiplyExact(descriptor.shape().knownElementCount().orElseThrow(),
                descriptor.dataType().byteWidth());
    }

    /**
     * Additionally requires any present leaf layout to be exactly canonical non-view geometry.
     * @param descriptor non-null prototype or state descriptor
     * @return checked logical byte count, without a file or JVM payload ceiling
     * @throws IllegalArgumentException if present layout is not zero-offset non-view contiguous
     * @throws ArithmeticException if checked geometry overflows
     */
    static long leafBytes(TensorDescriptor descriptor) {
        long bytes = bytes(descriptor);
        if (descriptor.layout().isPresent()
                && !descriptor.layout().orElseThrow().equals(LayoutDescriptor.contiguous(descriptor.shape())))
            throw new IllegalArgumentException("Leaf layout must be canonical, zero-offset and non-view");
        return bytes;
    }

    /**
     * Preflights all JVM payload sizes, then snapshots each host association exactly once.
     * Read-only and excess capacity are allowed; empty segments still need live thread access.
     * @param states unique named identities in registration encounter order
     * @return state-aligned segment snapshot, never null
     * @throws IllegalArgumentException if type/capacity or JVM payload size is incompatible
     * @throws IllegalStateException if state is absent, dead or inaccessible to this thread
     */
    static List<MemorySegment> admit(List<Tensor> states) {
        for (Tensor state : states) payloadLength(state.descriptor());
        var segments = new ArrayList<MemorySegment>(states.size());
        for (Tensor state : states) {
            var descriptor = state.descriptor();
            var storage = state.hostStorage().orElseThrow(
                    () -> new IllegalStateException("Named state has no host storage"));
            if (storage.dataType() != descriptor.dataType()
                    || storage.byteSize() < bytes(descriptor))
                throw new IllegalArgumentException("Named state host type/capacity mismatch");
            MemorySegment segment = storage.segment();
            if (!storage.isAlive() || !segment.isAccessibleBy(Thread.currentThread()))
                throw new IllegalStateException("Named state is dead or thread-inaccessible");
            segments.add(segment);
        }
        return List.copyOf(segments);
    }

    /**
     * Validates BOOL before any import or bulk copy; it never normalizes noncanonical bytes.
     * @param states unique named state descriptors
     * @param segments completely admitted aligned associations
     * @throws IllegalArgumentException if a BOOL lane is neither zero nor one
     */
    static void validateBools(List<Tensor> states, List<MemorySegment> segments) {
        for (int i = 0; i < states.size(); i++) {
            if (states.get(i).descriptor().dataType() != DataType.BOOL) continue;
            long length = bytes(states.get(i).descriptor());
            for (long offset = 0; offset < length; offset++) {
                byte value = segments.get(i).get(ValueLayout.JAVA_BYTE, offset);
                if (value != 0 && value != 1) throw new IllegalArgumentException("Noncanonical BOOL state");
            }
        }
    }

    /**
     * Applies only the JVM single-payload bound, not any implicit file limit.
     * @param descriptor admitted canonical leaf metadata
     * @return exact logical payload size in bytes
     * @throws IllegalArgumentException if the payload exceeds Integer.MAX_VALUE−8
     * @throws ArithmeticException if checked geometry overflows
     */
    static int payloadLength(TensorDescriptor descriptor) {
        long bytes = leafBytes(descriptor);
        if (bytes > Integer.MAX_VALUE - 8) throw new IllegalArgumentException("State payload exceeds JVM ceiling");
        return Math.toIntExact(bytes);
    }

    /**
     * Copies once directly into matching mutable primitive heap carriers, preserving the descriptor.
     * @param descriptor admitted source metadata, retained without layout closure
     * @param source admitted native-order raw logical lanes, borrowed only during copying
     * @return fresh provenance-free state with private ordinary heap storage
     */
    static Tensor copy(TensorDescriptor descriptor, MemorySegment source) {
        MemorySegmentStorage storage = allocate(descriptor);
        MemorySegment target = storage.segment();
        long length = bytes(descriptor);
        switch (descriptor.dataType().byteWidth()) {
            case 8 -> { for (long p = 0; p < length; p += 8) target.set(LONG, p, source.get(LONG, p)); }
            case 4 -> { for (long p = 0; p < length; p += 4) target.set(INT, p, source.get(INT, p)); }
            case 2 -> { for (long p = 0; p < length; p += 2) target.set(SHORT, p, source.get(SHORT, p)); }
            case 1 -> MemorySegment.copy(source, 0, target, 0, length);
            default -> throw new AssertionError("Unknown logical width");
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }

    /**
     * Encodes logical lanes to big-endian raw bytes after complete admission and BOOL validation.
     * @param descriptor admitted state metadata
     * @param source snapshotted readable native-order storage
     * @return fresh exact canonical payload, preserving signed zero and NaN bits
     */
    static byte[] encode(TensorDescriptor descriptor, MemorySegment source) {
        ByteBuffer out = ByteBuffer.allocate(payloadLength(descriptor)).order(ByteOrder.BIG_ENDIAN);
        int width = descriptor.dataType().byteWidth();
        for (long p = 0; out.hasRemaining(); p += width) {
            switch (width) {
                case 8 -> out.putLong(source.get(LONG, p));
                case 4 -> out.putInt(source.get(INT, p));
                case 2 -> out.putShort(source.get(SHORT, p));
                case 1 -> out.put(source.get(ValueLayout.JAVA_BYTE, p));
                default -> throw new AssertionError("Unknown logical width");
            }
        }
        return out.array();
    }

    /**
     * Imports prevalidated canonical bytes into a final candidate's private mutable heap storage.
     * The candidate Tensor can receive this storage without another wrapper or ownership copy.
     * @param descriptor fully preflighted state descriptor
     * @param canonical exact validated big-endian payload, cursor not mutated
     * @return private storage with matching logical primitive carrier and native-order raw bits
     */
    static MemorySegmentStorage decode(TensorDescriptor descriptor, ByteBuffer canonical) {
        MemorySegmentStorage storage = allocate(descriptor);
        ByteBuffer in = canonical.duplicate().order(ByteOrder.BIG_ENDIAN);
        MemorySegment target = storage.segment();
        int width = descriptor.dataType().byteWidth();
        for (long p = 0; in.hasRemaining(); p += width) {
            switch (width) {
                case 8 -> target.set(LONG, p, in.getLong());
                case 4 -> target.set(INT, p, in.getInt());
                case 2 -> target.set(SHORT, p, in.getShort());
                case 1 -> target.set(ValueLayout.JAVA_BYTE, p, in.get());
                default -> throw new AssertionError("Unknown logical width");
            }
        }
        return storage;
    }

    /**
     * Allocates no descriptor, layout-closing Tensor or native owner.
     * @param descriptor static canonical leaf metadata within the JVM payload ceiling
     * @return private zero-filled storage with the exact logical primitive carrier
     */
    private static MemorySegmentStorage allocate(TensorDescriptor descriptor) {
        int count = payloadLength(descriptor) / descriptor.dataType().byteWidth();
        MemorySegment segment = switch (descriptor.dataType()) {
            case FLOAT64 -> MemorySegment.ofArray(new double[count]);
            case FLOAT32 -> MemorySegment.ofArray(new float[count]);
            case INT64 -> MemorySegment.ofArray(new long[count]);
            case INT32 -> MemorySegment.ofArray(new int[count]);
            case BFLOAT16, FLOAT16 -> MemorySegment.ofArray(new short[count]);
            case BOOL -> MemorySegment.ofArray(new byte[count]);
        };
        return new MemorySegmentStorage(descriptor.dataType(), count, segment);
    }
}
