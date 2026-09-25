package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Prepared owner of one immutable source-splat buffer and its run-owned read bindings.
 *
 * <p>The owner allocates, fills, and uploads exactly once before publication. Every run receives a
 * fresh nominal binding and therefore preserves Runtime's exact representation-object ownership.
 * Closing the prepared owner rejects new bindings and releases its ownership reference. An already
 * issued binding is a child lease that may defer the one physical buffer release until the last
 * binding closes. The buffer is never exposed as a writable destination after construction.</p>
 */
final class MetalPreparedSplatResource implements PreparedResource {
    private final MetalDeviceContext context;
    private final long byteSize;
    private final DataType dataType;
    private final int rawBits;
    private final MetalBufferRepresentation buffer;
    private int activeBindings;
    private boolean ownerClosed;
    private boolean releaseClaimed;

    private MetalPreparedSplatResource(
            MetalDeviceContext context,
            long byteSize,
            DataType dataType,
            int rawBits,
            MetalBufferRepresentation buffer) {
        this.context = Objects.requireNonNull(context, "context");
        this.byteSize = byteSize;
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.rawBits = rawBits;
        this.buffer = Objects.requireNonNull(buffer, "buffer");
    }

    /**
     * Allocates and initializes one prepared immutable splat.
     *
     * @param context exact non-null device context
     * @param byteSize positive four-byte-aligned logical extent
     * @param scalar exact non-null FLOAT32 or INT32 scalar
     * @return a new open prepared owner
     * @throws IllegalArgumentException if type or extent is outside the Metal splat domain
     * @throws RuntimeException if native allocation, upload, or cleanup fails
     * @throws Error if allocation, upload, or cleanup reports an error
     */
    static MetalPreparedSplatResource create(
            MetalDeviceContext context, long byteSize, ScalarValue scalar) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(scalar, "scalar");
        if (byteSize <= 0L || byteSize % Integer.BYTES != 0L
                || (scalar.dataType() != DataType.FLOAT32
                        && scalar.dataType() != DataType.INT32)) {
            throw new IllegalArgumentException(
                    "Metal prepared splat requires a positive four-byte FLOAT32 or INT32 extent");
        }
        int rawBits = raw32Bits(scalar);
        MetalBufferRepresentation buffer = context.createBuffer(byteSize);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(byteSize, Integer.BYTES);
            long elements = byteSize / Integer.BYTES;
            for (long index = 0L; index < elements; index++) {
                source.setAtIndex(JAVA_INT, index, rawBits);
            }
            buffer.upload(0L, source, 0L, byteSize);
            return new MetalPreparedSplatResource(
                    context, byteSize, scalar.dataType(), rawBits, buffer);
        } catch (RuntimeException | Error failure) {
            closeAfterFailure(buffer, failure);
            throw failure;
        }
    }

    /**
     * Creates one fresh read-only run-owned binding.
     *
     * @return a new binding that owns one child lease
     * @throws IllegalStateException if prepared-owner close has begun
     */
    BufferRepresentation newRunBinding() {
        Binding binding = new Binding(this);
        synchronized (this) {
            if (ownerClosed) {
                throw new IllegalStateException("Metal prepared splat resource is closed");
            }
            activeBindings++;
        }
        return binding;
    }

    /** Returns this binding's live physical source when exact immutable facts agree. */
    static MetalBufferRepresentation exactReadBuffer(
            BufferRepresentation representation,
            MetalDeviceContext expectedContext,
            long expectedBytes,
            ScalarValue expectedScalar) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(expectedContext, "expectedContext");
        Objects.requireNonNull(expectedScalar, "expectedScalar");
        if (!(representation instanceof Binding binding)) {
            return null;
        }
        return binding.owner.exactReadBuffer(
                binding, expectedContext, expectedBytes, expectedScalar);
    }

    /** Returns a live physical source for transfer/materialization, or {@code null}. */
    static MetalBufferRepresentation readableBuffer(BufferRepresentation representation) {
        Objects.requireNonNull(representation, "representation");
        if (representation instanceof MetalBufferRepresentation metal) {
            return metal;
        }
        if (representation instanceof Binding binding) {
            return binding.owner.readableBuffer(binding);
        }
        return null;
    }

    /** @return whether this nominal representation is a prepared immutable binding */
    static boolean isBinding(BufferRepresentation representation) {
        return representation instanceof Binding;
    }

    @Override
    public void close() {
        MetalBufferRepresentation release = null;
        synchronized (this) {
            if (ownerClosed) {
                return;
            }
            ownerClosed = true;
            if (activeBindings == 0 && !releaseClaimed) {
                releaseClaimed = true;
                release = buffer;
            }
        }
        if (release != null) {
            release.close();
        }
    }

    private synchronized MetalBufferRepresentation exactReadBuffer(
            Binding binding,
            MetalDeviceContext expectedContext,
            long expectedBytes,
            ScalarValue expectedScalar) {
        if (binding.closed
                || context != expectedContext
                || byteSize != expectedBytes
                || dataType != expectedScalar.dataType()
                || rawBits != raw32Bits(expectedScalar)
                || !buffer.belongsTo(context)) {
            return null;
        }
        return buffer;
    }

    private synchronized MetalBufferRepresentation readableBuffer(Binding binding) {
        return binding.closed || !buffer.belongsTo(context) ? null : buffer;
    }

    private void closeBinding(Binding binding) {
        MetalBufferRepresentation release = null;
        synchronized (this) {
            if (binding.closed) {
                return;
            }
            binding.closed = true;
            activeBindings--;
            if (activeBindings < 0) {
                throw new AssertionError("Metal prepared splat binding count underflow");
            }
            if (ownerClosed && activeBindings == 0 && !releaseClaimed) {
                releaseClaimed = true;
                release = buffer;
            }
        }
        if (release != null) {
            release.close();
        }
    }

    private static int raw32Bits(ScalarValue scalar) {
        return switch (scalar.dataType()) {
            case FLOAT32 -> Float.floatToRawIntBits(scalar.float32Value());
            case INT32 -> scalar.int32Value();
            default -> throw new IllegalArgumentException(
                    "Metal prepared splat requires FLOAT32 or INT32");
        };
    }

    private static void closeAfterFailure(
            MetalBufferRepresentation buffer, Throwable failure) {
        try {
            buffer.close();
        } catch (RuntimeException | Error cleanup) {
            if (cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
        }
    }

    /** Fresh run-owned nominal representation leasing the immutable prepared buffer. */
    private static final class Binding implements BufferRepresentation {
        private final MetalPreparedSplatResource owner;
        private boolean closed;

        private Binding(MetalPreparedSplatResource owner) {
            this.owner = owner;
        }

        @Override
        public void close() {
            owner.closeBinding(this);
        }
    }
}
