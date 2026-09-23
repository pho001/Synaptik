package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Immutable direct CPU/Metal FLOAT32 transfer recipe for one prepared logical value.
 *
 * <p>Construction retains the exact two backend integrations and logical descriptor after checking
 * static canonical contiguous FLOAT32 geometry against the shared prepared memory slot. Cold
 * binding validates both concrete representations and captures one native CPU segment plus one
 * exact Metal upload or download action. The hot action performs no allocation, route search,
 * registry lookup, backend discovery, representation lookup, or validity mutation.</p>
 */
final class CpuMetalPreparedBufferTransfer extends PreparedBufferTransfer {
    private enum Direction {
        CPU_TO_METAL,
        METAL_TO_CPU
    }

    private final CpuBackendIntegration cpu;
    private final MetalBackendIntegration metal;
    private final TensorDescriptor descriptor;
    private final Direction direction;

    private CpuMetalPreparedBufferTransfer(
            CpuBackendIntegration cpu,
            MetalBackendIntegration metal,
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex,
            TensorDescriptor descriptor,
            Direction direction) {
        super(
                memoryPlan,
                bufferIndex,
                sourceRepresentationIndex,
                destinationRepresentationIndex);
        this.cpu = Objects.requireNonNull(cpu, "cpu");
        this.metal = Objects.requireNonNull(metal, "metal");
        this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
        this.direction = Objects.requireNonNull(direction, "direction");
        long byteCount = transferByteCount(descriptor);
        if (byteCount < 0L) {
            throw new IllegalArgumentException(
                    "CPU/Metal transfer requires positive rank-1..16 static canonical contiguous "
                            + "FLOAT32 with a checked byte extent");
        }
        if (memoryPlan.buffers().get(bufferIndex).byteSize() != byteCount) {
            throw new IllegalArgumentException(
                    "transfer descriptor byte extent does not match prepared memory slot");
        }
    }

    /**
     * Reports support for the one currently implemented mixed-owner transfer geometry.
     *
     * @param descriptor non-null exact logical descriptor
     * @return whether it has rank 1..16, positive fully static extents, canonical contiguous
     *     FLOAT32 layout, and a checked element and byte count
     */
    static boolean supports(TensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        return transferByteCount(descriptor) >= 0L;
    }

    private static long transferByteCount(TensorDescriptor descriptor) {
        if (descriptor.dataType() != DataType.FLOAT32
                || !descriptor.shape().isFullyStatic()
                || descriptor.shape().rank() < 1
                || descriptor.shape().rank() > 16
                || descriptor.layout().isEmpty()) {
            return -1L;
        }
        long elementCount = 1L;
        long byteCount;
        try {
            for (long dimension : descriptor.shape().toLongArray()) {
                if (dimension <= 0L) {
                    return -1L;
                }
                elementCount = Math.multiplyExact(elementCount, dimension);
            }
            byteCount = Math.multiplyExact(elementCount, Float.BYTES);
        } catch (ArithmeticException overflow) {
            return -1L;
        }
        return descriptor.layout().orElseThrow().equals(
                LayoutDescriptor.contiguous(descriptor.shape()))
                ? byteCount
                : -1L;
    }

    /** Creates one checked CPU-to-Metal upload recipe. */
    static PreparedBufferTransfer cpuToMetal(
            CpuBackendIntegration cpu,
            MetalBackendIntegration metal,
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex,
            TensorDescriptor descriptor) {
        return new CpuMetalPreparedBufferTransfer(
                cpu,
                metal,
                memoryPlan,
                bufferIndex,
                sourceRepresentationIndex,
                destinationRepresentationIndex,
                descriptor,
                Direction.CPU_TO_METAL);
    }

    /** Creates one checked Metal-to-CPU download recipe. */
    static PreparedBufferTransfer metalToCpu(
            CpuBackendIntegration cpu,
            MetalBackendIntegration metal,
            PreparedMemoryPlan memoryPlan,
            int bufferIndex,
            int sourceRepresentationIndex,
            int destinationRepresentationIndex,
            TensorDescriptor descriptor) {
        return new CpuMetalPreparedBufferTransfer(
                cpu,
                metal,
                memoryPlan,
                bufferIndex,
                sourceRepresentationIndex,
                destinationRepresentationIndex,
                descriptor,
                Direction.METAL_TO_CPU);
    }

    @Override
    protected boolean acceptsSourceBufferRepresentation(BufferRepresentation representation) {
        return switch (direction) {
            case CPU_TO_METAL -> cpu.acceptsContiguousFloat32Transfer(
                    representation, descriptor, false);
            case METAL_TO_CPU -> metal.acceptsContiguousFloat32Transfer(
                    representation, descriptor);
        };
    }

    @Override
    protected boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation) {
        return switch (direction) {
            case CPU_TO_METAL -> metal.acceptsContiguousFloat32Transfer(
                    representation, descriptor);
            case METAL_TO_CPU -> cpu.acceptsContiguousFloat32Transfer(
                    representation, descriptor, true);
        };
    }

    @Override
    protected BoundBufferTransfer bindCompatible(
            RunState runState,
            BufferRepresentation sourceRepresentation,
            BufferRepresentation destinationRepresentation) {
        MemorySegment hostSegment;
        Consumer<MemorySegment> transfer;
        switch (direction) {
            case CPU_TO_METAL -> {
                hostSegment = cpu.bindContiguousFloat32Transfer(
                        sourceRepresentation, descriptor, false);
                transfer = metal.bindContiguousFloat32Upload(
                        destinationRepresentation, descriptor);
            }
            case METAL_TO_CPU -> {
                hostSegment = cpu.bindContiguousFloat32Transfer(
                        destinationRepresentation, descriptor, true);
                transfer = metal.bindContiguousFloat32Download(
                        sourceRepresentation, descriptor);
            }
            default -> throw new AssertionError("unknown CPU/Metal transfer direction");
        }
        return new Bound(
                runState,
                bufferIndex(),
                sourceRepresentationIndex(),
                destinationRepresentationIndex(),
                hostSegment,
                transfer);
    }

    /** Direct per-run action retaining only the exact native host segment and Metal operation. */
    private static final class Bound extends BoundBufferTransfer {
        private final MemorySegment hostSegment;
        private final Consumer<MemorySegment> transfer;

        private Bound(
                RunState runState,
                int bufferIndex,
                int sourceRepresentationIndex,
                int destinationRepresentationIndex,
                MemorySegment hostSegment,
                Consumer<MemorySegment> transfer) {
            super(
                    runState,
                    bufferIndex,
                    sourceRepresentationIndex,
                    destinationRepresentationIndex);
            this.hostSegment = Objects.requireNonNull(hostSegment, "hostSegment");
            this.transfer = Objects.requireNonNull(transfer, "transfer");
        }

        @Override
        protected void executeTransfer() {
            transfer.accept(hostSegment);
        }
    }
}
