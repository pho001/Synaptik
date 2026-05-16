package tensor;

import backend.cpu.nativecpu.NativeCpuAllocation;

import java.lang.foreign.MemorySegment;

/**
 * {@link MemorySegment}-backed CPU tensor storage for native CPU kernels.
 *
 * <p>This is an internal runtime/storage contract, not a public tensor-construction API. Public
 * {@link Tensor} instances remain logical tensors; prepared execution state decides whether a runtime value
 * also has a native CPU representation.</p>
 */
public interface NativeTensorStorage extends TensorStorage, AutoCloseable {
    MemorySegment segment();

    long byteOffset();

    long byteSize();

    long elementSizeBytes();

    boolean ownsSegment();

    NativeCpuAllocation allocation();

    boolean closed();

    void ensureOpen();

    @Override
    void close();
}
