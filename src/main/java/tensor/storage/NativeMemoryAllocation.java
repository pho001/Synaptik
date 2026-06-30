package tensor.storage;

import tensor.DataType;

import runtime.memory.ExecutionResource;

import java.lang.foreign.MemorySegment;

/**
 * Backend-neutral native memory allocation handle used by native tensor storage views.
 */
public interface NativeMemoryAllocation extends ExecutionResource {
    MemorySegment segment();

    long byteSize();

    boolean closed();

    void ensureOpen();
}
