package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import java.lang.foreign.MemorySegment;

/** Native operation whose invocation can borrow one pipeline's charged scratch and ordered stream. */
interface CpuOneDnnAtomicKernel extends CpuOneDnnKernel {
    /**
     * Acquires only invocation controls and a retained lease, never another scratch allocation.
     * @param scratchpad exact writable aligned scratch slice borrowed until workspace close
     * @param stream pipeline-owned in-order native stream, borrowed until workspace close
     * @return isolated unbound invocation; close it before either borrowed resource
     */
    CpuOneDnnWorkspace createWorkspace(MemorySegment scratchpad, MemorySegment stream);
}
