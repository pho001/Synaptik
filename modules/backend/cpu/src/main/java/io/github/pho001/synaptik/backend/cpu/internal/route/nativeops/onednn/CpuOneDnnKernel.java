package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.runtime.resource.PreparedResource;

/**
 * Immutable native code and descriptors owned as a persistent Runtime prepared resource.
 * Closing prevents new workspace creation and releases the owner's lease; already-issued
 * workspaces retain the kernel and library until their own close. No mutable invocation
 * state or data snapshot belongs to this resource.
 */
public interface CpuOneDnnKernel extends PreparedResource {
    /**
     * Returns the exact descriptor recipe authenticated during finalization.
     *
     * @return non-null immutable numeric specification; no ownership transfers
     */
    CpuOneDnnKernelSpec spec();
    /**
     * Reports the exact user-scratch requirement queried from the native descriptor.
     *
     * @return non-negative scratch byte count, excluding ingress copies and ABI control tables
     */
    long scratchpadBytes();
    /**
     * Identifies the implementation whose facts must match the analyzed recipe.
     *
     * @return non-null non-blank copied implementation identity, not a native pointer
     */
    String implementation();
    /**
     * Acquires isolated invocation resources before cold binding.
     *
     * @return fresh non-null run-owned workspace with stream, native data handles, ABI tables,
     *     user scratch and a retained kernel/library lease; its owner must close it
     * @throws IllegalStateException if this kernel is closed or native resource creation fails
     * @throws ArithmeticException if retaining the kernel lease overflows
     */
    CpuOneDnnWorkspace createWorkspace();
}
