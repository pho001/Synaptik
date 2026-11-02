package io.github.pho001.synaptik.runtime.run;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Authenticated immutable numeric sizing for one run, resolved before physical allocation.
 * Private arrays never escape; the exact memory-plan reference, not record equality, authenticates
 * subsequent binding. This owner knows no graph, shape, backend or physical address.
 */
public final class RunMemoryGeometry {
    private final PreparedMemoryPlan plan;
    private final OptionalLong batchSize;
    private final long[] buffers;
    private final long[] workspaces;

    private RunMemoryGeometry(PreparedMemoryPlan plan, OptionalLong batchSize,
            long[] buffers, long[] workspaces) {
        this.plan = plan;
        this.batchSize = batchSize;
        this.buffers = buffers;
        this.workspaces = workspaces;
    }

    /**
     * Resolves every declaration in buffer-then-workspace order without acquiring storage.
     * @param plan non-null exact prepared memory plan
     * @param batchSize non-null optional; present iff the plan explicitly requires B
     * @return fresh immutable geometry retaining the exact plan
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if mode, batch size or a resolved byte count is invalid
     * @throws ArithmeticException if checked recipe arithmetic overflows
     */
    public static RunMemoryGeometry resolve(PreparedMemoryPlan plan, OptionalLong batchSize) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(batchSize, "batchSize");
        if (plan.requiresBatchSize() != batchSize.isPresent()) {
            throw new IllegalArgumentException("batchSize presence differs from prepared mode");
        }
        long batch = batchSize.orElse(0);
        if (batch < 0) throw new IllegalArgumentException("batchSize must be non-negative");
        long[] buffers = new long[plan.buffers().size()];
        long[] workspaces = new long[plan.workspaces().size()];
        for (int i = 0; i < buffers.length; i++) {
            buffers[i] = checked(plan.buffers().get(i).size().byteSize(batch));
        }
        for (int i = 0; i < workspaces.length; i++) {
            workspaces[i] = checked(plan.workspaces().get(i).size().byteSize(batch));
        }
        return new RunMemoryGeometry(plan, batchSize, buffers, workspaces);
    }

    private static long checked(long bytes) {
        if (bytes < 0) throw new IllegalArgumentException("resolved byteSize must be non-negative");
        return bytes;
    }

    /** Returns the sizing authority by reference identity.
     * @return the exact non-null authenticated prepared plan */
    public PreparedMemoryPlan memoryPlan() { return plan; }
    /** Returns explicit invocation mode and extent.
     * @return the immutable optional actual non-negative B; empty only in static mode */
    public OptionalLong batchSize() { return batchSize; }

    /**
     * Reads an already resolved buffer size, never re-evaluating a recipe.
     * @param bufferIndex dense zero-based plan position
     * @return exact non-negative usable bytes
     * @throws IndexOutOfBoundsException if the index is outside the plan
     */
    public long bufferByteSize(int bufferIndex) { return buffers[bufferIndex]; }

    /**
     * Reads an already resolved workspace size, never re-evaluating a recipe.
     * @param workspaceIndex dense zero-based plan position
     * @return exact non-negative usable scratch bytes
     * @throws IndexOutOfBoundsException if the index is outside the plan
     */
    public long workspaceByteSize(int workspaceIndex) { return workspaces[workspaceIndex]; }
}
