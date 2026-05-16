package config.runtime;

/**
 * Native CPU memory allocation and pooling policy.
 *
 * @param poolPolicy native CPU memory pool lifetime policy
 * @param maxPoolBytes maximum bytes retained by a native CPU pool
 * @param alignmentBytes default native CPU allocation alignment
 * @param debugPoisonReleasedBuffers whether released buffers should be poisoned in debug paths
 * @param traceAllocations whether allocation-level trace detail should be captured
 */
public record NativeCpuMemoryConfig(
        NativeMemoryPoolPolicy poolPolicy,
        long maxPoolBytes,
        int alignmentBytes,
        boolean debugPoisonReleasedBuffers,
        boolean traceAllocations
) {
    public NativeCpuMemoryConfig {
        poolPolicy = poolPolicy == null ? NativeMemoryPoolPolicy.DISABLED : poolPolicy;
        maxPoolBytes = Math.max(0L, maxPoolBytes);
        alignmentBytes = alignmentBytes <= 0 ? (int) backend.cpu.nativecpu.NativeCpuAllocator.DEFAULT_ALIGNMENT_BYTES : alignmentBytes;
    }

    public static NativeCpuMemoryConfig disabled() {
        return new NativeCpuMemoryConfig(
                NativeMemoryPoolPolicy.DISABLED,
                0L,
                (int) backend.cpu.nativecpu.NativeCpuAllocator.DEFAULT_ALIGNMENT_BYTES,
                false,
                false
        );
    }

    public static NativeCpuMemoryConfig perExecution(long maxPoolBytes) {
        return new NativeCpuMemoryConfig(
                NativeMemoryPoolPolicy.PER_EXECUTION,
                maxPoolBytes,
                (int) backend.cpu.nativecpu.NativeCpuAllocator.DEFAULT_ALIGNMENT_BYTES,
                false,
                false
        );
    }
}
