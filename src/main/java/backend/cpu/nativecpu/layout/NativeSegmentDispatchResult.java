package backend.cpu.nativecpu.layout;

/**
 * Observable result of a MemorySegment kernel dispatch decision.
 */
public record NativeSegmentDispatchResult(
        NativeSegmentKernelFamily family,
        int logicalSize,
        int chunks,
        boolean parallel,
        String reason
) {
    public NativeSegmentDispatchResult {
        if (family == null) {
            throw new IllegalArgumentException("family cannot be null");
        }
        logicalSize = Math.max(0, logicalSize);
        chunks = Math.max(0, chunks);
        reason = reason == null ? "" : reason;
    }

    static NativeSegmentDispatchResult scalar(NativeSegmentKernelFamily family, int logicalSize, String reason) {
        return new NativeSegmentDispatchResult(family, logicalSize, 1, false, reason);
    }

    static NativeSegmentDispatchResult parallel(NativeSegmentKernelFamily family, int logicalSize, int chunks, String reason) {
        return new NativeSegmentDispatchResult(family, logicalSize, chunks, true, reason);
    }
}
