package backend.cpu1.kernels.dtype;

import backend.cpu1.kernels.dtype.cast.Cpu1CastLoops;

/**
 * Resolves prepared dtype kernel ids to concrete kernels outside the hot path.
 */
public final class Cpu1DTypeKernelDispatch {
    private Cpu1DTypeKernelDispatch() {
    }

    public static Cpu1DTypeKernel kernelFor(Cpu1DTypeKernelId kernelId) {
        if (kernelId == null) {
            throw new IllegalArgumentException("kernelId cannot be null");
        }
        return switch (kernelId) {
            case CAST_ARRAY_SCALAR, CAST_SEGMENT_SCALAR -> Cpu1CastLoops::cast;
        };
    }
}
