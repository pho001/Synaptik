package backend.kernels.cpu;

public final class CpuNodeWorkspace {
    private final int[] intWorkspace;
    private final float[] floatWorkspace;

    private CpuNodeWorkspace(int[] intWorkspace, float[] floatWorkspace) {
        this.intWorkspace = intWorkspace;
        this.floatWorkspace = floatWorkspace;
    }

    public static CpuNodeWorkspace withIntWorkspace(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("Workspace size cannot be negative.");
        }
        return new CpuNodeWorkspace(new int[size], null);
    }

    public static CpuNodeWorkspace withFloatWorkspace(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("Workspace size cannot be negative.");
        }
        return new CpuNodeWorkspace(null, new float[size]);
    }

    public static CpuNodeWorkspace withIntAndFloatWorkspace(int intSize, int floatSize) {
        if (intSize < 0 || floatSize < 0) {
            throw new IllegalArgumentException("Workspace sizes cannot be negative.");
        }
        return new CpuNodeWorkspace(
                intSize == 0 ? null : new int[intSize],
                floatSize == 0 ? null : new float[floatSize]
        );
    }

    public int[] requireIntWorkspace() {
        if (intWorkspace == null) {
            throw new IllegalStateException("CPU node workspace does not provide int[] storage.");
        }
        return intWorkspace;
    }

    public float[] requireFloatWorkspace() {
        if (floatWorkspace == null) {
            throw new IllegalStateException("CPU node workspace does not provide float[] storage.");
        }
        return floatWorkspace;
    }
}
