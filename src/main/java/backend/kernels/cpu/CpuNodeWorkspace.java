package backend.kernels.cpu;

public final class CpuNodeWorkspace {
    private final int[] intWorkspace;

    private CpuNodeWorkspace(int[] intWorkspace) {
        this.intWorkspace = intWorkspace;
    }

    public static CpuNodeWorkspace withIntWorkspace(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("Workspace size cannot be negative.");
        }
        return new CpuNodeWorkspace(new int[size]);
    }

    public int[] requireIntWorkspace() {
        if (intWorkspace == null) {
            throw new IllegalStateException("CPU node workspace does not provide int[] storage.");
        }
        return intWorkspace;
    }
}
