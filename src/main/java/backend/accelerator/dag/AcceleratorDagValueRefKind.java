package backend.accelerator.dag;

/**
 * Native ABI namespace for an accelerator DAG value reference.
 */
public enum AcceleratorDagValueRefKind {
    NONE(0),
    EXTERNAL_INPUT(1),
    NODE_OUTPUT(2);

    private final int abiCode;

    AcceleratorDagValueRefKind(int abiCode) {
        this.abiCode = abiCode;
    }

    /**
     * Returns the integer namespace code consumed by native accelerator shims.
     */
    public int abiCode() {
        return abiCode;
    }
}
