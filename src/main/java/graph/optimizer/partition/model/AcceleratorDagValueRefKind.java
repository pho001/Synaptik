package graph.optimizer.partition.model;

public enum AcceleratorDagValueRefKind {
    NONE(0),
    EXTERNAL_INPUT(1),
    NODE_OUTPUT(2);

    private final int abiCode;

    AcceleratorDagValueRefKind(int abiCode) {
        this.abiCode = abiCode;
    }

    public int abiCode() {
        return abiCode;
    }
}
