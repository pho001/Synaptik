package graph.optimizer.partition.model;

public enum AcceleratorPostOpType {
    RELU(1, false),
    TANH(2, false),
    SIGMOID(3, false),
    ABS(4, false),
    EXP(5, false),
    LOG(6, false),
    NEG(7, false),
    SQRT(8, false),
    INV(9, false),
    MUL(10, true),
    DIV(11, true),
    SUB(12, true),
    CLAMP_MIN(13, false),
    CLAMP_MAX(14, false),
    ADD(15, true),
    MUL_SCALAR(16, false);

    private final int abiCode;
    private final boolean binary;

    AcceleratorPostOpType(int abiCode, boolean binary) {
        this.abiCode = abiCode;
        this.binary = binary;
    }

    public int abiCode() {
        return abiCode;
    }

    public boolean binary() {
        return binary;
    }
}
