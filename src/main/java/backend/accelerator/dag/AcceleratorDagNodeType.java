package backend.accelerator.dag;

/**
 * Native ABI operation codes for lowered accelerator DAG nodes.
 */
public enum AcceleratorDagNodeType {
    MATMUL(1),
    LINEAR(2),
    ADD(3),
    SUB(4),
    MUL(5),
    DIV(6),
    RELU(7),
    TANH(8),
    SIGMOID(9),
    ABS(10),
    EXP(11),
    LOG(12),
    NEG(13),
    SQRT(14),
    INV(15),
    CLAMP_MIN(16),
    CLAMP_MAX(17),
    RESHAPE(18),
    CONTIGUOUS(19),
    PERMUTE(20),
    EXPAND_DIMS(21),
    SQUEEZE(22),
    MUL_SCALAR(23),
    WHERE(24),
    SOFTMAX(25),
    SDPA(26),
    SOFTMAX_GRAD(27),
    LOG_SOFTMAX_GRAD(28),
    REDUCE_MIN_GRAD(29),
    REDUCE_MAX_GRAD(30),
    MIN_GRAD(31),
    MAX_GRAD(32),
    SDPA_BACKWARD_QUERY(33),
    SDPA_BACKWARD_KEY(34),
    SDPA_BACKWARD_VALUE(35),
    SUM(36),
    MEAN(37),
    REDUCE_MIN(38),
    REDUCE_MAX(39),
    ADD_SCALAR(40),
    GT(41),
    GE(42),
    LT(43),
    LE(44),
    EQ(45),
    NE(46);

    private final int abiCode;

    AcceleratorDagNodeType(int abiCode) {
        this.abiCode = abiCode;
    }

    /**
     * Returns the integer operation code consumed by native CUDA and Metal shims.
     */
    public int abiCode() {
        return abiCode;
    }
}
