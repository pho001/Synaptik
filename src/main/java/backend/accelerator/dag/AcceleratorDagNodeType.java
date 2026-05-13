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
    NE(46),
    LOGICAL_AND(47),
    LOGICAL_OR(48),
    LOGICAL_NOT(49),
    REDUCE_ALL(50),
    REDUCE_ANY(51),
    GATHER(52),
    TAKE_ALONG_AXIS(53),
    CONV2D(54),
    MAX_POOL2D(55),
    AVG_POOL2D(56),
    MIN(57),
    MAX(58),
    POW_SCALAR(59),
    EXPAND(60),
    SELECT(61),
    SDPA_WEIGHTS(62),
    SCATTER_ADD(63),
    GATHER_GRAD(64),
    TAKE_ALONG_AXIS_GRAD(65),
    CONV2D_BACKWARD_INPUT(66),
    CONV2D_BACKWARD_WEIGHT(67),
    AVG_POOL2D_BACKWARD_INPUT(68),
    MAX_POOL2D_BACKWARD_INPUT(69),
    CROSS_ENTROPY_LOSS_INDICES(70),
    CROSS_ENTROPY_LOSS_INDICES_GRAD(71),
    GATHER_AXIS(72),
    GATHER_AXIS_GRAD(73),
    SLICE(74),
    CONCAT(75),
    PAD(76),
    TILE(77),
    REDUCE_PROD(78),
    ARGMAX(79),
    CUMSUM(80);

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
