package Operations;
import Backend.ComputeBackend;
import Tensor.Tensor;

import java.util.List;

public interface Operation {
    enum OpType {
        ADD,
        SUB,
        MUL,
        DIV,
        MIN,
        MAX,
        NEG,
        INV,
        LOG,
        EXP,
        TANH,
        POW,
        SQRT,
        MUL_SCALAR,
        SUM,
        RELU,
        SIGMOID,
        CONTIGUOUS,
        NOOP,
        FUSED,
        UNKNOWN
    }

    OpType opType();

    boolean isElementWise();
    void apply(List<Tensor> inputs, Tensor out);

    default void gradient(List<Tensor> inputs, Tensor out) {}
    ComputeBackend getPreferredBackend();
    boolean supportsBackend(ComputeBackend backend);

    String getExpression();

    default boolean requiresOutputForGradient() { return false; }

    default boolean isCheap() { return false; }


}
