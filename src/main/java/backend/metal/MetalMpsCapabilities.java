package backend.metal;

import graph.CompiledNode;
import operations.Operation;
import tensor.DataType;

/**
 * Capability contract for the current Metal MPS FFM bridge.
 *
 * <p>This class is the Java-side source of truth for the dtype subset exposed by the
 * native {@code synaptik_apple_mps_*_f32} ABI. It intentionally describes only what
 * the bridge can execute today: float32 compute/output tensors, float32 data inputs,
 * and bool external inputs in predicate roles.</p>
 */
public final class MetalMpsCapabilities {
    private MetalMpsCapabilities() {
    }

    /**
     * Returns whether the current Metal bridge can execute compute nodes with this dtype.
     *
     * @param dtype compiled node output/compute dtype
     * @return true only for {@link DataType#FLOAT32}
     */
    public static boolean supportsComputeDType(DataType dtype) {
        return dtype == DataType.FLOAT32;
    }

    /**
     * Returns whether the current Metal bridge can publish output tensors with this dtype.
     *
     * @param dtype output tensor dtype
     * @return true only for {@link DataType#FLOAT32}
     */
    public static boolean supportsOutputDType(DataType dtype) {
        return dtype == DataType.FLOAT32;
    }

    /**
     * Returns whether an external input dtype is representable by the native ABI.
     *
     * <p>This is a storage-level predicate. Planner legality should prefer
     * {@link #supportsExternalInputRole(CompiledNode, CompiledNode, int)} because bool
     * tensors are valid only in predicate positions.</p>
     *
     * @param dtype external input dtype
     * @return true for float32 data tensors and bool predicate tensors
     */
    public static boolean supportsExternalInputDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BOOL;
    }

    /**
     * Returns whether a producer may feed the given consumer input from outside a Metal partition.
     *
     * <p>Bool tensors are deliberately role-limited: {@code WHERE} input 0 may be bool
     * and all other data inputs must be float32. Direct SDPA with a Java bool mask is
     * rejected for Metal because the native MPSGraph SDPA mask operand on supported
     * systems expects a floating mask tensor, not the framework's public bool-mask
     * semantics.</p>
     *
     * @param producer compiled producer outside the Metal candidate
     * @param consumer compiled Metal consumer inside the candidate
     * @param inputIndex input position on {@code consumer}
     * @return true when the producer dtype is legal for this specific role
     */
    public static boolean supportsExternalInputRole(CompiledNode producer, CompiledNode consumer, int inputIndex) {
        if (producer == null || consumer == null || consumer.operation() == null || inputIndex < 0) {
            return false;
        }
        Operation.OpType opType = consumer.operation().opType();
        DataType dtype = producer.dataType();
        if (opType == Operation.OpType.WHERE) {
            return switch (inputIndex) {
                case 0 -> dtype == DataType.BOOL;
                case 1, 2 -> dtype == DataType.FLOAT32;
                default -> false;
            };
        }
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            if (inputIndex >= 0 && inputIndex <= 2) {
                return dtype == DataType.FLOAT32;
            }
            return false;
        }
        return dtype == DataType.FLOAT32;
    }

    /**
     * Maps a Java tensor dtype to the native Metal DAG dtype code.
     *
     * @param dtype supported bridge dtype
     * @return native ABI dtype code
     * @throws IllegalArgumentException when {@code dtype} is not supported by the current bridge
     */
    public static int abiDataTypeCode(DataType dtype) {
        return switch (dtype) {
            case FLOAT32 -> 1;
            case BOOL -> 2;
            default -> throw new IllegalArgumentException(unsupportedDTypeMessage(dtype));
        };
    }

    /**
     * Builds a readable message for unsupported Metal dtype diagnostics.
     *
     * @param dtype rejected dtype
     * @return diagnostic message naming the current supported set
     */
    public static String unsupportedDTypeMessage(DataType dtype) {
        return "Metal MPS bridge currently supports FLOAT32 compute/output tensors and BOOL only for predicate inputs; got " + dtype + ".";
    }
}
