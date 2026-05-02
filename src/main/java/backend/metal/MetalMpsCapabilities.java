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
     * Returns whether runtime Metal residency can represent this dtype as storage bytes.
     *
     * <p>This is not native compute support. It only says the runtime can name and carry
     * the dtype in metadata/buffer decisions.</p>
     */
    public static MetalDTypeCapabilityDecision storageDecision(DataType dtype) {
        return supported(
                MetalDTypeRole.STORAGE,
                dtype,
                false,
                false,
                dtype == DataType.FLOAT32
                        ? MetalDTypeReasonCode.SUPPORTED
                        : MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                "storage representable; dtype residency is not native dtype compute"
        );
    }

    /**
     * Returns storage-level external input support.
     *
     * <p>Use {@link #externalInputRoleDecision(CompiledNode, CompiledNode, int)} for
     * planner legality because BOOL is valid only in predicate roles.</p>
     */
    public static MetalDTypeCapabilityDecision externalInputDecision(DataType dtype) {
        if (dtype == DataType.FLOAT32) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "FLOAT32 external data input is supported");
        }
        if (dtype == DataType.BOOL) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT, dtype, false, false, MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY,
                    "BOOL external input is supported only for predicate roles");
        }
        return unsupported(
                MetalDTypeRole.EXTERNAL_INPUT,
                dtype,
                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_DTYPE,
                "external input dtype is not supported by the current Metal execution ABI"
        );
    }

    /**
     * Returns native compute dtype support for the current Metal bridge.
     */
    public static MetalDTypeCapabilityDecision computeDecision(DataType dtype) {
        if (dtype == DataType.FLOAT32) {
            return supported(MetalDTypeRole.COMPUTE, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "FLOAT32 native compute is supported");
        }
        return unsupported(
                MetalDTypeRole.COMPUTE,
                dtype,
                dtype == DataType.FLOAT64
                        ? MetalDTypeReasonCode.FLOAT64_UNSUPPORTED
                        : MetalDTypeReasonCode.UNSUPPORTED_NATIVE_COMPUTE_DTYPE,
                "native Metal compute is FLOAT32 only in the current bridge"
        );
    }

    /**
     * Returns native output dtype support for the current Metal bridge.
     */
    public static MetalDTypeCapabilityDecision outputDecision(DataType dtype) {
        if (dtype == DataType.FLOAT32) {
            return supported(MetalDTypeRole.OUTPUT, dtype, false, true, MetalDTypeReasonCode.SUPPORTED,
                    "FLOAT32 native output is supported");
        }
        return unsupported(
                MetalDTypeRole.OUTPUT,
                dtype,
                dtype == DataType.FLOAT64
                        ? MetalDTypeReasonCode.FLOAT64_UNSUPPORTED
                        : MetalDTypeReasonCode.UNSUPPORTED_NATIVE_OUTPUT_DTYPE,
                "native Metal output publication is FLOAT32 only in the current bridge"
        );
    }

    /**
     * Returns operation-specific dtype support for a Metal compute node output.
     */
    public static MetalDTypeCapabilityDecision operationDecision(Operation.OpType opType, DataType outputDType) {
        if (opType == null) {
            return unsupported(MetalDTypeRole.OPERATION, outputDType, MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE,
                    "operation type is unavailable");
        }
        if (outputDType != DataType.FLOAT32) {
            return unsupported(MetalDTypeRole.OPERATION, outputDType, MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE,
                    "operation " + opType + " cannot produce native " + outputDType + " output on the current Metal bridge");
        }
        return supported(MetalDTypeRole.OPERATION, outputDType, true, true, MetalDTypeReasonCode.SUPPORTED,
                "operation " + opType + " is dtype-legal for FLOAT32 subject to semantic/layout capability checks");
    }

    /**
     * Returns whether a producer may feed the given consumer input from outside a Metal partition.
     */
    public static MetalDTypeCapabilityDecision externalInputRoleDecision(CompiledNode producer, CompiledNode consumer, int inputIndex) {
        if (producer == null || consumer == null || consumer.operation() == null || inputIndex < 0) {
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, DataType.FLOAT32,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "external input role cannot be checked without producer, consumer, operation, and input index");
        }
        Operation.OpType opType = consumer.operation().opType();
        DataType dtype = producer.dataType();
        if (opType == Operation.OpType.WHERE) {
            return switch (inputIndex) {
                case 0 -> dtype == DataType.BOOL
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY,
                                "WHERE condition accepts BOOL predicate input")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                "WHERE condition requires BOOL predicate input");
                case 1, 2 -> dtype == DataType.FLOAT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                "WHERE branch input accepts FLOAT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                "WHERE branch input requires FLOAT32 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        "WHERE has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            if (inputIndex >= 0 && inputIndex <= 2 && dtype == DataType.FLOAT32) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "unmasked SDPA query/key/value inputs accept FLOAT32 data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "Metal direct SDPA currently accepts only FLOAT32 query/key/value inputs and no public BOOL mask input");
        }
        if (dtype == DataType.FLOAT32) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                    MetalDTypeReasonCode.SUPPORTED,
                    "default Metal external data input accepts FLOAT32");
        }
        return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                "default Metal external data input requires FLOAT32");
    }

    /**
     * Returns whether the current Metal bridge can execute compute nodes with this dtype.
     *
     * @param dtype compiled node output/compute dtype
     * @return true only for {@link DataType#FLOAT32}
     */
    public static boolean supportsComputeDType(DataType dtype) {
        return computeDecision(dtype).supported();
    }

    /**
     * Returns whether the current Metal bridge can publish output tensors with this dtype.
     *
     * @param dtype output tensor dtype
     * @return true only for {@link DataType#FLOAT32}
     */
    public static boolean supportsOutputDType(DataType dtype) {
        return outputDecision(dtype).supported();
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
        return externalInputDecision(dtype).supported();
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
        return externalInputRoleDecision(producer, consumer, inputIndex).supported();
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
     * Maps a Java tensor dtype to the optional dtype ABI v3 descriptor code.
     */
    public static int abiDescriptorDataTypeCode(DataType dtype) {
        return backend.metal.bridge.MetalDTypeAbiV3Support.descriptorCode(dtype);
    }

    /**
     * Builds a readable message for unsupported Metal dtype diagnostics.
     *
     * @param dtype rejected dtype
     * @return diagnostic message naming the current supported set
     */
    public static String unsupportedDTypeMessage(DataType dtype) {
        MetalDTypeCapabilityDecision compute = computeDecision(dtype);
        return "UNSUPPORTED_DTYPE: " + compute.detail()
                + "; Metal MPS bridge currently supports FLOAT32 compute/output tensors and BOOL only for predicate inputs; got "
                + dtype + ".";
    }

    private static MetalDTypeCapabilityDecision supported(
            MetalDTypeRole role,
            DataType dtype,
            boolean nativeCompute,
            boolean nativeOutput,
            MetalDTypeReasonCode reasonCode,
            String detail
    ) {
        return new MetalDTypeCapabilityDecision(
                role,
                dtype,
                true,
                true,
                nativeCompute,
                nativeOutput,
                reasonCode,
                detail(role, dtype, reasonCode, detail)
        );
    }

    private static MetalDTypeCapabilityDecision unsupported(
            MetalDTypeRole role,
            DataType dtype,
            MetalDTypeReasonCode reasonCode,
            String detail
    ) {
        return new MetalDTypeCapabilityDecision(
                role,
                dtype,
                false,
                true,
                false,
                false,
                reasonCode,
                detail(role, dtype, reasonCode, detail)
        );
    }

    private static String detail(MetalDTypeRole role, DataType dtype, MetalDTypeReasonCode reasonCode, String detail) {
        return "backend=GPU_METAL role=" + role.label()
                + " dtype=" + dtype
                + " code=" + reasonCode
                + " " + detail;
    }
}
