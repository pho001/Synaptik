package backend.metal;

import graph.CompiledNode;
import operations.Operation;
import tensor.DataType;

/**
 * Capability contract for the current Metal MPS FFM bridge.
 *
 * <p>This class is the Java-side source of truth for the dtype subset exposed by the
 * native {@code synaptik_apple_mps_*} ABI. It intentionally describes only what
 * the bridge can execute today: float32 compute/output tensors, bfloat16 parity
 * for Metal-supported floating operation families, scoped bool outputs, bool external inputs in predicate roles,
 * int32 external inputs in index roles, and scoped int32 index outputs.</p>
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
        if (dtype == DataType.BFLOAT16) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "BFLOAT16 external data input is supported for BF16 Metal floating operation families");
        }
        if (dtype == DataType.BOOL) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT, dtype, false, false, MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY,
                    "BOOL external input is supported only for predicate roles");
        }
        if (dtype == DataType.INT32) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT, dtype, false, false, MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                    "INT32 external input is supported only for index tensor roles");
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
        if (dtype == DataType.BFLOAT16) {
            return supported(MetalDTypeRole.COMPUTE, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "BFLOAT16 native compute is supported for Metal floating operation families");
        }
        if (dtype == DataType.BOOL) {
            return supported(MetalDTypeRole.COMPUTE, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "BOOL native compute is supported for scoped compare/logical/reduction and predicate layout operation families");
        }
        if (dtype == DataType.INT32) {
            return supported(MetalDTypeRole.COMPUTE, dtype, true, false, MetalDTypeReasonCode.SUPPORTED,
                    "INT32 native compute is supported only for scoped index-output operation families such as ARGMAX");
        }
        return unsupported(
                MetalDTypeRole.COMPUTE,
                dtype,
                dtype == DataType.FLOAT64
                        ? MetalDTypeReasonCode.FLOAT64_UNSUPPORTED
                        : MetalDTypeReasonCode.UNSUPPORTED_NATIVE_COMPUTE_DTYPE,
                "native Metal compute is FLOAT32/BFLOAT16 for supported floating operation families, BOOL compare/logical/reduction and predicate layout operation families, plus scoped INT32 index-output operation families in the current bridge"
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
        if (dtype == DataType.BFLOAT16) {
            return supported(MetalDTypeRole.OUTPUT, dtype, false, true, MetalDTypeReasonCode.SUPPORTED,
                    "BFLOAT16 native output is supported for Metal floating operation families");
        }
        if (dtype == DataType.BOOL) {
            return supported(MetalDTypeRole.OUTPUT, dtype, false, true, MetalDTypeReasonCode.SUPPORTED,
                    "BOOL native output is supported for scoped compare/logical/reduction and predicate layout operation families");
        }
        if (dtype == DataType.INT32) {
            return supported(MetalDTypeRole.OUTPUT, dtype, false, true, MetalDTypeReasonCode.SUPPORTED,
                    "INT32 native output is supported only for scoped index-output operation families such as ARGMAX");
        }
        return unsupported(
                MetalDTypeRole.OUTPUT,
                dtype,
                dtype == DataType.FLOAT64
                        ? MetalDTypeReasonCode.FLOAT64_UNSUPPORTED
                        : MetalDTypeReasonCode.UNSUPPORTED_NATIVE_OUTPUT_DTYPE,
                "native Metal output publication is FLOAT32/BFLOAT16 for supported floating operation families, BOOL compare/logical/reduction and predicate layout operation families, plus scoped INT32 index-output operation families in the current bridge"
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
        if (outputDType == DataType.BFLOAT16) {
            if (supportsBFloat16Operation(opType)) {
                return supported(MetalDTypeRole.OPERATION, outputDType, true, true, MetalDTypeReasonCode.SUPPORTED,
                        "operation " + opType + " is dtype-legal for BFLOAT16 in the Metal bridge");
            }
            return unsupported(MetalDTypeRole.OPERATION, outputDType, MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE,
                    "operation " + opType + " cannot produce native BFLOAT16 output on the current Metal bridge");
        }
        if (outputDType == DataType.BOOL) {
            if (supportsBoolOutputOperation(opType)) {
                return supported(MetalDTypeRole.OPERATION, outputDType, true, true, MetalDTypeReasonCode.SUPPORTED,
                        "operation " + opType + " is dtype-legal for BOOL output in the scoped Metal bridge");
            }
            return unsupported(MetalDTypeRole.OPERATION, outputDType, MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE,
                    "operation " + opType + " cannot produce native BOOL output on the current Metal bridge");
        }
        if (outputDType == DataType.INT32) {
            if (opType == Operation.OpType.ARGMAX) {
                return supported(MetalDTypeRole.OPERATION, outputDType, true, true, MetalDTypeReasonCode.SUPPORTED,
                        "operation " + opType + " is dtype-legal for INT32 index output in the scoped Metal bridge");
            }
            return unsupported(MetalDTypeRole.OPERATION, outputDType, MetalDTypeReasonCode.UNSUPPORTED_OPERATION_DTYPE,
                    "operation " + opType + " cannot produce native INT32 output on the current Metal bridge");
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
                case 1, 2 -> (dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16)
                        && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                "WHERE branch input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                "WHERE branch input requires dtype-matched FLOAT32/BFLOAT16 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        "WHERE has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            if (inputIndex >= 0 && inputIndex <= 2
                    && (dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16)
                    && consumer.dataType() == dtype) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "SDPA query/key/value inputs accept dtype-matched FLOAT32/BFLOAT16 data");
            }
            if (inputIndex == 3 && dtype == DataType.BOOL) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "SDPA mask input accepts verified dense BOOL predicate data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "Metal direct SDPA accepts dtype-matched FLOAT32/BFLOAT16 query/key/value inputs and optional dense BOOL mask input");
        }
        if (opType == Operation.OpType.GATHER || opType == Operation.OpType.GATHER_AXIS || opType == Operation.OpType.TAKE_ALONG_AXIS) {
            return switch (inputIndex) {
                case 0 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " value input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " value input requires dtype-matched FLOAT32/BFLOAT16 data");
                case 1 -> dtype == DataType.INT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                                opType + " index input accepts INT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " index input requires INT32 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.SCATTER_ADD) {
            return switch (inputIndex) {
                case 0, 2 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " value input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " value inputs require dtype-matched FLOAT32/BFLOAT16 data");
                case 1 -> dtype == DataType.INT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                                opType + " index input accepts INT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " index input requires INT32 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.GATHER_GRAD || opType == Operation.OpType.GATHER_AXIS_GRAD || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD) {
            return switch (inputIndex) {
                case 0 -> dtype == DataType.INT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                                opType + " index input accepts INT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " index input requires INT32 data");
                case 1 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " gradient input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " gradient input requires dtype-matched FLOAT32/BFLOAT16 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.NLL_LOSS || opType == Operation.OpType.CROSS_ENTROPY_LOSS) {
            return switch (inputIndex) {
                case 0, 1 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " dense loss input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " dense loss inputs require dtype-matched FLOAT32/BFLOAT16 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES) {
            return switch (inputIndex) {
                case 0 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " logits input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " logits input requires dtype-matched FLOAT32/BFLOAT16 data");
                case 1 -> dtype == DataType.INT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                                opType + " target-index input accepts INT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " target-index input requires INT32 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD) {
            return switch (inputIndex) {
                case 0, 2 -> isMetalFloatingDType(dtype) && consumer.dataType() == dtype
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                                MetalDTypeReasonCode.SUPPORTED,
                                opType + " floating input accepts dtype-matched FLOAT32/BFLOAT16 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " floating inputs require dtype-matched FLOAT32/BFLOAT16 data");
                case 1 -> dtype == DataType.INT32
                        ? supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, false, false,
                                MetalDTypeReasonCode.SUPPORTED_STORAGE_ONLY,
                                opType + " target-index input accepts INT32 data")
                        : unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                                opType + " target-index input requires INT32 data");
                default -> unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        opType + " has no supported input role at index " + inputIndex);
            };
        }
        if (opType == Operation.OpType.ARGMAX) {
            if (inputIndex == 0 && isMetalFloatingDType(dtype)) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "ARGMAX value input accepts FLOAT32/BFLOAT16 data and produces INT32 indices");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "ARGMAX value input requires FLOAT32/BFLOAT16 data");
        }
        if (opType == Operation.OpType.CAST) {
            if (inputIndex != 0) {
                return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                        "CAST has no supported input role at index " + inputIndex);
            }
            MetalCastPolicy.Decision decision = MetalCastPolicy.decide(dtype, consumer.dataType());
            if (decision.supported()) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                        dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16,
                        false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "CAST input accepts " + dtype + " for " + dtype + " -> " + consumer.dataType());
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    decision.detail());
        }
        if (supportsBoolCompareOperation(opType)) {
            if ((inputIndex == 0 || inputIndex == 1) && (dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16)) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "BOOL compare input accepts FLOAT32/BFLOAT16 data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "BOOL compare inputs require FLOAT32/BFLOAT16 data");
        }
        if (supportsBoolLogicalOperation(opType)) {
            if ((inputIndex == 0 || inputIndex == 1) && dtype == DataType.BOOL) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "BOOL logical input accepts BOOL predicate data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "BOOL logical inputs require BOOL data");
        }
        if (supportsBoolReductionOperation(opType)) {
            if (inputIndex == 0 && dtype == DataType.BOOL) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                        MetalDTypeReasonCode.SUPPORTED,
                        "BOOL reduction input accepts BOOL predicate data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    "BOOL reduction input requires BOOL data");
        }
        if (supportsDTypePreservingLayoutOperation(opType)) {
            boolean legalInputPosition = opType == Operation.OpType.CONCAT ? inputIndex >= 0 : inputIndex == 0;
            if (legalInputPosition
                    && dtype == consumer.dataType()
                    && (dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16 || dtype == DataType.BOOL)) {
                return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, dtype != DataType.BOOL, false,
                        dtype == DataType.BOOL
                                ? MetalDTypeReasonCode.SUPPORTED_PREDICATE_INPUT_ONLY
                                : MetalDTypeReasonCode.SUPPORTED,
                        opType + " layout input accepts dtype-preserving " + dtype + " data");
            }
            return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                    MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                    opType + " layout input requires matching FLOAT32/BFLOAT16/BOOL dtype");
        }
        if (dtype == DataType.FLOAT32) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                    MetalDTypeReasonCode.SUPPORTED,
                    "default Metal external data input accepts FLOAT32");
        }
        if (dtype == DataType.BFLOAT16
                && consumer.dataType() == DataType.BFLOAT16
                && operationDecision(opType, consumer.dataType()).supported()) {
            return supported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype, true, false,
                    MetalDTypeReasonCode.SUPPORTED,
                    "default Metal external data input accepts BFLOAT16 for BFLOAT16 operation family");
        }
        return unsupported(MetalDTypeRole.EXTERNAL_INPUT_ROLE, dtype,
                MetalDTypeReasonCode.UNSUPPORTED_EXTERNAL_INPUT_ROLE,
                "default Metal external data input requires FLOAT32 or supported BFLOAT16");
    }

    /**
     * Returns whether the current Metal bridge can execute compute nodes with this dtype.
     *
     * @param dtype compiled node output/compute dtype
     * @return true for {@link DataType#FLOAT32}, supported {@link DataType#BFLOAT16}, and scoped BOOL compare output
     */
    public static boolean supportsComputeDType(DataType dtype) {
        return computeDecision(dtype).supported();
    }

    /**
     * Returns whether the current Metal bridge can publish output tensors with this dtype.
     *
     * @param dtype output tensor dtype
     * @return true for {@link DataType#FLOAT32}, supported {@link DataType#BFLOAT16}, and scoped BOOL compare output
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
     * @return true for float32 data tensors, bfloat16 data tensors, and bool predicate tensors
     */
    public static boolean supportsExternalInputDType(DataType dtype) {
        return externalInputDecision(dtype).supported();
    }

    /**
     * Returns whether a producer may feed the given consumer input from outside a Metal partition.
     *
     * <p>Bool tensors are deliberately role-limited: {@code WHERE} input 0 may be bool,
     * direct SDPA input 3 may be a verified dense public BOOL mask, and all other data
     * inputs must be role-legal FLOAT32/BFLOAT16/INT32 values.</p>
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
                + "; Metal MPS bridge currently supports FLOAT32/BFLOAT16 compute/output tensors for supported floating operation families, scoped BOOL outputs, BOOL predicate inputs, INT32 index inputs, and scoped INT32 index outputs; got "
                + dtype + ".";
    }

    private static boolean supportsBFloat16Operation(Operation.OpType opType) {
        return switch (opType) {
            case MATMUL,
                 LINEAR,
                 ADD,
                 SUB,
                 MUL,
                 DIV,
                 MIN,
                 MAX,
                 RELU,
                 TANH,
                 FAST_TANH,
                 SIGMOID,
                 ABS,
                 EXP,
                 FAST_EXP,
                 LOG,
                 NEG,
                 SQRT,
                 INV,
                 MUL_SCALAR,
                 POW,
                 CLAMP_MIN,
                 CLAMP_MAX,
                 WHERE,
                 SOFTMAX,
                 LOG_SOFTMAX,
                 SOFTMAX_GRAD,
                 LOG_SOFTMAX_GRAD,
                 SUM,
                 MEAN,
                 REDUCE_MIN,
                 REDUCE_MAX,
                 REDUCE_PROD,
                 CUMSUM,
                 REDUCE_MIN_GRAD,
                 REDUCE_MAX_GRAD,
                 MIN_GRAD,
                 MAX_GRAD,
                 LAYER_NORM,
                 RMS_NORM,
                 NLL_LOSS,
                 CROSS_ENTROPY_LOSS,
                 CROSS_ENTROPY_LOSS_INDICES,
                 CROSS_ENTROPY_LOSS_INDICES_GRAD,
                 GATHER,
                 GATHER_AXIS,
                 TAKE_ALONG_AXIS,
                 SCATTER_ADD,
                 GATHER_GRAD,
                 GATHER_AXIS_GRAD,
                 TAKE_ALONG_AXIS_GRAD,
                 CONV2D,
                 CONV2D_GEMM,
                 CONV2D_BACKWARD_INPUT,
                 CONV2D_BACKWARD_INPUT_GEMM,
                 CONV2D_BACKWARD_WEIGHT,
                 CONV2D_BACKWARD_WEIGHT_GEMM,
                 MAX_POOL2D,
                 AVG_POOL2D,
                 MAX_POOL2D_BACKWARD_INPUT,
                 AVG_POOL2D_BACKWARD_INPUT,
                 SCALED_DOT_PRODUCT_ATTENTION,
                 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                 SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
                 RESHAPE,
                 CONTIGUOUS,
                 PERMUTE,
                 EXPAND,
                 SLICE,
                 CONCAT,
                 PAD,
                 TILE,
                 CAST,
                 EXPAND_DIMS,
                 SQUEEZE,
                 SELECT,
                 NOOP -> true;
            default -> false;
        };
    }

    private static boolean isMetalFloatingDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16;
    }

    private static boolean supportsBoolCompareOperation(Operation.OpType opType) {
        return switch (opType) {
            case GT, GE, LT, LE, EQ, NE -> true;
            default -> false;
        };
    }

    private static boolean supportsBoolLogicalOperation(Operation.OpType opType) {
        return switch (opType) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> true;
            default -> false;
        };
    }

    private static boolean supportsBoolReductionOperation(Operation.OpType opType) {
        return switch (opType) {
            case REDUCE_ALL, REDUCE_ANY -> true;
            default -> false;
        };
    }

    private static boolean supportsBoolOutputOperation(Operation.OpType opType) {
        return supportsBoolCompareOperation(opType)
                || supportsBoolLogicalOperation(opType)
                || supportsBoolReductionOperation(opType)
                || supportsDTypePreservingLayoutOperation(opType);
    }

    private static boolean supportsDTypePreservingLayoutOperation(Operation.OpType opType) {
        return switch (opType) {
            case CONTIGUOUS,
                 RESHAPE,
                 EXPAND,
                 SELECT,
                 SLICE,
                 CONCAT,
                 PAD,
                 TILE,
                 PERMUTE,
                 EXPAND_DIMS,
                 SQUEEZE,
                 NOOP -> true;
            default -> false;
        };
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
