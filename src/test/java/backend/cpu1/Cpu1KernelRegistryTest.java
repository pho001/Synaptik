package backend.cpu1;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1KernelRegistry;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.storage.Cpu1StorageKind;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1KernelRegistryTest {
    private static final Operation.OpType[] SUPPORTED_OPS = {
            Operation.OpType.ADD,
            Operation.OpType.SUB,
            Operation.OpType.MUL,
            Operation.OpType.DIV,
            Operation.OpType.MIN,
            Operation.OpType.MAX,
            Operation.OpType.POW_TENSOR,
            Operation.OpType.WHERE,
            Operation.OpType.GT,
            Operation.OpType.GE,
            Operation.OpType.LT,
            Operation.OpType.LE,
            Operation.OpType.EQ,
            Operation.OpType.NE,
            Operation.OpType.LOGICAL_AND,
            Operation.OpType.LOGICAL_OR,
            Operation.OpType.LOGICAL_NOT,
            Operation.OpType.MUL_SCALAR,
            Operation.OpType.RELU,
            Operation.OpType.NEG,
            Operation.OpType.ABS,
            Operation.OpType.INV,
            Operation.OpType.EXP,
            Operation.OpType.FAST_EXP,
            Operation.OpType.ERF,
            Operation.OpType.LOG,
            Operation.OpType.TANH,
            Operation.OpType.FAST_TANH,
            Operation.OpType.SIGMOID,
            Operation.OpType.SQRT,
            Operation.OpType.POW,
            Operation.OpType.CLAMP_MIN,
            Operation.OpType.CLAMP_MAX,
            Operation.OpType.FLOOR,
            Operation.OpType.CEIL,
            Operation.OpType.SIGN
    };

    private static final Operation.OpType[] VECTORIZED_OPS = {
            Operation.OpType.ADD,
            Operation.OpType.SUB,
            Operation.OpType.MUL,
            Operation.OpType.DIV,
            Operation.OpType.MIN,
            Operation.OpType.MAX,
            Operation.OpType.MUL_SCALAR,
            Operation.OpType.RELU,
            Operation.OpType.NEG,
            Operation.OpType.ABS,
            Operation.OpType.INV,
            Operation.OpType.EXP,
            Operation.OpType.LOG,
            Operation.OpType.TANH,
            Operation.OpType.SIGMOID,
            Operation.OpType.SQRT,
            Operation.OpType.CLAMP_MIN,
            Operation.OpType.CLAMP_MAX
    };

    @Test
    void resolvesInitialContiguousScalarKernels() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64}) {
            assertFloatingScalarOpsResolve(registry, dataType);
            for (Cpu1LayoutKind layoutKind : new Cpu1LayoutKind[]{
                    Cpu1LayoutKind.STRIDED_RANK2,
                    Cpu1LayoutKind.STRIDED_RANK3,
                    Cpu1LayoutKind.STRIDED_RANK4,
                    Cpu1LayoutKind.STRIDED_GENERIC
            }) {
                assertFloatingScalarOpsResolve(registry, dataType, layoutKind);
            }
        }
        assertBoolScalarOpsResolve(registry);
    }

    @Test
    void resolvesDistinctContiguousAndStridedImplementationVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64}) {
            assertNotNull(registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.CONTIGUOUS));
            assertNotNull(registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK2));
            assertNotEquals(
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.CONTIGUOUS),
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK2)
            );
            assertNotEquals(
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK2),
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK3)
            );
            assertNotEquals(
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK3),
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK4)
            );
            assertNotEquals(
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_RANK4),
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.STRIDED_GENERIC)
            );
        }
    }

    @Test
    void rejectsUnsupportedOperationOrDtype() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(Operation.OpType.MATMUL, DataType.FLOAT32));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(Operation.OpType.ADD, DataType.INT32));
    }

    @Test
    void resolvesInitialBfloat16ContiguousScalarArrayKernels() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (Operation.OpType opType : SUPPORTED_OPS) {
            if (requiresBoolInput(opType)) {
                continue;
            }
            assertNotNull(registry.resolve(opType, DataType.BFLOAT16));
        }
        assertBoolScalarOpsResolve(registry);
    }

    @Test
    void resolvesBfloat16StridedAndVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        assertNotNull(registry.resolve(
                Operation.OpType.ADD,
                DataType.BFLOAT16,
                Cpu1LayoutKind.STRIDED_RANK2
        ));
        assertNotNull(registry.resolve(
                Operation.OpType.ADD,
                DataType.BFLOAT16,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
    }

    @Test
    void resolvesMemorySegmentScalarAndContiguousVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16}) {
            for (Operation.OpType opType : SUPPORTED_OPS) {
                if (!supportsMemorySegment(opType) || requiresBoolInput(opType)) {
                    continue;
                }
                assertNotNull(registry.resolve(
                        opType,
                        dataType,
                        Cpu1LayoutKind.STRIDED_RANK3,
                        Cpu1StorageKind.MEMORY_SEGMENT,
                        Cpu1VectorizationKind.SCALAR
                ));
                if (supportsVector(opType)) {
                    assertNotNull(registry.resolve(
                            opType,
                            dataType,
                            Cpu1LayoutKind.CONTIGUOUS,
                            Cpu1StorageKind.MEMORY_SEGMENT,
                            Cpu1VectorizationKind.VECTOR
                    ));
                }
                assertNotEquals(
                        registry.resolve(
                                opType,
                                dataType,
                                Cpu1LayoutKind.CONTIGUOUS,
                                Cpu1StorageKind.MEMORY_SEGMENT,
                                Cpu1VectorizationKind.SCALAR
                        ),
                        registry.resolve(
                                opType,
                                dataType,
                                Cpu1LayoutKind.STRIDED_RANK2,
                                Cpu1StorageKind.MEMORY_SEGMENT,
                                Cpu1VectorizationKind.SCALAR
                        )
                );
                assertNotEquals(
                        registry.resolve(
                                opType,
                                dataType,
                                Cpu1LayoutKind.STRIDED_RANK2,
                                Cpu1StorageKind.MEMORY_SEGMENT,
                                Cpu1VectorizationKind.SCALAR
                        ),
                        registry.resolve(
                                opType,
                                dataType,
                                Cpu1LayoutKind.STRIDED_RANK3,
                                Cpu1StorageKind.MEMORY_SEGMENT,
                                Cpu1VectorizationKind.SCALAR
                        )
                );
            }
        }
    }

    @Test
    void resolvesContiguousVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16}) {
            for (Operation.OpType opType : VECTORIZED_OPS) {
                assertNotNull(registry.resolve(
                        opType,
                        dataType,
                        Cpu1LayoutKind.CONTIGUOUS,
                        Cpu1VectorizationKind.VECTOR
                ));
            }
            assertNotEquals(
                    registry.resolve(Operation.OpType.ADD, dataType, Cpu1LayoutKind.CONTIGUOUS),
                    registry.resolve(
                            Operation.OpType.ADD,
                            dataType,
                            Cpu1LayoutKind.CONTIGUOUS,
                            Cpu1VectorizationKind.VECTOR
                    )
            );
        }
    }

    @Test
    void resolvesBroadcastInnerVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16}) {
            for (Operation.OpType opType : VECTORIZED_OPS) {
                assertNotNull(registry.resolve(
                        opType,
                        dataType,
                        Cpu1LayoutKind.BROADCAST_INNER,
                        Cpu1VectorizationKind.VECTOR
                ));
                assertNotNull(registry.resolve(
                        opType,
                        dataType,
                        Cpu1LayoutKind.BROADCAST_INNER,
                        Cpu1StorageKind.MEMORY_SEGMENT,
                        Cpu1VectorizationKind.VECTOR
                ));
            }
        }
    }

    @Test
    void rejectsFastApproxVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.FAST_EXP,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.FAST_TANH,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.POW_TENSOR,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.WHERE,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.ERF,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.SIGN,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.POW,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.GT,
                DataType.FLOAT32,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.LOGICAL_AND,
                DataType.BOOL,
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1VectorizationKind.VECTOR
        ));
    }

    @Test
    void rejectsUnregisteredStridedVectorVariants() {
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();

        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                Cpu1LayoutKind.STRIDED_RANK2,
                Cpu1VectorizationKind.VECTOR
        ));
        assertThrows(UnsupportedOperationException.class, () -> registry.resolve(
                Operation.OpType.ADD,
                DataType.FLOAT32,
                Cpu1LayoutKind.STRIDED_RANK3,
                Cpu1VectorizationKind.VECTOR
        ));
    }

    private static boolean supportsVector(Operation.OpType opType) {
        return opType != Operation.OpType.FAST_EXP
                && opType != Operation.OpType.FAST_TANH
                && opType != Operation.OpType.POW_TENSOR
                && opType != Operation.OpType.WHERE
                && opType != Operation.OpType.POW
                && opType != Operation.OpType.ERF
                && opType != Operation.OpType.GT
                && opType != Operation.OpType.GE
                && opType != Operation.OpType.LT
                && opType != Operation.OpType.LE
                && opType != Operation.OpType.EQ
                && opType != Operation.OpType.NE
                && opType != Operation.OpType.LOGICAL_AND
                && opType != Operation.OpType.LOGICAL_OR
                && opType != Operation.OpType.LOGICAL_NOT
                && opType != Operation.OpType.FLOOR
                && opType != Operation.OpType.CEIL
                && opType != Operation.OpType.SIGN;
    }

    private static boolean supportsMemorySegment(Operation.OpType opType) {
        return opType != Operation.OpType.GT
                && opType != Operation.OpType.GE
                && opType != Operation.OpType.LT
                && opType != Operation.OpType.LE
                && opType != Operation.OpType.EQ
                && opType != Operation.OpType.NE
                && opType != Operation.OpType.LOGICAL_AND
                && opType != Operation.OpType.LOGICAL_OR
                && opType != Operation.OpType.LOGICAL_NOT;
    }

    private static boolean requiresBoolInput(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND
                || opType == Operation.OpType.LOGICAL_OR
                || opType == Operation.OpType.LOGICAL_NOT;
    }

    private static void assertFloatingScalarOpsResolve(Cpu1KernelRegistry registry, DataType dataType) {
        assertFloatingScalarOpsResolve(registry, dataType, Cpu1LayoutKind.CONTIGUOUS);
    }

    private static void assertFloatingScalarOpsResolve(
            Cpu1KernelRegistry registry,
            DataType dataType,
            Cpu1LayoutKind layoutKind
    ) {
        for (Operation.OpType opType : SUPPORTED_OPS) {
            if (requiresBoolInput(opType)) {
                continue;
            }
            assertNotNull(registry.resolve(opType, dataType, layoutKind));
        }
    }

    private static void assertBoolScalarOpsResolve(Cpu1KernelRegistry registry) {
        for (Operation.OpType opType : new Operation.OpType[]{
                Operation.OpType.LOGICAL_AND,
                Operation.OpType.LOGICAL_OR,
                Operation.OpType.LOGICAL_NOT
        }) {
            for (Cpu1LayoutKind layoutKind : new Cpu1LayoutKind[]{
                    Cpu1LayoutKind.CONTIGUOUS,
                    Cpu1LayoutKind.STRIDED_RANK2,
                    Cpu1LayoutKind.STRIDED_RANK3,
                    Cpu1LayoutKind.STRIDED_RANK4,
                    Cpu1LayoutKind.STRIDED_GENERIC
            }) {
                assertNotNull(registry.resolve(opType, DataType.BOOL, layoutKind));
            }
        }
    }
}
