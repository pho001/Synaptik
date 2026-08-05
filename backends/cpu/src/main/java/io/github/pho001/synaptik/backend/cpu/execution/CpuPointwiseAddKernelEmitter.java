package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/** Emits the exact scalar flat-loop body for one dense pointwise {@code ADD} lowering. */
final class CpuPointwiseAddKernelEmitter implements CpuFamilyKernelEmitter {
    private final CpuPointwiseAddLowering lowering;

    /**
     * Creates an emitter for one exact lowering.
     *
     * @param lowering non-null immutable ADD lowering retained exactly
     * @throws NullPointerException if {@code lowering} is {@code null}
     */
    CpuPointwiseAddKernelEmitter(CpuPointwiseAddLowering lowering) {
        this.lowering = Objects.requireNonNull(lowering, "lowering");
    }

    /**
     * Returns the identity-free fingerprint that must match the generated specialization.
     *
     * @return exact non-null lowering fingerprint
     */
    @Override public CpuLoweringFingerprint loweringFingerprint() { return lowering.fingerprint(); }

    /**
     * Emits two loads, one Java Virtual Machine primitive addition, and one store for every flat
     * element.
     *
     * @param scalar non-null scalar instruction helper for the selected specialization
     * @param carriers non-null direct carrier load/store helper
     * @param loops non-null flat-range loop helper
     * @param reductions non-null structural reduction helper; unused by pointwise ADD
     */
    @Override
    public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
            CpuLoopEmitter loops, CpuReductionEmitter reductions) {
        loops.emitRange(index -> {
            carriers.emitScalarLoad(0, index);
            carriers.emitScalarLoad(1, index);
            switch (lowering.dataType()) {
                case FLOAT64 -> scalar.code().dadd();
                case FLOAT32 -> scalar.code().fadd();
                case INT32 -> scalar.code().iadd();
                case INT64 -> scalar.code().ladd();
                default -> throw new IllegalStateException("unsupported pointwise ADD data type");
            }
            carriers.emitScalarStore(2, index);
        });
        scalar.code().return_();
    }

    /**
     * Rejects Vector emission because CPU 0005 implements no Vector ADD route.
     *
     * @param vector non-null Vector instruction helper
     * @param carriers non-null direct carrier helper
     * @param loops non-null loop helper
     * @param reductions non-null structural reduction helper
     * @throws IllegalStateException always, because Vector execution is unsupported
     */
    @Override
    public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
            CpuLoopEmitter loops, CpuReductionEmitter reductions) {
        throw new IllegalStateException("pointwise ADD Vector route is not implemented");
    }
}
