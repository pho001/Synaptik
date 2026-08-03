package io.github.pho001.synaptik.backend.cpu.execution;

/**
 * Typed family-owned collaboration that emits one complete already-selected lowering. The
 * generator supplies only low-level scalar or Vector, carrier, loop, and reduction structure;
 * this callback remains responsible for family meaning. No registry or production family
 * implementation is provided by the generator foundation.
 */
interface CpuFamilyKernelEmitter {
    /** Returns the identity that generation matches against its specialization.
     * @return immutable non-null fingerprint of the exact lowering implemented by this emitter */
    CpuLoweringFingerprint loweringFingerprint();

    /**
     * Emits one complete scalar {@code void} entry body.
     * @param scalar non-null primitive-local helper owned by the current generation call
     * @param carriers non-null direct-carrier helper owned by the current generation call
     * @param loops non-null range, tile, and tail helper owned by the current generation call
     * @param reductions non-null structural reduction helper owned by the current generation call
     */
    void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
            CpuLoopEmitter loops, CpuReductionEmitter reductions);

    /**
     * Emits one complete Vector API {@code void} entry body.
     * @param vector non-null exact-species helper owned by the current generation call
     * @param carriers non-null direct-carrier helper owned by the current generation call
     * @param loops non-null range, tile, and tail helper owned by the current generation call
     * @param reductions non-null structural reduction helper owned by the current generation call
     */
    void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
            CpuLoopEmitter loops, CpuReductionEmitter reductions);
}
