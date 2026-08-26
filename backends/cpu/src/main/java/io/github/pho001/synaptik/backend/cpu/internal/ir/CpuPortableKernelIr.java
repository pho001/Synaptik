package io.github.pho001.synaptik.backend.cpu.internal.ir;

/**
 * CPU-private structural identity shared by portable generated kernel families.
 *
 * <p>The permitted implementations distinguish the pointwise instruction form from the static
 * affine copy, movement, indexing, functional-scatter, overlap-fold, ordering, explicit-state
 * random, cumulative-scan, aggregate, softmax, trailing-normalization, and batch-normalization
 * inference forms. Every form exposes only
 * deterministic code-shaping identity; graph
 * identifiers, concrete extents, slots, carriers, addresses, and run state remain cold facts
 * outside this role.</p>
 */
public sealed interface CpuPortableKernelIr permits CpuKernelIr, CpuAffineCopyIr,
        CpuDataMovementIr, CpuIndexingIr, CpuScatterIr, CpuFoldIr, CpuOrderingIr, CpuRandomIr,
        CpuScanIr, CpuAggregateIr, CpuArgExtremaIr, CpuMaskedReductionIr,
        CpuAdvancedReductionIr, CpuSoftmaxIr, CpuTrailingNormalizationIr,
        CpuBatchNormInferenceIr, CpuBatchNormTrainingIr, CpuConv2dIr {
    /**
     * Returns the deterministic structural key used for generated-code compatibility.
     *
     * @return a non-null lowercase hexadecimal structural key
     */
    String structuralKey();
}
