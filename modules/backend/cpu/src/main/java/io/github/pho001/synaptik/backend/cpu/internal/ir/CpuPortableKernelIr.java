package io.github.pho001.synaptik.backend.cpu.internal.ir;

/**
 * CPU-private structural identity shared by portable generated kernel families.
 *
 * <p>The permitted implementations distinguish the pointwise instruction form from the static
 * affine copy, movement, indexing, functional-scatter, overlap-fold, ordering, explicit-state
 * random, cumulative-scan, aggregate, rounded-site ordinary aggregate fusion,
 * proved output-only constant-write, softmax, fixed two-pass softmax derivative,
 * trailing-normalization, and batch-normalization
 * inference forms, bounded typed general-scan and shared row-map/multiple-terminal forms. Every form exposes only
 * deterministic code-shaping identity; graph
 * identifiers, assigned slots, carrier instances, addresses, and run state remain cold facts
 * outside this role. Concrete geometry stays cold unless a selected family explicitly bakes it
 * into emitted loop/access shape; new ordinary fusion magnitudes stay in a separate baked
 * specialization payload rather than this canonical envelope.</p>
 * <p>Constant-write identity includes the exact typed literal because generation bakes those
 * represented bits into the entry. This form does not represent a new graph constant source.</p>
 */
public sealed interface CpuPortableKernelIr permits CpuKernelIr, CpuAffineCopyIr, CpuConstantWriteIr,
        CpuDataMovementIr, CpuIndexingIr, CpuScatterIr, CpuFoldIr, CpuOrderingIr, CpuRandomIr,
        CpuScanIr, CpuAggregateIr, CpuAggregateFusionIr, CpuSoftmaxBackwardIr, CpuArgExtremaIr, CpuMaskedReductionIr,
        CpuAdvancedReductionIr, CpuSoftmaxIr, CpuTrailingNormalizationIr,
        CpuBatchNormInferenceIr, CpuBatchNormTrainingIr, CpuConv2dIr, CpuConv3dIr, CpuPool2dIr,
        CpuPool3dIr, CpuAttentionIr, CpuLossIr, CpuGeneralScanIr, CpuRowMapReductionIr {
    /**
     * Returns the deterministic structural key used for generated-code compatibility.
     *
     * @return a non-null lowercase hexadecimal structural key
     */
    String structuralKey();
}
