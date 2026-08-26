package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import java.util.Objects;

/**
 * Immutable portable Class-File realization facts for one already-lowered CPU execution unit.
 * This unsupported internal plan performs no graph interpretation or native-route selection.
 * A selected contiguous pointwise input copy is already reflected in the supplied canonical
 * consumer IR and specialization. An affine plan instead retains its structural copy form while
 * exposing an instruction-free encoded form to the existing generator/cache boundary. The plan
 * does not retain source/workspace objects or concrete affine addresses. Functional-scatter IR
 * retains the structural family, reduction, boundary access, and optional scratch-entry facts
 * required to embed its typed output/contribution body; concrete indices, geometry, and scratch
 * slices remain outside the plan. Overlap-fold IR similarly retains the structural facts required
 * for a typed embedded output/contribution body while concrete layouts, mappings, and ranges stay
 * cold. Ordering IR retains the family, type, direction, output-order, boundary-access, and
 * scratch-entry facts needed for a typed embedded stable-merge body while concrete axis/layout
 * geometry and scratch assignment stay cold. Ordinary aggregate IR is encoded for compatibility
 * while its generated entry remains a direct typed body. Softmax and trailing Layer/RMS
 * normalization similarly retain only code-shaping identity; concrete slice ranges, typed
 * carriers, and optional Layer scratch stay cold.
 *
 * @param portableKernelIr non-null route-independent pointwise, affine, movement, indexing,
 *     functional-scatter, overlap-fold, ordering, random, cumulative-scan, aggregate, softmax, or
 *     trailing-normalization IR
 * @param specialization non-null selected exact/default scalar or preferred-species vector
 *     Class-File specialization; parallel orchestration, if selected, remains outside the artifact
 */
public record CpuPortableRoutePlan(CpuPortableKernelIr portableKernelIr,
        CpuKernelSpecialization specialization) {
    /**
     * Validates one portable realization plan.
     *
     * @param portableKernelIr non-null route-independent pointwise, affine, movement, indexing,
     *     scatter, fold, ordering, random, cumulative-scan, aggregate, softmax, or
     *     trailing-normalization IR
     * @param specialization non-null matching scalar or vector generated specialization
     * @throws NullPointerException if either component is {@code null}
     * @throws IllegalArgumentException if the specialization does not match the canonical IR, or
     *     a pointwise plan attempts to use BFLOAT16 or its affine-only {@code SHORT_ARRAY} carrier
     */
    public CpuPortableRoutePlan {
        Objects.requireNonNull(portableKernelIr, "portableKernelIr");
        Objects.requireNonNull(specialization, "specialization");
        if (!portableKernelIr.structuralKey().equals(specialization.loweringFingerprint().hex())) {
            throw new IllegalArgumentException("specialization must match canonical IR");
        }
        CpuKernelIr generated = encoded(portableKernelIr);
        if (portableKernelIr instanceof CpuKernelIr
                && (generated.values().stream().anyMatch(value -> value.dataType()
                        == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16)
                || specialization.carrierPattern().contains(
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY))) {
            throw new IllegalArgumentException("BFLOAT16 carriers are affine-copy-only");
        }
        var realizations = generated.instructions().stream()
                .filter(instruction -> instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
        if (!realizations.equals(specialization.scalarPowerRealizations())) {
            throw new IllegalArgumentException("specialization power realizations must match canonical IR");
        }
    }

    /**
     * Returns the cache-compatible generated IR form.
     *
     * @return the retained pointwise IR or a newly encoded instruction-free family IR; never
     *     {@code null}
     */
    public CpuKernelIr kernelIr() {
        return encoded(portableKernelIr);
    }

    private static CpuKernelIr encoded(CpuPortableKernelIr source) {
        if (source instanceof CpuKernelIr pointwise) return pointwise;
        if (source instanceof CpuAffineCopyIr affine) return affine.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr movement)
            return movement.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr indexing)
            return indexing.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr scatter)
            return scatter.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr fold)
            return fold.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr ordering)
            return ordering.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr random)
            return random.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr scan)
            return scan.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr argExtrema)
            return argExtrema.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr masked)
            return masked.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr advanced)
            return advanced.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr softmax)
            return softmax.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr normalization)
            return normalization.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr batch)
            return batch.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr training)
            return training.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr conv2d)
            return conv2d.encodedKernelIr();
        return ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr) source)
                .encodedKernelIr();
    }
}
