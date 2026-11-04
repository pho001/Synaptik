package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import java.util.Objects;
import java.util.List;
import java.util.Optional;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLowering;

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
 * while its generated entry remains a direct typed body.
 * Ordinary aggregate fusion instead retains its separate typed domain/binding composition in
 * specialization and exposes a selectively versioned physical-boundary envelope.
 * Fixed softmax derivatives likewise retain a typed two-pass recipe; both its canonical
 * envelope and baked bindings must match the exclusive specialization. Actual carriers,
 * invocation bases and mutable exact state are not route facts.
 * Softmax and trailing Layer/RMS
 * normalization similarly retain only code-shaping identity; concrete slice ranges, typed
 * carriers, and optional Layer scratch stay cold. Loss IR retains semantic role positions,
 * represented types, reduction, ignore form, and range ownership only; its normalized axes,
 * layouts, actual ignore bits, and bases stay cold. General-scan IR retains typed arithmetic,
 * positional bindings, demanded roles and temporal storage choices; its boundary envelope is
 * for compatibility, while concrete time/step extents and assigned storage remain cold.
 * An ordinary row region similarly exposes only its instruction-free envelope at this seam;
 * its exclusive projection-72 specialization must carry the identical proved recipe payload.
 * A proved constant count instead has an output-only envelope including exact typed literal
 * bits. Original source declarations remain with the partition, not this generated entry.
 *
 * @param portableKernelIr non-null route-independent pointwise, affine, movement, indexing,
 *     functional-scatter, overlap-fold, ordering, random, cumulative-scan, aggregate,
 *     output-only constant-write, softmax or its fixed two-pass derivative,
 *     trailing-normalization, direct-loss, bounded general-scan or ordinary row-region IR
 * @param specialization non-null selected exact/default scalar or preferred-species vector
 *     Class-File specialization; parallel orchestration, if selected, remains outside the artifact
 * @param batchBindings immutable ordered B-only access recipes; empty for static execution
 * @param batchMatmul optional actual-contraction recipe, only for ordinary batch MATMUL
 * @param batchAggregate optional exact-state recipe, only for ordinary batch SUM/MEAN
 * @param ordinaryBatchGeometry optional family-owned numeric binding recipe for ordinary
 *     low-precision symbolic execution; resolved once at the actual run batch
 */
public record CpuPortableRoutePlan(CpuPortableKernelIr portableKernelIr,
        CpuKernelSpecialization specialization, List<CpuAccessPlan.BatchBinding> batchBindings,
        Optional<CpuMatmulLowering.BatchGeometry> batchMatmul,
        Optional<CpuAggregateLowering.BatchGeometry> batchAggregate,
        Optional<io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering.BatchUnitGeometry> ordinaryBatchGeometry) {
    /**
     * Retains the static portable route constructor and its unchanged generated identities.
     * @param portableKernelIr non-null immutable typed IR
     * @param specialization non-null matching specialization
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if canonical identity or a family's baked recipe
     *     disagrees with the specialization
     */
    public CpuPortableRoutePlan(CpuPortableKernelIr portableKernelIr, CpuKernelSpecialization specialization) {
        this(portableKernelIr, specialization, List.of(), Optional.empty(), Optional.empty(), Optional.empty());
    }
    /**
     * Validates one portable realization plan.
     *
     * @param portableKernelIr non-null route-independent pointwise, affine, movement, indexing,
     *     scatter, fold, ordering, random, cumulative-scan, aggregate, softmax or its fixed derivative,
     *     trailing-normalization, direct-loss, bounded general-scan or ordinary row-region IR
     * @param specialization non-null matching scalar or vector generated specialization
     * @param batchBindings non-null immutable ordered B-only recipes, empty in static mode
     * @param batchMatmul non-null optional contraction recipe, exclusive with aggregate
     * @param batchAggregate non-null optional exact-state recipe, exclusive with MATMUL
     * @param ordinaryBatchGeometry non-null optional generic symbolic recipe, exclusive with
     *     legacy matrix/aggregate bindings and using the existing resolved family ABI
     * @throws NullPointerException if a required component or collection element is null
     * @throws IllegalArgumentException if the specialization does not match the canonical IR, or
     *     baked family facts, or a pointwise plan has an incompatible generated specialization
     */
    public CpuPortableRoutePlan {
        Objects.requireNonNull(portableKernelIr, "portableKernelIr");
        Objects.requireNonNull(specialization, "specialization");
        batchBindings = List.copyOf(batchBindings);
        batchMatmul = Objects.requireNonNull(batchMatmul, "batchMatmul");
        batchAggregate = Objects.requireNonNull(batchAggregate, "batchAggregate");
        ordinaryBatchGeometry = Objects.requireNonNull(ordinaryBatchGeometry, "ordinaryBatchGeometry");
        if (ordinaryBatchGeometry.isPresent()) {
            var geometry = ordinaryBatchGeometry.orElseThrow();
            if (specialization.ordinaryBatch() || !batchBindings.isEmpty()
                    || batchMatmul.isPresent() || batchAggregate.isPresent()
                    || !geometry.ir().structuralKey().equals(portableKernelIr.structuralKey())
                    || !geometry.boundaryTypes().equals(specialization.boundaryDataTypes())
                    || geometry.bindings().size() != specialization.carrierPattern().size()
                    || specialization.carrierPattern().stream().anyMatch(carrier ->
                            carrier != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT))
                throw new IllegalArgumentException("ordinary symbolic family recipe disagrees with specialization");
        }
        if ((portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr)
                != specialization.softmaxBackwardIr().isPresent())
            throw new IllegalArgumentException("softmax backward route and specialization disagree");
        if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr backward
                && !backward.bakedSpecializationIdentity().equals(specialization.softmaxBackwardIr().orElseThrow().bakedSpecializationIdentity()))
            throw new IllegalArgumentException("softmax backward baked geometry differs from specialization");
        if ((portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr)
                != specialization.aggregateFusionIr().isPresent())
            throw new IllegalArgumentException("ordinary aggregate fusion route and specialization disagree");
        if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr fusion
                && !fusion.bakedSpecializationIdentity().equals(specialization.aggregateFusionIr().orElseThrow().bakedSpecializationIdentity()))
            throw new IllegalArgumentException("ordinary aggregate fusion baked geometry differs from specialization");
        boolean regional = portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr;
        if (regional != specialization.rowMapReductionIr().isPresent())
            throw new IllegalArgumentException("row-region route and specialization disagree");
        if (regional) {
            var region = (io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr)portableKernelIr;
            if (!region.bakedSpecializationIdentity().equals(specialization.rowMapReductionIr().orElseThrow().bakedSpecializationIdentity())
                    || batchMatmul.isPresent() || batchAggregate.isPresent())
                throw new IllegalArgumentException("row-region route requires identical proved facts and no whole-operation recipe");
            if (specialization.ordinaryBatch()) {
                var boundary = java.util.stream.Stream.concat(region.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.INPUT),
                        region.arithmetic().values().stream().filter(v -> v.kind() == CpuKernelIr.Value.Kind.OUTPUT)).map(v -> region.accesses().get(v.ordinal())).toList();
                if (!boundary.equals(batchBindings)) throw new IllegalArgumentException("row-region boundary recipes disagree");
            }
        }
        if (specialization.ordinaryBatch() != !batchBindings.isEmpty()
                || !specialization.ordinaryBatch() && (batchMatmul.isPresent() || batchAggregate.isPresent())
                || batchMatmul.isPresent() && batchAggregate.isPresent()
                || !batchBindings.isEmpty() && batchBindings.size() != specialization.carrierPattern().size()) {
            throw new IllegalArgumentException("ordinary batch route geometry disagrees");
        }
        if (!portableKernelIr.structuralKey().equals(specialization.loweringFingerprint().hex())) {
            throw new IllegalArgumentException("specialization must match canonical IR");
        }
        CpuKernelIr generated = encoded(portableKernelIr);
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
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr backward)
            return backward.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr fusion)
            return fusion.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr constant)
            return constant.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region)
            return region.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr scan)
            return scan.encodedKernelIr();
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
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr conv3d)
            return conv3d.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr pool2d)
            return pool2d.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr pool3d)
            return pool3d.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr attention)
            return attention.encodedKernelIr();
        if (source instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr loss)
            return loss.encodedKernelIr();
        return ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr) source)
                .encodedKernelIr();
    }
}
