package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ByteVector;

/**
 * Whole-partition CPU analysis entry for the current bounded static portable families.
 * Analysis deterministically compares direct access with at most three one-input contiguous-copy
 * candidates, then selects scalar or preferred-species vector compute and single-thread or
 * bounded parallel orchestration before shared resource assignment. Exact vector eligibility is
 * typed across floating, signed-integral, canonical-BOOL, and narrowly virtual floating-mask
 * topologies; direct power and unsafe mask storage remain scalar. Analysis measures nothing and
 * performs no artifact or persistence access. Static affine chains instead retain scalar compute,
 * compose one exact distinct-write address domain, declare only source and final result buffers,
 * and use deterministic scalar fallback when vector compute was preferred.
 * One-node scatter, fold, and ordering plans remain scalar compute and use disjoint output or
 * logical-slice ranges. Fold selects neither materialization nor workspace; scatter declares
 * per-range scratch only for nonempty floating multiplication. Ordering always declares the
 * checked run-owned workspace required by its two primitive INT64 merge-index regions per
 * selected range and may expose one or two output stores.
 */
public final class CpuPartitionPreparer implements BackendPartitionPreparer<
        CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> {
    private final CpuPartitionLowering lowering;

    /** Creates a preparer with the permanent common lowering owner. */
    public CpuPartitionPreparer() { this(new CpuPartitionLowering()); }

    /**
     * Creates a preparer with an explicit lowering collaborator.
     * @param lowering non-null whole-partition lowering retained by the preparer
     * @throws NullPointerException if {@code lowering} is {@code null}
     */
    public CpuPartitionPreparer(CpuPartitionLowering lowering) {
        this.lowering = Objects.requireNonNull(lowering, "lowering");
    }

    /**
     * Lowers, fuses, selects one bounded complete plan, and declares exact post-fusion resources.
     * @param context non-null complete CPU analysis context
     * @return one immutable analysis with one unit, a cold-selected portable strategy, one exact
     *     declaration per derived boundary, and at most one workspace declaration; affine and
     *     fold plans have no workspace, while ordering has exact per-range scratch and TOP_K has
     *     ordered values then INT64-index outputs; never {@code null}
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        var lowered = lowering.lower(context);
        var requestedCarriers = context.backendInputs().carrierPattern().isEmpty()
                ? java.util.Collections.nCopies(lowered.boundaryValues().size(),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : context.backendInputs().carrierPattern();
        if (requestedCarriers.size() != lowered.boundaryValues().size()) {
            throw new IllegalArgumentException("carrier pattern count must match boundary count");
        }
        var budget = new CpuSpecializationBudget(4, 1, 0, 0);
        boolean movement = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
        boolean indexing = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
        boolean scatter = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
        boolean fold = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
        boolean ordering = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
        boolean random = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
        Optional<CpuMaterializationPlan> materialization = movement || indexing || scatter || fold || ordering || random ? Optional.empty()
                : selectMaterialization(lowered, context.backendInputs().materializationPolicy());
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(lowered.boundaryValues().size());
        for (int i = 0; i < lowered.boundaryValues().size(); i++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(i),
                        Math.multiplyExact(lowered.referencedElementSpans().get(i),
                                lowered.boundaryDataTypes().get(i).byteWidth()),
                        lowered.boundaryDataTypes().get(i).byteWidth()));
        var bindings = new ArrayList<>(lowered.accessBindings());
        var carriers = new ArrayList<>(requestedCarriers);
        CpuKernelIr kernelIr = lowered.kernelIr();
        boolean affineCopy = kernelIr.instructions().isEmpty();
        if (materialization.isPresent()) {
            var selected = materialization.orElseThrow();
            bindings.set(selected.sourceBoundaryIndex(), selected.consumerBinding());
            carriers.set(selected.sourceBoundaryIndex(), CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
            kernelIr = adjustedIr(kernelIr, selected.sourceBoundaryIndex(),
                    selected.consumerBinding().plan());
        }
        var config = context.backendInputs().portableExecution();
        DataType vectorType = vectorLaneType(kernelIr);
        int lanes = speciesLanes(vectorType);
        int speciesBits = speciesBits(vectorType);
        boolean vectorEligible = !affineCopy && !indexing && !scatter && !fold && !ordering && !random && config.computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorType != null && lanes > 1 && lowered.elementCount() >= lanes
                && vectorTopologyEligible(kernelIr, vectorType)
                && bindings.stream().allMatch(binding -> vectorEligible(binding, lanes));
        int usableParallelism = Math.min(config.configuredMaximumParallelism(),
                config.availableParallelism());
        int selectedRangeCount = lowered.elementCount() == 0 ? 1 : Math.min(usableParallelism,
                Math.toIntExact(Math.min(Integer.MAX_VALUE,
                        ceilDiv(lowered.elementCount(), config.minimumElementsPerWorker()))));
        boolean parallel = selectedRangeCount >= 2;
        var strategy = vectorEligible
                ? (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)
                : (parallel ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR);
        var artifactStrategy = vectorEligible
                ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR;
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(kernelIr.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                artifactStrategy, lowered.boundaryDataTypes(), carriers,
                vectorEligible ? speciesBits : 0,
                materialization.map(CpuMaterializationPlan::sourceBoundaryIndex).orElse(-1),
                powerRealizations(kernelIr), lowered.scatterGeometry()
                        .filter(g -> g.scratchSliceBytes() > 0).isPresent()
                        || lowered.orderingGeometry().isPresent());
        var selectedPortableIr = materialization.isPresent()
                ? kernelIr : lowered.portableKernelIr();
        var routePlan = new CpuPortableRoutePlan(selectedPortableIr, specialization);
        String manifest = "unit=0;fusion=" + lowered.fusionReason()
                + ";access=" + bindings.stream()
                        .map(binding -> binding.plan().regime().name()).toList()
                + ";carriers=" + carriers
                + ";route=PORTABLE;strategy=" + strategy + ";speciesBits="
                + (vectorEligible ? speciesBits : 0) + ";power="
                + powerRealizations(kernelIr) + ";key="
                + specialization.structuralKey() + ";buffers=" + lowered.boundaryValues();
        Optional<PreparationResourceRequirement.Workspace> workspace = materialization.map(copy ->
                    new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                            copy.byteCount(), copy.byteAlignment()));
        if (lowered.scatterGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent()) {
            var scatterGeometry = lowered.scatterGeometry().orElseThrow();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(0,
                    scatterGeometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        if (lowered.orderingGeometry().isPresent()) {
            var geometry = lowered.orderingGeometry().orElseThrow();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(0,
                    geometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        var workspaceUse = materialization.isPresent()
                ? CpuPartitionPreparationPlan.WorkspaceUse.MATERIALIZATION
                : workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.SCATTER_PRODUCT
                : CpuPartitionPreparationPlan.WorkspaceUse.NONE;
        if (lowered.orderingGeometry().isPresent() && workspace.isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.ORDERING_INDICES;
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.fusionReason())),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy,
                declarations, lowered.boundaryValues(), bindings,
                requestedCarriers, carriers,
                lowered.extents(), lowered.elementCount(), lowered.affineAddressPairs(),
                selectedRangeCount, config.minimumElementsPerWorker(),
                vectorEligible ? speciesBits : 0,
                context.backendInputs().loweringManifestEnabled() ? manifest : "",
                materialization, workspace, workspaceUse, budget,
                lowered.movementGeometry(), lowered.indexingGeometry(), lowered.scatterGeometry(),
                lowered.foldGeometry(), lowered.orderingGeometry(), lowered.randomGeometry());
        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.workspaceDeclaration().ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    private static Optional<CpuMaterializationPlan> selectMaterialization(
            CpuPartitionLowering.LoweredPartition lowered,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        if (!policy.enabled()) return Optional.empty();
        long elements = lowered.elementCount();
        long bytes = Math.multiplyExact(elements, Double.BYTES);
        if (bytes > policy.maximumAdditionalBytes()) return Optional.empty();
        CpuMaterializationPlan best = null;
        int considered = 0;
        for (int index = 0; index < lowered.boundaryValues().size() - 1 && considered < 3; index++) {
            int sourceOrdinal = index;
            CpuAccessPlan.Binding source = lowered.accessBindings().get(index);
            if (lowered.boundaryDataTypes().get(index) != DataType.FLOAT64) continue;
            if (source.plan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                    || source.plan().regime() == CpuAccessPlan.Regime.DENSE_LINEAR) continue;
            considered++;
            long useCount = lowered.kernelIr().instructions().stream()
                    .flatMap(instruction -> instruction.inputs().stream())
                    .filter(ordinal -> ordinal == sourceOrdinal)
                    .count();
            if (useCount == 0) continue;
            long directKernel = Math.multiplyExact(elements,
                    policy.directKernelCostUnitsPerElement());
            long contiguousKernel = Math.multiplyExact(elements,
                    policy.contiguousKernelCostUnitsPerElement());
            long copy = Math.addExact(policy.copyFixedCostUnits(), Math.multiplyExact(elements,
                    policy.copyCostUnitsPerElement()));
            long direct = Math.multiplyExact(policy.expectedRunCount(),
                    Math.multiplyExact(useCount, directKernel));
            long copied = Math.multiplyExact(policy.expectedRunCount(),
                    Math.addExact(copy, Math.multiplyExact(useCount, contiguousKernel)));
            if (direct == 0 || copied >= direct) continue;
            long benefit = Math.subtractExact(direct, copied);
            int basisPoints = Math.toIntExact(Math.floorDiv(
                    Math.multiplyExact(10_000L, benefit), direct));
            if (benefit < policy.minimumNetBenefitCostUnits()
                    || basisPoints < policy.minimumBenefitBasisPoints()) continue;
            CpuAccessPlan.Binding dense = denseBinding(lowered.extents(), elements);
            var candidate = new CpuMaterializationPlan(index, lowered.boundaryValues().get(index),
                    source, dense, elements, bytes, 0, Double.BYTES, useCount,
                    policy.expectedRunCount(), direct, copy, contiguousKernel, copied, benefit,
                    basisPoints, "selected: lower estimated total cost after all hard gates");
            if (best == null || copied < best.copiedTotalCost()) best = candidate;
        }
        return Optional.ofNullable(best);
    }

    private static List<CpuKernelIr.PowerRealization> powerRealizations(CpuKernelIr kernelIr) {
        return kernelIr.instructions().stream()
                .filter(instruction -> instruction.opcode()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
    }

    private static DataType vectorLaneType(CpuKernelIr kernelIr) {
        var numeric = kernelIr.values().stream().map(CpuKernelIr.Value::dataType)
                .filter(type -> type != DataType.BOOL).distinct().toList();
        if (numeric.size() > 1) return null;
        if (numeric.isEmpty()) return kernelIr.values().stream()
                .allMatch(value -> value.dataType() == DataType.BOOL) ? DataType.BOOL : null;
        DataType type = numeric.getFirst();
        return type == DataType.FLOAT32 || type == DataType.FLOAT64
                || type == DataType.INT32 || type == DataType.INT64 ? type : null;
    }

    private static boolean vectorTopologyEligible(CpuKernelIr ir, DataType laneType) {
        if (laneType == null) return false;
        boolean mixedMasks = laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64
                ? ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL) : false;
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.dataType() == DataType.BOOL && mixedMasks
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && !(value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && value.accessPlan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                return false;
            }
        }
        for (CpuKernelIr.Instruction instruction : ir.instructions()) {
            if (!instruction.opcode().vectorEligible()
                    || instruction.opcode() == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization() == CpuKernelIr.PowerRealization.DIRECT) {
                return false;
            }
            switch (instruction.opcode().vectorForm()) {
                case VALUE -> {
                    if (!valueOpcodeEligible(instruction.opcode(), laneType)) return false;
                }
                case MASK_PRODUCER -> {
                    if (!mixedMasks || ir.values().get(instruction.output()).kind()
                            != CpuKernelIr.Value.Kind.VIRTUAL) return false;
                }
                case VALUE_OR_MASK -> {
                    boolean byteValues = laneType == DataType.BOOL;
                    boolean virtualMasks = mixedMasks
                            && instruction.inputs().stream().allMatch(input ->
                                ir.values().get(input).kind() == CpuKernelIr.Value.Kind.VIRTUAL)
                            && ir.values().get(instruction.output()).kind()
                                == CpuKernelIr.Value.Kind.VIRTUAL;
                    if (!byteValues && !virtualMasks) return false;
                }
                case MASK_CONSUMER -> {
                    if (!mixedMasks) return false;
                    CpuKernelIr.Value condition = ir.values().get(instruction.inputs().getFirst());
                    if (condition.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) return false;
                }
                case NONE -> { return false; }
            }
        }
        return true;
    }

    private static boolean valueOpcodeEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode opcode,
            DataType laneType) {
        if (laneType == DataType.BOOL) return opcode
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST;
        if (laneType == DataType.INT32 || laneType == DataType.INT64) return switch (opcode) {
            case ADD, SUB, MUL, MIN, MAX, SCALAR_ADD, SCALAR_SUB, SCALAR_MUL,
                    SCALAR_MIN, SCALAR_MAX, CAST -> true;
            default -> false;
        };
        return laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64;
    }

    private static int speciesLanes(DataType type) {
        if (type == null) return 0;
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case INT32 -> IntVector.SPECIES_PREFERRED.length();
            case INT64 -> LongVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            default -> 0;
        };
    }

    private static int speciesBits(DataType type) {
        if (type == null) return 0;
        return switch (type) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> LongVector.SPECIES_PREFERRED.vectorBitSize();
            case BOOL -> ByteVector.SPECIES_PREFERRED.vectorBitSize();
            default -> 0;
        };
    }

    private static CpuAccessPlan.Binding denseBinding(long[] extents, long elementCount) {
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int i = 0; i < extents.length; i++) roles.add(CpuAccessPlan.AxisRole.CONTIGUOUS);
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, extents.length, roles, extents.length);
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, Math.max(1, extents[axis]));
        }
        return CpuAccessPlan.Binding.create(plan, extents, 0, strides, elementCount,
                0, elementCount, elementCount);
    }

    private static CpuKernelIr adjustedIr(CpuKernelIr source, int boundaryIndex,
            CpuAccessPlan plan) {
        var values = new ArrayList<>(source.values());
        CpuKernelIr.Value old = values.get(boundaryIndex);
        values.set(boundaryIndex, new CpuKernelIr.Value(old.ordinal(), old.dataType(), old.kind(), plan));
        return new CpuKernelIr(values, source.instructions(), source.loop(), source.stores());
    }

    private static boolean vectorEligible(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding,
            int lanes) {
        return switch (binding.plan().regime()) {
            case GENERAL_ODOMETER -> false;
            case SCALAR_ALL_ZERO -> binding.plan().accessKind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.READ;
            case DENSE_LINEAR -> true;
            case LAST_AXIS_BIAS, BLOCK_OUTER -> contiguousRun(binding) >= lanes;
        };
    }

    private static long contiguousRun(
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding) {
        long result = 1;
        int start = binding.extents().size() - binding.plan().contiguousSuffix();
        for (int axis = start; axis < binding.extents().size(); axis++) {
            result = Math.multiplyExact(result, binding.extents().get(axis));
        }
        return result;
    }

    private static long ceilDiv(long value, long divisor) {
        return value == 0 ? 0 : 1 + (value - 1) / divisor;
    }
}
