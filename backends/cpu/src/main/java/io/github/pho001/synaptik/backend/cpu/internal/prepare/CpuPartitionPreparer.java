package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSpecializedSubgraphRecognizer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
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
 * One-node cumulative scans remain scalar compute, declare exactly input and output with no
 * workspace or materialization, and may parallelize only across complete independent slices.
 * One-node ordinary aggregates likewise remain scalar compute, declare exactly input and output
 * with no workspace or materialization, and may parallelize only across complete output cells.
 * One-node masked SUM/MEAN plans declare ordered data, mask, and output buffers plus one exact
 * floating-state workspace sliced by the simultaneously used complete-output-cell ranges. They
 * never materialize the mask or declare selected-count, partial, or combine state.
 * One-node trailing Layer/RMS plans likewise keep scalar compute and partition complete leading
 * slices. Layer declares one exact-state slice per simultaneous range; RMS declares no workspace.
 * Neither family selects materialization, partial values, or combine state.
 */
public final class CpuPartitionPreparer implements BackendPartitionPreparer<
        CpuPartitionAnalysisInputs, CpuPartitionPreparationPlan> {
    private final CpuPartitionLowering lowering;
    private final CpuPartitionDagDecomposer decomposer = new CpuPartitionDagDecomposer();
    private final CpuSpecializedSubgraphRecognizer recognizer =
            new CpuSpecializedSubgraphRecognizer();

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
     * @return one immutable analysis with one through eight topologically ordered units, one exact
     *     deduplicated declaration per materialized value, and each unit's already-selected
     *     portable strategy and optional exact workspace; never {@code null}. Multi-unit workspace
     *     declarations are rebased to the final unit index before they enter the partition
     *     requirement list. A one-unit plan may retain the established optional external-read
     *     materialization, while a multi-unit plan disables it and retains only family-intrinsic
     *     unit workspaces. The returned plan also carries ordered CPU-private recognition facts
     *     only after their exact baseline-unit IR and resource snapshot has been validated;
     *     those facts do not alter declarations, artifact identity, finalization, or execution.
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        var units = decomposer.decompose(context, lowering);
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis;
        if (units.size() == 1) {
            analysis = annotateSingle(analyzeUnit(context, units.getFirst().lowering(), 0, true),
                    units.getFirst());
        } else {
            analysis = analyzeDag(context, units);
        }
        List<CpuSpecializedSubgraph> facts = recognizer.recognize(context,
                analysis.plan().units());
        return withRecognition(analysis, facts);
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withRecognition(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            List<CpuSpecializedSubgraph> facts) {
        var plan = analysis.plan();
        var enriched = new CpuPartitionPreparationPlan(plan.units(), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), facts);
        return new BackendPartitionAnalysis<>(analysis.partition(), enriched,
                analysis.requirements());
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> annotateSingle(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuPartitionDagDecomposer.Unit topology) {
        var plan = analysis.plan();
        var unit = plan.units().getFirst();
        var facts = new CpuPartitionPreparationPlan.UnitRuntimeFacts(plan.affineAddressPairs(),
                plan.materialization(), plan.workspaceDeclaration(), plan.workspaceUse(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry());
        var enriched = new CpuPartitionPreparationPlan.ExecutionUnitPlan(unit.portablePlan(),
                unit.boundaryValues(), unit.accessBindings(), unit.carrierPattern(),
                unit.generatedCarrierPattern(), unit.extents(), unit.elementCount(),
                unit.executionStrategy(), unit.selectedRangeCount(), unit.minimumElementsPerWorker(),
                unit.vectorSpeciesBitSize(), unit.conv2dGeometry(), unit.conv3dGeometry(),
                unit.outputCount(), unit.fusionReason(), topology.dependencies(),
                topology.memberNodeOrdinals(), facts);
        var annotated = new CpuPartitionPreparationPlan(List.of(enriched), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), List.of());
        return new BackendPartitionAnalysis<>(analysis.partition(), annotated,
                analysis.requirements());
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeUnit(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering.LoweredPartition lowered, int workspaceRequirementId,
            boolean allowMaterialization) {
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
        boolean scan = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
        boolean aggregate = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
        boolean argExtrema = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
        boolean maskedReduction = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
        boolean advancedReduction = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
        boolean softmax = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
        boolean trailingNormalization = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
        boolean batchNormalization = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
        boolean batchNormTraining = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr;
        boolean conv2d = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
        boolean conv3d = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
        Optional<CpuMaterializationPlan> materialization = !allowMaterialization || movement || indexing || scatter || fold || ordering || random || scan || aggregate || argExtrema || maskedReduction || advancedReduction || softmax || trailingNormalization || batchNormalization || batchNormTraining || conv2d || conv3d ? Optional.empty()
                : selectMaterialization(lowered, context.backendInputs().materializationPolicy(),
                        workspaceRequirementId);
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
        boolean vectorEligible = !affineCopy && !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate && !argExtrema && !maskedReduction && !advancedReduction && !softmax && !trailingNormalization && !batchNormalization && !batchNormTraining && !conv2d && !conv3d && config.computePreference()
                        == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && vectorType != null && lanes > 1 && lowered.elementCount() >= lanes
                && vectorTopologyEligible(kernelIr, vectorType)
                && bindings.stream().allMatch(binding -> vectorEligible(binding, lanes));
        int usableParallelism = Math.min(config.configuredMaximumParallelism(),
                config.availableParallelism());
        long iterationCount = lowered.elementCount();
        long minimumRangeItemsPerWorker = config.minimumElementsPerWorker();
        long[] selectedExtents = lowered.extents();
        var selectedPortableIr = lowered.portableKernelIr();
        var selectedBatchGeometry = lowered.batchNormInferenceGeometry();
        var selectedTrainingGeometry = lowered.batchNormTrainingGeometry();
        int selectedRangeCount;
        if (batchNormalization) {
            var geometry = lowered.batchNormInferenceGeometry().orElseThrow();
            var form = usableParallelism >= 2 && geometry.channelCount() < geometry.nonChannelCount()
                    ? io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.NON_CHANNEL_RANGE
                    : io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE;
            geometry = geometry.withRangeForm(form);
            iterationCount = geometry.rangeItemCount();
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(),
                            geometry.coordinatesPerRangeItem()));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            if (selectedRangeCount < 2
                    && form != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE) {
                geometry = geometry.withRangeForm(
                        io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr.RangeForm.CHANNEL_RANGE);
                iterationCount = geometry.rangeItemCount();
                minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                        ceilDiv(config.minimumElementsPerWorker(),
                                geometry.coordinatesPerRangeItem()));
                selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                        Math.toIntExact(Math.min(Integer.MAX_VALUE,
                                ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            }
            selectedBatchGeometry = Optional.of(geometry);
            selectedPortableIr = ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr)
                    lowered.portableKernelIr()).withRangeForm(geometry.rangeForm());
            kernelIr = ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr)
                    selectedPortableIr).encodedKernelIr();
            selectedExtents = new long[] {iterationCount};
        } else if (batchNormTraining) {
            var geometry = lowered.batchNormTrainingGeometry().orElseThrow();
            iterationCount = geometry.channelCount();
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(), geometry.reductionCount()));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            selectedExtents = new long[]{iterationCount};
        } else {
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
        }
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
                        || lowered.orderingGeometry().isPresent()
                        || lowered.aggregateGeometry().filter(g -> g.scratchSliceBytes() > 0)
                            .isPresent()
                        || lowered.maskedReductionGeometry().isPresent()
                        || lowered.advancedReductionGeometry().filter(g -> g.scratchSliceBytes() > 0)
                            .isPresent()
                        || lowered.trailingNormalizationGeometry()
                            .filter(g -> g.scratchSliceBytes() > 0).isPresent()
                        || lowered.batchNormTrainingGeometry()
                            .filter(g -> g.scratchSliceBytes() > 0).isPresent());
        selectedPortableIr = materialization.isPresent() ? kernelIr : selectedPortableIr;
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
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    scatterGeometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        if (lowered.orderingGeometry().isPresent()) {
            var geometry = lowered.orderingGeometry().orElseThrow();
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    geometry.workspaceBytes(selectedRangeCount), Long.BYTES));
        }
        if (lowered.aggregateGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            var geometry = lowered.aggregateGeometry().orElseThrow();
            long aggregateBytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && aggregateBytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "aggregate exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    aggregateBytes, Long.BYTES));
        }
        if (lowered.maskedReductionGeometry().filter(g -> g.outputCount() > 0).isPresent()) {
            var geometry = lowered.maskedReductionGeometry().orElseThrow();
            long maskedBytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && maskedBytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "masked-reduction exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    maskedBytes, Long.BYTES));
        }
        if (lowered.advancedReductionGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent()) {
            var geometry = lowered.advancedReductionGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "advanced-reduction exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    bytes, Long.BYTES));
        }
        if (lowered.trailingNormalizationGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent()) {
            var geometry = lowered.trailingNormalizationGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "trailing-normalization exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId, bytes,
                    Long.BYTES));
        }
        if (lowered.batchNormTrainingGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent()) {
            var geometry = lowered.batchNormTrainingGeometry().orElseThrow();
            long bytes = geometry.workspaceBytes(selectedRangeCount);
            var limit = context.backendInputs().materializationPolicy();
            if (limit.enabled() && bytes > limit.maximumAdditionalBytes())
                throw new IllegalArgumentException(
                        "batch-normalization training exact-state workspace exceeds the configured byte ceiling");
            workspace = Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId, bytes, Long.BYTES));
        }
        var workspaceUse = materialization.isPresent()
                ? CpuPartitionPreparationPlan.WorkspaceUse.MATERIALIZATION
                : workspace.isPresent() ? CpuPartitionPreparationPlan.WorkspaceUse.SCATTER_PRODUCT
                : CpuPartitionPreparationPlan.WorkspaceUse.NONE;
        if (lowered.orderingGeometry().isPresent() && workspace.isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.ORDERING_INDICES;
        if (lowered.aggregateGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.maskedReductionGeometry().filter(g -> g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.advancedReductionGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.outputCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.trailingNormalizationGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.normalizedCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        if (lowered.batchNormTrainingGeometry().filter(g -> g.scratchSliceBytes() > 0
                && g.channelCount() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE;
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.boundaryValues(), bindings, requestedCarriers, carriers,
                        selectedExtents, iterationCount, strategy, selectedRangeCount,
                        minimumRangeItemsPerWorker, vectorEligible ? speciesBits : 0,
                        lowered.conv2dGeometry(), lowered.conv3dGeometry(),
                        (int) kernelIr.values().stream()
                            .filter(value -> value.kind() == CpuKernelIr.Value.Kind.OUTPUT).count(),
                        lowered.fusionReason())),
                CpuPartitionPreparationPlan.Route.PORTABLE,
                strategy,
                declarations, lowered.boundaryValues(), bindings,
                requestedCarriers, carriers,
                selectedExtents, iterationCount, lowered.affineAddressPairs(),
                selectedRangeCount, minimumRangeItemsPerWorker,
                vectorEligible ? speciesBits : 0,
                context.backendInputs().loweringManifestEnabled() ? manifest : "",
                materialization, workspace, workspaceUse, budget,
                lowered.movementGeometry(), lowered.indexingGeometry(), lowered.scatterGeometry(),
                lowered.foldGeometry(), lowered.orderingGeometry(), lowered.randomGeometry(),
                lowered.scanGeometry(), lowered.aggregateGeometry(), lowered.argExtremaGeometry(),
                lowered.maskedReductionGeometry(), lowered.advancedReductionGeometry(),
                lowered.softmaxGeometry(), lowered.trailingNormalizationGeometry(),
                selectedBatchGeometry, selectedTrainingGeometry, lowered.conv2dGeometry(),
                List.of());

        var requirements = new ArrayList<PreparationResourceRequirement>(declarations);
        plan.workspaceDeclaration().ifPresent(requirements::add);
        return new BackendPartitionAnalysis<>(context.partition(), plan, requirements);
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeDag(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> topology) {
        var baseInputs = new CpuPartitionAnalysisInputs(
                context.backendInputs().loweringManifestEnabled(), List.of(),
                context.backendInputs().portableExecution(),
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, false);
        var analyses = new ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>>();
        for (int index = 0; index < topology.size(); index++) {
            var unit = topology.get(index);
            analyses.add(analyzeUnit(decomposer.unitContext(context, unit.nodes(), baseInputs),
                    unit.lowering(), 0, false));
        }
        var declarations = mergeDeclarations(analyses);
        List<CpuKernelSpecialization.CarrierAccess> requested =
                context.backendInputs().carrierPattern();
        if (!requested.isEmpty()) {
            if (requested.size() != declarations.size()) {
                throw new IllegalArgumentException(
                        "carrier pattern count must match distinct partition buffers: requested="
                                + requested.size() + ", buffers=" + declarations.size()
                                + ", values=" + declarations.stream()
                                    .map(PreparationResourceRequirement.Buffer::valueId).toList()
                                + ", members=" + topology.stream()
                                    .map(unit -> unit.memberNodeOrdinals() + ":"
                                            + unit.lowering().portableKernelIr().getClass().getSimpleName())
                                    .toList());
            }
            var carriers = new java.util.LinkedHashMap<ValueId,
                    CpuKernelSpecialization.CarrierAccess>();
            for (int i = 0; i < declarations.size(); i++)
                carriers.put(declarations.get(i).valueId(), requested.get(i));
            analyses.clear();
            for (int index = 0; index < topology.size(); index++) {
                var unit = topology.get(index);
                var local = unit.lowering().boundaryValues().stream().map(carriers::get).toList();
                if (local.stream().anyMatch(Objects::isNull)) {
                    throw new IllegalArgumentException("unit boundary has no carrier selection");
                }
                var inputs = new CpuPartitionAnalysisInputs(
                        context.backendInputs().loweringManifestEnabled(), local,
                        context.backendInputs().portableExecution(),
                        CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, false);
                analyses.add(analyzeUnit(decomposer.unitContext(context, unit.nodes(), inputs),
                        unit.lowering(), 0, false));
            }
            declarations = mergeDeclarations(analyses);
        }

        var bindingByValue = new java.util.LinkedHashMap<ValueId, CpuAccessPlan.Binding>();
        var requestedByValue = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        var generatedByValue = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        var units = new ArrayList<CpuPartitionPreparationPlan.ExecutionUnitPlan>();
        var requirements = new ArrayList<PreparationResourceRequirement>();
        requirements.addAll(declarations);
        for (int index = 0; index < analyses.size(); index++) {
            CpuPartitionPreparationPlan selected = analyses.get(index).plan();
            mergeUnitFacts(selected, bindingByValue, requestedByValue, generatedByValue);
            int unitIndex = index;
            Optional<PreparationResourceRequirement.Workspace> workspace =
                    selected.workspaceDeclaration().map(declaration ->
                            new PreparationResourceRequirement.Workspace(unitIndex,
                                    declaration.byteSize(), declaration.byteAlignment()));
            workspace.ifPresent(requirements::add);
            var unit = selected.units().getFirst();
            var runtimeFacts = new CpuPartitionPreparationPlan.UnitRuntimeFacts(
                    selected.affineAddressPairs(), selected.materialization(),
                    workspace, selected.workspaceUse(),
                    selected.movementGeometry(), selected.indexingGeometry(),
                    selected.scatterGeometry(), selected.foldGeometry(),
                    selected.orderingGeometry(), selected.randomGeometry(),
                    selected.scanGeometry(), selected.aggregateGeometry(),
                    selected.argExtremaGeometry(), selected.maskedReductionGeometry(),
                    selected.advancedReductionGeometry(), selected.softmaxGeometry(),
                    selected.trailingNormalizationGeometry(),
                    selected.batchNormInferenceGeometry(), selected.batchNormTrainingGeometry());
            CpuPartitionDagDecomposer.Unit topologyUnit = topology.get(index);
            units.add(new CpuPartitionPreparationPlan.ExecutionUnitPlan(unit.portablePlan(),
                    unit.boundaryValues(), unit.accessBindings(), unit.carrierPattern(),
                    unit.generatedCarrierPattern(), unit.extents(), unit.elementCount(),
                    unit.executionStrategy(), unit.selectedRangeCount(),
                    unit.minimumElementsPerWorker(), unit.vectorSpeciesBitSize(),
                    unit.conv2dGeometry(), unit.conv3dGeometry(), unit.outputCount(),
                    unit.fusionReason(), topologyUnit.dependencies(),
                    topologyUnit.memberNodeOrdinals(), runtimeFacts));
        }
        var boundaryValues = declarations.stream()
                .map(PreparationResourceRequirement.Buffer::valueId).toList();
        String manifest = context.backendInputs().loweringManifestEnabled()
                ? "form=GENERAL_PARTITION;units=" + units.size() + ";members="
                    + units.stream().map(CpuPartitionPreparationPlan.ExecutionUnitPlan::memberNodeOrdinals)
                        .toList() : "";
        var combined = new CpuPartitionPreparationPlan(units,
                CpuPartitionPreparationPlan.Route.PORTABLE,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                declarations, boundaryValues,
                boundaryValues.stream().map(bindingByValue::get).toList(),
                boundaryValues.stream().map(requestedByValue::get).toList(),
                boundaryValues.stream().map(generatedByValue::get).toList(),
                new long[0], 0, new long[0], 1, 1, 0, manifest,
                Optional.empty(), Optional.empty(), CpuPartitionPreparationPlan.WorkspaceUse.NONE,
                new CpuSpecializationBudget(4, 1, 0, 0), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), List.of());
        return new BackendPartitionAnalysis<>(context.partition(), combined, requirements);
    }

    private static List<PreparationResourceRequirement.Buffer> mergeDeclarations(
            List<BackendPartitionAnalysis<CpuPartitionPreparationPlan>> analyses) {
        var result = new java.util.LinkedHashMap<ValueId,
                PreparationResourceRequirement.Buffer>();
        for (var analysis : analyses) for (var declaration : analysis.plan().bufferDeclarations()) {
            var old = result.putIfAbsent(declaration.valueId(), declaration);
            if (old != null && (old.byteSize() != declaration.byteSize()
                    || old.byteAlignment() != declaration.byteAlignment())) {
                throw new IllegalArgumentException("unit buffer declarations disagree");
            }
        }
        return List.copyOf(result.values());
    }

    private static void mergeUnitFacts(CpuPartitionPreparationPlan unit,
            java.util.Map<ValueId, CpuAccessPlan.Binding> bindings,
            java.util.Map<ValueId, CpuKernelSpecialization.CarrierAccess> requested,
            java.util.Map<ValueId, CpuKernelSpecialization.CarrierAccess> generated) {
        for (int i = 0; i < unit.boundaryValues().size(); i++) {
            bindings.putIfAbsent(unit.boundaryValues().get(i), unit.accessBindings().get(i));
            requested.putIfAbsent(unit.boundaryValues().get(i), unit.carrierPattern().get(i));
            generated.putIfAbsent(unit.boundaryValues().get(i),
                    unit.generatedCarrierPattern().get(i));
        }
    }

    private static Optional<CpuMaterializationPlan> selectMaterialization(
            CpuPartitionLowering.LoweredPartition lowered,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy, int workspaceRequirementId) {
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
                    source, dense, elements, bytes, workspaceRequirementId, Double.BYTES, useCount,
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
