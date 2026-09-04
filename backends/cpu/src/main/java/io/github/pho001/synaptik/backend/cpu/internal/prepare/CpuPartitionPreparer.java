package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuSpecializationBudget;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSpecializedSubgraphRecognizer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFusionProfitabilitySelector;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRepresentationPlanner;
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
 * Analysis retains complete direct, eligible single-copy, and eligible disjoint-consumer pair
 * representation candidates for FLOAT64, FLOAT32, INT32, INT64, and canonical BOOL pointwise
 * work. It rejects a pair when one represented instruction consumes both copied sources, and one
 * compatible copy identity may serve repeated or cross-unit consumers. Ordinary preparation
 * selects the exact CPU 0008D direct topology with no representation-copy resources; retained
 * materialized forms remain complete candidate data for later explicit pre-Runtime promotion.
 * Analysis then selects scalar or preferred-species vector compute and single-thread or bounded
 * parallel orchestration before shared resource assignment. Exact vector eligibility is
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
    private final CpuFusionProfitabilitySelector selector =
            new CpuFusionProfitabilitySelector();
    private final CpuRepresentationPlanner representationPlanner =
            new CpuRepresentationPlanner();

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
     *     The plan also retains the authoritative logical-memory graph-publication boundary
     *     positions solely so its selected boundary roles can be independently recomputed.
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if complete-partition lowering rejects the occurrence or
     *     declared resource geometry is invalid
     * @throws ArithmeticException if exact byte geometry overflows {@code long}
     */
    @Override public BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        List<CpuPartitionDagDecomposer.Unit> baseline = decomposer.decompose(context, lowering);
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> baselineAnalysis =
                analyzeTopology(context, baseline, null);
        List<CpuSpecializedSubgraph> recognition = recognizer.recognize(context,
                baselineAnalysis.plan().units());
        CpuPartitionDagDecomposer.Enumeration enumeration = decomposer.enumerate(context,
                lowering, recognition);
        var analyses = new ArrayList<BackendPartitionAnalysis<CpuPartitionPreparationPlan>>();
        var candidates = new ArrayList<CpuFusionProfitabilitySelector.Candidate>();
        for (List<CpuPartitionDagDecomposer.Unit> topology : enumeration.candidates()) {
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis =
                    sameTopology(topology, baseline) ? baselineAnalysis
                            : analyzeTopology(context, topology,
                                    baselineAnalysis.plan().boundaryValues());
            analyses.add(analysis);
            candidates.add(new CpuFusionProfitabilitySelector.Candidate(topology, analysis.plan()));
        }
        CpuFusionProfitabilitySelector.Result selected = selector.select(context, enumeration,
                candidates);
        CpuRepresentationPlanner.Result representation = representationPlanner.select(context,
                candidates, selected.decisions());
        BackendPartitionAnalysis<CpuPartitionPreparationPlan> represented = withRepresentation(
                analyses.get(representation.candidateIndex()), representation);
        return withMetadata(context, represented, recognition, selected.decisions());
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withRepresentation(
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            CpuRepresentationPlanner.Result representation) {
        var plan = analysis.plan();
        var representedUnits = new ArrayList<CpuPartitionPreparationPlan.RepresentationUnitPlan>();
        for (int unitIndex = 0; !representation.materializations().isEmpty()
                && unitIndex < plan.units().size(); unitIndex++) {
            int representedUnitIndex = unitIndex;
            var unit = plan.units().get(unitIndex);
            var bindings = new ArrayList<>(unit.accessBindings());
            var carriers = new ArrayList<>(unit.generatedCarrierPattern());
            CpuKernelIr ir = unit.portablePlan().kernelIr();
            var representedMatmul=unit.portablePlan().specialization().matmulIr();
            boolean changed = false;
            for (CpuMaterializationPlan copy : representation.materializations()) {
                for (var consumer : copy.consumers()) {
                    if (consumer.unitPosition() != unitIndex) continue;
                    bindings.set(consumer.boundaryPosition(), copy.consumerBinding());
                    carriers.set(consumer.boundaryPosition(),
                            CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                    ir = adjustedIr(ir, consumer.boundaryPosition(),
                            copy.consumerBinding().plan());
                    if(representedMatmul.isPresent()) {
                        var matmul=representedMatmul.orElseThrow().withInputAccess(
                                consumer.boundaryPosition(),copy.consumerBinding().plan());
                        representedMatmul=Optional.of(matmul);ir=matmul.encodedKernelIr();
                    }
                    changed = true;
                }
            }
            CpuPortableRoutePlan route = unit.portablePlan();
            if (changed) {
                var old = route.specialization();
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()), old.numericalMode(),
                        old.executionStrategy(), old.boundaryDataTypes(), carriers,
                        old.vectorSpeciesBitSize(), representation.materializations().size() == 1
                            ? representation.materializations().getFirst().consumers().stream()
                                .filter(value -> value.unitPosition() == representedUnitIndex)
                                .mapToInt(value -> value.boundaryPosition()).findFirst().orElse(-1)
                            : -1,
                        old.scalarPowerRealizations(), old.scratchParameter(),
                        old.classIdentitySchema(),representedMatmul);
                route=new CpuPortableRoutePlan(ir,specialization);
            }
            representedUnits.add(new CpuPartitionPreparationPlan.RepresentationUnitPlan(unitIndex,
                    route, bindings, carriers));
        }
        var requirements = new ArrayList<PreparationResourceRequirement>(analysis.requirements());
        for (CpuMaterializationPlan copy : representation.materializations()) requirements.add(
                new PreparationResourceRequirement.Workspace(copy.workspaceRequirementId(),
                        copy.byteCount(), copy.byteAlignment()));
        var representedPlan = new CpuPartitionPreparationPlan(plan.units(), plan.route(),
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
                plan.conv2dGeometry(), plan.specializedSubgraphs(), plan.fusionDecisions(),
                plan.publicationBoundaryPositions(), representation.materializations(),
                representation.materializations().isEmpty() ? List.of() : representedUnits,
                representation.decisions());
        return new BackendPartitionAnalysis<>(analysis.partition(), representedPlan, requirements);
    }

    private BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyzeTopology(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> units, List<ValueId> carrierReferenceValues) {
        if (units.size() != 1) return analyzeDag(context, units, carrierReferenceValues);
        var unit = units.getFirst();
        PrepareContext<CpuPartitionAnalysisInputs> unitContext = context;
        if (carrierReferenceValues != null && !context.backendInputs().carrierPattern().isEmpty()) {
            var inputs = candidateInputs(context, unit.lowering().boundaryValues(),
                    carrierReferenceValues, true);
            unitContext = decomposer.unitContext(context, unit.nodes(), inputs);
        }
        return annotateSingle(analyzeUnit(unitContext, unit.lowering(), 0, true), unit);
    }

    private static CpuPartitionAnalysisInputs candidateInputs(
            PrepareContext<CpuPartitionAnalysisInputs> context, List<ValueId> boundaryValues,
            List<ValueId> carrierReferenceValues, boolean allowMaterialization) {
        List<CpuKernelSpecialization.CarrierAccess> requested =
                context.backendInputs().carrierPattern();
        if (requested.size() != carrierReferenceValues.size()) {
            throw new IllegalArgumentException(
                    "carrier pattern count must match the exact 0008B baseline buffers");
        }
        var carriers = new java.util.LinkedHashMap<ValueId,
                CpuKernelSpecialization.CarrierAccess>();
        for (int index = 0; index < carrierReferenceValues.size(); index++) {
            carriers.put(carrierReferenceValues.get(index), requested.get(index));
        }
        var projected = boundaryValues.stream().map(value -> carriers.getOrDefault(value,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)).toList();
        return new CpuPartitionAnalysisInputs(
                context.backendInputs().loweringManifestEnabled(), projected,
                context.backendInputs().portableExecution(),
                allowMaterialization ? context.backendInputs().materializationPolicy()
                        : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED,
                allowMaterialization
                        && context.backendInputs().conv2dMaterializedSuffixUnit());
    }

    private static boolean sameTopology(List<CpuPartitionDagDecomposer.Unit> left,
            List<CpuPartitionDagDecomposer.Unit> right) {
        return left.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList()
                .equals(right.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList());
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> withMetadata(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis,
            List<CpuSpecializedSubgraph> facts,
            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision> decisions) {
        var plan = analysis.plan();
        var publications = new java.util.HashSet<ValueId>();
        context.memoryRequirements().stream().filter(
                io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement::graphOutput)
                .map(io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement::valueId)
                .forEach(publications::add);
        List<Integer> publicationBoundaryPositions = java.util.stream.IntStream.range(
                0, plan.boundaryValues().size()).filter(position ->
                        publications.contains(plan.boundaryValues().get(position))).boxed().toList();
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
                plan.conv2dGeometry(), facts, decisions, publicationBoundaryPositions,
                plan.materializations(), plan.representationUnits(),
                plan.representationDecisions());
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
                unit.matmulGeometry(), unit.pool2dGeometry(), unit.pool3dGeometry(),
                unit.attentionGeometry(), unit.outputCount(),
                unit.fusionReason(), topology.dependencies(),
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
        boolean matmul = lowered.matmulIr().isPresent();
        boolean pool2d = lowered.pool2dGeometry().isPresent();
        boolean pool3d = lowered.pool3dGeometry().isPresent();
        boolean attention = lowered.attentionGeometry().isPresent();
        Optional<CpuMaterializationPlan> materialization = Optional.empty();
        var declarations = new ArrayList<PreparationResourceRequirement.Buffer>(lowered.boundaryValues().size());
        for (int i = 0; i < lowered.boundaryValues().size(); i++) declarations.add(
                new PreparationResourceRequirement.Buffer(lowered.boundaryValues().get(i),
                        Math.multiplyExact(lowered.referencedElementSpans().get(i),
                                lowered.boundaryDataTypes().get(i).byteWidth()),
                        lowered.boundaryDataTypes().get(i).byteWidth()));
        var bindings = new ArrayList<>(lowered.accessBindings());
        var carriers = new ArrayList<>(requestedCarriers);
        CpuKernelIr kernelIr = lowered.kernelIr();
        boolean affineCopy = lowered.portableKernelIr()
                instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
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
        boolean matmulVector = matmul && config.computePreference()
                == CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE
                && (lowered.matmulIr().orElseThrow().realization()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_N_VECTOR
                ||lowered.matmulIr().orElseThrow().realization()
                == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.TILED_N_VECTOR_2X2);
        boolean vectorEligible = matmulVector || !matmul && !pool2d && !pool3d && !attention && !affineCopy && !movement
                && !indexing && !scatter && !fold && !ordering && !random && !scan && !aggregate
                && !argExtrema && !maskedReduction && !advancedReduction && !softmax
                && !trailingNormalization && !batchNormalization && !batchNormTraining
                && !conv2d && !conv3d && config.computePreference()
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
        if (matmul) {
            var geometry = lowered.matmulGeometry().orElseThrow();
            var form=lowered.matmulIr().orElseThrow().realization();
            iterationCount = switch (form) {
                case DIRECT_SCALAR -> geometry.outputCount();
                case DIRECT_N_VECTOR -> Math.multiplyExact(geometry.batchCount(), geometry.m());
                case TILED_SCALAR_2X2 -> Math.multiplyExact(geometry.batchCount(),
                        Math.multiplyExact(ceilDiv(geometry.m(), 2), ceilDiv(geometry.n(), 2)));
                case TILED_N_VECTOR_2X2 -> Math.multiplyExact(geometry.batchCount(),
                        Math.multiplyExact(ceilDiv(geometry.m(), 2),
                                ceilDiv(geometry.n(), Math.multiplyExact(2L, lanes))));
            };
            long coordinatesPerWorkUnit = iterationCount == 0 ? 1
                    : Math.max(1, geometry.outputCount() / iterationCount);
            minimumRangeItemsPerWorker = iterationCount == 0 ? 1 : Math.max(1,
                    ceilDiv(config.minimumElementsPerWorker(), coordinatesPerWorkUnit));
            selectedRangeCount = iterationCount == 0 ? 1 : Math.min(usableParallelism,
                    Math.toIntExact(Math.min(Integer.MAX_VALUE,
                            ceilDiv(iterationCount, minimumRangeItemsPerWorker))));
            selectedExtents = new long[] {iterationCount};
        } else if (batchNormalization) {
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
                powerRealizations(kernelIr), attention || lowered.scatterGeometry()
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
                            .filter(g -> g.scratchSliceBytes() > 0).isPresent()
                        || lowered.attentionGeometry().filter(g -> g.scratchSliceBytes() > 0)
                            .isPresent(),
                attention ? 57 : pool3d ? 56 : pool2d ? 55 : matmul ? 54
                        : selectedPortableIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr
                            ? 58 : 52,
                lowered.matmulIr());
        selectedPortableIr = matmul||materialization.isPresent()?kernelIr:selectedPortableIr;
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
        if (lowered.attentionGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent()) {
            var geometry=lowered.attentionGeometry().orElseThrow();
            workspace=Optional.of(new PreparationResourceRequirement.Workspace(workspaceRequirementId,
                    geometry.workspaceBytes(selectedRangeCount),Long.BYTES));
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
        if (lowered.attentionGeometry().filter(g -> g.scratchSliceBytes() > 0).isPresent())
            workspaceUse = CpuPartitionPreparationPlan.WorkspaceUse.ATTENTION_ROW_STATE;
        var plan = new CpuPartitionPreparationPlan(
                List.of(new CpuPartitionPreparationPlan.ExecutionUnitPlan(
                        routePlan, lowered.boundaryValues(), bindings, requestedCarriers, carriers,
                        selectedExtents, iterationCount, strategy, selectedRangeCount,
                        minimumRangeItemsPerWorker, vectorEligible ? speciesBits : 0,
                        lowered.conv2dGeometry(), lowered.conv3dGeometry(), lowered.matmulGeometry(),
                        lowered.pool2dGeometry(), lowered.pool3dGeometry(), lowered.attentionGeometry(),
                        (int) kernelIr.values().stream()
                            .filter(value -> value.kind() == CpuKernelIr.Value.Kind.OUTPUT).count(),
                        lowered.fusionReason(), List.of(), List.of(),
                        CpuPartitionPreparationPlan.UnitRuntimeFacts.EMPTY)),
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
            List<CpuPartitionDagDecomposer.Unit> topology,
            List<ValueId> carrierReferenceValues) {
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
            if (carrierReferenceValues == null && requested.size() != declarations.size()) {
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
            List<ValueId> reference = carrierReferenceValues == null
                    ? declarations.stream().map(PreparationResourceRequirement.Buffer::valueId)
                            .toList()
                    : carrierReferenceValues;
            if (requested.size() != reference.size()) throw new IllegalArgumentException(
                    "carrier pattern count must match the exact 0008B baseline buffers");
            analyses.clear();
            for (int index = 0; index < topology.size(); index++) {
                var unit = topology.get(index);
                var inputs = candidateInputs(context, unit.lowering().boundaryValues(), reference,
                        false);
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
                    unit.conv2dGeometry(), unit.conv3dGeometry(), unit.matmulGeometry(),
                    unit.pool2dGeometry(), unit.pool3dGeometry(), unit.attentionGeometry(),
                    unit.outputCount(),
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
