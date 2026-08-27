package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Stateless CPU-private owner of bounded direct/single/pair representation enumeration and safe
 * ordinary selection over the unchanged legal CPU 0008D topologies. Complete materialized
 * variants retain their checked costs as candidate facts, while ordinary preparation selects the
 * direct representation already chosen by CPU 0008D. A pair is retained as a typed rejection,
 * before ranking, when one represented instruction consumes both proposed copied sources; its two
 * single-copy alternatives remain complete candidates. Compatible repeated and cross-unit uses
 * of one source share one copy identity and one generated copy unit. The planner performs no
 * measurement and exposes no choice to Runtime.
 */
public final class CpuRepresentationPlanner {
    /** Maximum complete eligible external-read sources. */
    static final int MAX_ELIGIBLE_SOURCES = 8;
    /** Maximum complete non-direct single and pair choices per topology. */
    static final int MAX_COPY_CHOICES = 36;
    /** Maximum direct plus copied variants per topology. */
    static final int MAX_VARIANTS_PER_TOPOLOGY = 37;

    /** Creates a planner with no mutable state, cache, measurement, or Runtime dependency. */
    public CpuRepresentationPlanner() { }

    /**
     * Selected candidate, its realized copies, and all retained representation facts.
     *
     * @param candidateIndex non-negative selected CPU 0008D topology index
     * @param materializations ordered realized copies; empty for ordinary direct selection
     * @param decisions complete retained variants/rejections followed by final selection
     */
    public record Result(int candidateIndex, List<CpuMaterializationPlan> materializations,
            List<CpuRepresentationDecision> decisions) {
        /** Snapshots and validates a bounded final result. */
        public Result {
            materializations = List.copyOf(materializations);
            decisions = List.copyOf(decisions);
            if (candidateIndex < 0 || materializations.size() > 2 || decisions.isEmpty()
                    || decisions.size() > CpuRepresentationDecision.MAX_VARIANTS + 1
                    || !(decisions.getLast() instanceof CpuRepresentationDecision.Selection)) {
                throw new IllegalArgumentException("CPU representation result is incomplete");
            }
        }
    }

    /**
     * Enumerates every complete bounded representation and selects the CPU 0008D direct topology.
     *
     * @param context complete CPU analysis context
     * @param candidates legal topology candidates in unchanged enumeration order
     * @param fusionDecisions unchanged CPU 0008D facts for those candidates
     * @return selected topology index, zero through two copies, and retained typed facts
     */
    public Result select(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuFusionProfitabilitySelector.Candidate> candidates,
            List<CpuFusionDecision> fusionDecisions) {
        Objects.requireNonNull(context, "context");
        candidates = List.copyOf(candidates);
        fusionDecisions = List.copyOf(fusionDecisions);
        var legal = fusionDecisions.stream().filter(CpuFusionDecision.LegalCandidate.class::isInstance)
                .map(CpuFusionDecision.LegalCandidate.class::cast).toList();
        CpuFusionDecision.Selection fusionSelection = fusionDecisions.stream()
                .filter(CpuFusionDecision.Selection.class::isInstance)
                .map(CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow();
        if (candidates.isEmpty() || candidates.size() != legal.size() || candidates.size() > 64)
            throw new IllegalArgumentException("CPU representation topology facts disagree");
        int canonicalIndex = -1;
        var work = new ArrayList<WorkVariant>();
        var retained = new ArrayList<CpuRepresentationDecision>();
        boolean complete = true;
        for (int candidateIndex = 0; candidateIndex < candidates.size(); candidateIndex++) {
            var candidate = candidates.get(candidateIndex);
            CpuFusionDecision.CandidateIdentity identity = CpuFusionProfitabilitySelector
                    .identityOf(context, candidate);
            CpuFusionDecision.LegalCandidate topology = legal.stream()
                    .filter(value -> value.identity().equals(identity)).findFirst().orElseThrow();
            if (topology.canonicalSplit()) canonicalIndex = candidateIndex;
            Optional<Long> topologyScore = topology.score().map(CpuFusionDecision.Score::totalScore);
            List<Eligible> eligible;
            try { eligible = eligible(candidate.plan(), identity, context); }
            catch (ArithmeticException | IllegalArgumentException uncertain) {
                eligible = List.of(); complete = false;
            }
            if (eligible.size() > MAX_ELIGIBLE_SOURCES || topologyScore.isEmpty()) {
                complete = false;
                eligible = List.of();
            }
            try {
                addVariants(work, retained, candidateIndex, topology, candidate.plan(), eligible,
                        topologyScore.orElse(0L), context.backendInputs().materializationPolicy());
            } catch (ArithmeticException uncertain) {
                complete = false;
            }
        }
        if (canonicalIndex < 0) throw new IllegalArgumentException(
                "CPU representation facts omit canonical split");
        final int canonicalCandidateIndex = canonicalIndex;
        complete &= fusionSelection.reason()
                != CpuFusionDecision.SelectionReason.ENUMERATION_BUDGET_FALLBACK
                && fusionSelection.reason() != CpuFusionDecision.SelectionReason.UNCERTAINTY_FALLBACK;
        CpuRepresentationDecision.Variant canonical = directVariant(work, canonicalIndex);
        WorkVariant fusionSelected = work.stream().filter(value -> value.plans().isEmpty()
                && value.fact().identity().topology().equals(fusionSelection.selected()))
                .findFirst().orElseThrow();
        CpuPartitionAnalysisInputs.MaterializationPolicy policy =
                context.backendInputs().materializationPolicy();
        WorkVariant selected = !policy.enabled() || complete ? fusionSelected : work.stream().filter(value ->
                value.candidateIndex() == canonicalCandidateIndex && value.plans().isEmpty())
                .findFirst().orElseThrow();
        CpuRepresentationDecision.SelectionReason reason = !policy.enabled()
                ? CpuRepresentationDecision.SelectionReason.DIRECT_POLICY_DISABLED
                : !complete ? CpuRepresentationDecision.SelectionReason.DIRECT_UNCERTAINTY
                : CpuRepresentationDecision.SelectionReason.DIRECT_MATERIALIZATION_UNPROVED;
        var facts = new ArrayList<CpuRepresentationDecision>(retained);
        facts.add(new CpuRepresentationDecision.Selection(selected.fact().identity(),
                canonical.identity(), selected.stableRank(), reason));
        if (facts.size() + fusionDecisions.size()
                > CpuRepresentationDecision.MAX_TOTAL_DECISION_FACTS) throw new IllegalArgumentException(
                        "CPU combined decision facts exceed the exact ceiling");
        return new Result(selected.candidateIndex(), selected.plans(), facts);
    }

    private static void addVariants(List<WorkVariant> result,
            List<CpuRepresentationDecision> retained, int candidateIndex,
            CpuFusionDecision.LegalCandidate topology, CpuPartitionPreparationPlan plan,
            List<Eligible> eligible, long topologyScore,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        addVariant(result, retained, candidateIndex, topology, plan, eligible, List.of(),
                topologyScore, policy);
        if (!policy.enabled()) return;
        for (int left = 0; left < eligible.size(); left++)
            addVariant(result, retained, candidateIndex, topology, plan, eligible, List.of(left),
                    topologyScore, policy);
        for (int left = 0; left < eligible.size(); left++)
            for (int right = left + 1; right < eligible.size(); right++)
                addVariant(result, retained, candidateIndex, topology, plan, eligible,
                        List.of(left, right), topologyScore, policy);
        int count = 1 + eligible.size() + eligible.size() * (eligible.size() - 1) / 2;
        if (count > MAX_VARIANTS_PER_TOPOLOGY || count - 1 > MAX_COPY_CHOICES)
            throw new IllegalArgumentException("CPU representation variant ceiling exceeded");
    }

    private static void addVariant(List<WorkVariant> result,
            List<CpuRepresentationDecision> retained, int candidateIndex,
            CpuFusionDecision.LegalCandidate topology, CpuPartitionPreparationPlan plan,
            List<Eligible> eligible, List<Integer> selected, long topologyScore,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        var copies = new ArrayList<CpuMaterializationPlan>();
        for (int index : selected) copies.add(withWorkspace(eligible.get(index).plan(),
                8 + copies.size()));
        var identity = new CpuRepresentationDecision.VariantIdentity(topology.identity(),
                copies.stream().map(CpuMaterializationPlan::identity).toList());
        int stableRank = retained.size();
        if (copies.size() == 2 && coConsumedPair(plan, copies)) {
            retained.add(new CpuRepresentationDecision.Rejection(identity,
                    CpuRepresentationDecision.RejectionReason.CO_CONSUMED_PAIR));
            return;
        }
        long representation = 0, bytes = 0, direct = 0, copied = 0;
        for (int index = 0; index < eligible.size(); index++) {
            Eligible source = eligible.get(index);
            boolean copy = selected.contains(index);
            representation = Math.addExact(representation, copy ? source.plan().copiedTotalCost()
                    : source.plan().directCost());
            if (copy) {
                CpuMaterializationPlan selectedPlan = copies.get(selected.indexOf(index));
                bytes = Math.addExact(bytes, selectedPlan.byteCount());
                direct = Math.addExact(direct, selectedPlan.directCost());
                copied = Math.addExact(copied, selectedPlan.copiedTotalCost());
            }
        }
        long comparison = Math.addExact(topologyScore, representation);
        Optional<Long> directFact = copies.isEmpty() ? Optional.empty() : Optional.of(direct);
        Optional<Long> copiedFact = copies.isEmpty() ? Optional.empty() : Optional.of(copied);
        Optional<Long> benefit = copies.isEmpty() ? Optional.empty()
                : Optional.of(Math.subtractExact(direct, copied));
        Optional<Integer> basis = copies.isEmpty() ? Optional.empty() : Optional.of(
                Math.toIntExact(Math.floorDiv(Math.multiplyExact(10_000L,
                        benefit.orElseThrow()), direct)));
        var fact = new CpuRepresentationDecision.Variant(identity, topologyScore, representation,
                comparison, bytes, plan.bufferDeclarations().size()
                        + plan.units().stream().mapToInt(unit -> unit.runtimeFacts()
                                .workspaceDeclaration().isPresent() ? 1 : 0).sum() + copies.size(),
                topology.stableRank(), topology.canonicalSplit(), topology.compatibilityBaseline(),
                directFact, copiedFact, benefit, basis);
        retained.add(fact);
        result.add(new WorkVariant(candidateIndex, copies, fact, topology.facts(), stableRank));
    }

    private static boolean coConsumedPair(CpuPartitionPreparationPlan plan,
            List<CpuMaterializationPlan> copies) {
        CpuMaterializationPlan left = copies.get(0), right = copies.get(1);
        for (CpuRepresentationDecision.ConsumerPosition leftConsumer : left.consumers()) {
            for (CpuRepresentationDecision.ConsumerPosition rightConsumer : right.consumers()) {
                if (leftConsumer.unitPosition() != rightConsumer.unitPosition()) continue;
                var unit = plan.units().get(leftConsumer.unitPosition());
                if (!(unit.portablePlan().portableKernelIr() instanceof CpuKernelIr ir)) {
                    throw new IllegalArgumentException("CPU represented pair has non-pointwise IR");
                }
                var values = ir.values().stream()
                        .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
                int leftOrdinal = values.get(leftConsumer.boundaryPosition()).ordinal();
                int rightOrdinal = values.get(rightConsumer.boundaryPosition()).ordinal();
                if (ir.instructions().stream().anyMatch(instruction ->
                        instruction.inputs().contains(leftOrdinal)
                                && instruction.inputs().contains(rightOrdinal))) return true;
            }
        }
        return false;
    }

    private static boolean selectable(WorkVariant value, CpuRepresentationDecision.Variant canonical,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy, boolean complete) {
        if (!complete) return value.fact().identity().equals(canonical.identity());
        if (value.topologyFacts().maximumGeneratedCodeSizeUnits()
                    > CpuFusionProfitabilitySelector.MAX_PROFITABLE_CODE_UNITS
                || value.topologyFacts().maximumSimultaneouslyLiveValues()
                    > CpuFusionProfitabilitySelector.MAX_PROFITABLE_LIVE_VALUES) return false;
        if (value.plans().isEmpty()) {
            return value.fact().identity().equals(canonical.identity())
                    || differenceAtLeast(canonical.comparisonScore(),
                            value.fact().comparisonScore(),
                            CpuFusionProfitabilitySelector.REQUIRED_MARGIN);
        }
        if (!policy.enabled() || value.fact().copiedBytes() > policy.maximumAdditionalBytes()
                || value.plans().stream().anyMatch(plan -> plan.netBenefit() <= 0)
                || value.fact().netBenefit().orElseThrow() < policy.minimumNetBenefitCostUnits()
                || value.fact().benefitBasisPoints().orElseThrow()
                        < policy.minimumBenefitBasisPoints()
                || !differenceAtLeast(canonical.comparisonScore(),
                        value.fact().comparisonScore(),
                        CpuFusionProfitabilitySelector.REQUIRED_MARGIN)) return false;
        return true;
    }

    private static boolean differenceAtLeast(long baseline, long candidate, long margin) {
        try { return Math.subtractExact(baseline, candidate) >= margin; }
        catch (ArithmeticException uncertain) { return false; }
    }

    private static CpuRepresentationDecision.Variant directVariant(List<WorkVariant> work,
            int candidateIndex) {
        return work.stream().filter(value -> value.candidateIndex() == candidateIndex
                && value.plans().isEmpty()).map(WorkVariant::fact).findFirst().orElseThrow();
    }

    private static List<Eligible> eligible(CpuPartitionPreparationPlan plan,
            CpuFusionDecision.CandidateIdentity identity,
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        var result = new ArrayList<Eligible>();
        for (int boundary = 0; boundary < plan.boundaryValues().size(); boundary++) {
            ValueId sourceValue = plan.boundaryValues().get(boundary);
            boolean producedInside = plan.units().stream().anyMatch(unit -> {
                int local = unit.boundaryValues().indexOf(sourceValue);
                if (local < 0) return false;
                var values = unit.portablePlan().kernelIr().values().stream()
                        .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
                return values.get(local).kind() == CpuKernelIr.Value.Kind.OUTPUT;
            });
            if (producedInside) continue;
            DataType type = null;
            CpuAccessPlan.Binding sourceBinding = null;
            CpuKernelSpecialization.CarrierAccess carrier = null;
            var consumers = new ArrayList<CpuRepresentationDecision.ConsumerPosition>();
            boolean compatible = true;
            for (int unitIndex = 0; unitIndex < plan.units().size(); unitIndex++) {
                var unit = plan.units().get(unitIndex);
                int local = unit.boundaryValues().indexOf(sourceValue);
                if (local < 0) continue;
                if (!(unit.portablePlan().portableKernelIr() instanceof CpuKernelIr ir)) {
                    compatible = false; break;
                }
                CpuFusionDecision.BoundaryRole role = identity.units().get(unitIndex)
                        .boundaries().get(local).role();
                if (role != CpuFusionDecision.BoundaryRole.EXTERNAL_READ
                        && role != CpuFusionDecision.BoundaryRole.CROSS_UNIT) {
                    compatible = false; break;
                }
                var values = ir.values().stream()
                        .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
                CpuKernelIr.Value value = values.get(local);
                if (value.kind() != CpuKernelIr.Value.Kind.INPUT || value.dataType() == DataType.BFLOAT16
                        || !admitted(value.dataType())) { compatible = false; break; }
                long uses = ir.instructions().stream().flatMap(instruction ->
                        instruction.inputs().stream()).filter(input -> input == value.ordinal()).count();
                if (uses == 0) { compatible = false; break; }
                CpuAccessPlan.Binding binding = unit.accessBindings().get(local);
                CpuKernelSpecialization.CarrierAccess localCarrier = unit.carrierPattern().get(local);
                if (type != null && (type != value.dataType() || !sourceBinding.equals(binding)
                        || carrier != localCarrier || binding.elementCount()
                            != sourceBinding.elementCount())) { compatible = false; break; }
                type = value.dataType(); sourceBinding = binding; carrier = localCarrier;
                consumers.add(new CpuRepresentationDecision.ConsumerPosition(unitIndex, local, uses));
            }
            if (!compatible || consumers.isEmpty() || sourceBinding.plan().regime()
                    == CpuAccessPlan.Regime.DENSE_LINEAR || sourceBinding.plan().regime()
                    == CpuAccessPlan.Regime.SCALAR_ALL_ZERO) continue;
            long elements = sourceBinding.elementCount();
            CpuAccessPlan.Binding dense = denseBinding(sourceBinding.extents().stream()
                    .mapToLong(Long::longValue).toArray(), elements);
            long uses = consumers.stream().mapToLong(
                    CpuRepresentationDecision.ConsumerPosition::instructionUseCount)
                    .reduce(0, Math::addExact);
            result.add(new Eligible(materialization(boundary, type, carrier, sourceBinding, dense,
                    consumers, uses, context.backendInputs())));
        }
        return List.copyOf(result);
    }

    private static CpuMaterializationPlan materialization(int boundary, DataType type,
            CpuKernelSpecialization.CarrierAccess carrier, CpuAccessPlan.Binding source,
            CpuAccessPlan.Binding dense,
            List<CpuRepresentationDecision.ConsumerPosition> consumers, long uses,
            CpuPartitionAnalysisInputs inputs) {
        var policy = inputs.materializationPolicy();
        long elements = source.elementCount();
        long bytes = Math.multiplyExact(elements, type.byteWidth());
        long directKernel = Math.multiplyExact(elements, policy.directKernelCostUnitsPerElement());
        long contiguous = Math.multiplyExact(elements,
                policy.contiguousKernelCostUnitsPerElement());
        long copy = Math.addExact(policy.copyFixedCostUnits(),
                Math.multiplyExact(elements, policy.copyCostUnitsPerElement()));
        long direct = Math.multiplyExact(policy.expectedRunCount(),
                Math.multiplyExact(uses, directKernel));
        long copied = Math.multiplyExact(policy.expectedRunCount(),
                Math.addExact(copy, Math.multiplyExact(uses, contiguous)));
        long benefit = Math.subtractExact(direct, copied);
        int basis = direct == 0 ? 0 : Math.toIntExact(Math.floorDiv(
                Math.multiplyExact(10_000L, benefit), direct));
        CpuAccessPlan write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                dense.plan().regime(), dense.plan().iterationRank(), dense.plan().axisRoles(),
                dense.plan().contiguousSuffix());
        var copyIr = new CpuAffineCopyIr(type, source.plan(), write,
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.CONTIGUOUS,
                        source.plan().iterationRank(), source.plan().iterationRank(), List.of())),
                CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(copyIr.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, List.of(type, type),
                List.of(carrier, CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT), 0, -1,
                List.of(), false);
        return new CpuMaterializationPlan(boundary, type, carrier, source, dense, consumers,
                elements, bytes, 8, type.byteWidth(), uses, policy.expectedRunCount(), direct,
                copy, contiguous, copied, benefit, basis,
                CpuRepresentationDecision.CopyStrategy.SCALAR, copyIr, specialization,
                affinePairs(source));
    }

    private static CpuMaterializationPlan withWorkspace(CpuMaterializationPlan source, int id) {
        return new CpuMaterializationPlan(source.sourceBoundaryIndex(), source.dataType(),
                source.sourceCarrier(), source.sourceBinding(), source.consumerBinding(),
                source.consumers(), source.elementCount(), source.byteCount(), id,
                source.byteAlignment(), source.useCount(), source.expectedRunCount(),
                source.directCost(), source.copyCost(), source.contiguousCost(),
                source.copiedTotalCost(), source.netBenefit(), source.benefitBasisPoints(),
                source.copyStrategy(), source.copyIr(), source.copySpecialization(),
                source.affineAddressPairs());
    }

    private static long[] affinePairs(CpuAccessPlan.Binding binding) {
        int count = Math.toIntExact(binding.elementCount());
        long[] pairs = new long[Math.multiplyExact(count, 2)];
        long[] extents = binding.extents().stream().mapToLong(Long::longValue).toArray();
        long[] strides = binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
        long[] coordinates = new long[extents.length];
        long address = binding.baseElementOffset();
        for (int logical = 0; logical < count; logical++) {
            pairs[logical * 2] = address;
            pairs[logical * 2 + 1] = logical;
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                coordinates[axis]++;
                address = Math.addExact(address, strides[axis]);
                if (coordinates[axis] < extents[axis]) break;
                address = Math.subtractExact(address,
                        Math.multiplyExact(extents[axis], strides[axis]));
                coordinates[axis] = 0;
            }
        }
        return pairs;
    }

    private static CpuAccessPlan.Binding denseBinding(long[] extents, long count) {
        var roles = java.util.Collections.nCopies(extents.length,
                CpuAccessPlan.AxisRole.CONTIGUOUS);
        var plan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, extents.length, roles, extents.length);
        long[] strides = new long[extents.length];
        long stride = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, Math.max(1, extents[axis]));
        }
        return CpuAccessPlan.Binding.create(plan, extents, 0, strides, count, 0, count, count);
    }

    private static boolean admitted(DataType type) {
        return type == DataType.FLOAT64 || type == DataType.FLOAT32 || type == DataType.INT32
                || type == DataType.INT64 || type == DataType.BOOL;
    }

    /**
     * Operation-free immutable ceiling probe used only by focused boundary tests.
     *
     * @param topologyCount non-negative topology count
     * @param eligibleSourceCount non-negative eligible-source count
     * @param existingDecisionFacts non-negative existing fact count
     * @param additionalSourceRemains whether discovery found another source
     * @param additionalVariantRemains whether discovery found another representation variant
     */
    record RepresentationProbe(int topologyCount, int eligibleSourceCount,
            int existingDecisionFacts, boolean additionalSourceRemains,
            boolean additionalVariantRemains) {
        /** Validates non-negative probe counts. */
        RepresentationProbe {
            if (topologyCount < 0 || eligibleSourceCount < 0 || existingDecisionFacts < 0)
                throw new IllegalArgumentException("CPU representation probe counts are negative");
        }
    }

    /**
     * Reports whether a typed probe proves complete enumeration within every exact ceiling.
     *
     * @param probe non-null operation-free boundary probe
     * @return true only when all sources, variants, and combined facts are complete and bounded
     */
    static boolean completeWithinBudgets(RepresentationProbe probe) {
        Objects.requireNonNull(probe, "probe");
        if (probe.topologyCount() > 64 || probe.eligibleSourceCount() > MAX_ELIGIBLE_SOURCES
                || probe.additionalSourceRemains() || probe.additionalVariantRemains()) return false;
        long choices = probe.eligibleSourceCount()
                + (long) probe.eligibleSourceCount() * (probe.eligibleSourceCount() - 1) / 2;
        long perTopology = 1 + choices;
        long variants = Math.multiplyExact(probe.topologyCount(), perTopology);
        return choices <= MAX_COPY_CHOICES && perTopology <= MAX_VARIANTS_PER_TOPOLOGY
                && variants <= CpuRepresentationDecision.MAX_VARIANTS
                && Math.addExact(Math.addExact(probe.existingDecisionFacts(), variants), 1)
                    <= CpuRepresentationDecision.MAX_TOTAL_DECISION_FACTS;
    }

    private static final Comparator<WorkVariant> ORDER = Comparator
            .comparingLong((WorkVariant value) -> value.fact().comparisonScore())
            .thenComparingLong(value -> value.fact().copiedBytes())
            .thenComparingInt(value -> value.plans().size())
            .thenComparingInt(value -> value.fact().stableTopologyRank());
    private record Eligible(CpuMaterializationPlan plan) { }
    private record WorkVariant(int candidateIndex, List<CpuMaterializationPlan> plans,
            CpuRepresentationDecision.Variant fact,
            CpuFusionDecision.CandidateFacts topologyFacts, int stableRank) {
        private WorkVariant { plans = List.copyOf(plans); }
    }
}
