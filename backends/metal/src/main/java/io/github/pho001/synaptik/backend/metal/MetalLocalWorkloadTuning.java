package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Supported Metal-owned collaboration for cold local route tuning.
 *
 * <p>One instance is retained by one {@link MetalBackendIntegration}. It wraps the existing
 * version-thirty-four singleton-NEG candidate and decision implementation in opaque public values
 * with exact association. It performs no representative execution, measurement, cache input/output,
 * selection, or fallback-policy work. Every trial and selected preparation repeats authoritative
 * Metal analysis before returning a preparation that retains the integration's trace producer.</p>
 *
 * <p>The only tunable local workload is an exact Metal singleton NEG for which both the custom
 * kernel and MPSGraph routes are valid. A valid MPSGraph-only partition returns no handoff. All
 * compatibility is limited to the current live integration session.</p>
 */
public final class MetalLocalWorkloadTuning {
    private static final int COMPLETE_CANDIDATE_COUNT = 2;

    private final MetalDeviceContext context;
    private final MetalTraceProducer traceProducer;
    private final MetalNegRouteCandidateGenerator generator = new MetalNegRouteCandidateGenerator();
    private final MetalNegTuningCodec codec = new MetalNegTuningCodec();
    private final MetalNegPartitionPreparer preparer = new MetalNegPartitionPreparer();

    MetalLocalWorkloadTuning(MetalDeviceContext context, MetalTraceProducer traceProducer) {
        this.context = Objects.requireNonNull(context, "context");
        this.traceProducer = traceProducer;
    }

    /**
     * Produces the complete current two-route handoff for one exact Metal projection.
     *
     * @param projectedContext non-null exact stable projection of one non-empty Metal partition
     * @return the decision-empty two-route handoff, or empty when the valid partition has no
     *     tunable peer route
     * @throws NullPointerException if {@code projectedContext} is {@code null}
     * @throws IllegalArgumentException if the projection does not belong to this integration or
     *     is invalid for Metal analysis
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<BackendPartitionTuningHandoff<CandidateBatch, SelectedDecision>>
            candidateHandoff(PrepareContext<?> projectedContext) {
        requireLive();
        PrepareContext<?> source = requireProjection(projectedContext);
        PreparedFacts facts = analyze(source, Optional.empty(), null);
        if (!isCompleteLocalBatch(facts.batch)) return Optional.empty();

        Association association = new Association(this, source, source.partition(), facts.batch);
        var candidates = new ArrayList<Candidate>(COMPLETE_CANDIDATE_COUNT);
        for (int index = 0; index < COMPLETE_CANDIDATE_COUNT; index++) {
            candidates.add(new Candidate(association, index));
        }
        CandidateBatch batch = new CandidateBatch(association, List.copyOf(candidates));
        return Optional.of(new BackendPartitionTuningHandoff<>(
                association.partition, batch, Optional.empty()));
    }

    /**
     * Returns the complete stable safe-heuristic-first route candidates.
     *
     * @param batch non-null batch issued by this collaboration
     * @return retained immutable list containing exactly two opaque candidates
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public List<Candidate> candidates(CandidateBatch batch) {
        return requireBatch(batch).candidates;
    }

    /**
     * Returns defensive canonical version-thirty-four session compatibility.
     *
     * @param batch non-null batch issued by this collaboration
     * @return immutable session-scoped compatibility with defensive bytes
     * @throws NullPointerException if {@code batch} is {@code null}
     * @throws IllegalArgumentException if the batch belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Compatibility compatibility(CandidateBatch batch) {
        Association association = requireBatch(batch).association;
        return new Compatibility(
                MetalNegTuningBatch.COMPATIBILITY_SCHEMA_VERSION,
                ReuseScope.SESSION,
                codec.encodeCompatibility(association.internal.compatibility()));
    }

    /**
     * Returns defensive canonical version-thirty-four identity bytes for one route candidate.
     *
     * @param candidate non-null candidate issued by this collaboration
     * @return immutable opaque identity with defensive bytes
     * @throws NullPointerException if {@code candidate} is {@code null}
     * @throws IllegalArgumentException if the candidate belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public CandidateIdentity candidateIdentity(Candidate candidate) {
        Association association = requireCandidate(candidate);
        return new CandidateIdentity(codec.encodeCandidate(
                association.internal.candidates().get(candidate.index)));
    }

    /**
     * Constructs one exact-batch selected decision.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return immutable decision associated with exactly {@code batch}
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if either value has another owner or batch association
     * @throws IllegalStateException if the owning integration is closed
     */
    public SelectedDecision selectedDecision(CandidateBatch batch, Candidate candidate) {
        Association association = requireBatch(batch).association;
        requireCandidate(candidate);
        if (candidate.association != association) {
            throw new IllegalArgumentException("candidate does not belong to batch");
        }
        MetalNegTuningBatch.Candidate internal = association.internal.candidates()
                .get(candidate.index);
        return new SelectedDecision(association, new MetalNegTuningDecision(
                MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                association.internal.compatibility(), internal));
    }

    /**
     * Encodes one current decision using the bounded version-thirty-four Metal codec.
     *
     * @param decision non-null decision issued by this collaboration
     * @return fresh canonical caller-owned bytes
     * @throws NullPointerException if {@code decision} is {@code null}
     * @throws IllegalArgumentException if it belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public byte[] encodeDecision(SelectedDecision decision) {
        requireDecision(decision);
        return codec.encodeDecision(decision.internal);
    }

    /**
     * Decodes only a decision compatible with the exact supplied current batch.
     *
     * @param batch non-null current batch issued by this collaboration
     * @param encodedDecision non-null caller-owned bytes, snapshotted before parsing
     * @return matching exact-batch decision, or empty for malformed, corrupt, stale,
     *     wrong-session, or unknown-candidate input
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if {@code batch} belongs to another collaboration
     * @throws IllegalStateException if the owning integration is closed
     */
    public Optional<SelectedDecision> decodeCompatibleDecision(
            CandidateBatch batch, byte[] encodedDecision) {
        Association association = requireBatch(batch).association;
        byte[] input = Objects.requireNonNull(encodedDecision, "encodedDecision").clone();
        return codec.decodeDecision(input, association.internal)
                .map(value -> new SelectedDecision(association, value));
    }

    /**
     * Freshly prepares one exact route candidate without executing or measuring it.
     *
     * @param batch non-null originating batch
     * @param candidate non-null exact member of {@code batch}
     * @return Metal-owned partition preparation for exactly the candidate
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association or fresh compatibility fails
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> trialPreparation(
            CandidateBatch batch, Candidate candidate) {
        return selectedPreparation(batch, selectedDecision(batch, candidate));
    }

    /**
     * Freshly prepares an authenticated decision without heuristic fallback.
     *
     * @param batch non-null originating batch
     * @param decision non-null decision associated with exactly {@code batch}
     * @return Metal-owned partition preparation for exactly the selected route
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if association, eligibility, compatibility, or membership
     *     fails during fresh analysis
     * @throws IllegalStateException if the owning integration is closed
     */
    public PartitionPreparation<?, ?> selectedPreparation(
            CandidateBatch batch, SelectedDecision decision) {
        Association association = requireBatch(batch).association;
        requireDecision(decision);
        if (decision.association != association) {
            throw new IllegalArgumentException("decision does not belong to batch");
        }
        return selectedPreparation(association.context, association.partition,
                association.internal, decision.internal);
    }

    AuthenticatedPhaseOne authenticateCompletePlanPhaseOne(
            PrepareContext<?> projectedContext, SelectedDecision decision) {
        requireLive();
        PrepareContext<?> source = requireProjection(projectedContext);
        requireDecision(decision);
        if (decision.association.partition != source.partition()
                || !sameProjection(decision.association.context, source)) {
            throw new IllegalArgumentException(
                    "Metal Phase-1 decision does not belong to the exact projection");
        }
        PreparedFacts facts = analyze(source, Optional.empty(), null);
        if (!isCompleteLocalBatch(facts.batch)) {
            throw new IllegalArgumentException(
                    "Metal Phase-1 decision is no longer eligible");
        }
        MetalNegTuningBatch.Candidate selected = decision.internal.match(facts.batch)
                .orElseThrow(() -> new IllegalArgumentException(
                        "Metal Phase-1 decision is stale or incompatible"));
        return new AuthenticatedPhaseOne(
                facts.batch,
                new MetalNegTuningDecision(
                        MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                        facts.batch.compatibility(), selected));
    }

    byte[] completePlanCandidateIdentity(SelectedDecision decision) {
        requireDecision(decision);
        return codec.encodeCandidate(decision.internal.selectedCandidate());
    }

    PartitionPreparation<?, ?> completePlanSelectedPreparation(
            PrepareContext<?> projectedContext, SelectedDecision decision) {
        AuthenticatedPhaseOne authenticated = authenticateCompletePlanPhaseOne(
                projectedContext, decision);
        return selectedPreparation(
                requireProjection(projectedContext), projectedContext.partition(),
                authenticated.batch(), authenticated.decision());
    }

    private PartitionPreparation<?, ?> selectedPreparation(
            PrepareContext<?> projectedContext,
            PlannedPartition partition,
            MetalNegTuningBatch batch,
            MetalNegTuningDecision decision) {
        requireLive();
        var handoff = generator.presentHandoff(
                partition, batch, decision.selectedCandidate());
        PreparedFacts validated = analyze(
                projectedContext, Optional.of(handoff), null);
        if (validated.analysis.plan().route() != decision.selectedCandidate().route()) {
            throw new IllegalArgumentException("Metal tuning decision was not selected");
        }
        return new PartitionPreparation<>(
                new MetalNegAnalysisInputs(context, Optional.of(handoff), traceProducer),
                preparer,
                new MetalNegPartitionFinalizer(context));
    }

    private PreparedFacts analyze(
            PrepareContext<?> projectedContext,
            Optional<BackendPartitionTuningHandoff<
                    MetalNegTuningBatch, MetalNegTuningDecision>> handoff,
            MetalTraceProducer producer) {
        PrepareContext<MetalNegAnalysisInputs> contextWithInputs = new PrepareContext<>(
                projectedContext.partitionDag(),
                projectedContext.values(),
                projectedContext.memoryRequirements(),
                projectedContext.constants(),
                new MetalNegAnalysisInputs(context, handoff, producer));
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                preparer.analyze(contextWithInputs);
        MetalNegTuningBatch batch = generator.generate(
                contextWithInputs,
                analysis.plan(),
                MetalNegTuningBatch.Candidate.values().length);
        return new PreparedFacts(analysis, batch);
    }

    private PrepareContext<?> requireProjection(PrepareContext<?> projectedContext) {
        Objects.requireNonNull(projectedContext, "projectedContext");
        if (!(projectedContext.backendInputs() instanceof MetalNegAnalysisInputs inputs)
                || inputs.context() != context) {
            throw new IllegalArgumentException(
                    "projection belongs to another Metal integration");
        }
        return projectedContext;
    }

    void requireLive() {
        context.access(() -> { });
    }

    private CandidateBatch requireBatch(CandidateBatch batch) {
        requireLive();
        Objects.requireNonNull(batch, "batch");
        if (batch.association.owner != this) {
            throw new IllegalArgumentException("batch belongs to another Metal integration");
        }
        return batch;
    }

    private Association requireCandidate(Candidate candidate) {
        requireLive();
        Objects.requireNonNull(candidate, "candidate");
        if (candidate.association.owner != this) {
            throw new IllegalArgumentException("candidate belongs to another Metal integration");
        }
        return candidate.association;
    }

    private void requireDecision(SelectedDecision decision) {
        requireLive();
        Objects.requireNonNull(decision, "decision");
        if (decision.association.owner != this) {
            throw new IllegalArgumentException("decision belongs to another Metal integration");
        }
    }

    private static boolean isCompleteLocalBatch(MetalNegTuningBatch batch) {
        return batch.candidates().equals(List.of(
                MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                MetalNegTuningBatch.Candidate.MPSGRAPH));
    }

    private static boolean sameProjection(PrepareContext<?> left, PrepareContext<?> right) {
        return left.partition() == right.partition()
                && sameReferences(left.partitionDag().nodes(), right.partitionDag().nodes())
                && sameReferences(left.values(), right.values())
                && sameReferences(left.memoryRequirements(), right.memoryRequirements())
                && left.constants().equals(right.constants());
    }

    private static boolean sameReferences(List<?> left, List<?> right) {
        if (left.size() != right.size()) return false;
        for (int index = 0; index < left.size(); index++) {
            if (left.get(index) != right.get(index)) return false;
        }
        return true;
    }

    /** Scope within which Metal compatibility and decisions may be reused. */
    public enum ReuseScope {
        /** Reuse is limited to the exact live {@link MetalBackendIntegration}. */
        SESSION
    }

    /** Opaque exact-association local candidate batch. */
    public static final class CandidateBatch implements BackendTuningCandidateBatch {
        private final Association association;
        private final List<Candidate> candidates;

        private CandidateBatch(Association association, List<Candidate> candidates) {
            this.association = association;
            this.candidates = candidates;
        }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof CandidateBatch value
                    && association == value.association;
        }

        @Override
        public int hashCode() { return System.identityHashCode(association); }

        @Override
        public String toString() { return "MetalLocalCandidateBatch[opaque]"; }
    }

    /** Opaque exact-association selected local decision. */
    public static final class SelectedDecision implements BackendTuningDecision {
        private final Association association;
        private final MetalNegTuningDecision internal;

        private SelectedDecision(Association association, MetalNegTuningDecision internal) {
            this.association = association;
            this.internal = internal;
        }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof SelectedDecision value
                    && association == value.association && internal.equals(value.internal);
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(association) + internal.hashCode();
        }

        @Override
        public String toString() { return "MetalLocalSelectedDecision[opaque]"; }
    }

    /** Opaque exact-association complete local route candidate. */
    public static final class Candidate {
        private final Association association;
        private final int index;

        private Candidate(Association association, int index) {
            this.association = association;
            this.index = index;
        }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof Candidate value
                    && association == value.association && index == value.index;
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(association) + index;
        }

        @Override
        public String toString() { return "MetalLocalCandidate[opaque]"; }
    }

    /** Immutable canonical local compatibility bytes and reuse scope. */
    public static final class Compatibility {
        private final int schemaVersion;
        private final ReuseScope reuseScope;
        private final byte[] encoded;

        private Compatibility(int schemaVersion, ReuseScope reuseScope, byte[] encoded) {
            this.schemaVersion = schemaVersion;
            this.reuseScope = reuseScope;
            this.encoded = encoded.clone();
        }

        /**
         * Returns the compatibility schema.
         * @return exact positive Metal compatibility schema version
         */
        public int schemaVersion() { return schemaVersion; }

        /**
         * Copies the compatibility payload.
         * @return fresh caller-owned canonical compatibility bytes
         */
        public byte[] bytes() { return encoded.clone(); }

        /**
         * Returns the declared reuse scope.
         * @return exact session-only reuse scope
         */
        public ReuseScope reuseScope() { return reuseScope; }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof Compatibility value
                    && schemaVersion == value.schemaVersion
                    && reuseScope == value.reuseScope
                    && Arrays.equals(encoded, value.encoded);
        }

        @Override
        public int hashCode() {
            return 31 * (31 * schemaVersion + reuseScope.hashCode())
                    + Arrays.hashCode(encoded);
        }

        @Override
        public String toString() {
            return "MetalLocalCompatibility[schema=" + schemaVersion
                    + ", scope=" + reuseScope + ", bytes=" + encoded.length + ']';
        }
    }

    /** Immutable canonical local candidate identity bytes. */
    public static final class CandidateIdentity {
        private final byte[] encoded;

        private CandidateIdentity(byte[] encoded) { this.encoded = encoded.clone(); }

        /**
         * Copies the candidate identity.
         * @return fresh caller-owned canonical candidate identity bytes
         */
        public byte[] bytes() { return encoded.clone(); }

        @Override
        public boolean equals(Object other) {
            return this == other || other instanceof CandidateIdentity value
                    && Arrays.equals(encoded, value.encoded);
        }

        @Override
        public int hashCode() { return Arrays.hashCode(encoded); }

        @Override
        public String toString() {
            return "MetalLocalCandidateIdentity[bytes=" + encoded.length + ']';
        }
    }

    private static final class Association {
        private final MetalLocalWorkloadTuning owner;
        private final PrepareContext<?> context;
        private final PlannedPartition partition;
        private final MetalNegTuningBatch internal;

        private Association(
                MetalLocalWorkloadTuning owner,
                PrepareContext<?> context,
                PlannedPartition partition,
                MetalNegTuningBatch internal) {
            this.owner = owner;
            this.context = context;
            this.partition = partition;
            this.internal = internal;
        }
    }

    private record PreparedFacts(
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis,
            MetalNegTuningBatch batch) { }

    record AuthenticatedPhaseOne(
            MetalNegTuningBatch batch, MetalNegTuningDecision decision) { }
}
