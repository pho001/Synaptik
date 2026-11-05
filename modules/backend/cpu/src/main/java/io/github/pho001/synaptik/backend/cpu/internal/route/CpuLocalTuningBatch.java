package io.github.pho001.synaptik.backend.cpu.internal.route;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasQualification;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasRoutePlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Immutable CPU-owned finite realization batch for a resolved common CPU partition. Portable
 * numeric MATMUL needs no native provider; qualified OpenBLAS mask/thread plans and oneDNN
 * primitive/Graph assignments are additional typed alternatives. The ordinary production portable
 * realization is first, followed by legal fixed portable forms/ranges, OpenBLAS mask/thread order,
 * and canonical-topology oneDNN assignment order. The local handoff requires one intrinsically
 * singleton bare-MATMUL topology/representation; fusion and nonlocal complete choices remain
 * exclusively Phase 2. Local discovery admits one MATMUL and at most two absorbed permutations, and
 * 8192 aggregate alternatives, rejecting excess before realization expansion rather than tuning
 * a prefix. Cold plans own no executable, native handle, physical workspace, or Runtime state.
 *
 * <p>Its method-free {@link BackendTuningCandidateBatch} role permits shared Prepare code to
 * transport the exact batch opaquely without changing or exposing its CPU-owned schema.</p>
 *
 * @param schemaVersion exact candidate schema, currently {@value #SCHEMA_VERSION}
 * @param workload canonical immutable workload and compatibility signature
 * @param candidates immutable non-empty ordered candidates, beginning with portable
 */
public record CpuLocalTuningBatch(int schemaVersion, WorkloadSignature workload,
        List<Candidate> candidates) implements BackendTuningCandidateBatch {
    /** Current meaning of the candidate and decision values in this package. */
    public static final int SCHEMA_VERSION = 4;
    /** All-or-nothing aggregate local candidate ceiling, checked before realization expansion. */
    public static final int MAXIMUM_CANDIDATES = 8_192;
    /** Current CPU local-realization route-policy meaning. */
    public static final int ROUTE_POLICY_VERSION = 3;
    /** Current checked whole-plan cost-policy meaning. */
    public static final int COST_POLICY_VERSION = 1;

    /** Candidate route kind. */
    public enum RouteKind { /** Existing generated portable route. */ PORTABLE,
        /** Qualified OpenBLAS SGEMM/DGEMM route. */ OPENBLAS, /** Qualified oneDNN primitive or Graph assignment. */ ONEDNN }
    /** Typed semantic projection discriminator. */
    public enum OperationKind { /** Bare rank-two matrix multiplication. */ MATMUL }
    /** Additional attributes are carried by the typed common-IR semantic fingerprint. */
    public enum OperationAttributes { /** MATMUL has no configurable operation attributes. */ NONE }
    /** Fixed production determinism request. */
    public enum DeterminismMode { /** Current default execution determinism contract. */ DEFAULT }

    /**
     * Caller-supplied immutable CPU identity used only for compatibility. Strings are opaque
     * canonical values supplied by composition; CPU analysis never discovers or normalizes them.
     *
     * @param schemaVersion hardware identity schema, currently one
     * @param architecture non-blank canonical processor architecture
     * @param vendor non-blank canonical processor vendor
     * @param model non-blank canonical processor model or target identifier
     * @param features sorted, unique, non-blank canonical feature tokens
     */
    public record HardwareIdentity(int schemaVersion, String architecture, String vendor,
            String model, List<String> features) {
        /** Compatibility default for callers that have not yet supplied host-specific facts. */
        public static final HardwareIdentity UNSPECIFIED = new HardwareIdentity(1,
                "unspecified", "unspecified", "unspecified", List.of());
        /**
         * Validates and snapshots the caller-owned canonical identity.
         *
         * @throws NullPointerException if a text value, {@code features}, or a feature is
         *     {@code null}
         * @throws IllegalArgumentException if the schema is unsupported, text is blank, or the
         *     feature list is not sorted and unique
         */
        public HardwareIdentity {
            Objects.requireNonNull(architecture, "architecture");
            Objects.requireNonNull(vendor, "vendor");
            Objects.requireNonNull(model, "model");
            features = List.copyOf(features);
            if (schemaVersion != 1 || architecture.isBlank() || vendor.isBlank()
                    || model.isBlank() || features.stream().anyMatch(String::isBlank)
                    || features.stream().distinct().count() != features.size()
                    || !features.equals(features.stream().sorted().toList())) {
                throw new IllegalArgumentException("invalid canonical CPU hardware identity");
            }
        }
    }

    /**
     * Expected repeated-use cohort supplied before analysis. It is compatibility metadata, not a
     * measured occurrence count or an objective.
     *
     * @param schemaVersion cohort schema, currently one
     * @param cohortId non-blank caller-defined canonical cohort identifier
     * @param expectedRunCount positive expected executions used by existing cost facts
     */
    public record WorkloadCohort(int schemaVersion, String cohortId, long expectedRunCount) {
        /** Ordinary single-run compatibility cohort. */
        public static final WorkloadCohort DEFAULT = new WorkloadCohort(1, "ordinary", 1);
        /**
         * Validates one immutable cohort.
         *
         * @throws NullPointerException if {@code cohortId} is {@code null}
         * @throws IllegalArgumentException if the schema is unsupported, the identifier is blank,
         *     or the expected run count is not positive
         */
        public WorkloadCohort {
            Objects.requireNonNull(cohortId, "cohortId");
            if (schemaVersion != 1 || cohortId.isBlank() || expectedRunCount <= 0) {
                throw new IllegalArgumentException("invalid workload cohort");
            }
        }
    }

    /**
     * Exact resolved boundary compatibility without physical storage.
     *
     * @param dataType exact boundary type
     * @param shape exact static logical shape
     * @param layout exact resolved logical layout
     * @param carrier exact cold carrier form
     * @param storage exact expected native provenance/alignment fact
     * @param access exact common-lowering access binding
     */
    public record BoundarySignature(DataType dataType, Shape shape, LayoutDescriptor layout,
            CpuKernelSpecialization.CarrierAccess carrier,
            CpuPartitionAnalysisInputs.BoundaryStorageFact storage,
            CpuAccessPlan.Binding access) {
        /**
         * Validates the complete immutable boundary projection. The referenced component values
         * are immutable and are retained without physical storage ownership.
         *
         * @throws NullPointerException if any component is {@code null}
         */
        public BoundarySignature {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(shape, "shape");
            Objects.requireNonNull(layout, "layout");
            Objects.requireNonNull(carrier, "carrier");
            Objects.requireNonNull(storage, "storage");
            Objects.requireNonNull(access, "access");
        }
    }

    /**
     * Qualification reuse scope. Persistent scope retains only stable target/binary facts;
     * session-only scope retains the exact immutable credential needed to prevent cross-session
     * reuse. Neither form retains a provider or native resource.
     *
     * @param scope exact qualification lifetime
     * @param targetFingerprint exact qualified target
     * @param persistentIdentity stable identity present exactly for persistent-binary scope
     * @param sessionCredential exact credential present exactly for session-only scope
     */
    public record QualificationScope(CpuOpenBlasQualification.Scope scope,
            CpuOpenBlasQualification.TargetFingerprint targetFingerprint,
            Optional<CpuOpenBlasQualification.PersistentIdentity> persistentIdentity,
            Optional<CpuOpenBlasQualification> sessionCredential) {
        /**
         * Projects one successful qualification without leaking a session token into persistent
         * compatibility.
         * @param qualification non-null successful credential
         * @throws NullPointerException if {@code qualification} is {@code null}
         */
        public QualificationScope(CpuOpenBlasQualification qualification) {
            this(Objects.requireNonNull(qualification, "qualification").scope(),
                    qualification.targetFingerprint(), qualification.persistentIdentity(),
                    qualification.scope() == CpuOpenBlasQualification.Scope.SESSION_ONLY
                            ? Optional.of(qualification) : Optional.empty());
        }
        /**
         * Validates exact scope-dependent presence. Optional containers are retained after null
         * validation; their immutable contents carry no native-resource ownership.
         *
         * @throws NullPointerException if a required component or optional container is
         *     {@code null}
         * @throws IllegalArgumentException if persistent or session evidence presence disagrees
         *     with {@code scope}
         */
        public QualificationScope {
            Objects.requireNonNull(scope, "scope");
            Objects.requireNonNull(targetFingerprint, "targetFingerprint");
            persistentIdentity = Objects.requireNonNull(persistentIdentity,
                    "persistentIdentity");
            sessionCredential = Objects.requireNonNull(sessionCredential, "sessionCredential");
            if ((scope == CpuOpenBlasQualification.Scope.PERSISTENT_BINARY)
                        != persistentIdentity.isPresent()
                    || (scope == CpuOpenBlasQualification.Scope.SESSION_ONLY)
                        != sessionCredential.isPresent()) {
                throw new IllegalArgumentException("qualification reuse scope disagrees");
            }
        }
    }

    /**
     * Canonical CPU-owned workload signature. It deliberately retains every policy input that can
     * change candidate meaning, even when two current values happen to produce equal costs.
     *
     * @param operationKind exact bare MATMUL semantics discriminator
     * @param operationAttributes no additional attributes outside the canonical semantic projection
     * @param boundaries exact typed materialized boundaries in declaration order
     * @param determinismMode default determinism contract
     * @param qualification optional qualified OpenBLAS target, binary/session, ABI and numerical scope;
     *     absent for portable-only or oneDNN-only domains
     * @param hardware caller-supplied canonical CPU hardware identity
     * @param cpuConcurrencyCapacity positive shared CPU capacity snapshot
     * @param portableExecution exact configured portable execution inputs
     * @param portableStrategy exact selected portable strategy
     * @param portableRangeCount positive selected portable range count
     * @param portableVectorSpeciesBits positive vector species size or zero for scalar
     * @param openBlasThreadCounts all positive configured counts in caller order; candidate
     *     generation separately filters counts above the shared capacity
     * @param cohort exact expected-use cohort
     * @param materializationPolicy exact existing representation policy
     * @param portableCosts exact portable cost coefficients
     * @param representationCosts exact representation cost coefficients
     * @param openBlasThreadCandidates all exact configured thread/cost values in caller order
     * @param minimumNetBenefitCostUnits exact optional non-negative heuristic threshold
     * @param minimumBenefitBasisPoints exact optional heuristic threshold from zero through 10,000
     * @param generatedArtifactSchema exact selected portable generated-artifact schema
     * @param routePolicyVersion exact CPU local-realization route-policy version
     * @param costPolicyVersion exact checked cost-policy version
     * @param semanticsFingerprint graph-identity-free common IR, unit membership, dependencies and
     *     ordered boundary associations; operation attributes remain bound by typed IR identity
     */
    public record WorkloadSignature(OperationKind operationKind,
            OperationAttributes operationAttributes,
            List<BoundarySignature> boundaries, DeterminismMode determinismMode,
            Optional<QualificationScope> qualification,
            HardwareIdentity hardware, int cpuConcurrencyCapacity,
            CpuPartitionAnalysisInputs.PortableExecutionConfig portableExecution,
            CpuPartitionPreparationPlan.ExecutionStrategy portableStrategy,
            int portableRangeCount, int portableVectorSpeciesBits,
            List<Integer> openBlasThreadCounts, WorkloadCohort cohort,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy,
            CpuPartitionAnalysisInputs.CostTerms portableCosts,
            CpuPartitionAnalysisInputs.RepresentationCostTerms representationCosts,
            List<CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate>
                    openBlasThreadCandidates,
            OptionalLong minimumNetBenefitCostUnits, OptionalInt minimumBenefitBasisPoints,
            int generatedArtifactSchema, int routePolicyVersion, int costPolicyVersion,
            String semanticsFingerprint) {
        /**
         * Validates and snapshots the complete compatibility signature. All lists become
         * unmodifiable snapshots; other components are immutable values retained by reference.
         *
         * @throws NullPointerException if a required component, list, or list entry is
         *     {@code null}
         * @throws IllegalArgumentException if type, boundary, concurrency, thread, cohort, cost,
         *     generated-artifact, or policy-version facts disagree
         */
        public WorkloadSignature {
            Objects.requireNonNull(operationKind, "operationKind");
            Objects.requireNonNull(semanticsFingerprint, "semanticsFingerprint");
            Objects.requireNonNull(operationAttributes, "operationAttributes");
            boundaries = List.copyOf(boundaries);
            Objects.requireNonNull(determinismMode, "determinismMode");
            Objects.requireNonNull(qualification, "qualification");
            Objects.requireNonNull(hardware, "hardware");
            Objects.requireNonNull(portableExecution, "portableExecution");
            Objects.requireNonNull(portableStrategy, "portableStrategy");
            openBlasThreadCounts = List.copyOf(openBlasThreadCounts);
            Objects.requireNonNull(cohort, "cohort");
            Objects.requireNonNull(materializationPolicy, "materializationPolicy");
            Objects.requireNonNull(portableCosts, "portableCosts");
            Objects.requireNonNull(minimumNetBenefitCostUnits, "minimumNetBenefitCostUnits");
            Objects.requireNonNull(minimumBenefitBasisPoints, "minimumBenefitBasisPoints");
            Objects.requireNonNull(representationCosts, "representationCosts");
            openBlasThreadCandidates = List.copyOf(openBlasThreadCandidates);

            if (semanticsFingerprint.isBlank() || boundaries.isEmpty()
                    || operationKind == OperationKind.MATMUL && boundaries.size() != 3
                    || cpuConcurrencyCapacity <= 0 || portableRangeCount <= 0
                    || portableVectorSpeciesBits < 0
                    || openBlasThreadCounts.stream().anyMatch(value -> value <= 0)
                    || openBlasThreadCounts.stream().distinct().count()
                            != openBlasThreadCounts.size()
                    || !openBlasThreadCounts.equals(openBlasThreadCandidates.stream()
                            .map(CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate
                                    ::threadCount).toList())
                    || cohort.expectedRunCount() != materializationPolicy.expectedRunCount()
                    || minimumNetBenefitCostUnits.isPresent() && minimumNetBenefitCostUnits.getAsLong() < 0
                    || minimumBenefitBasisPoints.isPresent() && (minimumBenefitBasisPoints.getAsInt() < 0
                        || minimumBenefitBasisPoints.getAsInt() > 10_000)
                    || generatedArtifactSchema <= 0
                    || routePolicyVersion != ROUTE_POLICY_VERSION
                    || costPolicyVersion != COST_POLICY_VERSION) {
                throw new IllegalArgumentException("CPU local workload-signature facts disagree");
            }
        }

        /**
         * Projects a reusable signature only when qualification provides a stable binary identity.
         * @return immutable persistent projection, or empty for session-only qualification
         */
        public Optional<PersistentWorkloadSignature> persistentProjection() {
            return qualification.flatMap(QualificationScope::persistentIdentity).map(identity ->
                    new PersistentWorkloadSignature(this, identity));
        }
    }

    /**
     * Persistent projection of a workload signature. The source must carry persistent
     * qualification; the embedded source is normalized by equality to all fields except the exact
     * session credential.
     *
     * @param workload source workload
     * @param qualification persistent provider/binary identity
     */
    public record PersistentWorkloadSignature(WorkloadSignature workload,
            CpuOpenBlasQualification.PersistentIdentity qualification) {
        /**
         * Validates that the source projects the supplied identity. Both immutable values are
         * retained without resource ownership.
         *
         * @throws NullPointerException if either component is {@code null}
         * @throws IllegalArgumentException if the workload does not project the supplied
         *     persistent qualification
         * @throws java.util.NoSuchElementException if the workload is session-only
         */
        public PersistentWorkloadSignature {
            Objects.requireNonNull(workload, "workload");
            Objects.requireNonNull(qualification, "qualification");
            if (!workload.qualification().flatMap(QualificationScope::persistentIdentity).orElseThrow()
                    .equals(qualification)) {
                throw new IllegalArgumentException("persistent qualification disagrees");
            }
        }
    }

    /** Deterministic candidate identity independent of live objects and native addresses. */
    public sealed interface CandidateIdentity permits PortableIdentity, OpenBlasIdentity, OneDnnIdentity { }

    /**
     * Graph-identity-free exact resource shape.
     * @param byteSize exact non-negative byte count
     * @param byteAlignment exact positive power-of-two alignment
     */
    public record ResourceIdentity(long byteSize, long byteAlignment) {
        /**
         * Validates an immutable resource shape.
         *
         * @throws IllegalArgumentException if the byte count is negative or alignment is not a
         *     positive power of two
         */
        public ResourceIdentity {
            if (byteSize < 0 || byteAlignment <= 0
                    || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException("invalid candidate resource identity");
            }
        }
    }

    /**
     * Exact portable candidate identity.
     * @param strategy selected portable compute/orchestration strategy
     * @param rangeCount selected range count
     * @param minimumElementsPerWorker positive chunk threshold
     * @param vectorSpeciesBits positive species size or zero for scalar
     * @param generatedArtifactSchema exact generated-artifact schema
     * @param structuralKey non-blank structural generated-kernel key
     * @param resources ordered graph-identity-free portable buffer/resource shapes
     * @param topology optional exact retained canonical topology; absent only on ordinary route decoration
     */
    public record PortableIdentity(CpuPartitionPreparationPlan.ExecutionStrategy strategy,
            int rangeCount, long minimumElementsPerWorker, int vectorSpeciesBits,
            int generatedArtifactSchema, String structuralKey,
            List<ResourceIdentity> resources,
            Optional<CpuRepresentationDecision.VariantIdentity> topology) implements CandidateIdentity {
        /**
         * Validates complete portable identity facts and snapshots {@code resources}.
         *
         * @throws NullPointerException if a required component, resource list, or entry is
         *     {@code null}
         * @throws IllegalArgumentException if counts, thresholds, species size, schema,
         *     structural key, or resource presence is invalid
         */
        public PortableIdentity {
            Objects.requireNonNull(strategy, "strategy");
            Objects.requireNonNull(structuralKey, "structuralKey");
            resources = List.copyOf(resources);
            Objects.requireNonNull(topology, "topology");
            if (rangeCount <= 0 || minimumElementsPerWorker <= 0 || vectorSpeciesBits < 0
                    || generatedArtifactSchema <= 0 || structuralKey.isBlank()
                    || resources.isEmpty()) {
                throw new IllegalArgumentException("invalid portable candidate identity");
            }
        }
    }

    /**
     * Exact OpenBLAS candidate identity.
     * @param representation exact copy mask
     * @param threadCount positive provider thread count
     * @param permitDemand identical shared-budget demand
     * @param configuredThreadOrder stable source-list position
     * @param workspaceRequirementIds exact ordered route-local workspace identities
     * @param workspaceBytes exact aggregate workspace bytes
     * @param inputCopiedElements exact copied input elements
     * @param outputCopiedElements exact copied output elements
     */
    public record OpenBlasIdentity(CpuOpenBlasRoutePlan.Representation representation,
            int threadCount, int permitDemand, int configuredThreadOrder,
            List<Long> workspaceRequirementIds, long workspaceBytes,
            long inputCopiedElements, long outputCopiedElements) implements CandidateIdentity {
        /**
         * Validates complete route identity facts and snapshots the workspace-identity list.
         *
         * @throws NullPointerException if {@code representation}, the workspace list, or an entry
         *     is {@code null}
         * @throws IllegalArgumentException if thread, permit, ordering, workspace, or copied-
         *     element facts disagree
         */
        public OpenBlasIdentity {
            Objects.requireNonNull(representation, "representation");
            workspaceRequirementIds = List.copyOf(workspaceRequirementIds);
            if (threadCount <= 0 || permitDemand != threadCount || configuredThreadOrder < 0
                    || workspaceRequirementIds.size() != representation.copyCount()
                    || workspaceBytes < 0 || inputCopiedElements < 0
                    || outputCopiedElements < 0) {
                throw new IllegalArgumentException("invalid OpenBLAS candidate identity");
            }
        }
    }

    /**
     * Exact qualified oneDNN assignment, including its API choices and declared resources.
     * @param fingerprint versioned descriptor, implementation and session fingerprint
     * @param resources exact ordered buffer and workspace geometry
     * @param topology exact retained canonical topology/representation identity
     */
    public record OneDnnIdentity(String fingerprint, List<ResourceIdentity> resources,
            CpuRepresentationDecision.VariantIdentity topology)
            implements CandidateIdentity {
        /** Validates immutable native assignment facts without acquiring native resources. */
        public OneDnnIdentity {
            Objects.requireNonNull(fingerprint, "fingerprint");
            resources = List.copyOf(resources);
            Objects.requireNonNull(topology, "topology");
            if (fingerprint.isBlank()) throw new IllegalArgumentException("blank native identity");
        }
    }

    /**
     * Canonicalizes common generated semantics and logical boundary associations without graph IDs.
     * @param plan immutable complete common plan
     * @return stable ordered per-unit specialization, membership, dependency and boundary key
     */
    public static String portableStructuralKey(CpuPartitionPreparationPlan plan) {
        return plan.units().stream().map(unit -> unit.portablePlan().specialization().structuralKey()
                + ":" + unit.memberNodeOrdinals() + ":" + unit.dependencies()
                + ":" + unit.selectedRangeCount() + ":" + unit.minimumElementsPerWorker()
                + ":" + unit.vectorSpeciesBitSize()
                + ":" + unit.boundaryValues().stream().map(plan.boundaryValues()::indexOf).toList())
                .collect(java.util.stream.Collectors.joining("|"));
    }

    /**
     * Projects exact ordered declarations without physical storage or executable ownership.
     * @param plan immutable complete realized plan
     * @return ordered buffer geometry followed by distinct common and native workspace geometry
     */
    public static List<ResourceIdentity> resourceIdentities(CpuPartitionPreparationPlan plan) {
        var resources = new java.util.ArrayList<ResourceIdentity>();
        for (var buffer : plan.bufferDeclarations())
            resources.add(new ResourceIdentity(buffer.byteSize(), buffer.byteAlignment()));
        var workspaces = new java.util.LinkedHashMap<Long,
                io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement.Workspace>();
        plan.workspaceDeclaration().ifPresent(workspace -> workspaces.put(workspace.requirementId(), workspace));
        for (var unit : plan.units()) unit.runtimeFacts().workspaceDeclaration()
                .ifPresent(workspace -> workspaces.putIfAbsent(workspace.requirementId(), workspace));
        plan.oneDnnPlan().ifPresent(nativePlan -> nativePlan.workspaceRequirements().forEach(
                workspace -> workspaces.putIfAbsent(workspace.requirementId(), workspace)));
        for (var workspace : workspaces.values())
            resources.add(new ResourceIdentity(workspace.byteSize(), workspace.byteAlignment()));
        return List.copyOf(resources);
    }
    /**
     * One exact cold realization. Portable and oneDNN retain immutable preparation plans;
     * OpenBLAS retains its route-local plan. No executable or native handle is retained.
     *
     * @param route route kind
     * @param identity deterministic typed identity
     * @param openBlasPlan plan present exactly for an OpenBLAS candidate
     * @param realization complete cold plan present exactly for portable and oneDNN
     */
    public record Candidate(RouteKind route, CandidateIdentity identity,
            Optional<CpuOpenBlasRoutePlan> openBlasPlan,
            Optional<CpuPartitionPreparationPlan> realization) {
        /**
         * Validates route, identity, and executable-plan agreement. The immutable identity and
         * optional plan are retained without acquiring executable or native-resource ownership.
         *
         * @throws NullPointerException if a required component or optional container is
         *     {@code null}
         * @throws IllegalArgumentException if route, identity, plan, or resource facts disagree
         */
        public Candidate {
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(identity, "identity");
            openBlasPlan = Objects.requireNonNull(openBlasPlan, "openBlasPlan");
            realization = Objects.requireNonNull(realization, "realization");
            if ((route == RouteKind.OPENBLAS) != openBlasPlan.isPresent()
                    || (route == RouteKind.PORTABLE) != (identity instanceof PortableIdentity)
                    || (route == RouteKind.OPENBLAS) != (identity instanceof OpenBlasIdentity)
                    || (route == RouteKind.ONEDNN) != (identity instanceof OneDnnIdentity)
                    || (route != RouteKind.OPENBLAS) != realization.isPresent()) {
                throw new IllegalArgumentException("candidate route facts disagree");
            }
            if (realization.isPresent()) {
                var plan = realization.orElseThrow();
                if (plan.localTuningBatch().isPresent()
                        || plan.route() != switch (route) {
                            case PORTABLE -> CpuPartitionPreparationPlan.Route.PORTABLE;
                            case OPENBLAS -> CpuPartitionPreparationPlan.Route.OPENBLAS;
                            case ONEDNN -> CpuPartitionPreparationPlan.Route.ONEDNN;
                        })
                    throw new IllegalArgumentException("cold realization route or tuning state disagrees");
                if (identity instanceof PortableIdentity portable
                        && (!portable.strategy().equals(plan.executionStrategy())
                            || portable.rangeCount() != plan.selectedRangeCount()
                            || portable.minimumElementsPerWorker() != plan.minimumElementsPerWorker()
                            || portable.vectorSpeciesBits() != plan.vectorSpeciesBitSize()
                            || portable.generatedArtifactSchema() != plan.units().getFirst()
                                .portablePlan().specialization().classIdentitySchema()
                            || !portable.structuralKey().equals(portableStructuralKey(plan))
                            || !portable.resources().equals(resourceIdentities(plan))))
                    throw new IllegalArgumentException("portable realization identity disagrees");
                if (identity instanceof OneDnnIdentity nativeIdentity
                        && !nativeIdentity.resources().equals(resourceIdentities(plan)))
                    throw new IllegalArgumentException("native realization resources disagree");
            }
            if (route == RouteKind.OPENBLAS) {
                CpuOpenBlasRoutePlan plan = openBlasPlan.orElseThrow();
                OpenBlasIdentity nativeIdentity = (OpenBlasIdentity) identity;
                if (nativeIdentity.representation() != plan.representation()
                        || nativeIdentity.threadCount() != plan.threadCount()
                        || nativeIdentity.permitDemand() != plan.permitDemand()
                        || nativeIdentity.configuredThreadOrder() != plan.candidateOrder()
                        || !nativeIdentity.workspaceRequirementIds().equals(
                            plan.workspaceRequirements().stream().map(
                                io.github.pho001.synaptik.prepare.analysis
                                    .PreparationResourceRequirement.Workspace::requirementId)
                                .toList())
                        || nativeIdentity.workspaceBytes() != plan.workspaceBytes()
                        || nativeIdentity.inputCopiedElements() != plan.inputCopiedElements()
                        || nativeIdentity.outputCopiedElements()
                            != plan.outputCopiedElements()) {
                    throw new IllegalArgumentException("candidate identity and plan disagree");
                }
            }
        }
    }

    /**
     * Validates ordering, uniqueness, and the mandatory portable fallback, then snapshots the
     * candidate list. Candidate plans remain immutable cold descriptions and transfer no resource
     * ownership.
     *
     * @throws NullPointerException if {@code workload}, {@code candidates}, or an entry is
     *     {@code null}
     * @throws IllegalArgumentException if the schema, portable fallback, uniqueness, ordering, or
     *     candidate/workload agreement is invalid
     */
    public CpuLocalTuningBatch {
        Objects.requireNonNull(workload, "workload");
        if (candidates.size() > MAXIMUM_CANDIDATES) {
            throw new IllegalArgumentException("CPU local candidate domain exceeds 8192 alternatives");
        }
        candidates = List.copyOf(candidates);
        if (schemaVersion != SCHEMA_VERSION || candidates.isEmpty()
                || candidates.getFirst().route() != RouteKind.PORTABLE
                || candidates.stream().map(Candidate::identity).distinct().count()
                        != candidates.size()) {
            throw new IllegalArgumentException("invalid CPU local tuning candidate batch");
        }
        int representationOrder = -1;
        int threadOrder = -1;
        for (int index = 1; index < candidates.size(); index++) {
            Candidate candidate = candidates.get(index);
            if (!(candidate.identity() instanceof OpenBlasIdentity identity)) continue;
            int currentRepresentation = identity.representation().ordinal();
            if (currentRepresentation < representationOrder
                    || currentRepresentation == representationOrder
                        && identity.configuredThreadOrder() <= threadOrder) {
                throw new IllegalArgumentException("OpenBLAS candidate order is not canonical");
            }
            representationOrder = currentRepresentation;
            threadOrder = identity.configuredThreadOrder();
            CpuOpenBlasRoutePlan plan = candidate.openBlasPlan().orElseThrow();
            if (plan.dataType() != workload.boundaries().getLast().dataType()
                    || identity.configuredThreadOrder() >= workload.openBlasThreadCandidates().size()
                    || !plan.threadCandidate().equals(workload.openBlasThreadCandidates()
                        .get(identity.configuredThreadOrder()))
                    || plan.threadCount() > workload.cpuConcurrencyCapacity()
                    || plan.workspaceBytes() > workload.materializationPolicy().maximumAdditionalBytes()
                    || plan.analysisCapacity() != workload.cpuConcurrencyCapacity()
                    || plan.expectedRunCount() != workload.cohort().expectedRunCount()
                    || !Optional.of(new QualificationScope(plan.qualification().orElseThrow()))
                            .equals(workload.qualification())) {
                throw new IllegalArgumentException("candidate and workload facts disagree");
            }
            for (int boundary = 0; boundary < 3; boundary++) {
                boolean copied = boundary == 0 ? plan.representation().copiesLeft()
                        : boundary == 1 ? plan.representation().copiesRight() : plan.representation().copiesOutput();
                var fact = workload.boundaries().get(boundary);
                if (!copied && (!fact.storage().nativeSegment()
                        || fact.storage().byteAlignment() < plan.dataType().byteWidth()
                        || fact.carrier() != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT))
                    throw new IllegalArgumentException("uncopied OpenBLAS boundary is not qualified native storage");
            }
        }
    }

    /**
     * Finds an exact candidate identity without interpreting it.
     * @param identity non-null candidate identity to find
     * @return non-null optional containing the retained matching complete candidate, or empty;
     *     the batch is not mutated
     * @throws NullPointerException if {@code identity} is {@code null}
     */
    public Optional<Candidate> find(CandidateIdentity identity) {
        Objects.requireNonNull(identity, "identity");
        return candidates.stream().filter(value -> value.identity().equals(identity)).findFirst();
    }
}
