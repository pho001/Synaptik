package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningBatch;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasTuningDecision;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Checked immutable CPU inputs that do not contain graph semantics or instance resources.
 *
 * @param loweringManifestEnabled whether cold diagnostics should retain a lowering manifest
 * @param carrierPattern non-null ordered direct carrier forms for the derived boundaries;
 *     membership is snapshotted and contains no physical carrier object; an empty list selects
 *     one exact {@code MEMORY_SEGMENT} form per derived boundary
 * @param portableExecution non-null immutable cold compute-preference and parallelism inputs
 * @param materializationPolicy non-null dimensionless cold candidate-comparison policy; disabled
 *     retains only direct forms, while enabled ordinary analysis retains complete bounded copied
 *     candidates but still selects direct execution
 * @param conv2dMaterializedSuffixUnit whether this input belongs to the sole tagged pointwise
 *     suffix unit of a two-unit Conv2d plan
 * @param partialReductionEvidence non-null diagnostic partial-reduction evidence; current
 *     production preparation treats it as non-authoritative and remains fail-closed
 * @param boundaryStorageFacts non-null ordered expected storage-provenance and alignment facts;
 *     an empty list supplies no OpenBLAS qualification, and no entry retains a physical segment
 * @param openBlasRoute non-null disabled-by-default OpenBLAS qualification and cost snapshot
 * @param cpuHardwareIdentity non-null caller-supplied canonical cold CPU identity; analysis never
 *     discovers or normalizes host facts
 * @param workloadCohort non-null expected-use cohort whose run count must agree with the existing
 *     materialization policy whenever OpenBLAS candidate generation is enabled
 * @param openBlasTuningDecision non-null optional explicit decision to validate against a freshly
 *     generated candidate batch; incompatibility is a miss, not an error
 */
public record CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
        List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
        MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit,
        PartialReductionEvidence partialReductionEvidence,
        List<BoundaryStorageFact> boundaryStorageFacts, OpenBlasRouteConfig openBlasRoute,
        CpuOpenBlasTuningBatch.HardwareIdentity cpuHardwareIdentity,
        CpuOpenBlasTuningBatch.WorkloadCohort workloadCohort,
        Optional<CpuOpenBlasTuningDecision> openBlasTuningDecision)
        implements BackendAnalysisInputs {
    /**
     * Preserves the complete pre-OpenBLAS construction surface and disables native routing.
     *
     * @param loweringManifestEnabled whether cold diagnostics should retain a lowering manifest
     * @param carrierPattern non-null ordered direct carrier forms; copied defensively
     * @param portableExecution non-null immutable cold execution inputs
     * @param materializationPolicy non-null dimensionless cold materialization policy
     * @param conv2dMaterializedSuffixUnit whether this is the tagged Conv2d suffix unit
     * @param partialReductionEvidence non-null diagnostic partial-reduction evidence
     * @throws NullPointerException if a required reference or list entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit,
            PartialReductionEvidence partialReductionEvidence) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                conv2dMaterializedSuffixUnit, partialReductionEvidence, List.of(),
                OpenBlasRouteConfig.DISABLED, CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty());
    }
    /**
     * Compatibility constructor that deliberately admits no partial route.
     *
     * @param loweringManifestEnabled whether diagnostics are retained
     * @param carrierPattern requested carrier forms
     * @param portableExecution cold execution limits
     * @param materializationPolicy cold materialization policy
     * @param conv2dMaterializedSuffixUnit tagged Conv2d suffix marker
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                conv2dMaterializedSuffixUnit, PartialReductionEvidence.NONE, List.of(),
                OpenBlasRouteConfig.DISABLED, CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty());
    }
    /**
     * Default input: manifest and materialization disabled, scalar single-thread execution, and
     * one exact-segment carrier selected for every boundary derived by lowering.
     */
    public static final CpuPartitionAnalysisInputs DEFAULT = new CpuPartitionAnalysisInputs(false,
            List.of(),
            PortableExecutionConfig.DEFAULT, MaterializationPolicy.DISABLED, false,
            PartialReductionEvidence.NONE, List.of(), OpenBlasRouteConfig.DISABLED,
            CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
            CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty());

    /**
     * Immutable expected boundary-storage fact used only during cold native-route analysis.
     * It promises provenance and minimum alignment but retains no segment or runtime resource.
     *
     * @param nativeSegment whether composition expects a genuine native memory segment
     * @param byteAlignment positive power-of-two minimum address alignment in bytes
     */
    public record BoundaryStorageFact(boolean nativeSegment, long byteAlignment) {
        /** Unknown/non-native fail-closed storage expectation. */
        public static final BoundaryStorageFact UNKNOWN = new BoundaryStorageFact(false, 1);

        /**
         * Validates one immutable expected storage fact.
         *
         * @param nativeSegment whether composition expects a genuine native memory segment
         * @param byteAlignment positive power-of-two minimum address alignment in bytes
         * @throws IllegalArgumentException if {@code byteAlignment} is not a positive power of two
         */
        public BoundaryStorageFact {
            if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
                throw new IllegalArgumentException(
                        "boundary byte alignment must be a positive power of two");
            }
        }
    }

    /**
     * Disabled-by-default immutable OpenBLAS route qualification and whole-plan cost snapshot.
     * Missing cost or threshold components are representable so analysis can fail closed without
     * guessing. The snapshot owns no provider, native handle, segment, slot, or measured result.
     *
     * @param qualification successful session-bound qualification; empty disables the route
     * @param threadCandidates immutable bounded positive provider-thread candidates
     * @param portableCosts complete or incomplete portable whole-plan cost terms
     * @param representationCosts complete or incomplete per-run representation cost terms
     * @param minimumNetBenefitCostUnits optional non-negative absolute benefit threshold
     * @param minimumBenefitBasisPoints optional relative threshold in {@code [0, 10_000]}
     */
    public record OpenBlasRouteConfig(
            Optional<io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasQualification> qualification,
            List<ThreadCandidate> threadCandidates, CostTerms portableCosts,
            RepresentationCostTerms representationCosts,
            OptionalLong minimumNetBenefitCostUnits,
            OptionalInt minimumBenefitBasisPoints) {
        /**
         * One immutable positive provider-thread candidate. Incomplete cost terms are retained so
         * route analysis can reject the containing configuration without guessing.
         *
         * @param threadCount positive provider count and fixed permit demand
         * @param openBlasCosts complete or deliberately incomplete non-negative cost terms for
         *     this exact count
         */
        public record ThreadCandidate(int threadCount, CostTerms openBlasCosts) {
            /**
             * Validates one candidate without converting missing costs into zero.
             *
             * @throws NullPointerException if {@code openBlasCosts} is {@code null}
             * @throws IllegalArgumentException if {@code threadCount} is not positive
             */
            public ThreadCandidate {
                if (threadCount <= 0) throw new IllegalArgumentException(
                        "OpenBLAS candidate thread count must be positive");
                java.util.Objects.requireNonNull(openBlasCosts, "openBlasCosts");
            }
        }

        /** Fail-closed route input used by every compatibility constructor. */
        public static final OpenBlasRouteConfig DISABLED = new OpenBlasRouteConfig(
                Optional.empty(), List.of(), CostTerms.MISSING,
                RepresentationCostTerms.MISSING, OptionalLong.empty(), OptionalInt.empty());

        /**
         * Creates one complete qualified single-thread route snapshot.
         *
         * @param qualification successful credential for the exact live provider session
         * @param portableFixed non-negative portable fixed cost per run
         * @param portablePerOutput non-negative portable cost per output element
         * @param portablePerMac non-negative portable cost per multiply-accumulate
         * @param openBlasFixed non-negative native transition and call cost per run
         * @param openBlasPerOutput non-negative OpenBLAS cost per output element
         * @param openBlasPerMac non-negative OpenBLAS cost per multiply-accumulate
         * @param absoluteThreshold non-negative minimum absolute benefit
         * @param relativeBasisPoints relative benefit threshold in {@code [0, 10_000]}
         * @return a complete immutable qualified configuration; never {@code null}
         * @throws NullPointerException if {@code qualification} is {@code null}
         * @throws IllegalArgumentException if a cost or threshold is negative or the relative
         *     threshold is greater than 10,000 basis points
         */
        public static OpenBlasRouteConfig qualifiedSingleThread(
                io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasQualification qualification,
                long portableFixed,
                long portablePerOutput, long portablePerMac, long openBlasFixed,
                long openBlasPerOutput, long openBlasPerMac, long absoluteThreshold,
                int relativeBasisPoints) {
            return new OpenBlasRouteConfig(Optional.of(java.util.Objects.requireNonNull(
                            qualification, "qualification")),
                    List.of(new ThreadCandidate(1,
                            CostTerms.complete(openBlasFixed, openBlasPerOutput, openBlasPerMac))),
                    CostTerms.complete(portableFixed, portablePerOutput, portablePerMac),
                    RepresentationCostTerms.ZERO,
                    OptionalLong.of(absoluteThreshold), OptionalInt.of(relativeBasisPoints));
        }

        /**
         * Creates one complete qualified single-thread route snapshot including representation
         * transition costs.
         *
         * @param qualification successful credential for the exact live provider session
         * @param portableCosts complete portable GEMM cost terms
         * @param openBlasCosts complete native GEMM/provider cost terms
         * @param representationCosts complete workspace and copy cost terms
         * @param absoluteThreshold non-negative minimum absolute benefit
         * @param relativeBasisPoints relative benefit threshold in {@code [0, 10_000]}
         * @return a complete immutable qualified configuration; never {@code null}
         * @throws NullPointerException if the qualification or a cost-term set is {@code null}
         * @throws IllegalArgumentException if a cost term or threshold is negative or the
         *     relative threshold is greater than 10,000 basis points
         */
        public static OpenBlasRouteConfig qualifiedSingleThread(
                io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasQualification qualification,
                CostTerms portableCosts,
                CostTerms openBlasCosts, RepresentationCostTerms representationCosts,
                long absoluteThreshold, int relativeBasisPoints) {
            return new OpenBlasRouteConfig(Optional.of(java.util.Objects.requireNonNull(
                            qualification, "qualification")),
                    List.of(new ThreadCandidate(1, openBlasCosts)), portableCosts,
                    representationCosts, OptionalLong.of(absoluteThreshold),
                    OptionalInt.of(relativeBasisPoints));
        }

        /**
         * Validates one immutable qualification snapshot without making it complete.
         *
         * @param qualification non-null optional successful qualification
         * @param threadCandidates non-null ordered candidates, copied defensively; at most 32,
         *     with unique positive counts
         * @param portableCosts non-null complete or incomplete portable cost terms
         * @param representationCosts non-null complete or incomplete workspace and copy terms
         * @param minimumNetBenefitCostUnits non-null optional absolute threshold
         * @param minimumBenefitBasisPoints non-null optional relative threshold
         * @throws NullPointerException if a required reference is {@code null}
         * @throws IllegalArgumentException if more than 32 candidates are supplied, candidate
         *     counts repeat, a present threshold is negative, or the relative threshold is
         *     greater than 10,000 basis points
         */
        public OpenBlasRouteConfig {
            qualification = java.util.Objects.requireNonNull(qualification, "qualification");
            threadCandidates = List.copyOf(threadCandidates);
            if (threadCandidates.size() > 32) throw new IllegalArgumentException(
                    "OpenBLAS thread-candidate count exceeds 32");
            if (threadCandidates.stream().map(ThreadCandidate::threadCount).distinct().count()
                    != threadCandidates.size()) throw new IllegalArgumentException(
                    "OpenBLAS thread-candidate counts must be unique");
            java.util.Objects.requireNonNull(portableCosts, "portableCosts");
            java.util.Objects.requireNonNull(representationCosts, "representationCosts");
            java.util.Objects.requireNonNull(minimumNetBenefitCostUnits,
                    "minimumNetBenefitCostUnits");
            java.util.Objects.requireNonNull(minimumBenefitBasisPoints,
                    "minimumBenefitBasisPoints");
            if (minimumNetBenefitCostUnits.isPresent()
                    && minimumNetBenefitCostUnits.getAsLong() < 0) {
                throw new IllegalArgumentException("OpenBLAS absolute threshold is negative");
            }
            if (minimumBenefitBasisPoints.isPresent()
                    && (minimumBenefitBasisPoints.getAsInt() < 0
                        || minimumBenefitBasisPoints.getAsInt() > 10_000)) {
                throw new IllegalArgumentException("OpenBLAS basis-point threshold is invalid");
            }
        }

        /** Reports whether every qualification and cost component is present.
         * @return whether selection may evaluate the complete snapshot */
        public boolean complete() {
            return qualification.isPresent()
                    && !threadCandidates.isEmpty() && portableCosts.complete()
                    && threadCandidates.stream().allMatch(value -> value.openBlasCosts().complete())
                    && representationCosts.complete()
                    && minimumNetBenefitCostUnits.isPresent()
                    && minimumBenefitBasisPoints.isPresent();
        }
    }

    /**
     * Six checked per-run representation coefficients for an OpenBLAS route.
     * Every workspace and copy is charged on every expected run.
     *
     * @param workspaceAllocationAndBindingFixed optional non-negative fixed cost for each
     *     run-owned workspace allocated and bound for one run
     * @param workspaceCostPerByte optional non-negative cost per declared workspace byte
     * @param copyInFixed optional non-negative fixed cost for each input copy invocation
     * @param copyInPerElement optional non-negative cost per copied input element
     * @param copyOutFixed optional non-negative fixed cost for the output copy invocation
     * @param copyOutPerElement optional non-negative cost per copied output element
     */
    public record RepresentationCostTerms(OptionalLong workspaceAllocationAndBindingFixed,
            OptionalLong workspaceCostPerByte, OptionalLong copyInFixed,
            OptionalLong copyInPerElement, OptionalLong copyOutFixed,
            OptionalLong copyOutPerElement) {
        /** Deliberately incomplete fail-closed representation term set. */
        public static final RepresentationCostTerms MISSING = new RepresentationCostTerms(
                OptionalLong.empty(), OptionalLong.empty(), OptionalLong.empty(),
                OptionalLong.empty(), OptionalLong.empty(), OptionalLong.empty());
        /** Complete zero-cost compatibility representation term set. */
        public static final RepresentationCostTerms ZERO = complete(0, 0, 0, 0, 0, 0);

        /**
         * Creates a complete non-negative representation term set.
         *
         * @param workspaceFixed fixed cost for each run-owned workspace allocation and binding
         * @param perByte cost per declared workspace byte
         * @param inputFixed fixed cost for each input copy invocation
         * @param inputPerElement cost per copied input element
         * @param outputFixed fixed cost for the output copy invocation
         * @param outputPerElement cost per copied output element
         * @return a complete immutable term set; never {@code null}
         * @throws IllegalArgumentException if any coefficient is negative
         */
        public static RepresentationCostTerms complete(long workspaceFixed, long perByte,
                long inputFixed, long inputPerElement, long outputFixed,
                long outputPerElement) {
            return new RepresentationCostTerms(OptionalLong.of(workspaceFixed),
                    OptionalLong.of(perByte), OptionalLong.of(inputFixed),
                    OptionalLong.of(inputPerElement), OptionalLong.of(outputFixed),
                    OptionalLong.of(outputPerElement));
        }

        /**
         * Validates and snapshots optional representation coefficients.
         *
         * @throws NullPointerException if any optional reference is {@code null}
         * @throws IllegalArgumentException if any present coefficient is negative
         */
        public RepresentationCostTerms {
            java.util.Objects.requireNonNull(workspaceAllocationAndBindingFixed,
                    "workspaceAllocationAndBindingFixed");
            java.util.Objects.requireNonNull(workspaceCostPerByte, "workspaceCostPerByte");
            java.util.Objects.requireNonNull(copyInFixed, "copyInFixed");
            java.util.Objects.requireNonNull(copyInPerElement, "copyInPerElement");
            java.util.Objects.requireNonNull(copyOutFixed, "copyOutFixed");
            java.util.Objects.requireNonNull(copyOutPerElement, "copyOutPerElement");
            if (java.util.stream.Stream.of(workspaceAllocationAndBindingFixed,
                    workspaceCostPerByte, copyInFixed, copyInPerElement, copyOutFixed,
                    copyOutPerElement).anyMatch(value -> value.isPresent()
                            && value.getAsLong() < 0)) {
                throw new IllegalArgumentException(
                        "OpenBLAS representation costs must be non-negative");
            }
        }

        /**
         * Reports whether exact route selection can consume all representation coefficients.
         *
         * @return whether every representation coefficient is present
         */
        public boolean complete() {
            return workspaceAllocationAndBindingFixed.isPresent()
                    && workspaceCostPerByte.isPresent() && copyInFixed.isPresent()
                    && copyInPerElement.isPresent() && copyOutFixed.isPresent()
                    && copyOutPerElement.isPresent();
        }
    }

    /**
     * Three checked dimensionless cost coefficients for one route.
     *
     * @param fixedCostUnits optional non-negative fixed cost per run
     * @param costUnitsPerOutput optional non-negative cost per logical output element
     * @param costUnitsPerMac optional non-negative cost per multiply-accumulate
     */
    public record CostTerms(OptionalLong fixedCostUnits, OptionalLong costUnitsPerOutput,
            OptionalLong costUnitsPerMac) {
        /** Deliberately incomplete fail-closed term set. */
        public static final CostTerms MISSING = new CostTerms(OptionalLong.empty(),
                OptionalLong.empty(), OptionalLong.empty());

        /**
         * Creates complete non-negative terms.
         * @param fixed non-negative fixed cost per run
         * @param perOutput non-negative cost per output element
         * @param perMac non-negative cost per multiply-accumulate
         * @return complete immutable terms
         * @throws IllegalArgumentException if a cost is negative
         */
        public static CostTerms complete(long fixed, long perOutput, long perMac) {
            return new CostTerms(OptionalLong.of(fixed), OptionalLong.of(perOutput),
                    OptionalLong.of(perMac));
        }

        /**
         * Validates every present cost term.
         *
         * @param fixedCostUnits non-null optional fixed cost per run
         * @param costUnitsPerOutput non-null optional cost per logical output
         * @param costUnitsPerMac non-null optional cost per multiply-accumulate
         * @throws NullPointerException if an optional reference is {@code null}
         * @throws IllegalArgumentException if a present cost is negative
         */
        public CostTerms {
            java.util.Objects.requireNonNull(fixedCostUnits, "fixedCostUnits");
            java.util.Objects.requireNonNull(costUnitsPerOutput, "costUnitsPerOutput");
            java.util.Objects.requireNonNull(costUnitsPerMac, "costUnitsPerMac");
            if (fixedCostUnits.isPresent() && fixedCostUnits.getAsLong() < 0
                    || costUnitsPerOutput.isPresent() && costUnitsPerOutput.getAsLong() < 0
                    || costUnitsPerMac.isPresent() && costUnitsPerMac.getAsLong() < 0) {
                throw new IllegalArgumentException("OpenBLAS cost terms must be non-negative");
            }
        }

        /** Reports whether all three coefficients are present.
         * @return whether exact whole-plan arithmetic can consume these terms */
        public boolean complete() {
            return fixedCostUnits.isPresent() && costUnitsPerOutput.isPresent()
                    && costUnitsPerMac.isPresent();
        }
    }

    /**
     * Immutable diagnostic record for the deliberately narrow partial-reduction route.
     *
     * <p>This record is not a production admission credential.  Until a separately scoped
     * trusted reader verifies a complete frozen evidence root, the current selector ignores even
     * a syntactically passing instance and keeps the whole-cell route.</p>
     *
     * @param passed claimed matching-evidence outcome; it cannot authorize the current selector
     * @param kind exact ordinary aggregate kind authorized by a passing row
     * @param dataType exact represented primitive type authorized by a passing row
     * @param form exact aggregate form authorized by a passing row
     * @param partialCount exact prepared count, restricted to two or four when passed
     */
    public record PartialReductionEvidence(boolean passed, CpuAggregateIr.Kind kind,
            DataType dataType, CpuAggregateIr.Form form, int partialCount) {
        /** Fail-closed default with no admitted partial count. */
        public static final PartialReductionEvidence NONE = new PartialReductionEvidence(false,
                null, null, null, 0);
        /**
         * Compatibility constructor for the former untyped evidence shape.
         *
         * <p>Only the fail-closed {@link #NONE} form is meaningful without an aggregate identity.
         * A passing row must name the exact kind, type, and form that it authorizes.</p>
         *
         * @param passed whether the evidence row passed
         * @param partialCount requested partial count
         */
        public PartialReductionEvidence(boolean passed, int partialCount) {
            this(passed, null, null, null, partialCount);
        }
        /** Validates the diagnostic shape without performing measurement or admission. */
        public PartialReductionEvidence {
            if (passed && (kind == null || dataType == null || form == null
                    || partialCount != 2 && partialCount != 4)
                    || !passed && (kind != null || dataType != null || form != null
                    || partialCount != 0))
                throw new IllegalArgumentException("partial-reduction evidence facts disagree");
        }
    }

    /**
     * Creates ordinary analysis inputs outside the tagged Conv2d suffix exception.
     *
     * @param loweringManifestEnabled whether cold diagnostics retain a lowering manifest
     * @param carrierPattern ordered direct carrier forms; copied defensively by the canonical
     *     constructor
     * @param portableExecution immutable compute and parallelism preferences
     * @param materializationPolicy dimensionless cold materialization policy
     * @throws NullPointerException if a required reference or list element is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                false, PartialReductionEvidence.NONE, List.of(), OpenBlasRouteConfig.DISABLED,
                CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                new CpuOpenBlasTuningBatch.WorkloadCohort(1, "ordinary",
                        materializationPolicy.expectedRunCount()), Optional.empty());
    }

    /**
     * Cold, dimensionless materialization evidence. No value is measured during preparation.
     *
     * @param enabled whether analysis may retain bounded direct, single-copy, and two-copy
     *     complete representation candidates; it does not authorize ordinary copy promotion
     * @param copyFixedCostUnits non-negative fixed copy estimate per run
     * @param copyCostUnitsPerElement non-negative copy estimate per logical element
     * @param directKernelCostUnitsPerElement non-negative direct-consumer estimate per element/use
     * @param contiguousKernelCostUnitsPerElement non-negative contiguous-consumer estimate per
     *     element/use
     * @param expectedRunCount positive repeated-run estimate
     * @param maximumAdditionalBytes non-negative combined copy-candidate workspace byte ceiling
     * @param minimumNetBenefitCostUnits non-negative diagnostic absolute benefit threshold
     * @param minimumBenefitBasisPoints diagnostic relative benefit threshold from {@code 0} through
     *     {@code 10_000}
     */
    public record MaterializationPolicy(boolean enabled, long copyFixedCostUnits,
            long copyCostUnitsPerElement, long directKernelCostUnitsPerElement,
            long contiguousKernelCostUnitsPerElement, long expectedRunCount,
            long maximumAdditionalBytes, long minimumNetBenefitCostUnits,
            int minimumBenefitBasisPoints) {
        /** Direct-only compatibility policy. */
        public static final MaterializationPolicy DISABLED = new MaterializationPolicy(false,
                0, 0, 0, 0, 1, 0, 0, 0);
        /**
         * Validates one cold comparison policy.
         *
         * @throws IllegalArgumentException if a cost or byte limit is negative, expected runs are
         *     not positive, or the basis-point threshold is outside {@code [0, 10_000]}
         */
        public MaterializationPolicy {
            if (copyFixedCostUnits < 0 || copyCostUnitsPerElement < 0
                    || directKernelCostUnitsPerElement < 0
                    || contiguousKernelCostUnitsPerElement < 0 || expectedRunCount <= 0
                    || maximumAdditionalBytes < 0 || minimumNetBenefitCostUnits < 0
                    || minimumBenefitBasisPoints < 0 || minimumBenefitBasisPoints > 10_000) {
                throw new IllegalArgumentException("invalid materialization policy");
            }
        }
    }

    /**
     * Cold CPU-private strategy-selection inputs. CPU analysis bounds usable parallelism by the
     * configured and available snapshots, then applies the minimum range size. These facts do not
     * discover workers, retain a worker group, or enter generated artifact identity.
     *
     * @param computePreference non-null scalar or vector-if-eligible preference
     * @param configuredMaximumParallelism positive caller-configured upper bound
     * @param availableParallelism positive composition/platform availability snapshot
     * @param minimumElementsPerWorker positive minimum logical elements per parallel chunk
     */
    public record PortableExecutionConfig(ComputePreference computePreference,
            int configuredMaximumParallelism, int availableParallelism,
            long minimumElementsPerWorker) {
        /** Requested compute-axis preference with deterministic scalar fallback. */
        public enum ComputePreference {
            /** Require scalar generated compute. */ SCALAR,
            /** Prefer vector compute when every boundary is eligible; otherwise use scalar. */
            VECTOR_IF_ELIGIBLE
        }
        /** Compatibility configuration used by {@link CpuPartitionAnalysisInputs#DEFAULT}. */
        public static final PortableExecutionConfig DEFAULT = new PortableExecutionConfig(
                ComputePreference.SCALAR, 1, 1, 1);

        /**
         * Validates one immutable cold strategy-selection input.
         *
         * @param computePreference non-null scalar or vector-if-eligible preference
         * @param configuredMaximumParallelism positive caller-configured upper bound
         * @param availableParallelism positive availability snapshot
         * @param minimumElementsPerWorker positive minimum logical elements per parallel chunk
         * @throws NullPointerException if {@code computePreference} is {@code null}
         * @throws IllegalArgumentException if a parallelism or minimum-range value is not positive
         */
        public PortableExecutionConfig {
            java.util.Objects.requireNonNull(computePreference, "computePreference");
            if (configuredMaximumParallelism <= 0 || availableParallelism <= 0
                    || minimumElementsPerWorker <= 0) {
                throw new IllegalArgumentException("portable execution limits must be positive");
            }
        }
    }

    /**
     * Creates compatibility inputs selecting scalar, single-thread execution.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; snapshotted
     * @throws NullPointerException if {@code carrierPattern} or an entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern) {
        this(loweringManifestEnabled, carrierPattern, PortableExecutionConfig.DEFAULT,
                MaterializationPolicy.DISABLED, false, PartialReductionEvidence.NONE, List.of(),
                OpenBlasRouteConfig.DISABLED, CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty());
    }

    /**
     * Creates direct-only inputs with explicit portable execution selection.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; snapshotted
     * @param portableExecution non-null immutable cold execution inputs
     * @throws NullPointerException if a reference or carrier entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution) {
        this(loweringManifestEnabled, carrierPattern, portableExecution,
                MaterializationPolicy.DISABLED, false, PartialReductionEvidence.NONE, List.of(),
                OpenBlasRouteConfig.DISABLED, CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty());
    }

    /**
     * Preserves the complete pre-0010E construction surface while supplying typed compatibility
     * defaults and no explicit tuning decision.
     *
     * @param loweringManifestEnabled whether cold diagnostics retain a lowering manifest
     * @param carrierPattern ordered derived-boundary carriers; copied defensively
     * @param portableExecution immutable portable execution inputs
     * @param materializationPolicy immutable materialization and expected-run policy
     * @param conv2dMaterializedSuffixUnit whether this is the tagged Conv2d suffix unit
     * @param partialReductionEvidence immutable diagnostic partial-reduction evidence
     * @param boundaryStorageFacts ordered expected native-storage facts; copied defensively
     * @param openBlasRoute disabled or complete qualified OpenBLAS route snapshot
     * @throws NullPointerException if a required reference or list entry is {@code null}
     */
    public CpuPartitionAnalysisInputs(boolean loweringManifestEnabled,
            List<CarrierAccess> carrierPattern, PortableExecutionConfig portableExecution,
            MaterializationPolicy materializationPolicy, boolean conv2dMaterializedSuffixUnit,
            PartialReductionEvidence partialReductionEvidence,
            List<BoundaryStorageFact> boundaryStorageFacts, OpenBlasRouteConfig openBlasRoute) {
        this(loweringManifestEnabled, carrierPattern, portableExecution, materializationPolicy,
                conv2dMaterializedSuffixUnit, partialReductionEvidence, boundaryStorageFacts,
                openBlasRoute, CpuOpenBlasTuningBatch.HardwareIdentity.UNSPECIFIED,
                new CpuOpenBlasTuningBatch.WorkloadCohort(1, "ordinary",
                        materializationPolicy.expectedRunCount()), Optional.empty());
    }

    /**
     * Snapshots the non-null ordered pattern, rejecting null entries in encounter order.
     *
     * @param loweringManifestEnabled whether to retain cold lowering diagnostics
     * @param carrierPattern non-null ordered derived-boundary carrier pattern, or empty for the
     *     exact-segment-per-boundary policy; copied defensively
     * @param portableExecution non-null immutable cold execution inputs
     * @param materializationPolicy non-null dimensionless cold materialization policy
     * @param conv2dMaterializedSuffixUnit whether this is the tagged Conv2d suffix unit
     * @param partialReductionEvidence non-null diagnostic partial-reduction evidence
     * @param boundaryStorageFacts non-null ordered expected native-storage facts; copied
     * @param openBlasRoute non-null disabled or complete OpenBLAS route snapshot
     * @param cpuHardwareIdentity non-null caller-supplied canonical CPU identity
     * @param workloadCohort non-null caller-supplied expected-use cohort
     * @param openBlasTuningDecision non-null optional explicit matching decision
     * @throws NullPointerException if a required reference or list entry is {@code null}
     * @throws IllegalArgumentException if the workload cohort and materialization policy have
     *     different expected-run counts
     */
    public CpuPartitionAnalysisInputs {
        carrierPattern = List.copyOf(carrierPattern);
        java.util.Objects.requireNonNull(portableExecution, "portableExecution");
        java.util.Objects.requireNonNull(materializationPolicy, "materializationPolicy");
        java.util.Objects.requireNonNull(partialReductionEvidence, "partialReductionEvidence");
        boundaryStorageFacts = List.copyOf(boundaryStorageFacts);
        java.util.Objects.requireNonNull(openBlasRoute, "openBlasRoute");
        java.util.Objects.requireNonNull(cpuHardwareIdentity, "cpuHardwareIdentity");
        java.util.Objects.requireNonNull(workloadCohort, "workloadCohort");
        openBlasTuningDecision = java.util.Objects.requireNonNull(openBlasTuningDecision,
                "openBlasTuningDecision");
        if (workloadCohort.expectedRunCount() != materializationPolicy.expectedRunCount()) {
            throw new IllegalArgumentException(
                    "workload cohort and materialization expected-run counts disagree");
        }
    }
}
