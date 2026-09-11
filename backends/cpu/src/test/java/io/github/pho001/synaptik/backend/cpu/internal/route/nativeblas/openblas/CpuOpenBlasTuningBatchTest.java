package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

final class CpuOpenBlasTuningBatchTest {
    private record WorkloadMutation(String component,
            CpuOpenBlasTuningBatch.WorkloadSignature workload) { }

    @Test void emitsPortableThenEveryRepresentationAndFittingThreadInStableOrder() {
        var config = config(CpuOpenBlasRouteSelectorTest.qualification(), List.of(
                CpuOpenBlasRouteSelectorTest.candidate(2, 2),
                CpuOpenBlasRouteSelectorTest.candidate(1, 1)));
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, config);
        var inputs = replace(base, hardware("x86_64", "one"), cohort("training", 1),
                Optional.empty(), 2);
        CpuPartitionPreparationPlan plan = analyze(inputs);
        CpuOpenBlasTuningBatch batch = plan.openBlasTuningBatch().orElseThrow();
        assertAll(() -> assertEquals(17, batch.candidates().size()),
                () -> assertEquals(CpuOpenBlasTuningBatch.RouteKind.PORTABLE,
                        batch.candidates().getFirst().route()),
                () -> assertEquals(List.of(
                        CpuOpenBlasRoutePlan.Representation.DIRECT,
                        CpuOpenBlasRoutePlan.Representation.DIRECT,
                        CpuOpenBlasRoutePlan.Representation.COPY_LEFT,
                        CpuOpenBlasRoutePlan.Representation.COPY_LEFT),
                        batch.candidates().subList(1, 5).stream()
                                .map(value -> value.openBlasPlan().orElseThrow().representation())
                                .toList()),
                () -> assertEquals(List.of(2, 1, 2, 1), batch.candidates().subList(1, 5)
                        .stream().map(value -> value.openBlasPlan().orElseThrow().threadCount())
                        .toList()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> batch.candidates().add(batch.candidates().getFirst())),
                () -> assertEquals(hardware("x86_64", "one"), batch.workload().hardware()),
                () -> assertEquals(cohort("training", 1), batch.workload().cohort()));
    }

    @Test void matchingDecisionSelectsAnyExistingNativeOrPortableCandidate() {
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, CpuOpenBlasRouteSelectorTest.defaultConfig());
        var first = analyze(replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty(), 1));
        CpuOpenBlasTuningBatch batch = first.openBlasTuningBatch().orElseThrow();
        var outputCopy = batch.candidates().stream().filter(value -> value.openBlasPlan()
                .map(plan -> plan.representation() == CpuOpenBlasRoutePlan.Representation.COPY_OUTPUT)
                .orElse(false)).findFirst().orElseThrow();
        var decision = new CpuOpenBlasTuningDecision(batch.schemaVersion(), batch.workload(),
                outputCopy.identity());
        var selected = analyze(replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.of(decision), 1));
        var portableDecision = new CpuOpenBlasTuningDecision(batch.schemaVersion(),
                batch.workload(), batch.candidates().getFirst().identity());
        var portable = analyze(replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT,
                Optional.of(portableDecision), 1));
        assertAll(() -> assertEquals(CpuOpenBlasRoutePlan.Representation.COPY_OUTPUT,
                        selected.openBlasPlan().orElseThrow().representation()),
                () -> assertEquals(outputCopy.identity(),
                        selected.selectedOpenBlasTuningCandidate().orElseThrow()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE, portable.route()),
                () -> assertTrue(portable.openBlasPlan().isEmpty()),
                () -> assertTrue(portable.openBlasTuningBatch().isPresent()));
    }

    @Test void incompatibleSchemaWorkloadScopeAndCandidateAreMissesUsingHeuristic() {
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, CpuOpenBlasRouteSelectorTest.defaultConfig());
        var initial = analyze(replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty(), 1));
        CpuOpenBlasTuningBatch batch = initial.openBlasTuningBatch().orElseThrow();
        var nativeIdentity = batch.candidates().stream()
                .filter(value -> value.route() == CpuOpenBlasTuningBatch.RouteKind.OPENBLAS)
                .findFirst().orElseThrow().identity();
        var wrongSchema = new CpuOpenBlasTuningDecision(99, batch.workload(), nativeIdentity);
        var wrongHardware = new CpuOpenBlasTuningDecision(batch.schemaVersion(), batch.workload(),
                nativeIdentity);
        var absent = new CpuOpenBlasTuningDecision(batch.schemaVersion(), batch.workload(),
                new CpuOpenBlasTuningBatch.OpenBlasIdentity(
                        CpuOpenBlasRoutePlan.Representation.DIRECT, 31, 31, 30, List.of(),
                        0, 0, 0));
        for (CpuPartitionAnalysisInputs inputs : List.of(
                replace(base, hardware("x86_64", "one"),
                        CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT,
                        Optional.of(wrongSchema), 1),
                replace(base, hardware("aarch64", "two"),
                        CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT,
                        Optional.of(wrongHardware), 1),
                replace(base, hardware("x86_64", "one"),
                        CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT,
                        Optional.of(absent), 1))) {
            var plan = analyze(inputs);
            assertEquals(CpuOpenBlasRoutePlan.Representation.DIRECT,
                    plan.openBlasPlan().orElseThrow().representation());
        }
    }

    @Test void everyMutableWorkloadCompatibilityFactIndependentlyMissesToSafeHeuristic() {
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, CpuOpenBlasRouteSelectorTest.defaultConfig());
        var inputs = replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty(), 1);
        CpuPartitionPreparationPlan heuristic = analyze(inputs);
        CpuOpenBlasTuningBatch batch = heuristic.openBlasTuningBatch().orElseThrow();
        CpuOpenBlasTuningBatch.WorkloadSignature original = batch.workload();
        var mutations = new ArrayList<WorkloadMutation>();

        add(mutations, original, "hardware.architecture", "hardware",
                with(original.hardware(), "architecture", "aarch64"));
        add(mutations, original, "hardware.vendor", "hardware",
                with(original.hardware(), "vendor", "other-vendor"));
        add(mutations, original, "hardware.model", "hardware",
                with(original.hardware(), "model", "other-model"));
        add(mutations, original, "hardware.features", "hardware",
                with(original.hardware(), "features", List.of("one", "two")));
        add(mutations, original, "cpuConcurrencyCapacity", "cpuConcurrencyCapacity", 2);

        var execution = original.portableExecution();
        add(mutations, original, "portableExecution.computePreference", "portableExecution",
                with(execution, "computePreference",
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference
                                .VECTOR_IF_ELIGIBLE));
        add(mutations, original, "portableExecution.configuredMaximumParallelism",
                "portableExecution", with(execution, "configuredMaximumParallelism", 2));
        add(mutations, original, "portableExecution.availableParallelism", "portableExecution",
                with(execution, "availableParallelism", 2));
        add(mutations, original, "portableExecution.minimumElementsPerWorker",
                "portableExecution", with(execution, "minimumElementsPerWorker", 2L));
        add(mutations, original, "portableStrategy.compute", "portableStrategy",
                with(original.portableStrategy(), "compute",
                        CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR));
        add(mutations, original, "portableStrategy.orchestration", "portableStrategy",
                with(original.portableStrategy(), "orchestration",
                        CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL));
        add(mutations, original, "portableRangeCount", "portableRangeCount",
                original.portableRangeCount() + 1);
        add(mutations, original, "portableVectorSpeciesBits", "portableVectorSpeciesBits", 64);

        add(mutations, original, "cohort.cohortId", "cohort",
                with(original.cohort(), "cohortId", "other-cohort"));
        var materialization = original.materializationPolicy();
        add(mutations, original, "materializationPolicy.enabled", "materializationPolicy",
                with(materialization, "enabled", !materialization.enabled()));
        addLongComponents(mutations, original, "materializationPolicy", materialization,
                "copyFixedCostUnits", "copyCostUnitsPerElement",
                "directKernelCostUnitsPerElement", "contiguousKernelCostUnitsPerElement",
                "maximumAdditionalBytes", "minimumNetBenefitCostUnits");
        add(mutations, original, "materializationPolicy.minimumBenefitBasisPoints",
                "materializationPolicy", with(materialization, "minimumBenefitBasisPoints", 1));

        addOptionalLongComponents(mutations, original, "portableCosts", original.portableCosts(),
                "fixedCostUnits", "costUnitsPerOutput", "costUnitsPerMac");
        addOptionalLongComponents(mutations, original, "representationCosts",
                original.representationCosts(), "workspaceAllocationAndBindingFixed",
                "workspaceCostPerByte", "copyInFixed", "copyInPerElement", "copyOutFixed",
                "copyOutPerElement");
        var thread = original.openBlasThreadCandidates().getFirst();
        addOptionalLongComponents(mutations, original,
                "openBlasThreadCandidates[0].openBlasCosts", thread.openBlasCosts(),
                changed -> List.of(with(thread, "openBlasCosts", changed)),
                "fixedCostUnits", "costUnitsPerOutput", "costUnitsPerMac");
        add(mutations, original, "minimumNetBenefitCostUnits", "minimumNetBenefitCostUnits",
                original.minimumNetBenefitCostUnits() + 1);
        add(mutations, original, "minimumBenefitBasisPoints", "minimumBenefitBasisPoints",
                original.minimumBenefitBasisPoints() + 1);
        add(mutations, original, "generatedArtifactSchema", "generatedArtifactSchema",
                original.generatedArtifactSchema() + 1);

        for (int boundaryIndex = 0; boundaryIndex < original.boundaries().size(); boundaryIndex++) {
            addBoundaryMutations(mutations, original, boundaryIndex);
        }
        addSessionQualificationMutations(mutations, original);

        assertEquals(97, mutations.size(), "every independently mutable session workload fact");
        for (WorkloadMutation mutation : mutations) {
            assertCompatibilityMissAndSafeFallback(inputs, heuristic, batch, mutation);
        }
    }

    @Test void schemaLockedOrRelationalWorkloadFactsRejectIndependentConstruction() {
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, CpuOpenBlasRouteSelectorTest.defaultConfig());
        var inputs = replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty(), 1);
        var workload = analyze(inputs).openBlasTuningBatch().orElseThrow().workload();

        assertRecordComponents(CpuOpenBlasTuningBatch.WorkloadSignature.class,
                "operationKind", "operationAttributes", "leftType", "rightType",
                "accumulationType", "outputType", "boundaries", "numericalMode",
                "determinismMode", "qualification", "hardware", "cpuConcurrencyCapacity",
                "portableExecution", "portableStrategy", "portableRangeCount",
                "portableVectorSpeciesBits", "openBlasThreadCounts", "cohort",
                "materializationPolicy", "portableCosts", "representationCosts",
                "openBlasThreadCandidates", "minimumNetBenefitCostUnits",
                "minimumBenefitBasisPoints", "generatedArtifactSchema", "routePolicyVersion",
                "costPolicyVersion");
        assertNestedCompatibilityComponentInventories();

        assertClosedEnum(CpuOpenBlasTuningBatch.OperationKind.values(), "operationKind");
        assertClosedEnum(CpuOpenBlasTuningBatch.OperationAttributes.values(),
                "operationAttributes");
        assertClosedEnum(CpuOpenBlasTuningBatch.NumericalMode.values(), "numericalMode");
        assertClosedEnum(CpuOpenBlasTuningBatch.DeterminismMode.values(), "determinismMode");
        for (String component : List.of("operationKind", "operationAttributes", "numericalMode",
                "determinismMode")) {
            assertThrows(NullPointerException.class, () -> with(workload, component, null),
                    component + " rejects null and has no second valid schema value");
        }
        for (String type : List.of("leftType", "rightType", "accumulationType", "outputType")) {
            assertConstructionRejected(workload, type, DataType.FLOAT64,
                    type + " cannot differ independently because schema 1 requires one exact type");
        }
        assertConstructionRejected(workload, "openBlasThreadCounts", List.of(2),
                "thread counts cannot differ independently from thread-candidate identities");
        assertConstructionRejected(workload, "boundaries", workload.boundaries().subList(0, 2),
                "boundaries must contain exactly left, right, and output");
        assertConstructionRejected(workload, "cohort",
                with(workload.cohort(), "expectedRunCount", 2L),
                "cohort run count cannot differ independently from materialization policy");
        assertConstructionRejected(workload, "materializationPolicy",
                with(workload.materializationPolicy(), "expectedRunCount", 2L),
                "materialization run count cannot differ independently from workload cohort");
        assertConstructionRejected(workload, "openBlasThreadCandidates", List.of(
                with(workload.openBlasThreadCandidates().getFirst(), "threadCount", 2)),
                "thread-candidate count cannot differ independently from the count list");
        assertConstructionRejected(workload, "routePolicyVersion",
                CpuOpenBlasTuningBatch.ROUTE_POLICY_VERSION + 1,
                "routePolicyVersion has one supported schema-1 value");
        assertConstructionRejected(workload, "costPolicyVersion",
                CpuOpenBlasTuningBatch.COST_POLICY_VERSION + 1,
                "costPolicyVersion has one supported schema-1 value");
        assertThrows(IllegalArgumentException.class,
                () -> new CpuOpenBlasTuningBatch.HardwareIdentity(2, "x86_64", "vendor",
                        "model", List.of()),
                "hardware.schemaVersion has one supported value, so inequality is impossible");
        assertThrows(IllegalArgumentException.class,
                () -> new CpuOpenBlasTuningBatch.WorkloadCohort(2, "ordinary", 1),
                "cohort.schemaVersion has one supported value, so inequality is impossible");
        assertQualificationConstructionRejections(workload.qualification());
    }

    @Test void everyPersistentQualificationIdentityFactIndependentlyMissesOrRejects() {
        var qualification = persistentQualification("a".repeat(64), 64,
                CpuOpenBlasQualification.ExecutableFormat.ELF_64,
                CpuOpenBlasQualification.Machine.X86_64);
        var config = config(qualification,
                List.of(CpuOpenBlasRouteSelectorTest.candidate(1, 1)));
        var base = CpuOpenBlasRouteSelectorTest.maskInputs(DataType.FLOAT32, false, false,
                false, 1, config);
        var inputs = replace(base, hardware("x86_64", "one"),
                CpuOpenBlasTuningBatch.WorkloadCohort.DEFAULT, Optional.empty(), 1);
        CpuPartitionPreparationPlan heuristic = analyze(inputs);
        CpuOpenBlasTuningBatch batch = heuristic.openBlasTuningBatch().orElseThrow();
        var original = batch.workload();
        var scope = original.qualification();
        var identity = scope.persistentIdentity().orElseThrow();
        var binary = identity.binaryIdentity();
        var mutations = new ArrayList<WorkloadMutation>();

        addPersistentIdentityMutation(mutations, original, scope, identity,
                "targetFingerprint.operatingSystem", "targetFingerprint",
                with(identity.targetFingerprint(), "operatingSystem",
                        CpuOpenBlasQualification.OperatingSystem.MACOS));
        addPersistentIdentityMutation(mutations, original, scope, identity,
                "targetFingerprint.machine", "targetFingerprint",
                with(identity.targetFingerprint(), "machine",
                        CpuOpenBlasQualification.Machine.AARCH64));
        addPersistentIdentityMutation(mutations, original, scope, identity,
                "binaryIdentity.sha256", "binaryIdentity",
                with(binary, "sha256", "b".repeat(64)));
        addPersistentIdentityMutation(mutations, original, scope, identity,
                "binaryIdentity.byteLength", "binaryIdentity",
                with(binary, "byteLength", binary.byteLength() + 1));
        addPersistentIdentityMutation(mutations, original, scope, identity,
                "binaryIdentity.executableFormat", "binaryIdentity",
                with(binary, "executableFormat",
                        CpuOpenBlasQualification.ExecutableFormat.PE_32_PLUS));
        addPersistentIdentityMutation(mutations, original, scope, identity,
                "binaryIdentity.machine", "binaryIdentity",
                with(binary, "machine", CpuOpenBlasQualification.Machine.AARCH64));

        assertEquals(6, mutations.size(), "every mutable persistent identity fact");
        for (WorkloadMutation mutation : mutations) {
            assertCompatibilityMissAndSafeFallback(inputs, heuristic, batch, mutation);
            assertNotEquals(original.persistentProjection(), mutation.workload()
                    .persistentProjection(), mutation.component() + " must alter persistence");
        }
        assertThrows(IllegalArgumentException.class,
                () -> with(identity, "qualificationSchemaVersion", 2),
                "persistent qualification schema has one supported value");
        assertThrows(IllegalArgumentException.class,
                () -> with(identity, "requiredSymbols", List.of("cblas_sgemm")),
                "required-symbol inventory is schema-locked, so inequality is impossible");
        assertClosedEnum(CpuOpenBlasQualification.BlasIntAbi.values(),
                "persistentIdentity.blasIntAbi");
        assertThrows(NullPointerException.class, () -> with(identity, "blasIntAbi", null),
                "persistentIdentity.blasIntAbi rejects null and has no second schema value");
        assertThrows(IllegalArgumentException.class,
                () -> with(identity, "numericalCaseVersion", "other"),
                "numerical qualification case is schema-locked, so inequality is impossible");
        assertThrows(IllegalArgumentException.class,
                () -> with(binary, "schemaVersion", 2),
                "binary identity schema has one supported value");
        assertThrows(IllegalArgumentException.class,
                () -> with(binary, "digestAlgorithm", "SHA-512"),
                "binary digest algorithm is schema-locked, so inequality is impossible");
        assertThrows(IllegalArgumentException.class,
                () -> with(scope, "sessionCredential", Optional.of(qualification)),
                "persistent scope cannot independently acquire a session credential");
    }

    @Test void sessionScopeCannotProjectPersistentlyButBinaryScopeCanReuseAcrossSessions() {
        var session = CpuOpenBlasRouteSelectorTest.qualification();
        assertTrue(new CpuOpenBlasTuningBatch.QualificationScope(session)
                .persistentIdentity().isEmpty());
        var binary = new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256", "a".repeat(64),
                64, CpuOpenBlasQualification.ExecutableFormat.ELF_64,
                CpuOpenBlasQualification.Machine.X86_64);
        var target = new CpuOpenBlasQualification.TargetFingerprint(1,
                CpuOpenBlasQualification.OperatingSystem.LINUX,
                CpuOpenBlasQualification.Machine.X86_64, 64, ByteOrder.LITTLE_ENDIAN);
        var first = new CpuOpenBlasQualification(CpuOpenBlasQualification.Scope.PERSISTENT_BINARY,
                target, Optional.of(binary), new CpuOpenBlasQualification.SessionKey());
        var second = new CpuOpenBlasQualification(CpuOpenBlasQualification.Scope.PERSISTENT_BINARY,
                target, Optional.of(binary), new CpuOpenBlasQualification.SessionKey());
        assertAll(() -> assertEquals(new CpuOpenBlasTuningBatch.QualificationScope(first),
                        new CpuOpenBlasTuningBatch.QualificationScope(second)),
                () -> assertTrue(new CpuOpenBlasTuningBatch.QualificationScope(first)
                        .persistentIdentity().isPresent()),
                () -> assertNotEquals(new CpuOpenBlasTuningBatch.QualificationScope(session),
                        new CpuOpenBlasTuningBatch.QualificationScope(
                                CpuOpenBlasRouteSelectorTest.qualification())));
    }

    @Test void malformedColdCompatibilityValuesFailConstruction() {
        assertAll(() -> assertThrows(IllegalArgumentException.class,
                        () -> hardware("x86_64", "z", "a")),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasTuningBatch.WorkloadCohort(1, "", 1)),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuOpenBlasTuningDecision(0, null,
                                new CpuOpenBlasTuningBatch.OpenBlasIdentity(
                                        CpuOpenBlasRoutePlan.Representation.DIRECT, 1, 1, 0,
                                        List.of(), 0, 0, 0))));
    }

    @Test void ineligibleBfloat16ProducesNoNativeCandidateOrTuningBatch() {
        DataType type = DataType.BFLOAT16;
        var config = CpuOpenBlasRouteSelectorTest.defaultConfig();
        var nativeFact = new CpuPartitionAnalysisInputs.BoundaryStorageFact(true,
                type.byteWidth());
        var inputs = new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(3,
                        io.github.pho001.synaptik.backend.cpu.internal.cache
                                .CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE,
                java.util.Collections.nCopies(3, nativeFact), config);
        var plan = new CpuPartitionPreparer().analyze(CpuOpenBlasRouteSelectorTest.context(type,
                dense(io.github.pho001.synaptik.model.shape.Shape.of(2, 3)),
                dense(io.github.pho001.synaptik.model.shape.Shape.of(3, 4)),
                dense(io.github.pho001.synaptik.model.shape.Shape.of(2, 4)), inputs)).plan();
        assertAll(() -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE, plan.route()),
                () -> assertTrue(plan.openBlasPlan().isEmpty()),
                () -> assertTrue(plan.openBlasTuningBatch().isEmpty()));
    }

    private static void addBoundaryMutations(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, int index) {
        var boundary = original.boundaries().get(index);
        String prefix = "boundaries[" + index + "].";
        addBoundary(mutations, original, index, prefix + "dataType", "dataType",
                boundary.dataType() == DataType.FLOAT32 ? DataType.FLOAT64 : DataType.FLOAT32);
        addBoundary(mutations, original, index, prefix + "shape", "shape", Shape.of(1, 1));
        addBoundary(mutations, original, index, prefix + "layout", "layout",
                LayoutDescriptor.of(boundary.shape(), new long[] {100, 1}, 0, true));
        addBoundary(mutations, original, index, prefix + "carrier", "carrier",
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization
                        .CarrierAccess.FLOAT_ARRAY);
        addBoundary(mutations, original, index, prefix + "storage.nativeSegment", "storage",
                with(boundary.storage(), "nativeSegment", !boundary.storage().nativeSegment()));
        addBoundary(mutations, original, index, prefix + "storage.byteAlignment", "storage",
                with(boundary.storage(), "byteAlignment",
                        boundary.storage().byteAlignment() * 2));

        CpuAccessPlan.Binding binding = boundary.access();
        addBoundary(mutations, original, index, prefix + "access.plan", "access",
                with(binding, "plan", with(binding.plan(), "regime",
                        CpuAccessPlan.Regime.GENERAL_ODOMETER)));
        addBoundary(mutations, original, index, prefix + "access.baseElementOffset", "access",
                with(binding, "baseElementOffset", binding.baseElementOffset() + 1));
        var strides = new ArrayList<>(binding.effectiveStrides());
        strides.set(0, strides.getFirst() + 1);
        addBoundary(mutations, original, index, prefix + "access.effectiveStrides", "access",
                with(binding, "effectiveStrides", strides));
        addBoundary(mutations, original, index, prefix + "access.start", "access",
                with(binding, "start", 1L));
        addBoundary(mutations, original, index, prefix + "access.end", "access",
                with(binding, "end", binding.end() - 1));
        addBoundary(mutations, original, index, prefix + "access.referencedElementSpan", "access",
                with(binding, "referencedElementSpan", binding.referencedElementSpan() + 1));
        var coordinates = new ArrayList<>(binding.startCoordinates());
        coordinates.set(0, 1L);
        addBoundary(mutations, original, index, prefix + "access.startCoordinates", "access",
                with(binding, "startCoordinates", coordinates));
        addBoundary(mutations, original, index, prefix + "access.startAddress", "access",
                with(binding, "startAddress", binding.startAddress() + 1));
        addBoundary(mutations, original, index, prefix + "access.accessedElementStart", "access",
                with(binding, "accessedElementStart", binding.accessedElementStart() + 1));
        addBoundary(mutations, original, index, prefix + "access.accessedElementEnd", "access",
                with(binding, "accessedElementEnd", binding.accessedElementEnd() - 1));

        CpuAccessPlan plan = binding.plan();
        addBoundary(mutations, original, index, prefix + "access.plan.accessKind", "access",
                with(binding, "plan", with(plan, "accessKind",
                        plan.accessKind() == CpuAccessPlan.AccessKind.READ
                                ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ)));
        var roles = new ArrayList<>(plan.axisRoles());
        roles.set(0, roles.getFirst() == CpuAccessPlan.AxisRole.CONTIGUOUS
                ? CpuAccessPlan.AxisRole.STRIDED : CpuAccessPlan.AxisRole.CONTIGUOUS);
        addBoundary(mutations, original, index, prefix + "access.plan.axisRoles", "access",
                with(binding, "plan", with(plan, "axisRoles", roles)));
        addBoundary(mutations, original, index, prefix + "access.plan.contiguousSuffix", "access",
                with(binding, "plan", with(plan, "contiguousSuffix",
                        plan.contiguousSuffix() == 0 ? 1 : plan.contiguousSuffix() - 1)));

        assertThrows(IllegalArgumentException.class,
                () -> with(binding, "extents", List.of(1L, 1L)),
                prefix + "access.extents cannot differ independently from elementCount");
        assertThrows(IllegalArgumentException.class,
                () -> with(binding, "elementCount", binding.elementCount() + 1),
                prefix + "access.elementCount cannot differ independently from extents");
        assertThrows(IllegalArgumentException.class,
                () -> with(plan, "iterationRank", plan.iterationRank() + 1),
                prefix + "access.plan.iterationRank cannot differ independently from axisRoles");
    }

    private static void addSessionQualificationMutations(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original) {
        var qualification = original.qualification();
        var target = qualification.targetFingerprint();
        add(mutations, original, "qualification.targetFingerprint.operatingSystem",
                "qualification", with(qualification, "targetFingerprint", with(target,
                        "operatingSystem", CpuOpenBlasQualification.OperatingSystem.MACOS)));
        add(mutations, original, "qualification.targetFingerprint.machine", "qualification",
                with(qualification, "targetFingerprint", with(target, "machine",
                        CpuOpenBlasQualification.Machine.AARCH64)));
        add(mutations, original, "qualification.sessionCredential", "qualification",
                new CpuOpenBlasTuningBatch.QualificationScope(
                        CpuOpenBlasRouteSelectorTest.qualification()));
    }

    private static void addPersistentIdentityMutation(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original,
            CpuOpenBlasTuningBatch.QualificationScope scope,
            CpuOpenBlasQualification.PersistentIdentity identity, String label,
            String identityComponent, Object value) {
        var changedIdentity = with(identity, identityComponent, value);
        var changedScope = with(scope, "persistentIdentity", Optional.of(changedIdentity));
        add(mutations, original, "qualification.persistentIdentity." + label,
                "qualification", changedScope);
    }

    private static void assertQualificationConstructionRejections(
            CpuOpenBlasTuningBatch.QualificationScope qualification) {
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification, "scope",
                        CpuOpenBlasQualification.Scope.PERSISTENT_BINARY),
                "qualification.scope cannot differ independently from evidence presence");
        var binary = new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256", "a".repeat(64),
                64, CpuOpenBlasQualification.ExecutableFormat.ELF_64,
                CpuOpenBlasQualification.Machine.X86_64);
        var persistent = new CpuOpenBlasQualification.PersistentIdentity(1,
                qualification.targetFingerprint(), CpuOpenBlasQualification.REQUIRED_SYMBOLS,
                CpuOpenBlasQualification.BlasIntAbi.C_INT_32,
                CpuOpenBlasQualification.NUMERICAL_CASE_VERSION, binary);
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification, "persistentIdentity", Optional.of(persistent)),
                "qualification.persistentIdentity cannot be present in session-only scope");
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification, "sessionCredential", Optional.empty()),
                "qualification.sessionCredential cannot be absent in session-only scope");
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification.targetFingerprint(), "schemaVersion", 2),
                "target schema has one supported value, so inequality is impossible");
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification.targetFingerprint(), "addressWidthBits", 32),
                "target address width is schema-locked to 64 bits");
        assertThrows(IllegalArgumentException.class,
                () -> with(qualification.targetFingerprint(), "byteOrder", ByteOrder.BIG_ENDIAN),
                "target byte order is schema-locked to little endian");
    }

    private static void addBoundary(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, int index, String label,
            String component, Object value) {
        var boundaries = new ArrayList<>(original.boundaries());
        boundaries.set(index, with(boundaries.get(index), component, value));
        add(mutations, original, label, "boundaries", boundaries);
    }

    private static void addLongComponents(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, String workloadComponent,
            Object nested, String... components) {
        for (String component : components) {
            long current = (long) component(nested, component);
            add(mutations, original, workloadComponent + "." + component, workloadComponent,
                    with(nested, component, current + 1));
        }
    }

    private static void addOptionalLongComponents(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, String workloadComponent,
            Object nested, String... components) {
        addOptionalLongComponents(mutations, original, workloadComponent, nested,
                changed -> changed, components);
    }

    private static void addOptionalLongComponents(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, String labelPrefix,
            Object nested, java.util.function.UnaryOperator<Object> outer, String... components) {
        String workloadComponent = labelPrefix.startsWith("openBlasThreadCandidates")
                ? "openBlasThreadCandidates" : labelPrefix;
        for (String component : components) {
            OptionalLong current = (OptionalLong) component(nested, component);
            Object changed = outer.apply(with(nested, component,
                    OptionalLong.of(current.orElseThrow() + 1)));
            add(mutations, original, labelPrefix + "." + component, workloadComponent, changed);
        }
    }

    private static void add(List<WorkloadMutation> mutations,
            CpuOpenBlasTuningBatch.WorkloadSignature original, String label,
            String component, Object value) {
        mutations.add(new WorkloadMutation(label, with(original, component, value)));
    }

    private static void assertCompatibilityMissAndSafeFallback(CpuPartitionAnalysisInputs inputs,
            CpuPartitionPreparationPlan heuristic, CpuOpenBlasTuningBatch batch,
            WorkloadMutation mutation) {
        var portable = batch.candidates().getFirst();
        var originalDecision = new CpuOpenBlasTuningDecision(batch.schemaVersion(),
                batch.workload(), portable.identity());
        var changedBatch = new CpuOpenBlasTuningBatch(batch.schemaVersion(), mutation.workload(),
                List.of(portable));
        assertAll(mutation.component(),
                () -> assertNotEquals(batch.workload(), mutation.workload(),
                        mutation.component() + " must participate in workload equality"),
                () -> assertTrue(originalDecision.match(changedBatch).isEmpty(),
                        mutation.component() + " must reject a decision carrying the original workload"));

        var incompatible = new CpuOpenBlasTuningDecision(batch.schemaVersion(),
                mutation.workload(), portable.identity());
        var fallback = analyze(replace(inputs, inputs.cpuHardwareIdentity(), inputs.workloadCohort(),
                Optional.of(incompatible), inputs.portableExecution().availableParallelism()));
        assertAll(mutation.component() + " fallback",
                () -> assertEquals(heuristic.route(), fallback.route()),
                () -> assertEquals(heuristic.selectedOpenBlasTuningCandidate(),
                        fallback.selectedOpenBlasTuningCandidate()),
                () -> assertEquals(heuristic.openBlasPlan().map(CpuOpenBlasRoutePlan::representation),
                        fallback.openBlasPlan().map(CpuOpenBlasRoutePlan::representation)));
    }

    private static void assertConstructionRejected(
            CpuOpenBlasTuningBatch.WorkloadSignature workload, String component, Object value,
            String message) {
        assertThrows(IllegalArgumentException.class, () -> with(workload, component, value), message);
    }

    private static void assertClosedEnum(Object[] values, String component) {
        assertEquals(1, values.length,
                component + " has one schema value, so valid inequality is impossible");
    }

    private static void assertNestedCompatibilityComponentInventories() {
        assertRecordComponents(CpuOpenBlasTuningBatch.BoundarySignature.class, "dataType",
                "shape", "layout", "carrier", "storage", "access");
        assertEquals(List.of("nativeSegment", "byteAlignment"), java.util.Arrays.stream(
                        CpuPartitionAnalysisInputs.BoundaryStorageFact.class.getRecordComponents())
                .map(RecordComponent::getName).toList(),
                "BoundaryStorageFact components must remain exactly nativeSegment and byteAlignment");
        assertRecordComponents(CpuOpenBlasTuningBatch.HardwareIdentity.class, "schemaVersion",
                "architecture", "vendor", "model", "features");
        assertRecordComponents(CpuOpenBlasTuningBatch.WorkloadCohort.class, "schemaVersion",
                "cohortId", "expectedRunCount");
        assertRecordComponents(CpuOpenBlasTuningBatch.QualificationScope.class, "scope",
                "targetFingerprint", "persistentIdentity", "sessionCredential");
        assertRecordComponents(CpuOpenBlasQualification.TargetFingerprint.class, "schemaVersion",
                "operatingSystem", "machine", "addressWidthBits", "byteOrder");
        assertRecordComponents(CpuOpenBlasQualification.PersistentIdentity.class,
                "qualificationSchemaVersion", "targetFingerprint", "requiredSymbols",
                "blasIntAbi", "numericalCaseVersion", "binaryIdentity");
        assertRecordComponents(CpuOpenBlasQualification.BinaryIdentity.class, "schemaVersion",
                "digestAlgorithm", "sha256", "byteLength", "executableFormat", "machine");
        assertRecordComponents(CpuPartitionAnalysisInputs.PortableExecutionConfig.class,
                "computePreference", "configuredMaximumParallelism", "availableParallelism",
                "minimumElementsPerWorker");
        assertRecordComponents(CpuPartitionPreparationPlan.ExecutionStrategy.class, "compute",
                "orchestration");
        assertRecordComponents(CpuPartitionAnalysisInputs.MaterializationPolicy.class, "enabled",
                "copyFixedCostUnits", "copyCostUnitsPerElement",
                "directKernelCostUnitsPerElement", "contiguousKernelCostUnitsPerElement",
                "expectedRunCount", "maximumAdditionalBytes", "minimumNetBenefitCostUnits",
                "minimumBenefitBasisPoints");
        assertRecordComponents(CpuPartitionAnalysisInputs.CostTerms.class, "fixedCostUnits",
                "costUnitsPerOutput", "costUnitsPerMac");
        assertRecordComponents(CpuPartitionAnalysisInputs.RepresentationCostTerms.class,
                "workspaceAllocationAndBindingFixed", "workspaceCostPerByte", "copyInFixed",
                "copyInPerElement", "copyOutFixed", "copyOutPerElement");
        assertRecordComponents(
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate.class,
                "threadCount", "openBlasCosts");
        assertRecordComponents(CpuAccessPlan.class, "accessKind", "regime", "iterationRank",
                "axisRoles", "contiguousSuffix");
        assertRecordComponents(CpuAccessPlan.Binding.class, "plan", "extents",
                "baseElementOffset", "effectiveStrides", "elementCount", "start", "end",
                "referencedElementSpan", "startCoordinates", "startAddress",
                "accessedElementStart", "accessedElementEnd");
    }

    private static void assertRecordComponents(Class<?> recordType, String... expected) {
        assertEquals(List.of(expected), java.util.Arrays.stream(recordType.getRecordComponents())
                .map(RecordComponent::getName).toList(), recordType.getSimpleName());
    }

    private static Object component(Object record, String name) {
        try {
            for (RecordComponent component : record.getClass().getRecordComponents()) {
                if (component.getName().equals(name)) return component.getAccessor().invoke(record);
            }
            throw new AssertionError("unknown record component " + name);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> T with(T record, String name, Object replacement) {
        try {
            RecordComponent[] components = record.getClass().getRecordComponents();
            Class<?>[] types = new Class<?>[components.length];
            Object[] values = new Object[components.length];
            boolean found = false;
            for (int index = 0; index < components.length; index++) {
                types[index] = components[index].getType();
                values[index] = components[index].getAccessor().invoke(record);
                if (components[index].getName().equals(name)) {
                    values[index] = replacement;
                    found = true;
                }
            }
            if (!found) throw new AssertionError("unknown record component " + name);
            return (T) record.getClass().getDeclaredConstructor(types).newInstance(values);
        } catch (InvocationTargetException failure) {
            if (failure.getCause() instanceof RuntimeException runtime) throw runtime;
            if (failure.getCause() instanceof Error error) throw error;
            throw new AssertionError(failure.getCause());
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static CpuPartitionPreparationPlan analyze(CpuPartitionAnalysisInputs inputs) {
        return new CpuPartitionPreparer().analyze(CpuOpenBlasRouteSelectorTest.context(
                DataType.FLOAT32, dense(io.github.pho001.synaptik.model.shape.Shape.of(2, 3)),
                dense(io.github.pho001.synaptik.model.shape.Shape.of(3, 4)),
                dense(io.github.pho001.synaptik.model.shape.Shape.of(2, 4)), inputs)).plan();
    }

    private static io.github.pho001.synaptik.model.tensor.TensorDescriptor dense(
            io.github.pho001.synaptik.model.shape.Shape shape) {
        return new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static CpuPartitionAnalysisInputs.OpenBlasRouteConfig config(
            CpuOpenBlasQualification qualification,
            List<CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate> threads) {
        return new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(Optional.of(qualification),
                threads, CpuPartitionAnalysisInputs.CostTerms.complete(1_000, 1, 1),
                CpuPartitionAnalysisInputs.RepresentationCostTerms.ZERO,
                java.util.OptionalLong.of(0), java.util.OptionalInt.of(0));
    }

    private static CpuOpenBlasQualification persistentQualification(String digest, long length,
            CpuOpenBlasQualification.ExecutableFormat format,
            CpuOpenBlasQualification.Machine machine) {
        var binary = new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256", digest, length,
                format, machine);
        var target = new CpuOpenBlasQualification.TargetFingerprint(1,
                CpuOpenBlasQualification.OperatingSystem.LINUX,
                CpuOpenBlasQualification.Machine.X86_64, 64, ByteOrder.LITTLE_ENDIAN);
        return new CpuOpenBlasQualification(CpuOpenBlasQualification.Scope.PERSISTENT_BINARY,
                target, Optional.of(binary), new CpuOpenBlasQualification.SessionKey());
    }

    private static CpuPartitionAnalysisInputs replace(CpuPartitionAnalysisInputs source,
            CpuOpenBlasTuningBatch.HardwareIdentity hardware,
            CpuOpenBlasTuningBatch.WorkloadCohort cohort,
            Optional<CpuOpenBlasTuningDecision> decision, int capacity) {
        return new CpuPartitionAnalysisInputs(source.loweringManifestEnabled(),
                source.carrierPattern(), new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        source.portableExecution().computePreference(), capacity, capacity,
                        source.portableExecution().minimumElementsPerWorker()),
                source.materializationPolicy(), source.conv2dMaterializedSuffixUnit(),
                source.partialReductionEvidence(), source.boundaryStorageFacts(),
                source.openBlasRoute(), hardware, cohort, decision);
    }

    private static CpuOpenBlasTuningBatch.HardwareIdentity hardware(String architecture,
            String... features) {
        var sorted = new ArrayList<>(List.of(features));
        return new CpuOpenBlasTuningBatch.HardwareIdentity(1, architecture, "test-vendor",
                "test-model", sorted);
    }

    private static CpuOpenBlasTuningBatch.WorkloadCohort cohort(String id, long runs) {
        return new CpuOpenBlasTuningBatch.WorkloadCohort(1, id, runs);
    }
}
