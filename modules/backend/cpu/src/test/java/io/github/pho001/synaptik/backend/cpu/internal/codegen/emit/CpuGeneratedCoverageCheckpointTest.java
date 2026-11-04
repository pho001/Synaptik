package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import java.util.Map;
import java.util.TreeMap;

/** Shared live diagnostic inventory; frozen byte snapshots are not current behavior tests. */
class CpuGeneratedCoverageCheckpointTest {
    private static final String HEADER = "owner-id\tevidence-key\toperation-form\tlowered-ir-family"
            + "\tdescriptor-alias-roles\tordered-boundary-values\tvirtual-values\tunit-count"
            + "\taffine-address-pairs\tordered-carriers\tlayout-access\trequested-strategy"
            + "\tselected-strategy\tmaterialization-candidate\tmaterializations-selected"
            + "\tprovider-supported\tpreparer-admitted\tprepared-ir-structural-key"
            + "\tgenerated-entry-descriptor\tclass-sha256\tnormalized-body-key\toutcome\treason"
            + "\tordinary-production-selection\tfallback-disposition\toracle-disposition\tperformance-disposition\n";

    /**
     * Executes the current canonical fixtures without pinning generated class bytes.
     * Historical inventory resources remain frozen evidence, not a golden for current code.
     *
     * @return complete current tab-separated fixture facts with validated disposition references
     * @throws Exception if fixture preparation, generation or execution fails
     */
    static String liveInventory() throws Exception {
        Map<String, CpuGeneratedCoverageEvidenceRegistry.Record> observed = executeCanonicalFixtures();
        CpuGeneratedCoverageDispositionRegistry.verifyCanonicalRegistry(observed);
        return serialize(observed);
    }


    /** Runs each finite owner matrix once; clearing makes this independent of JUnit order. */
    private static Map<String, CpuGeneratedCoverageEvidenceRegistry.Record> executeCanonicalFixtures()
            throws Exception {
        CpuGeneratedCoverageEvidenceRegistry.clear();
        new CpuGeneratedDirectEvidenceClosureTest().pointwiseCandidateUniverseClosesRequestedCarrierLayoutAndStrategyAxes();
        for (var form : CpuScalarImmediateClampMatrixOracle.forms()) {
            CpuScalarImmediateClampMatrixOracle.formArtifact(form);
        }
        for (int index = 0; index < CpuAffineGeneratedCoverageFixtureTest.coverageFixtureContexts().size(); index++) {
            var context = CpuAffineGeneratedCoverageFixtureTest.coverageFixtureContexts().get(index);
            CpuGeneratedCoverageEvidenceRegistry.generated("affine-matrix:" + index, context,
                    new CpuPartitionPreparer().analyze(context).plan());
        }
        var ordinary = new CpuOrdinaryNonPointwiseGeneratedMatrixTest();
        ordinary.everyModelValidOrdinaryBaseRoleTypeAndRuntimeRequestUsesProductionBoundaries();
        var specialized = new CpuSpecializedGeneratedMatrixTest();
        specialized.everyAdmittedArgMaskedAdvancedAndSoftmaxFormGeneratesAcrossRequests();
        specialized.everyAdmittedTrailingAndBatchNormRoleCombinationGeneratesAcrossRequests();
        specialized.everyCurrentDirectMatmulFormGeneratesAcrossRequests();
        specialized.everyCurrentDirectConvolutionAndPoolingFormGeneratesAcrossRequests();
        specialized.everyCurrentDirectAttentionAndLossRoleFormGeneratesAcrossRequests();
        var composition = new CpuOneDimensionalCompositionGeneratedMatrixTest();
        composition.everyModelValidConv1dCompositionRoleAndGeometryGeneratesAcrossRequests();
        composition.everyModelValidPool1dCompositionTypeAndGeometryGeneratesAcrossRequests();
        for (var candidate : CpuGeneratedDirectEvidenceClosureTest.rejectedPointwiseCandidates()) {
            var fixture = candidate.coverageFixture();
            CpuGeneratedCoverageEvidenceRegistry.rejectedByProvider("rejected:pointwise:" + fixture.id(),
                    fixture.operationForm(), fixture.operation(), fixture.inputs(), fixture.outputs(), fixture.reason());
        }
        for (var fixture : CpuOrdinaryNonPointwiseGeneratedMatrixTest.rejectedProviderFixtures()) {
            CpuGeneratedCoverageEvidenceRegistry.rejectedByProvider("rejected:ordinary:" + fixture.id(),
                    fixture.operationForm(), fixture.operation(), fixture.inputs(), fixture.outputs(), fixture.reason());
        }
        for (var fixture : CpuSpecializedGeneratedMatrixTest.rejectedProviderFixtures()) {
            CpuGeneratedCoverageEvidenceRegistry.rejectedByProvider("rejected:specialized:" + fixture.id(),
                    fixture.operationForm(), fixture.operation(), fixture.inputs(), fixture.outputs(), fixture.reason());
        }
        for (var fixture : CpuGeneratedCoverageUnsupportedBoundaryTest.rejectedFixtures()) {
            CpuGeneratedCoverageEvidenceRegistry.rejectedByProvider("rejected:unsupported:" + fixture.id(),
                    fixture.operation().kind().name(), fixture.operation(), fixture.inputs(), fixture.outputs(), fixture.reason());
        }
        for (var fixture : CpuOneDimensionalCompositionGeneratedMatrixTest.rejectedPoolFixtures()) {
            CpuGeneratedCoverageEvidenceRegistry.rejectedByPreparer("rejected:composition:" + fixture.id(),
                    fixture.operationForm(), fixture.context(), fixture.reason());
        }
        return new TreeMap<>(CpuGeneratedCoverageEvidenceRegistry.snapshot());
    }


    private static String serialize(Map<String, CpuGeneratedCoverageEvidenceRegistry.Record> records) {
        StringBuilder text = new StringBuilder(HEADER);
        records.forEach((owner, record) -> text.append(String.join("\t", owner, record.evidenceKey(),
                record.operationForm(), record.loweredForm(), record.descriptorRoles(), record.boundaryIds(),
                record.virtualValues(), Integer.toString(record.unitCount()), Integer.toString(record.affineAddressPairCount()),
                record.orderedCarrierTypes(), record.layoutAccessRegime(), record.requestedStrategy(),
                record.selectedStrategy(), Boolean.toString(record.materializationCandidate()),
                Integer.toString(record.materializationsSelected()), Boolean.toString(record.providerSupported()),
                Boolean.toString(record.preparerAdmitted()), record.preparedIrStructuralKey(),
                record.generatedEntryDescriptor(), record.classHash(), record.normalizedBodyKey(), record.outcome(),
                record.reason(), record.ordinarySelection(), record.fallbackDisposition(), canonicalInventoryDisposition(record,
                        CpuGeneratedCoverageDispositionRegistry.Kind.ORACLE), canonicalInventoryDisposition(record,
                        CpuGeneratedCoverageDispositionRegistry.Kind.PERFORMANCE))).append('\n'));
        return text.toString();
    }

    /** Keeps the SHA-frozen inventory as an execution baseline; live promotion belongs to the registry. */
    private static String canonicalInventoryDisposition(CpuGeneratedCoverageEvidenceRegistry.Record record,
            CpuGeneratedCoverageDispositionRegistry.Kind kind) {
        return record.outcome().equals("GENERATED")
                ? kind == CpuGeneratedCoverageDispositionRegistry.Kind.ORACLE
                ? "PARTIAL_CLASSFILE_NO_CLEAN_JAVA_ORACLE" : "PARTIAL_NO_REPRESENTATIVE_BENCHMARK"
                : "N/A_NO_GENERATED_CODE";
    }

}
