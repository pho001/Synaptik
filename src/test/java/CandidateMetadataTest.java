import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tuning.candidate.Candidate;
import tuning.candidate.CandidateIdentityFingerprint;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;
import tuning.candidate.ExecutableProfileFingerprint;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CandidateMetadataTest {
    @Test
    void metadataSurvivesCandidateConstruction() {
        CandidateMetadata metadata = CandidateMetadata.graphStandard("current")
                .withAttribute("graphParameter", "CURRENT_GRAPH_POLICY");

        Candidate candidate = new Candidate(
                "graphPolicy=current",
                profile(),
                CandidateKind.GRAPH_STANDARD,
                metadata
        );

        assertEquals(CandidateKind.GRAPH_STANDARD, candidate.kind());
        assertEquals("graph-autotune", candidate.metadata().candidateSpaceId());
        assertEquals("CURRENT_GRAPH_POLICY", candidate.metadata().attributes().get("graphParameter"));
        assertTrue(candidate.metadata().productionEligible());
    }

    @Test
    void metadataRestoresReservedFieldsAndAttributesFromMap() {
        CandidateMetadata metadata = CandidateMetadata.fromMap(Map.of(
                "candidateSpaceId", "graph-autotune",
                "candidateSpaceVersion", "1",
                "parameterFamily", "graphPolicy",
                "parameterVariant", "current",
                "graphAutotuneMode", "STANDARD",
                "runtimeFrozen", "true",
                "graphPolicyMutated", "false",
                "productionEligible", "true",
                "graphParameter", "CURRENT_GRAPH_POLICY"
        ));

        assertEquals("graph-autotune", metadata.candidateSpaceId());
        assertEquals("current", metadata.parameterVariant());
        assertEquals("STANDARD", metadata.graphAutotuneMode());
        assertEquals("CURRENT_GRAPH_POLICY", metadata.attributes().get("graphParameter"));
        assertTrue(metadata.runtimeFrozen());
        assertTrue(metadata.productionEligible());
    }

    @Test
    void executableAndCandidateIdentityFingerprintsAreSeparate() {
        var profile = profile();
        Candidate standard = new Candidate(
                "graphPolicy=current",
                profile,
                CandidateKind.GRAPH_STANDARD,
                CandidateMetadata.graphStandard("current")
        );
        Candidate research = new Candidate(
                "piecewise=current",
                profile,
                CandidateKind.GRAPH_RESEARCH,
                CandidateMetadata.graphResearch("PIECEWISE_LOWERING", "piecewise=current", false)
        );

        assertEquals(ExecutableProfileFingerprint.of(standard), ExecutableProfileFingerprint.of(research));
        assertTrue(!CandidateIdentityFingerprint.of(standard).equals(CandidateIdentityFingerprint.of(research)));
    }

    private static ExecutionProfile profile() {
        return new ExecutionProfile(
                "candidate",
                "candidate",
                tensor.DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.optimizer.OptimizerConfig.noOptimization(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}
