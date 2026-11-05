package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Identity-only DTO checks; constructed declarations do not assert native implementation availability. */
final class CpuOneDnnCandidateIdentityTest {
    @Test
    void identityIsFixedSizeAndBindsPolicySessionApiProjectionAndResources() {
        var baseline = route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 0, "implementation");
        String fingerprint = CpuOneDnnCandidateIdentity.fingerprint(CpuOneDnnRoutePlan.Api.AUTO, baseline);
        assertEquals(87, fingerprint.length());
        assertTrue(fingerprint.matches("onednn-finite-1:sha256:[0-9a-f]{64}"));
        assertEquals(fingerprint, CpuOneDnnCandidateIdentity.fingerprint(CpuOneDnnRoutePlan.Api.AUTO,
                route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 0, "implementation")));
        assertNotEquals(fingerprint, CpuOneDnnCandidateIdentity.fingerprint(CpuOneDnnRoutePlan.Api.PRIMITIVE, baseline));
        for (var changed : List.of(
                route("other-session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 0, "implementation"),
                route("session", CpuOneDnnRoutePlan.Api.GRAPH, 0, 0, "implementation"),
                route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 4, 0, "implementation"),
                route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 64, "implementation"),
                route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 0, "other-implementation"))) {
            assertNotEquals(fingerprint, CpuOneDnnCandidateIdentity.fingerprint(CpuOneDnnRoutePlan.Api.AUTO, changed));
        }
    }

    @Test
    void oversizedSourceIsRejectedWithoutConstructingAnAggregateDescriptorString() {
        var oversized = route("session", CpuOneDnnRoutePlan.Api.PRIMITIVE, 0, 0, "x".repeat(131_072));
        var failure = assertThrows(IllegalArgumentException.class,
                () -> CpuOneDnnCandidateIdentity.fingerprint(CpuOneDnnRoutePlan.Api.AUTO, oversized));
        assertTrue(failure.getMessage().contains("identity source exceeds one MiB"));
    }

    private static CpuOneDnnRoutePlan route(String session, CpuOneDnnRoutePlan.Api api,
            long origin, long scratch, String implementation) {
        var tensor = new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, List.of(2L), List.of(1L));
        var spec = new CpuOneDnnKernelSpec.Unary(CpuOneDnnKernelSpec.UnaryAlgorithm.ABS, tensor, tensor);
        var projection = new CpuOneDnnProjection(spec, List.of(origin, 0L), List.of(0), 1);
        var ingress = new PreparationResourceRequirement.Workspace(1, tensor.byteSize(), 64);
        var invocation = new PreparationResourceRequirement.Workspace(2, scratch, 64);
        var unit = new CpuOneDnnRoutePlan.Unit(0, projection, api, scratch, implementation,
                List.of(new CpuOneDnnRoutePlan.InputCopy(List.of(0), ingress, true)), invocation);
        return new CpuOneDnnRoutePlan(session, List.of(unit), 0, 0);
    }
}
