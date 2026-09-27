package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.EnumSet;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class MetalOperationRouteCatalogTest {
    @Test
    void exhaustivelyDescribesAllRegisteredKindsAndStructuralExecutionSet() {
        MetalMpsGraphProgram.NodeKind[] kinds = MetalMpsGraphProgram.NodeKind.values();
        assertEquals(115, kinds.length);

        int direct = 0;
        int composed = 0;
        int unavailable = 0;
        int customAvailable = 0;
        int customPending = 0;
        int customUnavailableWithProof = 0;
        int executable = 0;
        for (int index = 0; index < kinds.length; index++) {
            MetalMpsGraphProgram.NodeKind kind = kinds[index];
            assertEquals(index + 1, kind.wireIdentity());
            MetalOperationRouteCatalog.Entry entry = MetalOperationRouteCatalog.entry(kind);
            assertNotNull(entry);
            assertNotNull(entry.mpsGraphReason());
            assertNotNull(entry.customKernelReason());
            switch (entry.mpsGraphState()) {
                case DIRECT -> direct++;
                case COMPOSED -> composed++;
                case UNAVAILABLE -> unavailable++;
            }
            switch (entry.customKernelState()) {
                case AVAILABLE -> customAvailable++;
                case PENDING -> customPending++;
                case UNAVAILABLE_WITH_PROOF -> customUnavailableWithProof++;
            }
            if (kind.executable()) executable++;
            assertEquals(
                    kind == MetalMpsGraphProgram.NodeKind.NEG
                            || kind == MetalMpsGraphProgram.NodeKind.MATMUL
                            || kind.isCustomProgramOperation(),
                    entry.customKernelState()
                            == MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                    kind.name());
        }

        assertEquals(75, direct);
        assertEquals(35, composed);
        assertEquals(5, unavailable);
        assertEquals(58, customAvailable);
        assertEquals(57, customPending);
        assertEquals(0, customUnavailableWithProof);
        assertEquals(99, executable);
        assertEquals(16, kinds.length - executable);
        assertThrows(NullPointerException.class, () -> MetalOperationRouteCatalog.entry(null));
    }

    @Test
    void closedReasonsAndStructuralStatesRemainDescriptiveRatherThanAuthoritative() {
        Set<MetalOperationRouteCatalog.MpsGraphReason> usedMpsGraphReasons =
                Arrays.stream(MetalMpsGraphProgram.NodeKind.values())
                        .map(MetalOperationRouteCatalog::entry)
                        .map(MetalOperationRouteCatalog.Entry::mpsGraphReason)
                        .collect(Collectors.toCollection(
                                () -> EnumSet.noneOf(MetalOperationRouteCatalog.MpsGraphReason.class)));
        Set<MetalOperationRouteCatalog.CustomKernelReason> usedCustomReasons =
                Arrays.stream(MetalMpsGraphProgram.NodeKind.values())
                        .map(MetalOperationRouteCatalog::entry)
                        .map(MetalOperationRouteCatalog.Entry::customKernelReason)
                        .collect(Collectors.toCollection(
                                () -> EnumSet.noneOf(MetalOperationRouteCatalog.CustomKernelReason.class)));
        assertEquals(EnumSet.allOf(MetalOperationRouteCatalog.MpsGraphReason.class),
                usedMpsGraphReasons);
        assertEquals(EnumSet.allOf(MetalOperationRouteCatalog.CustomKernelReason.class),
                usedCustomReasons);

        assertCatalog(
                MetalMpsGraphProgram.NodeKind.NEG,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_ARITH,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_NEG);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.GT,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_PRED,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0052);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.IS_FINITE,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_PRED,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0057);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.FLOOR,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_ARITH,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0058);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.TENSOR_POW,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_ARITH,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_POWER);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_MSE,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_AGGREGATE);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_SCALAR,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_POINT);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.RECIPROCAL,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_UNARY,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_POINT);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.LOG,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_ARITH,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_ELEMENTARY);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_UNARY,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_ELEMENTARY);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.SILU,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_UNARY,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_RECURSIVE_SITES);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.CAST,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_CAST,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0059);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.MATMUL,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_MATMUL,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0061);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                MetalOperationRouteCatalog.MpsGraphReason.MC_LOGSUMEXP,
                MetalOperationRouteCatalog.CustomKernelState.PENDING,
                MetalOperationRouteCatalog.CustomKernelReason.CP_AGGREGATE);
        assertCatalog(
                MetalMpsGraphProgram.NodeKind.SORT,
                MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                MetalOperationRouteCatalog.MpsGraphReason.MD_SORT,
                MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                MetalOperationRouteCatalog.CustomKernelReason.CA_0063);
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.ARGSORT,
                MetalMpsGraphProgram.NodeKind.TOP_K,
                MetalMpsGraphProgram.NodeKind.ARG_MAX,
                MetalMpsGraphProgram.NodeKind.ARG_MIN)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                    MetalOperationRouteCatalog.MpsGraphReason.MC_INDEX64,
                    MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                    MetalOperationRouteCatalog.CustomKernelReason.CA_0063);
        }
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                MetalMpsGraphProgram.NodeKind.CONV3D)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                    MetalOperationRouteCatalog.MpsGraphReason.MD_CONV_BASE,
                    MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                    MetalOperationRouteCatalog.CustomKernelReason.CA_0064);
        }
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.DIRECT,
                    MetalOperationRouteCatalog.MpsGraphReason.MD_POOL2D,
                    MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                    MetalOperationRouteCatalog.CustomKernelReason.CA_0064);
        }
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.MAX_POOL3D,
                MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.COMPOSED,
                    MetalOperationRouteCatalog.MpsGraphReason.MC_WINDOW3D,
                    MetalOperationRouteCatalog.CustomKernelState.AVAILABLE,
                    MetalOperationRouteCatalog.CustomKernelReason.CA_0064);
        }
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.DROPOUT,
                MetalMpsGraphProgram.NodeKind.INITIAL_STATE)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.UNAVAILABLE,
                    MetalOperationRouteCatalog.MpsGraphReason.MU_RNG,
                    MetalOperationRouteCatalog.CustomKernelState.PENDING,
                    MetalOperationRouteCatalog.CustomKernelReason.CP_STATE);
        }
        for (MetalMpsGraphProgram.NodeKind kind : Set.of(
                MetalMpsGraphProgram.NodeKind.RNN_TANH,
                MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER,
                MetalMpsGraphProgram.NodeKind.LSTM)) {
            assertCatalog(
                    kind,
                    MetalOperationRouteCatalog.MpsGraphState.UNAVAILABLE,
                    MetalOperationRouteCatalog.MpsGraphReason.MU_RECURRENT,
                    MetalOperationRouteCatalog.CustomKernelState.PENDING,
                    MetalOperationRouteCatalog.CustomKernelReason.CP_STATE);
        }

        assertTrue(MetalMpsGraphProgram.NodeKind.CAST.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.VARIANCE.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.L1_NORM.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.L2_NORM.executable());
        assertFalse(MetalMpsGraphProgram.NodeKind.DROPOUT.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.NEG.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.GT.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.FLOOR.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.SCALAR_ADD.executable());
        assertTrue(MetalMpsGraphProgram.NodeKind.RECIPROCAL.executable());
    }

    private static void assertCatalog(
            MetalMpsGraphProgram.NodeKind kind,
            MetalOperationRouteCatalog.MpsGraphState mpsGraphState,
            MetalOperationRouteCatalog.MpsGraphReason mpsGraphReason,
            MetalOperationRouteCatalog.CustomKernelState customKernelState,
            MetalOperationRouteCatalog.CustomKernelReason customKernelReason) {
        MetalOperationRouteCatalog.Entry entry = MetalOperationRouteCatalog.entry(kind);
        assertSame(mpsGraphState, entry.mpsGraphState());
        assertSame(mpsGraphReason, entry.mpsGraphReason());
        assertSame(customKernelState, entry.customKernelState());
        assertSame(customKernelReason, entry.customKernelReason());
    }
}
