package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Self-contained topology/provenance negatives for the shared cold region proof. */
public final class CpuRowMapReductionLoweringTest {
    /** Known view aliases deduplicate; separate source origins never merge by equal geometry. */
    @Test void invariantSelectionDeduplicatesViewsButPreservesDistinctOrigins() {
        var ir = candidate(DataType.FLOAT32,false,7);
        for (boolean alias : List.of(true,false)) {
            var values = new ArrayList<>(ir.arithmetic().values()); var accesses = new ArrayList<>(ir.accesses());
            var origins = new ArrayList<>(ir.viewOrigins()); var rows = new ArrayList<>(ir.rowAxes());
            values.add(new CpuKernelIr.Value(7,DataType.FLOAT32,alias ? CpuKernelIr.Value.Kind.VIRTUAL : CpuKernelIr.Value.Kind.INPUT,ir.accesses().get(1).plan()));
            values.add(new CpuKernelIr.Value(8,DataType.FLOAT32,CpuKernelIr.Value.Kind.OUTPUT,ir.accesses().get(4).plan()));
            accesses.add(ir.accesses().get(1)); accesses.add(ir.accesses().get(4));
            origins.add(alias ? 1 : 7); origins.add(8); rows.add(-1); rows.add(0);
            var stores = new ArrayList<>(ir.arithmetic().stores()); stores.add(new CpuKernelIr.Store(8,2));
            var arithmetic = new CpuKernelIr(values,ir.arithmetic().instructions(),ir.arithmetic().loop(),stores);
            var regional = new CpuRowMapReductionIr(arithmetic,List.of(2,3,4,8),List.of(ir.contractions().getFirst(),
                    ir.contractions().getLast(),new CpuRowMapReductionIr.Contraction(8,2,7)),List.of(),ir.terminalOutputs(),
                    accesses,origins,rows,List.of(4,6,8));
            assertEquals(alias ? List.of(1) : List.of(1,7),CpuRowMapReductionLowering.invariantLoadStaging(regional,
                    alias ? List.of(1) : List.of(1,7),List.of(),112,116));
        }
    }

    /** Typed cells and readiness share the same two independently bounded phase envelopes. */
    @Test void genericInvariantSelectionUsesActualViewsAndTypedJointPhaseBudgets() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var ir = candidate(type,false,7);
            assertEquals(List.of(1),CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(0,1),List.of(),112,116));
            assertEquals(List.of(1),CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(),List.of(1),112,116));
            assertTrue(CpuRowMapReductionLowering.invariantLoadStaging(candidate(type,false,0),List.of(1),List.of(),112,116).isEmpty(),
                    "a statically empty row domain has no staging consumer");
            assertEquals(List.of(),CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(0),List.of(),112,116),
                    "ROW input is not made invariant by a coincident feature size");
            assertTrue(CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1),List.of(),128,128).isEmpty());
            assertTrue(CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1),List.of(),1,1).isEmpty(),
                    "a caller cannot substitute an undercounted base");
            assertTrue(CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1),List.of(),112,126).isEmpty(),
                    "helper staging/flag must fit independently of caller");
            var selected = CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1),List.of(),112,116);
            assertThrows(UnsupportedOperationException.class,() -> selected.add(0));
        }
    }

    /** Invalid metadata is a caller error, not an optimization fallback. */
    @Test void malformedStagingScopesAreNotMistakenForAnOptimizationMiss() {
        var ir = candidate(DataType.FLOAT32,false);
        assertThrows(IllegalArgumentException.class,() -> CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1),List.of(1),112,116));
        assertThrows(IllegalArgumentException.class,() -> CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(5),List.of(),112,116));
        assertThrows(IllegalArgumentException.class,() -> CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(1,1),List.of(),112,116));
        assertThrows(NullPointerException.class,() -> CpuRowMapReductionLowering.invariantLoadStaging(ir,null,List.of(),112,116));
        assertThrows(IllegalArgumentException.class,() -> CpuRowMapReductionLowering.invariantLoadStaging(ir,List.of(),List.of(),-1,116));
    }

    /** A partial consumption set cannot authorize eager reads of an entire feature view. */
    @Test void wholeViewStagingNeverInventsUnconsumedFeatureCells() {
        var ir = candidate(DataType.FLOAT32,false);
        var accesses = new ArrayList<>(ir.accesses()); var a = accesses.get(1);
        accesses.set(1,new CpuAccessPlan.BatchBinding(a.plan(),List.of(2L,3L),a.extentBatchFactors(),0,
                List.of(3L,1L),a.strideBatchFactors()));
        var partial = new CpuRowMapReductionIr(ir.arithmetic(),ir.rowSites(),ir.contractions(),ir.sums(),
                ir.terminalOutputs(),accesses,ir.viewOrigins(),ir.rowAxes(),ir.retainedStores());
        assertTrue(CpuRowMapReductionLowering.invariantLoadStaging(partial,List.of(1),List.of(),112,116).isEmpty(),
                "only two of three view columns have requests; do not read the third");
    }

    @Test void temporalOrderMustMatchTheActualRowAndTerminalEmissionOrder() {
        var ir = candidate(DataType.FLOAT32,false); var uses = uses();
        assertTrue(CpuRowMapReductionLowering.prove(ir,uses,emptyOrder()).isPresent());
        var reversed = emptyOrder(); reversed.set(3,List.of(2));
        assertTrue(CpuRowMapReductionLowering.prove(ir,uses,reversed).isEmpty(),"independent B before A cannot emit A before B");
        var terminalBeforeRow = emptyOrder(); terminalBeforeRow.set(6,List.of(3));
        assertTrue(CpuRowMapReductionLowering.prove(ir,uses,terminalBeforeRow).isEmpty(),"finish cannot precede row work");
    }

    @Test void squareFeaturesDoNotAuthorizeRowOneProjectionOutputs() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            assertTrue(CpuRowMapReductionLowering.prove(candidate(type,false),uses(),emptyOrder()).isPresent());
            assertTrue(CpuRowMapReductionLowering.prove(candidate(type,true),uses(),emptyOrder()).isEmpty());
        }
    }

    @Test void outsideLeaveAndReentryIsRejectedEvenWithAnAcyclicFullDag() {
        var uses = uses(); uses.add(new ArrayList<>()); uses.get(2).add(7); uses.get(7).add(6);
        var order = emptyOrder(); order.add(List.of());
        assertTrue(CpuRowMapReductionLowering.prove(candidate(DataType.FLOAT32,false),uses,order).isEmpty());
    }

    @Test void terminalFeedbackUnknownAliasesAndMissingCompleteUsesRejectBeforeResources() {
        var ir = candidate(DataType.FLOAT32,false);
        var missing = uses();missing.get(2).remove(Integer.valueOf(6));
        assertTrue(CpuRowMapReductionLowering.prove(ir,missing,emptyOrder()).isEmpty());
        var roots = new ArrayList<>(ir.viewOrigins());roots.set(5,1);
        var unknownAlias = new CpuRowMapReductionIr(ir.arithmetic(),ir.rowSites(),ir.contractions(),ir.sums(),ir.terminalOutputs(),ir.accesses(),roots,ir.rowAxes(),ir.retainedStores());
        var aliasUses = uses();aliasUses.get(1).add(5);
        assertTrue(CpuRowMapReductionLowering.prove(unknownAlias,aliasUses,emptyOrder()).isEmpty());
        var feedback = new CpuRowMapReductionIr(ir.arithmetic(),ir.rowSites(),List.of(
                new CpuRowMapReductionIr.Contraction(4,6,1),ir.contractions().getLast()),ir.sums(),ir.terminalOutputs(),ir.accesses(),ir.viewOrigins(),ir.rowAxes(),ir.retainedStores());
        var feedbackUses = uses();feedbackUses.get(6).add(4);
        assertTrue(CpuRowMapReductionLowering.prove(feedback,feedbackUses,emptyOrder()).isEmpty(),"acyclic terminal-to-row use still violates single-pass availability");
        assertThrows(IllegalArgumentException.class,() -> CpuRowMapReductionLowering.prove(ir,uses(),List.of()));
        assertTrue(CpuRowMapReductionLowering.prove(candidate(DataType.BFLOAT16,false),uses(),emptyOrder()).isEmpty());
    }

    /**
     * Causal admission witness, not a reconstruction of a lost historical failure diagnostic:
     * input movement may feed dX/dW but is not a shared computed row map. The identical graph
     * becomes eligible when an actual typed ADD producer supplies both sinks.
     */
    @Test void inputMovementAloneDoesNotRetireStandaloneMatmulArtifacts() {
        var ir = candidate(DataType.FLOAT32,false);
        var movement = new CpuKernelIr(ir.arithmetic().values(),List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST,List.of(0),2),
                ir.arithmetic().instructions().getLast()),ir.arithmetic().loop(),ir.arithmetic().stores());
        var inputOnly = new CpuRowMapReductionIr(movement,ir.rowSites(),ir.contractions(),ir.sums(),ir.terminalOutputs(),
                ir.accesses(),ir.viewOrigins(),ir.rowAxes(),ir.retainedStores());
        assertTrue(CpuRowMapReductionLowering.prove(inputOnly,uses(),emptyOrder()).isEmpty());
        assertTrue(CpuRowMapReductionLowering.prove(ir,uses(),emptyOrder()).isPresent());
    }

    /**
     * Produces independent row A/B sites, one feature projection and one matrix terminal.
     * X is [2,2], so a false ROW(1) cannot be detected by numerical extent comparison.
     * @param type homogeneous represented dtype
     * @param wrongProjectionRole whether to corrupt only the feature projection's row role
     * @return immutable self-contained candidate with no graph or filesystem dependencies
     */
    public static CpuRowMapReductionIr candidate(DataType type,boolean wrongProjectionRole) {
        return candidate(type,wrongProjectionRole,2);
    }

    /**
     * Builds the same source topology at a separately fixed row count, including empty R.
     * @param type homogeneous represented dtype
     * @param wrongProjectionRole whether to corrupt the projection's row provenance
     * @param rows fixed nonnegative row count
     * @return complete immutable candidate
     */
    public static CpuRowMapReductionIr candidate(DataType type,boolean wrongProjectionRole,long rows) {
        var values = new ArrayList<CpuKernelIr.Value>(); var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int v = 0; v < 7; v++) {
            boolean output = v == 4 || v == 6;
            var plan = new CpuAccessPlan(output ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ,
                    v == 5 ? CpuAccessPlan.Regime.GENERAL_ODOMETER : CpuAccessPlan.Regime.DENSE_LINEAR,
                    2,Collections.nCopies(2,CpuAccessPlan.AxisRole.CONTIGUOUS),v == 5 ? 0 : 2);
            accesses.add(new CpuAccessPlan.BatchBinding(plan,v == 1 || v == 6 ? List.of(2L,2L)
                    : v == 5 ? List.of(2L,rows) : List.of(rows,2L),List.of(0L,0L),0,
                    v == 5 ? List.of(1L,2L) : List.of(2L,1L),List.of(0L,0L)));
            values.add(new CpuKernelIr.Value(v,type,v < 2 ? CpuKernelIr.Value.Kind.INPUT
                    : output ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.VIRTUAL,plan));
        }
        var arithmetic = new CpuKernelIr(values,List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD,List.of(0,0),2),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.MUL,List.of(0,0),3)),new CpuKernelIr.Loop("start","end"),
                List.of(new CpuKernelIr.Store(4,0),new CpuKernelIr.Store(6,1)));
        return new CpuRowMapReductionIr(arithmetic,List.of(2,3,4),List.of(
                new CpuRowMapReductionIr.Contraction(4,2,1),new CpuRowMapReductionIr.Contraction(6,5,2)),
                List.of(),List.of(6),accesses,List.of(0,1,2,3,4,0,6),
                List.of(0,-1,0,0,wrongProjectionRole ? 1 : 0,1,-1),List.of(4,6));
    }

    private static ArrayList<List<Integer>> emptyOrder() {
        return new ArrayList<>(Collections.nCopies(7,List.of()));
    }

    private static ArrayList<List<Integer>> uses() {
        return new ArrayList<>(List.of(new ArrayList<>(List.of(2,3,5)),new ArrayList<>(List.of(4)),
                new ArrayList<>(List.of(4,6)),new ArrayList<>(),new ArrayList<>(),
                new ArrayList<>(List.of(6)),new ArrayList<>()));
    }
}
