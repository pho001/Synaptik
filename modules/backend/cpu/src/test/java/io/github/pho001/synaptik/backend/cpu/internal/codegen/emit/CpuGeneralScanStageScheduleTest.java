package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Cold scheduling proof probes for genuine raw unannotated recipes. Mutated tables test
 * conservative proof boundaries only; they are never admitted as controlled derivative scans.
 */
class CpuGeneralScanStageScheduleTest {

    /**
     * Actual lowering may choose a demanded history as a next-carry cut's primary destination
     * while still updating its own final port. The negative probe adds a later old-formal read
     * without an ordinal dependency, isolating the additional physical write hazard.
     */
    @Test void historyBackedNextCarryIncludesItsAdditionalFinalDestination() {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("historyAlias");
            var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            var sequence = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(2),b,new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            var weights = io.github.pho001.synaptik.model.shape.Shape.of(2,2);
            var h = CpuSymbolicLengthScanFixture.tensor(type,false,row); var x = CpuSymbolicLengthScanFixture.tensor(type,false,row);
            var w = CpuSymbolicLengthScanFixture.tensor(type,false,weights); var next = h.add(x); var emitted = x.matmul(w);
            var scan = io.github.pho001.synaptik.model.tensor.Scan.apply(
                    List.of(CpuSymbolicLengthScanFixture.tensor(type,false,row)),List.of(CpuSymbolicLengthScanFixture.tensor(type,false,sequence)),
                    List.of(CpuSymbolicLengthScanFixture.tensor(type,false,weights)),
                    CpuSymbolicLengthScanFixture.tensor(DataType.INT64,false,io.github.pho001.synaptik.model.shape.Shape.ofDimensions(b)),1,List.of(OptionalInt.of(0)),
                    new io.github.pho001.synaptik.model.tensor.ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(next,emitted)),false);
            var roots = new ArrayList<>(scan.finalCarry()); roots.addAll(scan.outputs());
            var ir = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.analyze(
                    io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.query(roots)).ir();
            var plan = ir.stages().orElseThrow();
            assertEquals(0,plan.carryPorts().getFirst()); assertTrue(plan.carryOffsets().getFirst() < 0);
            var cut = plan.cuts().stream().filter(c -> c.value() == ir.nextOrdinals().getFirst()).findFirst().orElseThrow();
            assertTrue(cut.port() >= ir.carryCount(),"genuine history-backed next carry, with a second final-port write");
            assertNotEquals(plan.order(),CpuGeneralScanStageSchedule.order(ir),"independent actual stages may reorder");
            var stages = new ArrayList<>(plan.order());
            int matrix = java.util.stream.IntStream.range(0,stages.size()).filter(i -> stages.get(i).right() >= 0).findFirst().orElseThrow();
            stages.set(matrix,input(stages.get(matrix),0));
            var changed = with(ir,stages,plan.cuts(),plan.carryOffsets());
            assertSame(changed.stages().orElseThrow().order(),CpuGeneralScanStageSchedule.order(changed),"additional final-port zero write conflicts with formal-zero read");
        }
    }


    /**
     * Changes only an ordered special-stage operand for a cold proof probe.
     * @param stage original special stage
     * @param input replacement source ordinal
     * @return new stage retaining all other producing facts
     */
    private static CpuGeneralScanIr.Stage input(CpuGeneralScanIr.Stage stage,int input) {
        return new CpuGeneralScanIr.Stage(stage.values(),stage.output(),input,stage.axes(),stage.reduction(),stage.epilogue(),stage.right(),stage.sumToShape());
    }

    /**
     * Replaces only the scoped scheduling/resource tables of a cold proof probe.
     * @param ir original combined reverse IR
     * @param order replacement immutable-stage sequence, copied by the IR
     * @param cuts replacement assigned scratch/port facts, copied by the IR
     * @param offsets replacement carry-bank offset constants, copied by the IR
     * @return independent IR; no production resource assignment or executable is changed
     */
    private static CpuGeneralScanIr with(CpuGeneralScanIr ir,List<CpuGeneralScanIr.Stage> order,List<CpuGeneralScanIr.Cut> cuts,List<Long> offsets) {
        var p = ir.stages().orElseThrow();
        var changed = new CpuGeneralScanIr.Stages(p.views(),order,cuts,offsets,p.carryPorts(),p.scratchBytes(),p.carryOffsetFactors(),
                p.scratchBytesPerBatch(),p.symbolic(),p.sequenceStrides(),p.rowAxes(),p.terminal(),p.rowCarryCount(),p.lengthRead(),p.batch());
        return new CpuGeneralScanIr(ir.arithmetic(),ir.formalOrdinals(),ir.actualReads(),ir.scalarReads(),ir.nextOrdinals(),ir.portOrdinals(),
                ir.demandedPorts(),ir.carries(),ir.boundaryAccesses(),ir.carryCount(),ir.sequenceCount(),ir.reverse(),ir.scratchBanks(),Optional.of(changed));
    }
}
