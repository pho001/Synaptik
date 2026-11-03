package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

/** Shared actual-Compiler recurrence fixture for numerical, lifecycle and stale-artifact tests. */
public final class CpuConditionalScanFixture {
    private CpuConditionalScanFixture() {}

    /**
     * Compiles the exact T4/B7/F3/H2 conditional fixture, joint nonuniform seed inputs and
     * original-forward/actual-reverse traversal; it performs no timing or physical allocation.
     * @param type homogeneous admitted floating type
     * @param length whether the recurrence is length-aware TANH, otherwise h+x@W
     * @return actual combined compile artifact with two independently qualified Scan occurrences
     */
    public static CompileArtifacts conditionalFamily(DataType type,boolean length) {
        Tensor h = CpuGeneralScanLoweringTest.tensor(type,true,7,2), x = CpuGeneralScanLoweringTest.tensor(type,true,7,3);
        Tensor w = CpuGeneralScanLoweringTest.tensor(type,true,3,2), u = CpuGeneralScanLoweringTest.tensor(type,true,2,2);
        Tensor bias = CpuGeneralScanLoweringTest.tensor(type,true,2);
        Tensor next = length ? x.matmul(w).add(h.matmul(u)).add(bias).tanh() : h.add(x.matmul(w));
        Tensor initial = CpuGeneralScanLoweringTest.tensor(type,true,7,2), seq = CpuGeneralScanLoweringTest.tensor(type,true,4,7,3);
        Tensor wx = CpuGeneralScanLoweringTest.tensor(type,true,3,2), wh = CpuGeneralScanLoweringTest.tensor(type,true,2,2), offset = CpuGeneralScanLoweringTest.tensor(type,true,2);
        var actual = length ? List.of(wx,wh,offset) : List.of(wx);
        var formal = length ? List.of(w,u,bias) : List.of(w);
        var body = new ScanBody(List.of(h),List.of(x),formal,List.of(next),List.of(next));
        var scan = length ? Scan.apply(List.of(initial),List.of(seq),actual,CpuGeneralScanLoweringTest.tensor(DataType.INT64,false,7),
                1,List.of(OptionalInt.of(0)),body,false) : Scan.apply(List.of(initial),List.of(seq),actual,body,false);
        var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        var wrt = new ArrayList<>(List.of(initial,seq)); wrt.addAll(actual);
        return CpuGeneralScanCapabilityTest.compile(roots,Optional.of(CpuGeneralScanCapabilityTest.request(roots,wrt)),false,new CpuCapabilityProvider());
    }
}
