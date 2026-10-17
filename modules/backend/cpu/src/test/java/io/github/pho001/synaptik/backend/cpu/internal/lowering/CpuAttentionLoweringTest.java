package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuAttentionLoweringTest {
  @Test
  void lowersDuplicateRolesAsOneAtomicSchema57RowFamily() {
    var lowered = new CpuPartitionLowering().lower(context(true));
    var ir = assertInstanceOf(CpuAttentionIr.class, lowered.portableKernelIr());
    var geometry = lowered.attentionGeometry().orElseThrow();
    assertAll(
        () -> assertEquals(List.of(0, 0, 1), ir.roleBoundaryPositions()),
        () -> assertEquals(3, lowered.boundaryValues().size()),
        () -> assertEquals(2, geometry.rowCount()),
        () -> assertEquals(8, geometry.scratchSliceBytes()),
        () -> assertTrue(lowered.virtualValues().isEmpty()));
    var plan = new CpuPartitionPreparer().analyze(context(true)).plan();
    assertAll(
        () -> assertEquals(1, plan.units().size()),
        () ->
            assertEquals(
                57, plan.units().getFirst().portablePlan().specialization().classIdentitySchema()),
        () -> assertEquals(8, plan.workspaceDeclaration().orElseThrow().byteSize()));
  }

  static PrepareContext<CpuPartitionAnalysisInputs> context(boolean duplicateQueryKey) {
    Shape q = Shape.of(2, 2), k = Shape.of(2, 2), v = Shape.of(2, 2);
    TensorDescriptor qd = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
    TensorDescriptor kd = CpuScatterLoweringTest.desc(DataType.FLOAT32, k);
    TensorDescriptor vd = CpuScatterLoweringTest.desc(DataType.FLOAT32, v);
    TensorDescriptor out = CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 2));
    return CpuScatterLoweringTest.context(
        new Operation(
            ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
            new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
        duplicateQueryKey ? List.of(0, 0, 1) : List.of(0, 1, 2),
        duplicateQueryKey ? List.of(qd, vd) : List.of(qd, kd, vd),
        out);
  }
}
