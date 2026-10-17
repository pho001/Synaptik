package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.NumericalForm;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import java.util.List;
import org.junit.jupiter.api.Test;

final class CpuMatmulIrTest {
    private static final CpuAccessPlan READ = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
            CpuAccessPlan.Regime.GENERAL_ODOMETER, 2,
            List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    private static final CpuAccessPlan WRITE = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
            CpuAccessPlan.Regime.DENSE_LINEAR, 2,
            List.of(CpuAccessPlan.AxisRole.CONTIGUOUS, CpuAccessPlan.AxisRole.CONTIGUOUS), 2);

    @Test void retainsExactAddOrderTerminalClampAndNumericalFormInIdentity() {
        var clamp = new ClampRangeAttrs(ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f));
        var left = ir(new Epilogue(Epilogue.AddInputOrder.MATMUL_LEFT,
                Epilogue.Terminal.CLAMP, clamp), NumericalForm.SEQUENTIAL);
        var right = ir(new Epilogue(Epilogue.AddInputOrder.MATMUL_RIGHT,
                Epilogue.Terminal.CLAMP, clamp), NumericalForm.SEQUENTIAL);
        var fused = ir(left.epilogue(), NumericalForm.FUSED_MULTIPLY_ADD);
        assertNotEquals(left.structuralKey(), right.structuralKey());
        assertNotEquals(left.structuralKey(), fused.structuralKey());
        assertEquals(3, left.inputAccesses().size());
    }

    @Test void rejectsIntegralFmaFloatingSuffixAndTerminalVectorForms() {
        assertThrows(IllegalArgumentException.class, () -> new CpuMatmulIr(DataType.INT32,
                DataType.INT32, DataType.INT32, Realization.DIRECT_SCALAR, Epilogue.none(), 0,
                NumericalForm.FUSED_MULTIPLY_ADD, List.of(READ, READ), WRITE));
        var relu = new Epilogue(Epilogue.AddInputOrder.MATMUL_LEFT, Epilogue.Terminal.RELU, null);
        assertThrows(IllegalArgumentException.class, () -> new CpuMatmulIr(DataType.INT32,
                DataType.INT32, DataType.INT32, Realization.DIRECT_SCALAR, relu, 0,
                NumericalForm.SEQUENTIAL, List.of(READ, READ, READ), WRITE));
        assertThrows(IllegalArgumentException.class, () -> new CpuMatmulIr(DataType.FLOAT32,
                DataType.FLOAT32, DataType.FLOAT32, Realization.DIRECT_N_VECTOR, relu, 128,
                NumericalForm.SEQUENTIAL, List.of(READ, READ, READ), WRITE));
    }

    @Test void realizationVocabularyIsClosedToExactlyFourForms() {
        assertEquals(List.of("DIRECT_SCALAR", "DIRECT_N_VECTOR", "TILED_SCALAR_2X2",
                "TILED_N_VECTOR_2X2"), java.util.Arrays.stream(Realization.values())
                        .map(Enum::name).toList());
    }

    private static CpuMatmulIr ir(Epilogue epilogue, NumericalForm form) {
        return new CpuMatmulIr(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
                Realization.DIRECT_SCALAR, epilogue, 0, form,
                List.of(READ, READ, READ), WRITE);
    }
}
