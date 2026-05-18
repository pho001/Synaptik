import backend.cpu.fused.codegen.FusedDTypeOps;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.BFloat16Bits;
import tensor.DataType;
import tensor.BFloat16Storage;
import tensor.Float32Storage;
import tensor.Float64Storage;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TensorStorageDataTypeTest {
    @Test
    void float32StorageQuantizesNonFusedOps() {
        Tensor a = new Tensor(new double[]{1.23456789, -2.34567891, 3.45678912}, new int[]{3}, null, "a");
        Tensor b = new Tensor(new double[]{0.11111111, 0.22222222, -0.33333333}, new int[]{3}, null, "b");
        a.setDataType(DataType.FLOAT32);
        b.setDataType(DataType.FLOAT32);

        Tensor out = a.add(b).mul(a);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(out.getStorage() instanceof Float32Storage, "Output tensor should use Float32Storage");

        double[] expected = new double[out.toDoubleArrayCopy().length];
        double[] aVals = a.toDoubleArrayCopy();
        double[] bVals = b.toDoubleArrayCopy();
        for (int i = 0; i < expected.length; i++) {
            double s = FusedDTypeOps.add(aVals[i], bVals[i], FusedDTypeOps.MODE_F32);
            expected[i] = FusedDTypeOps.mul(s, aVals[i], FusedDTypeOps.MODE_F32);
        }
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void bfloat16StorageQuantizesNonFusedOps() {
        Tensor a = new Tensor(new double[]{0.123456, 1.654321, -2.222222}, new int[]{3}, null, "a16");
        Tensor b = new Tensor(new double[]{0.333333, -0.777777, 0.999999}, new int[]{3}, null, "b16");
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.add(b).sigmoid();
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(out.getStorage() instanceof BFloat16Storage, "Output tensor should use BFloat16Storage");

        double[] expected = new double[out.toDoubleArrayCopy().length];
        double[] aVals = a.toDoubleArrayCopy();
        double[] bVals = b.toDoubleArrayCopy();
        for (int i = 0; i < expected.length; i++) {
            double s = FusedDTypeOps.add(aVals[i], bVals[i], FusedDTypeOps.MODE_BF16);
            expected[i] = FusedDTypeOps.sigmoid(s, FusedDTypeOps.MODE_BF16);
        }
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-3);
    }

    @Test
    void float64UsesFloat64StorageByDefault() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a64");
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b64", DataType.FLOAT64);

        assertTrue(a.getStorage() instanceof Float32Storage, "Default dtype should map to Float32Storage");
        assertTrue(b.getStorage() instanceof Float64Storage, "Explicit FLOAT64 should map to Float64Storage");
    }

    @Test
    void float32AddReadsTypedStorageNotStaleDoubleCache() {
        Tensor a = new Tensor(new double[]{1.25, 2.5, -3.75}, new int[]{3}, null, "a32", DataType.FLOAT32);
        Tensor b = new Tensor(new double[]{0.5, -1.0, 2.0}, new int[]{3}, null, "b32", DataType.FLOAT32);

        float[] aStorage = ((Float32Storage) a.getStorage()).getFloatArray().clone();
        float[] bStorage = ((Float32Storage) b.getStorage()).getFloatArray().clone();

        // Typed path must read typed storage directly.

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expected = new double[aStorage.length];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (double) (aStorage[i] + bStorage[i]);
        }
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void bfloat16NonFusedAddAppliesPerOpPrecision() {
        Tensor a = new Tensor(new double[]{0.1000123, 0.2000456, -0.3000789}, new int[]{3}, null, "a16Strict");
        Tensor b = new Tensor(new double[]{0.000031, -0.000047, 0.000059}, new int[]{3}, null, "b16Strict");
        a.setDataType(DataType.BFLOAT16);
        b.setDataType(DataType.BFLOAT16);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expected = new double[out.toDoubleArrayCopy().length];
        double[] aVals = a.toDoubleArrayCopy();
        double[] bVals = b.toDoubleArrayCopy();
        for (int i = 0; i < expected.length; i++) {
            expected[i] = FusedDTypeOps.add(aVals[i], bVals[i], FusedDTypeOps.MODE_BF16);
        }
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 0.0);
    }

    @Test
    void setFloat32DataRebindsTypedStorage() {
        Tensor t = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "t32", DataType.FLOAT32);
        float[] replacement = new float[]{3.5f, -2.25f, 9.75f};

        t.setFloat32Data(replacement);

        assertSame(replacement, t.getFloat32Data());
        assertArrayEquals(new double[]{3.5, -2.25, 9.75}, t.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void setFloat32DataRejectsNonFloat32Tensor() {
        Tensor t = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "t64", DataType.FLOAT64);
        assertThrows(UnsupportedOperationException.class, () -> t.setFloat32Data(new float[]{1.0f, 2.0f}));
    }

    @Test
    void typedRawGettersRejectWrongStorageType() {
        Tensor f64 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "f64", DataType.FLOAT64);
        Tensor f32 = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "f32", DataType.FLOAT32);

        assertThrows(UnsupportedOperationException.class, f64::getFloat32Data);
        assertThrows(UnsupportedOperationException.class, f32::getFloat64Data);
    }

    @Test
    void typedCopyHelpersReturnLogicalRowMajorCopies() {
        Tensor f32 = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "f32", DataType.FLOAT32);
        Tensor f32View = f32.select(1, 1);

        float[] f32Copy = f32View.toFloat32ArrayCopy();
        f32Copy[0] = -99f;

        assertArrayEquals(new float[]{2f, 5f}, f32View.toFloat32ArrayCopy(), 0f);

        Tensor i32 = new Tensor(new int[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "i32", DataType.INT32);
        assertArrayEquals(new int[]{2, 5}, i32.select(1, 1).toInt32ArrayCopy());

        Tensor bool = new Tensor(new byte[]{
                1, 0, 1,
                0, 1, 0
        }, new int[]{2, 3}, null, "bool", DataType.BOOL);
        assertArrayEquals(new byte[]{0, 1}, bool.select(1, 1).toBoolByteArrayCopy());

        Tensor bf16 = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "bf16", DataType.BFLOAT16);
        assertArrayEquals(new short[]{
                BFloat16Bits.fromFloat(1.0f),
                BFloat16Bits.fromFloat(-2.0f)
        }, bf16.toBFloat16BitsArrayCopy());
    }
}
