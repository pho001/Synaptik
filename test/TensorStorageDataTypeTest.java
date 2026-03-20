import Graph.codegen.FusedDTypeOps;
import Graph.optimizer.GraphOptimizer;
import Tensor.DataType;
import Tensor.Float16Storage;
import Tensor.Float32Storage;
import Tensor.Tensor;
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
        out.compute(new GraphOptimizer());

        assertTrue(out.getStorage() instanceof Float32Storage, "Output tensor should use Float32Storage");

        double[] expected = new double[out.getData().length];
        for (int i = 0; i < expected.length; i++) {
            double s = FusedDTypeOps.add(a.getData()[i], b.getData()[i], FusedDTypeOps.MODE_F32);
            expected[i] = FusedDTypeOps.mul(s, a.getData()[i], FusedDTypeOps.MODE_F32);
        }
        assertArrayEquals(expected, out.getData(), 1e-6);
    }

    @Test
    void float16StorageQuantizesNonFusedOps() {
        Tensor a = new Tensor(new double[]{0.123456, 1.654321, -2.222222}, new int[]{3}, null, "a16");
        Tensor b = new Tensor(new double[]{0.333333, -0.777777, 0.999999}, new int[]{3}, null, "b16");
        a.setDataType(DataType.FLOAT16);
        b.setDataType(DataType.FLOAT16);

        Tensor out = a.add(b).sigmoid();
        out.compute(new GraphOptimizer());

        assertTrue(out.getStorage() instanceof Float16Storage, "Output tensor should use Float16Storage");

        double[] expected = new double[out.getData().length];
        for (int i = 0; i < expected.length; i++) {
            double s = FusedDTypeOps.add(a.getData()[i], b.getData()[i], FusedDTypeOps.MODE_F16);
            expected[i] = FusedDTypeOps.sigmoid(s, FusedDTypeOps.MODE_F16);
        }
        assertArrayEquals(expected, out.getData(), 2e-3);
    }
}
