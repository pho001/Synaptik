import operations.layout.noop;
import tensor.DataType;
import tensor.storage.BFloat16Storage;
import tensor.storage.Float32Storage;
import tensor.storage.Float64Storage;
import tensor.storage.Int64Storage;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorConstructorDataTypeTest {

    @Test
    void defaultConstructorsUseFloat32() {
        Tensor t1 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "t1");
        Tensor t2 = new Tensor(new int[]{2, 3}, null, "t2");
        Tensor t3 = new Tensor(new int[]{2}, List.of(t1), new noop(), "t3");
        Tensor t4 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, new int[]{2, 1}, null, "t4");
        Tensor t5 = new Tensor(new double[][]{{1.0, 2.0}, {3.0, 4.0}}, null, "t5");
        Tensor scalar = Tensor.scalar(1.5);

        assertEquals(DataType.FLOAT32, t1.getDataType());
        assertEquals(DataType.FLOAT32, t2.getDataType());
        assertEquals(DataType.FLOAT32, t3.getDataType());
        assertEquals(DataType.FLOAT32, t4.getDataType());
        assertEquals(DataType.FLOAT32, t5.getDataType());
        assertEquals(DataType.FLOAT32, scalar.getDataType());

        assertTrue(t1.getStorage() instanceof Float32Storage);
        assertTrue(t2.getStorage() instanceof Float32Storage);
        assertTrue(t3.getStorage() instanceof Float32Storage);
        assertTrue(t4.getStorage() instanceof Float32Storage);
        assertTrue(t5.getStorage() instanceof Float32Storage);
        assertTrue(scalar.getStorage() instanceof Float32Storage);
    }

    @Test
    void constructorsExposeExplicitDataTypeOverloads() throws Exception {
        Constructor<Tensor> c1 = Tensor.class.getConstructor(double[].class, int[].class, List.class, String.class, DataType.class);
        Constructor<Tensor> c2 = Tensor.class.getConstructor(double[].class, int[].class, int[].class, List.class, String.class, DataType.class);
        Constructor<Tensor> c3 = Tensor.class.getConstructor(int[].class, List.class, String.class, DataType.class);
        Constructor<Tensor> c4 = Tensor.class.getConstructor(int[].class, List.class, operations.Operation.class, String.class, DataType.class);
        Constructor<Tensor> c5 = Tensor.class.getConstructor(Object.class, List.class, String.class, DataType.class);

        assertNotNull(c1);
        assertNotNull(c2);
        assertNotNull(c3);
        assertNotNull(c4);
        assertNotNull(c5);

        Tensor e1 = c1.newInstance(new double[]{1.0, 2.0}, new int[]{2}, null, "e1", DataType.FLOAT64);
        Tensor e2 = c2.newInstance(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, new int[]{2, 1}, null, "e2", DataType.BFLOAT16);
        Tensor e3 = c3.newInstance(new int[]{2, 2}, null, "e3", DataType.FLOAT32);
        Tensor e4 = c4.newInstance(new int[]{2}, List.of(e1), new noop(), "e4", DataType.FLOAT64);
        Tensor e5 = c5.newInstance(new double[][]{{1.0, 2.0}}, null, "e5", DataType.BFLOAT16);

        assertEquals(DataType.FLOAT64, e1.getDataType());
        assertEquals(DataType.BFLOAT16, e2.getDataType());
        assertEquals(DataType.FLOAT32, e3.getDataType());
        assertEquals(DataType.FLOAT64, e4.getDataType());
        assertEquals(DataType.BFLOAT16, e5.getDataType());

        assertTrue(e1.getStorage() instanceof Float64Storage);
        assertTrue(e2.getStorage() instanceof BFloat16Storage);
        assertTrue(e3.getStorage() instanceof Float32Storage);
        assertTrue(e4.getStorage() instanceof Float64Storage);
        assertTrue(e5.getStorage() instanceof BFloat16Storage);
    }

    @Test
    void int64ConstructorsExposeLongStorageAndAccessors() {
        Tensor values = new Tensor(new long[]{7L, 11L, 13L, 17L}, new int[]{2, 2}, null, "values", DataType.INT64);
        Tensor scalar = Tensor.scalar(42, DataType.INT64);
        Tensor zeros = Tensor.zerosLike(values);
        Tensor ones = Tensor.onesLike(values);

        assertEquals(DataType.INT64, values.getDataType());
        assertTrue(values.getStorage() instanceof Int64Storage);
        assertArrayEquals(new long[]{7L, 11L, 13L, 17L}, values.getInt64Data());
        assertEquals(13L, values.getInt64ByFlatIndex(2));
        assertEquals(17L, values.getIntegralByFlatIndex(3));

        assertEquals(DataType.INT64, scalar.getDataType());
        assertArrayEquals(new long[]{42L}, scalar.getInt64Data());
        assertArrayEquals(new long[]{0L, 0L, 0L, 0L}, zeros.getInt64Data());
        assertArrayEquals(new long[]{1L, 1L, 1L, 1L}, ones.getInt64Data());
    }
}
