import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageKind;
import backend.cpu.storage.CpuStorageResolver;
import backend.cpu.storage.CpuStorageView;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.storage.TensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class CpuStorageResolverTest {
    private final CpuStorageResolver resolver = new CpuStorageResolver();

    @Test
    void bindArrayOnlyPreservesArrayAndLayoutMetadata() {
        Tensor tensor = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "f32",
                DataType.FLOAT32
        );

        CpuStorageView view = resolver.bindArrayOnly(tensor);

        assertEquals(DataType.FLOAT32, view.dtype());
        assertEquals(CpuStorageKind.JAVA_ARRAY, view.kind());
        assertTrue(view.isArray());
        assertFalse(view.isMemorySegment());
        assertSame(TensorInternalAccess.float32Data(tensor), view.requireF32Array());
        assertArrayEquals(new int[]{2, 3}, view.shape());
        assertArrayEquals(new int[]{3, 1}, view.strides());
        assertEquals(0, view.storageOffset());
        assertEquals(6, view.logicalSize());

        int[] shapeSnapshot = view.shape();
        int[] stridesSnapshot = view.strides();
        shapeSnapshot[0] = 99;
        stridesSnapshot[0] = 99;

        assertArrayEquals(new int[]{2, 3}, view.shape());
        assertArrayEquals(new int[]{3, 1}, view.strides());
    }

    @Test
    void bindArrayOnlySupportsExistingArrayDtypes() {
        Tensor f64 = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "f64", DataType.FLOAT64);
        Tensor f32 = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "f32", DataType.FLOAT32);
        Tensor bf16 = new Tensor(new short[]{1, 2}, new int[]{2}, null, "bf16", DataType.BFLOAT16);
        Tensor i32 = new Tensor(new int[]{1, 2}, new int[]{2}, null, "i32", DataType.INT32);
        Tensor i64 = new Tensor(new long[]{1L, 2L}, new int[]{2}, null, "i64", DataType.INT64);
        Tensor bool = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "bool", DataType.BOOL);

        assertSame(TensorInternalAccess.float64Data(f64), resolver.bindArrayOnly(f64).requireF64Array());
        assertSame(TensorInternalAccess.float32Data(f32), resolver.bindArrayOnly(f32).requireF32Array());
        assertSame(TensorInternalAccess.bfloat16Data(bf16), resolver.bindArrayOnly(bf16).requireBF16Array());
        assertSame(TensorInternalAccess.int32Data(i32), resolver.bindArrayOnly(i32).requireI32Array());
        assertSame(TensorInternalAccess.int64Data(i64), resolver.bindArrayOnly(i64).requireI64Array());
        assertSame(TensorInternalAccess.boolData(bool), resolver.bindArrayOnly(bool).requireBoolArray());
    }

    @Test
    void bindArrayOnlyReturnsImmutableBindings() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{2}, null, "right", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{2}, null, "out", DataType.FLOAT32);

        CpuStorageBindings bindings = resolver.bindArrayOnly(List.of(left, right), output);

        assertEquals(2, bindings.inputs().size());
        assertSame(TensorInternalAccess.float32Data(left), bindings.input(0).requireF32Array());
        assertSame(TensorInternalAccess.float32Data(right), bindings.input(1).requireF32Array());
        assertSame(TensorInternalAccess.float32Data(output), bindings.output().requireF32Array());
        assertThrows(UnsupportedOperationException.class, () -> bindings.inputs().add(bindings.output()));
    }

    @Test
    void bindArrayOnlyRejectsNonArrayStorage() {
        Tensor tensor = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "f32", DataType.FLOAT32);
        TensorInternalAccess.replaceStorage(tensor, new TensorStorage() {
            @Override
            public DataType getType() {
                return DataType.FLOAT32;
            }

            @Override
            public int getSize() {
                return 2;
            }

            @Override
            public long version() {
                return 0;
            }

            @Override
            public void markModified() {
            }
        });

        UnsupportedOperationException failure = assertThrows(
                UnsupportedOperationException.class,
                () -> resolver.bindArrayOnly(tensor)
        );
        assertTrue(failure.getMessage().contains("FLOAT32 array storage required"));
    }

    @Test
    void viewTypedAccessorsRequireMatchingStorageKindAndDtype() {
        CpuStorageView f32Array = CpuStorageView.array(
                DataType.FLOAT32,
                new float[]{1.0f, 2.0f},
                new int[]{2},
                new int[]{1},
                0,
                2
        );
        MemorySegment segment = MemorySegment.ofArray(new float[]{1.0f, 2.0f});
        CpuStorageView f32Segment = CpuStorageView.segment(
                DataType.FLOAT32,
                segment,
                new int[]{2},
                new int[]{1},
                0,
                2
        );

        assertSame(segment, f32Segment.requireSegment());
        assertThrows(IllegalStateException.class, f32Array::requireF64Array);
        assertThrows(IllegalStateException.class, f32Array::requireSegment);
        assertThrows(IllegalStateException.class, f32Segment::requireF32Array);
    }

    @Test
    void viewFactoriesValidateDtypeAndStorageCapacity() {
        assertThrows(IllegalArgumentException.class, () -> CpuStorageView.array(
                DataType.FLOAT32,
                new double[]{1.0, 2.0},
                new int[]{2},
                new int[]{1},
                0,
                2
        ));
        assertThrows(IllegalArgumentException.class, () -> CpuStorageView.array(
                DataType.FLOAT32,
                new float[]{1.0f},
                new int[]{2},
                new int[]{1},
                0,
                2
        ));
        assertThrows(IllegalArgumentException.class, () -> CpuStorageView.segment(
                DataType.FLOAT64,
                MemorySegment.ofArray(new float[]{1.0f}),
                new int[]{1},
                new int[]{1},
                0,
                1
        ));
    }
}
