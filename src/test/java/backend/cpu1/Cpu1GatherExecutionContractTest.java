package backend.cpu1;

import backend.contract.ComputeBackend;
import runtime.memory.nativecpu.NativeCpuStorageFactory;
import backend.cpu1.exec.Cpu1IndexExecutableUnit;
import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.execution.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import graph.execution.PreparedExecutionStep;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.ExecutionState;
import runtime.runner.StepExecutionTracer;
import operations.index.ScatterReduction;
import operations.index.gather;
import operations.index.gatherNd;
import operations.index.scatterAdd;
import operations.index.scatterAxisAdd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.index.takeAlongAxis;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeBoolStorage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeInt32Storage;
import tensor.storage.NativeInt64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1GatherExecutionContractTest {
    @Test
    void preparedF32GatherDimOneReadsInt32Indices() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new float[]{3.0f, 4.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
        assertIndexPolicy(artifact, Cpu1StorageKind.JAVA_ARRAY);
    }

    @Test
    void preparedF64GatherDimZeroReadsInt64Indices() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new long[]{1L, 0L, 1L}, new int[]{3}, null, "indices", DataType.INT64);
        Fixture fixture = fixture(input.gather(indices, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_F64_I64_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{4.0d, 2.0d, 6.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedBf16GatherCopiesRawBfloat16Values() {
        Tensor input = new Tensor(
                new double[]{
                        1.25d, 2.5d, 3.75d,
                        4.5d, 5.25d, 6.5d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.BFLOAT16
        );
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_BF16_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new float[]{2.5f, 6.5f}, bf16ToF32(output), 1.0e-3f);
    }

    @Test
    void preparedGatherSupportsIntegerAndBoolValues() {
        Tensor ints = new Tensor(new int[]{10, 20, 30}, new int[]{3}, null, "ints", DataType.INT32);
        Tensor intIndices = new Tensor(new long[]{2L}, new int[]{1}, null, "intIndices", DataType.INT64);
        Tensor intOutput = execute(
                fixture(ints.gather(intIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new int[]{30}, intOutput.toInt32ArrayCopy());

        Tensor bools = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "bools", DataType.BOOL);
        Tensor boolIndices = new Tensor(new int[]{0}, new int[]{1}, null, "boolIndices", DataType.INT32);
        Tensor boolOutput = execute(
                fixture(bools.gather(boolIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new boolean[]{true}, boolOutput.toBooleanArrayCopy());

        Tensor longs = new Tensor(new long[]{100L, 200L, 300L}, new int[]{3}, null, "longs", DataType.INT64);
        Tensor longOutput = execute(
                fixture(longs.gather(boolIndices, 0)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new long[]{100L}, longOutput.toInt64ArrayCopy());
    }

    @Test
    void preparedF32GatherAxisReadsRankTwoInt32Indices() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                new int[]{2, 3, 2},
                null,
                "gatherAxisInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0, 1, 2}, new int[]{2, 2}, null, "gatherAxisIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_AXIS_F32_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new int[]{2, 2, 2, 2}, output.getShape());
        assertArrayEquals(new float[]{
                5.0f, 6.0f,
                1.0f, 2.0f,
                3.0f, 4.0f,
                5.0f, 6.0f,
                11.0f, 12.0f,
                7.0f, 8.0f,
                9.0f, 10.0f,
                11.0f, 12.0f
        }, output.toFloat32ArrayCopy(), 1.0e-6f);
        assertEquals(4, artifact.preparedIndexUnit().indexElementCount());
    }

    @Test
    void preparedGatherAxisNormalizesNegativeIndexValues() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "negativeGatherAxisInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{-1, 0}, new int[]{2}, null, "negativeGatherAxisIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_AXIS_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{3.0d, 1.0d, 6.0d, 4.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedF64TakeAlongAxisReadsInt64IndicesAndNegativeValues() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "takeAlongAxisInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(
                new long[]{2L, -1L, 0L, 0L, 1L, -1L},
                new int[]{2, 3},
                null,
                "takeAlongAxisIndices",
                DataType.INT64
        );
        Fixture fixture = fixture(input.takeAlongAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertIndexKernel(artifact, Cpu1IndexKernelId.TAKE_ALONG_AXIS_F64_I64_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{3.0d, 3.0d, 1.0d, 4.0d, 5.0d, 6.0d}, output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(3, artifact.preparedIndexUnit().indexAxisSize());
    }

    @Test
    void preparedBf16GatherAxisCopiesRawBfloat16Values() {
        Tensor input = new Tensor(
                new double[]{
                        1.25d, 2.5d, 3.75d,
                        4.5d, 5.25d, 6.5d
                },
                new int[]{2, 3},
                null,
                "bf16GatherAxisInput",
                DataType.BFLOAT16
        );
        Tensor indices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "bf16GatherAxisIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_AXIS_BF16_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new float[]{3.75f, 3.75f, 6.5f, 6.5f}, bf16ToF32(output), 1.0e-3f);
    }

    @Test
    void preparedTakeAlongAxisSupportsIntegerAndBoolValues() {
        Tensor ints = new Tensor(new int[]{10, 20, 30, 40}, new int[]{2, 2}, null, "takeInts", DataType.INT32);
        Tensor intIndices = new Tensor(new int[]{1, 0, 0, 1}, new int[]{2, 2}, null, "takeIntIndices", DataType.INT32);
        Tensor intOutput = execute(
                fixture(ints.takeAlongAxis(intIndices, 1)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new int[]{20, 10, 30, 40}, intOutput.toInt32ArrayCopy());

        Tensor longs = new Tensor(new long[]{100L, 200L, 300L, 400L}, new int[]{2, 2}, null, "takeLongs", DataType.INT64);
        Tensor longIndices = new Tensor(new long[]{0L, 1L, 1L, 0L}, new int[]{2, 2}, null, "takeLongIndices", DataType.INT64);
        Tensor longOutput = execute(
                fixture(longs.takeAlongAxis(longIndices, 1)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new long[]{100L, 200L, 400L, 300L}, longOutput.toInt64ArrayCopy());

        Tensor bools = new Tensor(new byte[]{1, 0, 0, 1}, new int[]{2, 2}, null, "takeBools", DataType.BOOL);
        Tensor boolOutput = execute(
                fixture(bools.takeAlongAxis(intIndices, 1)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new boolean[]{false, true, false, true}, boolOutput.toBooleanArrayCopy());
    }

    @Test
    void preparedF64GatherNdReadsTupleIndexedElements() {
        Tensor input = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "gatherNdElementsInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "gatherNdElementsIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new int[]{2}, output.getShape());
        assertArrayEquals(new double[]{30.0d, 40.0d}, output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(2, artifact.preparedIndexUnit().tupleRank());
        assertEquals(0, artifact.preparedIndexUnit().batchDims());
    }

    @Test
    void preparedF64GatherNdReadsTupleIndexedSlices() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "gatherNdSlicesInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2, 1}, null, "gatherNdSlicesIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new int[]{2, 3}, output.getShape());
        assertArrayEquals(new double[]{
                4.0d, 5.0d, 6.0d,
                1.0d, 2.0d, 3.0d
        }, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedF64GatherNdBatchDimsOneReadsPerBatchSlices() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d,
                        3.0d, 4.0d,
                        5.0d, 6.0d,
                        7.0d, 8.0d,
                        9.0d, 10.0d,
                        11.0d, 12.0d
                },
                new int[]{2, 3, 2},
                null,
                "gatherNdBatchInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{2, 0, 1, 0}, new int[]{2, 2, 1}, null, "gatherNdBatchIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new int[]{2, 2, 2}, output.getShape());
        assertArrayEquals(new double[]{
                5.0d, 6.0d,
                1.0d, 2.0d,
                9.0d, 10.0d,
                7.0d, 8.0d
        }, output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(1, artifact.preparedIndexUnit().batchDims());
    }

    @Test
    void preparedGatherNdAcceptsProjectScalarShapeAndNegativeIndices() {
        Tensor input = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "gatherNdNegativeInput",
                DataType.FLOAT64
        );
        Tensor scalarIndices = new Tensor(new int[]{1, -1}, new int[]{2}, null, "gatherNdScalarIndices", DataType.INT32);
        Tensor scalarOutput = execute(
                fixture(input.gatherNd(scalarIndices)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new int[]{1}, scalarOutput.getShape());
        assertArrayEquals(new double[]{60.0d}, scalarOutput.toDoubleArrayCopy(), 1.0e-12);

        Tensor indices = new Tensor(new int[]{-1, 0, 0, -1}, new int[]{2, 2}, null, "gatherNdNegativeIndices", DataType.INT32);
        Tensor output = execute(
                fixture(input.gatherNd(indices)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new double[]{40.0d, 30.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedGatherNdCopiesBf16RawBitsAndBoolValues() {
        short[] bits = new short[]{
                TensorDTypeOps.toBFloat16Bits(10.0f),
                TensorDTypeOps.toBFloat16Bits(20.0f)
        };
        Tensor bf16 = new Tensor(bits, new int[]{2}, null, "gatherNdBf16Input", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "gatherNdBf16Indices", DataType.INT32);
        Fixture fixture = fixture(bf16.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_BF16_I32_DENSE_ARRAY);

        Tensor bf16Output = execute(fixture, artifact);

        assertArrayEquals(new short[]{bits[1]}, TensorInternalAccess.bfloat16Data(bf16Output));

        Tensor bools = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "gatherNdBoolInput", DataType.BOOL);
        Tensor boolOutput = execute(
                fixture(bools.gatherNd(indices)),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertArrayEquals(new boolean[]{false}, boolOutput.toBooleanArrayCopy());
    }

    @Test
    void preparedF32GatherNdMemorySegmentReadsNegativeIndices() {
        Tensor input = new Tensor(
                new float[]{
                        10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f
                },
                new int[]{2, 3},
                null,
                "gatherNdSegmentNegativeInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{-1, 0, 0, -1}, new int[]{2, 2}, null, "gatherNdSegmentNegativeIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_F32_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new float[]{40.0f, 30.0f}, nativeF32(result.nativeOutput(), 2), 1.0e-6f);
    }

    @Test
    void preparedF64GatherNdMemorySegmentReadsSlices() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "gatherNdSegmentSlicesInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new long[]{1L, 0L}, new int[]{2, 1}, null, "gatherNdSegmentSlicesIndices", DataType.INT64);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{
                4.0d, 5.0d, 6.0d,
                1.0d, 2.0d, 3.0d
        }, nativeF64(result.nativeOutput(), 6), 1.0e-12);
    }

    @Test
    void preparedBf16GatherNdMemorySegmentCopiesRawBits() {
        short[] bits = new short[]{
                TensorDTypeOps.toBFloat16Bits(10.0f),
                TensorDTypeOps.toBFloat16Bits(20.0f)
        };
        Tensor input = new Tensor(bits, new int[]{2}, null, "gatherNdSegmentBf16Input", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "gatherNdSegmentBf16Indices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_BF16_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new short[]{bits[1]}, nativeBf16Bits(result.nativeOutput(), 1));
    }

    @Test
    void preparedBoolGatherNdMemorySegmentCopiesBytes() {
        Tensor input = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "gatherNdSegmentBoolInput", DataType.BOOL);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "gatherNdSegmentBoolIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherNd(indices));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_ND_BOOL_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new byte[]{0}, nativeBool(result.nativeOutput(), 1));
    }

    @Test
    void preparedGatherRejectsOutOfRangeIndexAtExecution() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> execute(fixture, artifact));

        assertTrue(exception.getMessage().contains("Gather index out of bounds"));
    }

    @Test
    void prepareRejectsNonContiguousGatherInput() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(view.gather(indices, 0));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread())
        );

        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void prepareRejectsNonContiguousGatherOutput() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2},
                new int[]{2},
                List.of(input, indices),
                new gather(1),
                "gather_strided_output",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(output);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("output access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void preparedF32GatherMemorySegmentReadsInt32Indices() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_F32_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new float[]{3.0f, 4.0f}, nativeF32(result.nativeOutput(), 2), 1.0e-6f);
        assertIndexPolicy(artifact, Cpu1StorageKind.MEMORY_SEGMENT);
    }

    @Test
    void preparedTakeAlongAxisMemorySegmentReadsInt64IndicesAndNegativeValues() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "takeMemoryInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(
                new long[]{2L, -1L, 0L, 0L, 1L, -1L},
                new int[]{2, 3},
                null,
                "takeMemoryIndices",
                DataType.INT64
        );
        Fixture fixture = fixture(input.takeAlongAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.TAKE_ALONG_AXIS_F32_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new float[]{3.0f, 3.0f, 1.0f, 4.0f, 5.0f, 6.0f},
                nativeF32(result.nativeOutput(), 6), 1.0e-6f);
    }

    @Test
    void preparedGatherAxisMemorySegmentReadsInt64IndicesAndNegativeValues() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "gatherAxisMemoryInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new long[]{-1L, 0L}, new int[]{2}, null, "gatherAxisMemoryIndices", DataType.INT64);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_AXIS_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{3.0d, 1.0d, 6.0d, 4.0d}, nativeF64(result.nativeOutput(), 4), 1.0e-12);
    }

    @Test
    void preparedBf16GatherMemorySegmentCopiesRawBfloat16Bits() {
        short[] bits = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.25f),
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(3.75f),
                TensorDTypeOps.toBFloat16Bits(4.5f),
                TensorDTypeOps.toBFloat16Bits(5.25f),
                TensorDTypeOps.toBFloat16Bits(6.5f)
        };
        Tensor input = new Tensor(bits, new int[]{2, 3}, null, "bf16GatherMemoryInput", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1, 2}, new int[]{2}, null, "bf16GatherMemoryIndices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.GATHER_BF16_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new short[]{bits[1], bits[5]}, nativeBf16Bits(result.nativeOutput(), 2));
    }

    @Test
    void preparedBoolTakeAlongAxisMemorySegmentCopiesBytes() {
        Tensor input = new Tensor(new byte[]{1, 0, 0, 1}, new int[]{2, 2}, null, "boolTakeMemoryInput", DataType.BOOL);
        Tensor indices = new Tensor(new int[]{1, 0, 0, 1}, new int[]{2, 2}, null, "boolTakeMemoryIndices", DataType.INT32);
        Fixture fixture = fixture(input.takeAlongAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.TAKE_ALONG_AXIS_BOOL_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new byte[]{0, 1, 0, 1}, nativeBool(result.nativeOutput(), 4));
    }

    @Test
    void prepareRejectsMemorySegmentGatherOutputOffset() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "offsetMemoryInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "offsetMemoryIndices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2},
                new int[]{1},
                1,
                List.of(input, indices),
                new gather(1),
                "gather_offset_output",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(output);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread())
        );

        assertTrue(exception.getMessage().contains("output access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.DENSE_WITH_OFFSET.name()));
    }

    @Test
    void prepareRejectsGatherNdOutputOffset() {
        Tensor input = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "gatherNdOffsetInput",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "gatherNdOffsetIndices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2},
                new int[]{1},
                1,
                List.of(input, indices),
                new gatherNd(),
                "gather_nd_offset_output",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(output);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread())
        );

        assertTrue(exception.getMessage().contains("output access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.DENSE_WITH_OFFSET.name()));
    }

    @Test
    void prepareRejectsNonContiguousGatherAxisInput() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "gatherAxisBase",
                DataType.FLOAT32
        );
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "gatherAxisViewIndices", DataType.INT32);
        Fixture fixture = fixture(view.gatherAxis(indices, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void prepareRejectsNonContiguousTakeAlongAxisOutput() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "takeStridedOutputInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{1, 0, 0, 1}, new int[]{2, 2}, null, "takeStridedOutputIndices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2, 2},
                new int[]{1, 2},
                List.of(input, indices),
                new takeAlongAxis(1),
                "take_strided_output",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(output);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("output access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void prepareRejectsFloatingIndexDType() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new float[]{2.0f, 0.0f}, new int[]{2}, null, "indices", DataType.FLOAT32);
        Fixture fixture = fixture(input.gather(indices, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("INT32/INT64 indices"));
        assertTrue(exception.getMessage().contains("FLOAT32"));
    }

    @Test
    void prepareRejectsFloatingGatherAxisIndexDType() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "gatherAxisFloatIndexInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new float[]{1.0f, 0.0f}, new int[]{2}, null, "gatherAxisFloatIndices", DataType.FLOAT32);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("GATHER_AXIS"));
        assertTrue(exception.getMessage().contains("INT32/INT64 indices"));
        assertTrue(exception.getMessage().contains("FLOAT32"));
    }

    @Test
    void prepareRejectsInvalidTakeAlongAxisShape() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "takeInvalidShapeInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "takeInvalidShapeIndices", DataType.INT32);
        Tensor output = new Tensor(
                new int[]{2},
                List.of(input, indices),
                new takeAlongAxis(1),
                "take_invalid_shape",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(output);

        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("TAKE_ALONG_AXIS indices rank must match input rank"));
    }

    @Test
    void artifactAndTraceExposeGatherKernelAndStorageKind() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertEquals(Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY, artifact.preparedIndexUnit().kernelId());
        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY.name(), trace.kernel());
        assertEquals(
                Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY.name(),
                trace.metadata().attributes().get("cpu1IndexKernelId")
        );
        assertEquals(Cpu1StorageKind.JAVA_ARRAY.name(), trace.metadata().attributes().get("cpu1StorageKind"));
        assertEquals("GATHER", trace.metadata().attributes().get("cpu1IndexOpType"));
        assertEquals("FLOAT32", trace.metadata().attributes().get("cpu1IndexValueDType"));
        assertEquals("INT32", trace.metadata().attributes().get("cpu1IndexDType"));
        assertEquals(1, trace.metadata().attributes().get("cpu1IndexDimension"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexLaunchWorkers"));
    }

    @Test
    void artifactAndTraceExposeMemorySegmentKernelAndStorageKind() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "traceSegmentInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "traceSegmentIndices", DataType.INT32);
        Fixture fixture = fixture(input.gather(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.GATHER_F32_I32_DENSE_SEGMENT.name(), trace.kernel());
        assertEquals(
                Cpu1IndexKernelId.GATHER_F32_I32_DENSE_SEGMENT.name(),
                trace.metadata().attributes().get("cpu1IndexKernelId")
        );
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT.name(), trace.metadata().attributes().get("cpu1StorageKind"));
        assertEquals("GATHER", trace.metadata().attributes().get("cpu1IndexOpType"));
    }

    @Test
    void artifactAndTraceExposeGatherAxisKernelAxisAndLaunch() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "traceGatherAxisInput",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "traceGatherAxisIndices", DataType.INT32);
        Fixture fixture = fixture(input.gatherAxis(indices, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(2));
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.GATHER_AXIS_F32_I32_DENSE_ARRAY.name(), trace.kernel());
        assertEquals(
                Cpu1IndexKernelId.GATHER_AXIS_F32_I32_DENSE_ARRAY.name(),
                trace.metadata().attributes().get("cpu1IndexKernelId")
        );
        assertEquals("GATHER_AXIS", trace.metadata().attributes().get("cpu1IndexOpType"));
        assertEquals(1, trace.metadata().attributes().get("cpu1IndexDimension"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexElementCount"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexLaunchWorkers"));
    }

    @Test
    void preparedF64ScatterAddAddsIntoCopiedBase() {
        Tensor base = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "scatterBase",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "scatterIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1.0d, 5.0d}, new int[]{2}, null, "scatterUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{
                10.0d, 20.0d, 31.0d,
                45.0d, 50.0d, 60.0d
        }, output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(1, artifact.preparedIndexUnit().launchConfig().workerCount());
        assertEquals(2, artifact.preparedIndexUnit().updateElementCount());
        assertIndexPolicy(artifact, Cpu1StorageKind.JAVA_ARRAY);
    }

    @Test
    void preparedScatterAddAccumulatesDuplicateIndicesInLogicalOrder() {
        Tensor base = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "scatterDuplicateBase",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{0, 0, 1}, new int[]{3}, null, "scatterDuplicateIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1.0d, 2.0d, 3.0d}, new int[]{3}, null, "scatterDuplicateUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{
                11.0d, 22.0d, 30.0d,
                40.0d, 50.0d, 63.0d
        }, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedBf16ScatterAddAccumulatesAndWritesBfloat16Bits() {
        short[] baseBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] updateBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(0.5f),
                TensorDTypeOps.toBFloat16Bits(1.5f)
        };
        Tensor base = new Tensor(baseBits, new int[]{2, 2}, null, "bf16ScatterBase", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "bf16ScatterIndices", DataType.INT32);
        Tensor updates = new Tensor(updateBits, new int[]{2}, null, "bf16ScatterUpdates", DataType.BFLOAT16);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_BF16_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(4.5f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        }, TensorInternalAccess.bfloat16Data(output));
    }

    @Test
    void preparedF32ScatterAddMemorySegmentAddsIntoCopiedBase() {
        Tensor base = new Tensor(
                new float[]{
                        10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f
                },
                new int[]{2, 3},
                null,
                "scatterSegmentF32Base",
                DataType.FLOAT32
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "scatterSegmentF32Indices", DataType.INT32);
        Tensor updates = new Tensor(new float[]{1.0f, 5.0f}, new int[]{2}, null, "scatterSegmentF32Updates", DataType.FLOAT32);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_F32_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new float[]{
                10.0f, 20.0f, 31.0f,
                45.0f, 50.0f, 60.0f
        }, nativeF32(result.nativeOutput(), 6), 1.0e-6f);
    }

    @Test
    void preparedF64ScatterAddMemorySegmentAccumulatesDuplicateIndices() {
        Tensor base = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "scatterSegmentF64Base",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new long[]{0L, 0L, 1L}, new int[]{3}, null, "scatterSegmentF64Indices", DataType.INT64);
        Tensor updates = new Tensor(new double[]{1.0d, 2.0d, 3.0d}, new int[]{3}, null, "scatterSegmentF64Updates", DataType.FLOAT64);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{
                11.0d, 22.0d, 30.0d,
                40.0d, 50.0d, 63.0d
        }, nativeF64(result.nativeOutput(), 6), 1.0e-12);
    }

    @Test
    void preparedBf16ScatterAddMemorySegmentAccumulatesBits() {
        short[] baseBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] updateBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(0.5f),
                TensorDTypeOps.toBFloat16Bits(1.5f)
        };
        Tensor base = new Tensor(baseBits, new int[]{2, 2}, null, "bf16ScatterSegmentBase", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "bf16ScatterSegmentIndices", DataType.INT32);
        Tensor updates = new Tensor(updateBits, new int[]{2}, null, "bf16ScatterSegmentUpdates", DataType.BFLOAT16);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ADD_BF16_I32_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(4.5f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        }, nativeBf16Bits(result.nativeOutput(), 4));
    }

    @Test
    void preparedF64ScatterAxisAddAccumulatesRankChangingUpdates() {
        Tensor base = new Tensor(new double[6], new int[]{2, 3}, null, "scatterAxisBase", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0, 2, 1}, new int[]{2, 2}, null, "scatterAxisIndices", DataType.INT32);
        Tensor updates = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d, 4.0d,
                        5.0d, 6.0d, 7.0d, 8.0d
                },
                new int[]{2, 2, 2},
                null,
                "scatterAxisUpdates",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(base.scatterAxisAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{
                2.0d, 4.0d, 4.0d,
                6.0d, 8.0d, 12.0d
        }, output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(1, artifact.preparedIndexUnit().launchConfig().workerCount());
        assertEquals(8, artifact.preparedIndexUnit().updateElementCount());
    }

    @Test
    void preparedBf16ScatterAxisAddAccumulatesRankChangingUpdates() {
        short[] baseBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f),
                TensorDTypeOps.toBFloat16Bits(4.0f)
        };
        short[] updateBits = new short[]{
                TensorDTypeOps.toBFloat16Bits(0.5f),
                TensorDTypeOps.toBFloat16Bits(1.5f),
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(3.5f)
        };
        Tensor base = new Tensor(baseBits, new int[]{2, 2}, null, "bf16ScatterAxisBase", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "bf16ScatterAxisIndices", DataType.INT32);
        Tensor updates = new Tensor(updateBits, new int[]{2, 2}, null, "bf16ScatterAxisUpdates", DataType.BFLOAT16);
        Fixture fixture = fixture(base.scatterAxisAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_AXIS_ADD_BF16_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new short[]{
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(2.5f),
                TensorDTypeOps.toBFloat16Bits(6.5f),
                TensorDTypeOps.toBFloat16Bits(6.5f)
        }, TensorInternalAccess.bfloat16Data(output));
    }

    @Test
    void preparedF64ScatterAxisAddMemorySegmentNormalizesNegativeIndices() {
        Tensor base = new Tensor(new double[6], new int[]{2, 3}, null, "scatterAxisSegmentBase", DataType.FLOAT64);
        Tensor indices = new Tensor(new long[]{-1L, 0L, -1L, 1L}, new int[]{2, 2}, null, "scatterAxisSegmentIndices", DataType.INT64);
        Tensor updates = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d, 4.0d,
                        5.0d, 6.0d, 7.0d, 8.0d
                },
                new int[]{2, 2, 2},
                null,
                "scatterAxisSegmentUpdates",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(base.scatterAxisAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{
                2.0d, 4.0d, 4.0d,
                6.0d, 8.0d, 12.0d
        }, nativeF64(result.nativeOutput(), 6), 1.0e-12);
    }

    @Test
    void preparedScatterAddRejectsOutOfRangeIndexAtExecution() {
        Tensor base = new Tensor(new double[6], new int[]{2, 3}, null, "scatterOobBase", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "scatterOobIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1.0d, 5.0d}, new int[]{2}, null, "scatterOobUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> execute(fixture, artifact));

        assertTrue(exception.getMessage().contains("Gather index out of bounds"));
    }

    @Test
    void prepareRejectsNonDenseScatterInputUpdateAndOutputAccess() {
        Tensor base = new Tensor(
                new double[]{1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d},
                new int[]{2, 3},
                null,
                "scatterAccessBase",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{0, 1}, new int[]{2}, null, "scatterAccessIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1.0d, 2.0d}, new int[]{2}, null, "scatterAccessUpdates", DataType.FLOAT64);

        UnsupportedOperationException inputException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(base.permute(1, 0).scatterAdd(indices, updates, 0)), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(inputException.getMessage().contains("input access"));
        assertTrue(inputException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor stridedUpdates = new Tensor(
                new int[]{2},
                new int[]{2},
                (List<Tensor>) null,
                (operations.Operation) null,
                "scatterStridedUpdates",
                DataType.FLOAT64
        );
        Tensor stridedUpdateOutput = new Tensor(
                new int[]{2, 3},
                List.of(base, indices, stridedUpdates),
                new scatterAdd(1),
                "scatter_strided_updates_output",
                DataType.FLOAT64
        );
        UnsupportedOperationException updateException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(stridedUpdateOutput), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(updateException.getMessage().contains("updates access"));
        assertTrue(updateException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor offsetOutput = new Tensor(
                new int[]{2, 3},
                new int[]{3, 1},
                1,
                List.of(base, indices, updates),
                new scatterAdd(1),
                "scatter_offset_output",
                DataType.FLOAT64
        );
        UnsupportedOperationException outputException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(offsetOutput), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(outputException.getMessage().contains("output access"));
        assertTrue(outputException.getMessage().contains(Cpu1StorageAccessKind.DENSE_WITH_OFFSET.name()));
    }

    @Test
    void preparedF64ScatterElementsNoneWritesJavaArray() {
        Tensor data = new Tensor(
                new double[]{10.0d, 20.0d, 30.0d, 40.0d, 50.0d, 60.0d},
                new int[]{2, 3},
                null,
                "scatterElementsData",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{2, 1, 0, 2}, new int[]{2, 2}, null, "scatterElementsIndices", DataType.INT32);
        Tensor updates = new Tensor(
                new double[]{1.0d, 2.0d, 3.0d, 4.0d},
                new int[]{2, 2},
                null,
                "scatterElementsUpdates",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(data.scatterElements(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{10.0d, 2.0d, 1.0d, 3.0d, 50.0d, 4.0d},
                output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(ScatterReduction.NONE, artifact.preparedIndexUnit().reduction());
        assertEquals(1, artifact.preparedIndexUnit().launchConfig().workerCount());
    }

    @Test
    void preparedScatterElementsAddAccumulatesDuplicateTargets() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterElementsAddData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1}, new int[]{1, 2}, null, "scatterElementsAddIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2.0d, 3.0d}, new int[]{1, 2}, null,
                "scatterElementsAddUpdates", DataType.FLOAT64);
        Tensor output = execute(fixture(data.scatterElements(indices, updates, 1, ScatterReduction.ADD)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new double[]{10.0d, 25.0d, 30.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedScatterElementsNoneRejectsDuplicateTargets() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterElementsDuplicateData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1}, new int[]{1, 2}, null, "scatterElementsDuplicateIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2.0d, 3.0d}, new int[]{1, 2}, null,
                "scatterElementsDuplicateUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterElements(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> execute(fixture, artifact));

        assertTrue(exception.getMessage().contains("NONE reduction does not allow duplicate"));
    }

    @Test
    void preparedScatterElementsSupportsBf16AndBoolNone() {
        short[] bf16Data = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f)
        };
        short[] bf16Updates = new short[]{
                TensorDTypeOps.toBFloat16Bits(4.0f),
                TensorDTypeOps.toBFloat16Bits(5.0f)
        };
        Tensor bf16 = new Tensor(bf16Data, new int[]{1, 3}, null, "scatterElementsBf16Data", DataType.BFLOAT16);
        Tensor bf16Indices = new Tensor(new int[]{2, 0}, new int[]{1, 2}, null, "scatterElementsBf16Indices", DataType.INT32);
        Tensor bf16UpdatesTensor = new Tensor(bf16Updates, new int[]{1, 2}, null,
                "scatterElementsBf16Updates", DataType.BFLOAT16);

        Tensor bf16Output = execute(fixture(bf16.scatterElements(bf16Indices, bf16UpdatesTensor, 1)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new short[]{bf16Updates[1], bf16Data[1], bf16Updates[0]},
                TensorInternalAccess.bfloat16Data(bf16Output));

        Tensor bools = new Tensor(new byte[]{1, 0, 1}, new int[]{1, 3}, null, "scatterElementsBoolData", DataType.BOOL);
        Tensor boolUpdates = new Tensor(new byte[]{0, 1}, new int[]{1, 2}, null,
                "scatterElementsBoolUpdates", DataType.BOOL);

        Tensor boolOutput = execute(fixture(bools.scatterElements(bf16Indices, boolUpdates, 1)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new boolean[]{true, false, false}, boolOutput.toBooleanArrayCopy());
        Tensor boolAddOutput = new Tensor(
                new int[]{1, 3},
                List.of(bools, bf16Indices, boolUpdates),
                new scatterElements(1, ScatterReduction.ADD),
                "scatter_elements_bool_add_output",
                DataType.BOOL
        );
        assertThrows(UnsupportedOperationException.class,
                () -> prepareRoot(fixture(boolAddOutput), Cpu1PrepareConfig.scalarSingleThread()));
    }

    @Test
    void preparedScatterElementsMemorySegmentWritesDensePath() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterElementsSegmentData", DataType.FLOAT64);
        Tensor indices = new Tensor(new long[]{-1L, 0L}, new int[]{1, 2}, null,
                "scatterElementsSegmentIndices", DataType.INT64);
        Tensor updates = new Tensor(new double[]{7.0d, 8.0d}, new int[]{1, 2}, null,
                "scatterElementsSegmentUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterElements(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{8.0d, 20.0d, 7.0d}, nativeF64(result.nativeOutput(), 3), 1.0e-12);
    }

    @Test
    void preparedF64ScatterNdNoneWritesTupleIndexedElements() {
        Tensor data = new Tensor(
                new double[]{10.0d, 20.0d, 30.0d, 40.0d, 50.0d, 60.0d},
                new int[]{2, 3},
                null,
                "scatterNdData",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "scatterNdIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{9.0d, 8.0d}, new int[]{2}, null, "scatterNdUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterNd(indices, updates));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ND_F64_I32_DENSE_ARRAY);

        Tensor output = execute(fixture, artifact);

        assertArrayEquals(new double[]{10.0d, 20.0d, 9.0d, 8.0d, 50.0d, 60.0d},
                output.toDoubleArrayCopy(), 1.0e-12);
        assertEquals(1, artifact.preparedIndexUnit().launchConfig().workerCount());
    }

    @Test
    void preparedScatterNdWritesTupleIndexedSlices() {
        Tensor data = new Tensor(new double[]{1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d}, new int[]{2, 3}, null,
                "scatterNdSliceData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1}, new int[]{1, 1}, null, "scatterNdSliceIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{7.0d, 8.0d, 9.0d}, new int[]{1, 3}, null,
                "scatterNdSliceUpdates", DataType.FLOAT64);

        Tensor output = execute(fixture(data.scatterNd(indices, updates)), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new double[]{1.0d, 2.0d, 3.0d, 7.0d, 8.0d, 9.0d},
                output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedScatterNdAddAccumulatesDuplicateTargets() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterNdAddData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1}, new int[]{2, 2}, null, "scatterNdAddIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2.0d, 3.0d}, new int[]{2}, null, "scatterNdAddUpdates", DataType.FLOAT64);

        Tensor output = execute(fixture(data.scatterNd(indices, updates, ScatterReduction.ADD)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new double[]{10.0d, 25.0d, 30.0d}, output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedScatterNdNoneRejectsDuplicateTargets() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterNdDuplicateData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 0, 1}, new int[]{2, 2}, null,
                "scatterNdDuplicateIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2.0d, 3.0d}, new int[]{2}, null,
                "scatterNdDuplicateUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterNd(indices, updates));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> execute(fixture, artifact));

        assertTrue(exception.getMessage().contains("NONE reduction does not allow duplicate"));
    }

    @Test
    void preparedScatterNdSupportsBatchDimsOne() {
        Tensor data = new Tensor(new double[]{1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d}, new int[]{2, 3}, null,
                "scatterNdBatchData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2, 1, 1}, null, "scatterNdBatchIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{7.0d, 8.0d}, new int[]{2, 1}, null,
                "scatterNdBatchUpdates", DataType.FLOAT64);

        Tensor output = execute(fixture(data.scatterNd(indices, updates, ScatterReduction.ADD, 1)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new double[]{1.0d, 2.0d, 10.0d, 12.0d, 5.0d, 6.0d},
                output.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void preparedScatterNdSupportsBf16AndBoolNone() {
        short[] bf16Data = new short[]{
                TensorDTypeOps.toBFloat16Bits(1.0f),
                TensorDTypeOps.toBFloat16Bits(2.0f),
                TensorDTypeOps.toBFloat16Bits(3.0f)
        };
        short[] bf16Updates = new short[]{TensorDTypeOps.toBFloat16Bits(9.0f)};
        Tensor bf16 = new Tensor(bf16Data, new int[]{1, 3}, null, "scatterNdBf16Data", DataType.BFLOAT16);
        Tensor indices = new Tensor(new int[]{0, -1}, new int[]{1, 2}, null, "scatterNdBf16Indices", DataType.INT32);
        Tensor bf16UpdatesTensor = new Tensor(bf16Updates, new int[]{1}, null, "scatterNdBf16Updates", DataType.BFLOAT16);

        Tensor bf16Output = execute(fixture(bf16.scatterNd(indices, bf16UpdatesTensor)),
                Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new short[]{bf16Data[0], bf16Data[1], bf16Updates[0]},
                TensorInternalAccess.bfloat16Data(bf16Output));

        Tensor bools = new Tensor(new byte[]{1, 0, 1}, new int[]{1, 3}, null, "scatterNdBoolData", DataType.BOOL);
        Tensor boolUpdates = new Tensor(new byte[]{0}, new int[]{1}, null, "scatterNdBoolUpdates", DataType.BOOL);

        Tensor boolOutput = execute(fixture(bools.scatterNd(indices, boolUpdates)), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new boolean[]{true, false, false}, boolOutput.toBooleanArrayCopy());
        Tensor boolAddOutput = new Tensor(
                new int[]{1, 3},
                List.of(bools, indices, boolUpdates),
                new scatterNd(ScatterReduction.ADD),
                "scatter_nd_bool_add_output",
                DataType.BOOL
        );
        assertThrows(UnsupportedOperationException.class,
                () -> prepareRoot(fixture(boolAddOutput), Cpu1PrepareConfig.scalarSingleThread()));
    }

    @Test
    void preparedScatterNdMemorySegmentWritesDensePath() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "scatterNdSegmentData", DataType.FLOAT64);
        Tensor indices = new Tensor(new long[]{0L, -1L}, new int[]{1, 2}, null, "scatterNdSegmentIndices", DataType.INT64);
        Tensor updates = new Tensor(new double[]{7.0d}, new int[]{1}, null, "scatterNdSegmentUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterNd(indices, updates));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertIndexKernel(artifact, Cpu1IndexKernelId.SCATTER_ND_F64_I64_DENSE_SEGMENT);

        ExecutionResult result = executeNative(fixture, artifact);

        assertArrayEquals(new double[]{10.0d, 20.0d, 7.0d}, nativeF64(result.nativeOutput(), 3), 1.0e-12);
    }

    @Test
    void prepareRejectsNonDenseScatterWriteDataIndicesUpdatesAndOutputAccess() {
        Tensor data = new Tensor(new double[]{1.0d, 2.0d, 3.0d, 4.0d}, new int[]{2, 2}, null,
                "scatterWriteAccessData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 1, 1, 0}, new int[]{2, 2}, null,
                "scatterWriteAccessIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{5.0d, 6.0d, 7.0d, 8.0d}, new int[]{2, 2}, null,
                "scatterWriteAccessUpdates", DataType.FLOAT64);

        UnsupportedOperationException dataException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(data.permute(1, 0).scatterElements(indices, updates, 1)),
                        Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(dataException.getMessage().contains("input access"));
        assertTrue(dataException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor stridedIndices = new Tensor(
                new int[]{2, 2},
                new int[]{1, 2},
                (List<Tensor>) null,
                (operations.Operation) null,
                "scatterWriteStridedIndices",
                DataType.INT32
        );
        Tensor stridedIndexOutput = new Tensor(
                new int[]{2, 2},
                List.of(data, stridedIndices, updates),
                new scatterElements(1, ScatterReduction.NONE),
                "scatter_write_strided_indices_output",
                DataType.FLOAT64
        );
        UnsupportedOperationException indexException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(stridedIndexOutput), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(indexException.getMessage().contains("indices access"));
        assertTrue(indexException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor stridedUpdates = new Tensor(
                new int[]{2, 2},
                new int[]{1, 2},
                (List<Tensor>) null,
                (operations.Operation) null,
                "scatterWriteStridedUpdates",
                DataType.FLOAT64
        );
        Tensor stridedUpdateOutput = new Tensor(
                new int[]{2, 2},
                List.of(data, indices, stridedUpdates),
                new scatterElements(1, ScatterReduction.NONE),
                "scatter_write_strided_updates_output",
                DataType.FLOAT64
        );
        UnsupportedOperationException updateException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(stridedUpdateOutput), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(updateException.getMessage().contains("updates access"));
        assertTrue(updateException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor offsetOutput = new Tensor(
                new int[]{2, 2},
                new int[]{2, 1},
                1,
                List.of(data, indices, updates),
                new scatterElements(1, ScatterReduction.NONE),
                "scatter_write_offset_output",
                DataType.FLOAT64
        );
        UnsupportedOperationException outputException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture(offsetOutput), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(outputException.getMessage().contains("output access"));
        assertTrue(outputException.getMessage().contains(Cpu1StorageAccessKind.DENSE_WITH_OFFSET.name()));
    }

    @Test
    void artifactAndTraceExposeScatterWriteReductionAndDeterministicLaunch() {
        Tensor data = new Tensor(new double[]{10.0d, 20.0d, 30.0d}, new int[]{1, 3}, null,
                "traceScatterWriteData", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1}, new int[]{1, 2}, null, "traceScatterWriteIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{2.0d, 3.0d}, new int[]{1, 2}, null,
                "traceScatterWriteUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(data.scatterElements(indices, updates, 1, ScatterReduction.ADD));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(8));
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I32_DENSE_ARRAY.name(), trace.kernel());
        assertEquals("SCATTER_ELEMENTS", trace.metadata().attributes().get("cpu1IndexOpType"));
        assertEquals("ADD", trace.metadata().attributes().get("cpu1IndexReduction"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexUpdateElements"));
        assertEquals(1, trace.metadata().attributes().get("cpu1IndexLaunchWorkers"));
    }

    @Test
    void artifactAndTraceExposeScatterKernelStorageAndDeterministicLaunch() {
        Tensor base = new Tensor(
                new double[]{
                        10.0d, 20.0d, 30.0d,
                        40.0d, 50.0d, 60.0d
                },
                new int[]{2, 3},
                null,
                "traceScatterBase",
                DataType.FLOAT64
        );
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "traceScatterIndices", DataType.INT32);
        Tensor updates = new Tensor(new double[]{1.0d, 5.0d}, new int[]{2}, null, "traceScatterUpdates", DataType.FLOAT64);
        Fixture fixture = fixture(base.scatterAdd(indices, updates, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(8));
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_ARRAY.name(), trace.kernel());
        assertEquals(
                Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_ARRAY.name(),
                trace.metadata().attributes().get("cpu1IndexKernelId")
        );
        assertEquals(Cpu1StorageKind.JAVA_ARRAY.name(), trace.metadata().attributes().get("cpu1StorageKind"));
        assertEquals("SCATTER_ADD", trace.metadata().attributes().get("cpu1IndexOpType"));
        assertEquals(true, trace.metadata().attributes().get("cpu1IndexHasUpdates"));
        assertEquals(2, trace.metadata().attributes().get("cpu1IndexUpdateElements"));
        assertEquals(1, trace.metadata().attributes().get("cpu1IndexLaunchWorkers"));
    }

    private static Tensor execute(Fixture fixture, Cpu1PrepareConfig config) {
        return execute(fixture, prepareRoot(fixture, config));
    }

    private static Tensor execute(Fixture fixture, Cpu1PreparedArtifact artifact) {
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context.runtimeTensorForNodeId(fixture.node().id());
    }

    private static ExecutionResult executeNative(Fixture fixture, Cpu1PreparedArtifact artifact) {
        PreparedStepMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeLeaves(context, fixture);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        NativeTensorStorage nativeOutput = context.nativeStorageForNodeId(fixture.node().id());
        assertNotNull(nativeOutput, "cpu1 segment index execution must attach native output storage");
        return new ExecutionResult(context.runtimeTensorForNodeId(fixture.node().id()), nativeOutput, context);
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertIndexKernel(Cpu1PreparedArtifact artifact, Cpu1IndexKernelId expected) {
        Cpu1IndexExecutableUnit executable = assertInstanceOf(Cpu1IndexExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedIndexUnit().kernelId());
        assertSame(artifact.preparedIndexUnit(), executable.preparedUnit());
    }

    private static void assertIndexPolicy(Cpu1PreparedArtifact artifact, Cpu1StorageKind storageKind) {
        Cpu1PreparedIndexUnit unit = artifact.preparedIndexUnit();
        assertEquals(storageKind, unit.storageKind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.inputAccessPlan().kind());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.indexAccessPlan().kind());
        if (unit.hasUpdateInput()) {
            assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.updateAccessPlan().kind());
        }
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.outputAccessPlan().kind());
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(
            Fixture fixture,
            Map<Integer, PreparedStepMetadata> metadataIndex
    ) {
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static PreparedStepMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact,
                runtime.execution.InputResidencyRequirement.cpuReadableAll(),
                runtime.execution.OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private static void attachNativeLeaves(ExecutionContext context, Fixture fixture) {
        NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
        for (CompiledNode node : fixture.nodes()) {
            if (!node.leaf()) {
                continue;
            }
            Tensor tensor = context.runtimeTensorForNodeId(node.id());
            NativeTensorStorage storage = storageFactory.allocate(
                    tensor.getDataType(),
                    tensor.getFlatDataSize(),
                    "cpu1-index-test-native-input-" + node.id()
            );
            copyToNative(tensor, storage);
            context.attachNativeStorage(node.id(), storage, "cpu1 index test native leaf");
        }
    }

    private static void copyToNative(Tensor tensor, NativeTensorStorage storage) {
        switch (tensor.getDataType()) {
            case FLOAT32 -> {
                NativeFloat32Storage out = assertInstanceOf(NativeFloat32Storage.class, storage);
                float[] source = TensorInternalAccess.float32Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setFloat32At(i, source[i]);
                }
            }
            case FLOAT64 -> {
                NativeFloat64Storage out = assertInstanceOf(NativeFloat64Storage.class, storage);
                double[] source = TensorInternalAccess.float64Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setFloat64At(i, source[i]);
                }
            }
            case BFLOAT16 -> {
                NativeBFloat16Storage out = assertInstanceOf(NativeBFloat16Storage.class, storage);
                short[] source = TensorInternalAccess.bfloat16Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setBFloat16BitsAt(i, source[i]);
                }
            }
            case INT32 -> {
                NativeInt32Storage out = assertInstanceOf(NativeInt32Storage.class, storage);
                int[] source = TensorInternalAccess.int32Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setInt32At(i, source[i]);
                }
            }
            case INT64 -> {
                NativeInt64Storage out = assertInstanceOf(NativeInt64Storage.class, storage);
                long[] source = TensorInternalAccess.int64Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setInt64At(i, source[i]);
                }
            }
            case BOOL -> {
                NativeBoolStorage out = assertInstanceOf(NativeBoolStorage.class, storage);
                byte[] source = TensorInternalAccess.boolData(tensor);
                for (int i = 0; i < source.length; i++) {
                    out.setBoolAt(i, source[i]);
                }
            }
        }
    }

    private static float[] nativeF32(NativeTensorStorage storage, int elements) {
        NativeFloat32Storage source = assertInstanceOf(NativeFloat32Storage.class, storage);
        float[] out = new float[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = source.getFloat32At(i);
        }
        return out;
    }

    private static double[] nativeF64(NativeTensorStorage storage, int elements) {
        NativeFloat64Storage source = assertInstanceOf(NativeFloat64Storage.class, storage);
        double[] out = new double[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = source.getFloat64At(i);
        }
        return out;
    }

    private static short[] nativeBf16Bits(NativeTensorStorage storage, int elements) {
        NativeBFloat16Storage source = assertInstanceOf(NativeBFloat16Storage.class, storage);
        short[] out = new short[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = source.getBFloat16BitsAt(i);
        }
        return out;
    }

    private static byte[] nativeBool(NativeTensorStorage storage, int elements) {
        NativeBoolStorage source = assertInstanceOf(NativeBoolStorage.class, storage);
        byte[] out = new byte[elements];
        for (int i = 0; i < elements; i++) {
            out[i] = source.getBoolAt(i);
        }
        return out;
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] source = TensorInternalAccess.bfloat16Data(tensor);
        float[] out = new float[source.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(source[i]);
        }
        return out;
    }

    private record ExecutionResult(Tensor output, NativeTensorStorage nativeOutput, ExecutionContext context) {
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
