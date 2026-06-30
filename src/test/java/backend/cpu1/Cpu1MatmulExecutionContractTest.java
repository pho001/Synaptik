package backend.cpu1;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import backend.contract.ComputeBackend;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.provider.matmul.Cpu1JavaScalarMatmulProvider;
import backend.cpu1.provider.matmul.Cpu1MatmulProvider;
import backend.cpu1.provider.matmul.Cpu1MatmulProviders;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.provider.matmul.Cpu1OpenBlasArrayMatmulProvider;
import backend.cpu1.provider.matmul.Cpu1OpenBlasNativeSegmentMatmulProvider;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1MatmulPreparer;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.CompiledGraph;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import trace.execution.ExecutionStepTrace;
import trace.execution.RunTrace;
import runtime.runner.StepExecutionTracer;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1MatmulExecutionContractTest {
    @Test
    void matmulRouteEnumIncludesPlannedRoutes() {
        assertArrayEquals(
                new Cpu1MatmulRoute[]{
                        Cpu1MatmulRoute.JAVA_SCALAR,
                        Cpu1MatmulRoute.AUTO,
                        Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING,
                        Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT
                },
                Cpu1MatmulRoute.values()
        );
    }

    @Test
    void prepareConfigFactoriesDefaultMatmulRouteToJavaScalar() {
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);
        Cpu1PrepareConfig[] configs = new Cpu1PrepareConfig[]{
                Cpu1PrepareConfig.scalarSingleThread(),
                Cpu1PrepareConfig.vectorSingleThread(),
                Cpu1PrepareConfig.vectorParallel(2),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread(),
                Cpu1PrepareConfig.vectorMemorySegmentSingleThread(),
                Cpu1PrepareConfig.automatic(),
                Cpu1PrepareConfig.automatic(DataType.FLOAT32, ExecutionMode.FORWARD),
                Cpu1PrepareConfig.automatic(2),
                Cpu1PrepareConfig.automatic(DataType.FLOAT32, ExecutionMode.FORWARD, 2),
                Cpu1PrepareConfig.automatic(tuned, 2),
                Cpu1PrepareConfig.automatic(tuned, 2, Cpu1StorageKind.MEMORY_SEGMENT)
        };

        for (Cpu1PrepareConfig config : configs) {
            assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, config.matmulRoute());
        }
    }

    @Test
    void withMatmulRouteOverridesRouteAndPreservesOtherConfigValues() {
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);
        BlasConfig blasConfig = openBlasConfig(10, BlasStorageMode.CPU_ARRAY);
        Cpu1PrepareConfig base = Cpu1PrepareConfig
                .automatic(tuned, 3, Cpu1StorageKind.MEMORY_SEGMENT)
                .withBlasConfig(blasConfig)
                .withApproximation(true, false);

        Cpu1PrepareConfig overridden = base.withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);

        assertEquals(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT, overridden.matmulRoute());
        assertEquals(base.vectorizationKind(), overridden.vectorizationKind());
        assertEquals(base.launchConfig(), overridden.launchConfig());
        assertEquals(base.storageKind(), overridden.storageKind());
        assertEquals(base.useFastExpApprox(), overridden.useFastExpApprox());
        assertEquals(base.useFastTanhApprox(), overridden.useFastTanhApprox());
        assertEquals(base.automaticVectorization(), overridden.automaticVectorization());
        assertEquals(base.automaticLaunch(), overridden.automaticLaunch());
        assertSame(base.cpuKernelConfig(), overridden.cpuKernelConfig());
        assertSame(base.blasConfig(), overridden.blasConfig());
    }

    @Test
    void automaticFactoryCarriesRuntimeBlasConfig() {
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(7);
        BlasConfig blasConfig = openBlasConfig(11, BlasStorageMode.AUTO);
        RuntimeConfig runtimeConfig = new RuntimeConfig(cpuKernelConfig, ApproximationConfig.defaults(), blasConfig);

        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(runtimeConfig, 2);

        assertSame(cpuKernelConfig, config.cpuKernelConfig());
        assertSame(blasConfig, config.blasConfig());
    }

    @Test
    void prepareCarriesRouteSpecificOpenBlasThreadOverrides() {
        BlasConfig blasConfig = openBlasConfig(1L, BlasStorageMode.AUTO)
                .withThreads(2)
                .withOpenBlasRouteThreads(4, 16);

        Cpu1PreparedArtifact arrayArtifact = prepareRoot(
                simpleF32MatmulFixture(),
                Cpu1PrepareConfig.scalarSingleThread()
                        .withBlasConfig(blasConfig)
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING)
        );
        Cpu1PreparedArtifact nativeArtifact = prepareRoot(
                simpleF32MatmulFixture(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                        .withBlasConfig(blasConfig)
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT)
        );
        Cpu1PreparedArtifact javaArtifact = prepareRoot(
                simpleF32MatmulFixture(),
                Cpu1PrepareConfig.scalarSingleThread()
                        .withBlasConfig(blasConfig)
                        .withMatmulRoute(Cpu1MatmulRoute.JAVA_SCALAR)
        );

        assertEquals(4, arrayArtifact.preparedMatmulUnit().openBlasThreads());
        assertEquals(16, nativeArtifact.preparedMatmulUnit().openBlasThreads());
        assertEquals(0, javaArtifact.preparedMatmulUnit().openBlasThreads());
    }

    @Test
    void prepareFallsBackToGenericOpenBlasThreadConfigWhenRouteOverrideIsUnset() {
        BlasConfig blasConfig = openBlasConfig(1L, BlasStorageMode.AUTO).withThreads(6);

        Cpu1PreparedArtifact arrayArtifact = prepareRoot(
                simpleF32MatmulFixture(),
                Cpu1PrepareConfig.scalarSingleThread()
                        .withBlasConfig(blasConfig)
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING)
        );
        Cpu1PreparedArtifact nativeArtifact = prepareRoot(
                simpleF32MatmulFixture(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                        .withBlasConfig(blasConfig)
                        .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT)
        );

        assertEquals(6, arrayArtifact.preparedMatmulUnit().openBlasThreads());
        assertEquals(6, nativeArtifact.preparedMatmulUnit().openBlasThreads());
    }

    @Test
    void javaScalarProviderExposesPreparedRouteAndDenseScalarKernels() {
        Cpu1MatmulProvider provider = new Cpu1JavaScalarMatmulProvider();

        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, provider.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR, provider.kernelId(DataType.FLOAT32));
        assertEquals(Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR, provider.kernelId(DataType.FLOAT64));
        assertEquals(Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR, provider.kernelId(DataType.BFLOAT16));
    }

    @Test
    void matmulProviderFactoryReturnsJavaScalarProviderForJavaScalarRoute() {
        Cpu1MatmulProvider provider = Cpu1MatmulProviders.forRoute(Cpu1MatmulRoute.JAVA_SCALAR);

        assertInstanceOf(Cpu1JavaScalarMatmulProvider.class, provider);
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, provider.route());
    }

    @Test
    void openBlasArrayProviderExposesF32AndF64KernelIds() {
        Cpu1MatmulProvider provider = Cpu1MatmulProviders.forRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);

        assertInstanceOf(Cpu1OpenBlasArrayMatmulProvider.class, provider);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING, provider.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING, provider.kernelId(DataType.FLOAT32));
        assertEquals(Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING, provider.kernelId(DataType.FLOAT64));

        UnsupportedOperationException bf16 = assertThrows(
                UnsupportedOperationException.class,
                () -> provider.kernelId(DataType.BFLOAT16)
        );
        assertTrue(bf16.getMessage().contains("OPENBLAS_ARRAY_COPYING"));
        assertTrue(bf16.getMessage().contains(DataType.BFLOAT16.name()));
    }

    @Test
    void openBlasNativeSegmentProviderExposesF32AndF64KernelIds() {
        Cpu1MatmulProvider provider = Cpu1MatmulProviders.forRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);

        assertInstanceOf(Cpu1OpenBlasNativeSegmentMatmulProvider.class, provider);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT, provider.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT, provider.kernelId(DataType.FLOAT32));
        assertEquals(Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_NATIVE_SEGMENT, provider.kernelId(DataType.FLOAT64));

        UnsupportedOperationException bf16 = assertThrows(
                UnsupportedOperationException.class,
                () -> provider.kernelId(DataType.BFLOAT16)
        );
        assertTrue(bf16.getMessage().contains("OPENBLAS_NATIVE_SEGMENT"));
        assertTrue(bf16.getMessage().contains(DataType.BFLOAT16.name()));
    }

    @Test
    void matmulProviderFactoryRejectsAutoSelectorRoute() {
        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> Cpu1MatmulProviders.forRoute(Cpu1MatmulRoute.AUTO)
        );

        assertTrue(exception.getMessage().contains(Cpu1MatmulRoute.AUTO.name()));
        assertTrue(exception.getMessage().contains("must be resolved"));
    }

    @Test
    void explicitJavaScalarRoutePreparesSameDenseScalarKernel() {
        Fixture fixture = simpleF32MatmulFixture();

        Cpu1PreparedArtifact artifact = prepareRoot(
                fixture,
                Cpu1PrepareConfig.scalarSingleThread().withMatmulRoute(Cpu1MatmulRoute.JAVA_SCALAR)
        );

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
    }

    @Test
    void prepareDefaultsMatmulPostOpToNone() {
        Cpu1PreparedArtifact artifact = prepareRoot(simpleF32MatmulFixture(), Cpu1PrepareConfig.scalarSingleThread());

        assertEquals(Cpu1MatmulPostOp.NONE, artifact.preparedMatmulUnit().postOp());
    }

    @Test
    void prepareKeepsScalarMatmulForExplicitScalarF32AndBf16VectorRequests() {
        Cpu1PreparedArtifact f32Artifact = prepareRoot(simpleF32MatmulFixture(), Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(f32Artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1VectorizationKind.SCALAR, f32Artifact.preparedMatmulUnit().vectorizationKind());

        Tensor bf16Left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "bf16Left",
                DataType.BFLOAT16
        );
        Tensor bf16Right = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "bf16Right",
                DataType.BFLOAT16
        );
        Cpu1PreparedArtifact bf16Artifact = prepareRoot(fixture(bf16Left.matmul(bf16Right)), Cpu1PrepareConfig.vectorSingleThread());
        assertMatmulKernel(bf16Artifact, Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR);
        assertEquals(Cpu1VectorizationKind.SCALAR, bf16Artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void prepareSelectsPackedBVectorKernelForExplicitF32VectorConfig() {
        Fixture fixture = simpleF32MatmulFixture();

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(6, artifact.scratchBufferSpec().f32ArrayElements());
        assertEquals(0, artifact.scratchBufferSpec().f64ArrayElements());
    }

    @Test
    void prepareSelectsPackedBVectorKernelForExplicitF64VectorConfig() {
        Fixture fixture = f64MatmulFixture(2, 3, 2);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(0, artifact.scratchBufferSpec().f32ArrayElements());
        assertEquals(6, artifact.scratchBufferSpec().f64ArrayElements());
    }

    @Test
    void preparedMatmulStoresConfiguredLaunchConfig() {
        Fixture fixture = simpleF32MatmulFixture();
        Cpu1LaunchConfig launchConfig = Cpu1LaunchConfig.parallel(3, 7);
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                Cpu1VectorizationKind.SCALAR,
                launchConfig,
                Cpu1StorageKind.JAVA_ARRAY
        );

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertSame(launchConfig, artifact.preparedMatmulUnit().launchConfig());
    }

    @Test
    void automaticMatmulLaunchKeepsSmallWorkSingleThread() {
        Fixture fixture = simpleF32MatmulFixture();
        CpuKernelConfig cpuKernelConfig = matmulLaunchCpuKernelConfig(13, 2);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 4);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1LaunchConfig.singleThread(), artifact.preparedMatmulUnit().launchConfig());
    }

    @Test
    void automaticMatmulLaunchChoosesParallelForLargerWorkWhenThresholdAllows() {
        Fixture fixture = f32MatmulFixture(10, 3, 2);
        CpuKernelConfig cpuKernelConfig = matmulLaunchCpuKernelConfig(1, 2);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 3);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1LaunchConfig.parallel(3, 2), artifact.preparedMatmulUnit().launchConfig());
    }

    @Test
    void automaticMatmulVectorizationKeepsF32ScalarWhenCheapVectorThresholdIsAboveWork() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work + 1);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulVectorizationSelectsF32PackedBVectorWhenCheapVectorThresholdAllows() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulVectorizationSelectsF64PackedBVectorWhenCheapVectorThresholdAllows() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f64MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulRouteFallsBackToJavaPolicyWhenBlasDisabled() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1)
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulRouteFallsBackToJavaPolicyBelowBlasMinWork() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work + 1);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1)
                .withBlasConfig(openBlasConfig(work + 1L, BlasStorageMode.CPU_ARRAY))
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulRouteFallsBackToJavaPolicyForNativeBlasStorageMode() {
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(work);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1)
                .withBlasConfig(openBlasConfig(work, BlasStorageMode.CPU_NATIVE))
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulRouteFallsBackToJavaPolicyForBf16() {
        Tensor bf16Left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "bf16Left",
                DataType.BFLOAT16
        );
        Tensor bf16Right = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "bf16Right",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(bf16Left.matmul(bf16Right));
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(1);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 1)
                .withBlasConfig(openBlasConfig(1L, BlasStorageMode.CPU_ARRAY))
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
    }

    @Test
    void automaticMatmulRouteSelectsF32OpenBlasArrayCopyingWhenEligible() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f32MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(Integer.MAX_VALUE);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(cpuKernelConfig, 4)
                .withBlasConfig(openBlasConfig(work, BlasStorageMode.CPU_ARRAY))
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), artifact.preparedMatmulUnit().launchConfig());
    }

    @Test
    void automaticMatmulRouteSelectsF64OpenBlasArrayCopyingFromRuntimeConfigWhenEligible() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat64GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int m = 2;
        int k = 3;
        int n = 2;
        Fixture fixture = f64MatmulFixture(m, k, n);
        int work = matmulWork(m, k, n);
        CpuKernelConfig cpuKernelConfig = matmulVectorCpuKernelConfig(Integer.MAX_VALUE);
        RuntimeConfig runtimeConfig = new RuntimeConfig(
                cpuKernelConfig,
                ApproximationConfig.defaults(),
                openBlasConfig(work, BlasStorageMode.AUTO)
        );
        Cpu1PrepareConfig config = Cpu1PrepareConfig.automatic(runtimeConfig, 4)
                .withMatmulRoute(Cpu1MatmulRoute.AUTO);

        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);

        assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING, artifact.preparedMatmulUnit().route());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), artifact.preparedMatmulUnit().launchConfig());
    }

    @Test
    void openBlasArrayCopyingRoutePreparesF32AndF64Kernels() {
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);

        Cpu1PreparedArtifact f32 = prepareRoot(simpleF32MatmulFixture(), config);
        Cpu1PreparedArtifact f64 = prepareRoot(f64MatmulFixture(2, 3, 2), config);

        assertMatmulKernel(f32, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING, f32.preparedMatmulUnit().route());
        assertEquals(Cpu1VectorizationKind.SCALAR, f32.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), f32.preparedMatmulUnit().launchConfig());
        assertMatmulKernel(f64, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING, f64.preparedMatmulUnit().route());
        assertEquals(Cpu1VectorizationKind.SCALAR, f64.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), f64.preparedMatmulUnit().launchConfig());
    }

    @Test
    void openBlasNativeSegmentRouteRejectsJavaArrayStorageAtPrepare() {
        Fixture fixture = simpleF32MatmulFixture();

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(
                        fixture,
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT)
                )
        );

        assertTrue(exception.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT.name()));
        assertTrue(exception.getMessage().contains(Cpu1StorageKind.MEMORY_SEGMENT.name()));
    }

    @Test
    void openBlasNativeSegmentRoutePreparesF32AndF64KernelsForMemorySegmentStorage() {
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);

        Cpu1PreparedArtifact f32 = prepareRoot(simpleF32MatmulFixture(), config);
        Cpu1PreparedArtifact f64 = prepareRoot(f64MatmulFixture(2, 3, 2), config);

        assertMatmulKernel(f32, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT, f32.preparedMatmulUnit().route());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, f32.preparedMatmulUnit().storageKind());
        assertEquals(Cpu1VectorizationKind.SCALAR, f32.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), f32.preparedMatmulUnit().launchConfig());
        assertMatmulKernel(f64, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT, f64.preparedMatmulUnit().route());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, f64.preparedMatmulUnit().storageKind());
        assertEquals(Cpu1VectorizationKind.SCALAR, f64.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), f64.preparedMatmulUnit().launchConfig());
    }

    @Test
    void openBlasNativeSegmentRouteRejectsBf16AtPrepare() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "left",
                DataType.BFLOAT16
        );
        Tensor right = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "right",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, config)
        );

        assertTrue(exception.getMessage().contains("OPENBLAS_NATIVE_SEGMENT"));
        assertTrue(exception.getMessage().contains(DataType.BFLOAT16.name()));
    }

    @Test
    void openBlasRoutesRejectMatmulPostOpAtPrepare() {
        Fixture fixture = simpleF32MatmulFixture();

        UnsupportedOperationException arrayException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(
                        fixture,
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING),
                        Cpu1MatmulPostOp.RELU
                )
        );
        UnsupportedOperationException nativeException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(
                        fixture,
                        Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT),
                        Cpu1MatmulPostOp.RELU
                )
        );
        Tensor bias = new Tensor(new float[]{1.0f, -1.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Fixture biasFixture = fixture(fixture.root().add(bias).relu());
        CompiledNode matmulNode = node(biasFixture.nodes(), Operation.OpType.MATMUL);
        CompiledNode addNode = node(biasFixture.nodes(), Operation.OpType.ADD);
        UnsupportedOperationException biasException = assertThrows(
                UnsupportedOperationException.class,
                () -> new Cpu1MatmulPreparer().prepareMatmulBiasEpilogue(
                        matmulNode,
                        addNode,
                        biasFixture.node(),
                        biasFixture.descriptorIndex(),
                        Cpu1PrepareConfig.scalarSingleThread()
                                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING),
                        Cpu1MatmulPostOp.ADD_BIAS_RELU
                )
        );

        assertTrue(arrayException.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING.name()));
        assertTrue(arrayException.getMessage().contains(Cpu1MatmulPostOp.RELU.name()));
        assertTrue(nativeException.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT.name()));
        assertTrue(nativeException.getMessage().contains(Cpu1MatmulPostOp.RELU.name()));
        assertTrue(biasException.getMessage().contains(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING.name()));
        assertTrue(biasException.getMessage().contains(Cpu1MatmulPostOp.ADD_BIAS_RELU.name()));
    }

    @Test
    void preparedF32MatmulRunsDenseJavaScalarRoute() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                new int[]{3, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{58.0f, 64.0f, 139.0f, 154.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF32ScalarMatmulAppliesReluPostOpBeforeStore() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, -2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        2.0f, -1.0f,
                        3.0f, -5.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(
                fixture,
                Cpu1PrepareConfig.scalarSingleThread(),
                Cpu1MatmulPostOp.RELU
        );
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(Cpu1MatmulPostOp.RELU, artifact.preparedMatmulUnit().postOp());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{0.0f, 9.0f, 18.0f, 0.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedGraphSpecializesMatmulReluAsSingleCpu1MatmulPostOpStep() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, -2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        2.0f, -1.0f,
                        3.0f, -5.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Tensor relu = left.matmul(right).relu();
        PreparedExecution execution = CompiledGraph.compile(relu, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults(DataType.FLOAT32));

        PreparedExecutionStep step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.metadata().artifact() instanceof Cpu1PreparedArtifact artifact
                        && artifact.executableUnit() instanceof Cpu1MatmulExecutableUnit)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, step.metadata().artifact());

        assertEquals(Operation.OpType.RELU, step.compiledNode().operation().opType());
        assertEquals(2, step.orderedNodeIds().size());
        assertEquals(List.of(step.compiledNode().id()), step.boundaryOutputNodeIds());
        assertEquals(step.compiledNode().id(), artifact.preparedMatmulUnit().nodeId());
        assertEquals(Cpu1MatmulPostOp.RELU, artifact.preparedMatmulUnit().postOp());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertEquals(2, step.metadata().executionInputNodeIds().size());

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{0.0f, 9.0f, 18.0f, 0.0f}, relu.toFloat32ArrayCopy(), 1.0e-6f);
        ExecutionStepTrace tracedStep = trace.steps().stream()
                .filter(candidate -> "RELU".equals(candidate.metadata().attributes().get("cpu1MatmulPostOp")))
                .findFirst()
                .orElseThrow();
        assertEquals("RELU", tracedStep.metadata().attributes().get("cpu1MatmulPostOp"));
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR.name(), tracedStep.metadata().attributes().get("cpu1MatmulRoute"));
    }

    @Test
    void preparedGraphSpecializesMatmulBiasReluAsSingleCpu1MatmulPostOpStep() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, -2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        2.0f, -1.0f,
                        3.0f, -5.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{5.0f, -20.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor relu = left.matmul(right).add(bias).relu();
        PreparedExecution execution = CompiledGraph.compile(relu, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults(DataType.FLOAT32));

        PreparedExecutionStep step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.metadata().artifact() instanceof Cpu1PreparedArtifact artifact
                        && artifact.executableUnit() instanceof Cpu1MatmulExecutableUnit)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, step.metadata().artifact());

        assertEquals(Operation.OpType.RELU, step.compiledNode().operation().opType());
        assertEquals(2, step.orderedNodeIds().size());
        assertEquals(List.of(step.compiledNode().id()), step.boundaryOutputNodeIds());
        assertEquals(step.compiledNode().id(), artifact.preparedMatmulUnit().nodeId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS_RELU, artifact.preparedMatmulUnit().postOp());
        assertTrue(artifact.preparedMatmulUnit().hasBias());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertEquals(3, step.metadata().executionInputNodeIds().size());

        RunTrace trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(new float[]{1.0f, 0.0f, 23.0f, 0.0f}, relu.toFloat32ArrayCopy(), 1.0e-6f);
        ExecutionStepTrace tracedStep = trace.steps().stream()
                .filter(candidate -> "ADD_BIAS_RELU".equals(candidate.metadata().attributes().get("cpu1MatmulPostOp")))
                .findFirst()
                .orElseThrow();
        assertEquals("ADD_BIAS_RELU", tracedStep.metadata().attributes().get("cpu1MatmulPostOp"));
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR.name(), tracedStep.metadata().attributes().get("cpu1MatmulRoute"));
    }

    @Test
    void preparedF32ScalarMatmulAppliesBiasReluPostOpBeforeStore() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, -2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        2.0f, -1.0f,
                        3.0f, -5.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{5.0f, -20.0f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor relu = left.matmul(right).add(bias).relu();
        Fixture fixture = fixture(relu);
        CompiledNode matmulNode = node(fixture.nodes(), Operation.OpType.MATMUL);
        CompiledNode addNode = node(fixture.nodes(), Operation.OpType.ADD);
        Cpu1PreparedArtifact artifact = new Cpu1MatmulPreparer().prepareMatmulBiasEpilogue(
                matmulNode,
                addNode,
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarSingleThread(),
                Cpu1MatmulPostOp.ADD_BIAS_RELU
        );

        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        assertEquals(fixture.node().id(), artifact.preparedMatmulUnit().nodeId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS_RELU, artifact.preparedMatmulUnit().postOp());
        assertTrue(artifact.preparedMatmulUnit().hasBias());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertEquals(nodeId(fixture.nodes(), "bias"), artifact.preparedMatmulUnit().biasNodeId());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                List.of(matmulNode.inputIds().get(0), matmulNode.inputIds().get(1), artifact.preparedMatmulUnit().biasNodeId()),
                artifact
        );
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{1.0f, 0.0f, 23.0f, 0.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF32VectorMatmulAppliesBiasReluPostOpBeforeStore() {
        int m = 2;
        int k = FloatVector.SPECIES_PREFERRED.length() + 3;
        int n = 3;
        float[] leftData = new float[m * k];
        float[] rightData = new float[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 5) - 2.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 7) - 3.0f;
        }
        float[] biasData = new float[]{3.0f, -10.0f, 0.5f};
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT32);
        Tensor bias = new Tensor(biasData, new int[]{n}, null, "bias", DataType.FLOAT32);
        Tensor relu = left.matmul(right).add(bias).relu();
        Fixture fixture = fixture(relu);
        CompiledNode matmulNode = node(fixture.nodes(), Operation.OpType.MATMUL);
        CompiledNode addNode = node(fixture.nodes(), Operation.OpType.ADD);
        Cpu1PreparedArtifact artifact = new Cpu1MatmulPreparer().prepareMatmulBiasEpilogue(
                matmulNode,
                addNode,
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread(),
                Cpu1MatmulPostOp.ADD_BIAS_RELU
        );
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS_RELU, artifact.preparedMatmulUnit().postOp());
        assertEquals(nodeId(fixture.nodes(), "bias"), artifact.preparedMatmulUnit().biasNodeId());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                List.of(matmulNode.inputIds().get(0), matmulNode.inputIds().get(1), artifact.preparedMatmulUnit().biasNodeId()),
                artifact
        );
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                relu(addColumnBias(expectedBatchedF32Matmul(leftData, rightData, 1, m, k, n), biasData, m, n)),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-5f
        );
    }

    @Test
    void preparedF32VectorMatmulPacksBAndReturnsExpectedResults() {
        int m = 2;
        int k = FloatVector.SPECIES_PREFERRED.length() + 3;
        int n = 3;
        float[] leftData = new float[m * k];
        float[] rightData = new float[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 3.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.5f;
        }
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                expectedBatchedF32Matmul(leftData, rightData, 1, m, k, n),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-5f
        );
    }

    @Test
    void preparedF32VectorMatmulAppliesReluPostOpBeforeStore() {
        int m = 2;
        int k = FloatVector.SPECIES_PREFERRED.length() + 3;
        int n = 3;
        float[] leftData = new float[m * k];
        float[] rightData = new float[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 5) - 2.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 7) - 3.0f;
        }
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(
                fixture,
                Cpu1PrepareConfig.vectorSingleThread(),
                Cpu1MatmulPostOp.RELU
        );
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Cpu1MatmulPostOp.RELU, artifact.preparedMatmulUnit().postOp());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                relu(expectedBatchedF32Matmul(leftData, rightData, 1, m, k, n)),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-5f
        );
    }

    @Test
    void preparedF32MatmulUsesExplicitParallelLaunchForBatchedRows() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f,
                        10.0f, 11.0f, 12.0f
                },
                new int[]{2, 2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
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
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1LaunchConfig launchConfig = Cpu1LaunchConfig.parallel(3, 1);
        Cpu1PreparedArtifact artifact = prepareRoot(
                fixture,
                new Cpu1PrepareConfig(Cpu1VectorizationKind.SCALAR, launchConfig, Cpu1StorageKind.JAVA_ARRAY)
        );
        assertSame(launchConfig, artifact.preparedMatmulUnit().launchConfig());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(new int[]{2, 2, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new float[]{
                        22.0f, 28.0f,
                        49.0f, 64.0f,
                        220.0f, 244.0f,
                        301.0f, 334.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedBatchedF32VectorMatmulUsesExplicitParallelLaunch() {
        int batchCount = 2;
        int m = 2;
        int k = FloatVector.SPECIES_PREFERRED.length() + 1;
        int n = 2;
        float[] leftData = new float[batchCount * m * k];
        float[] rightData = new float[batchCount * k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 11) - 5.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 13) * 0.25f;
        }
        Tensor left = new Tensor(leftData, new int[]{batchCount, m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{batchCount, k, n}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1LaunchConfig launchConfig = Cpu1LaunchConfig.parallel(3, 1);
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                launchConfig,
                Cpu1StorageKind.JAVA_ARRAY
        );
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertSame(launchConfig, artifact.preparedMatmulUnit().launchConfig());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_PACKED_B_VECTOR);
        assertEquals(Math.multiplyExact(batchCount, Math.multiplyExact(n, k)), artifact.scratchBufferSpec().f32ArrayElements());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(new int[]{batchCount, m, n}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                expectedBatchedF32Matmul(leftData, rightData, batchCount, m, k, n),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-5f
        );
    }

    @Test
    void preparedF64MatmulSupportsBroadcastBatch() {
        Tensor left = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d,
                        7.0d, 8.0d, 9.0d,
                        10.0d, 11.0d, 12.0d
                },
                new int[]{2, 2, 3},
                null,
                "left",
                DataType.FLOAT64
        );
        Tensor right = new Tensor(
                new double[]{
                        1.0d, 2.0d,
                        3.0d, 4.0d,
                        5.0d, 6.0d
                },
                new int[]{1, 3, 2},
                null,
                "right",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 2, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new double[]{
                        22.0d, 28.0d,
                        49.0d, 64.0d,
                        76.0d, 100.0d,
                        103.0d, 136.0d
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedF64VectorMatmulPacksBAndReturnsExpectedResults() {
        int m = 2;
        int k = DoubleVector.SPECIES_PREFERRED.length() + 3;
        int n = 3;
        double[] leftData = new double[m * k];
        double[] rightData = new double[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 3.0d;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.25d;
        }
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT64);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT64);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_DENSE_PACKED_B_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                expectedBatchedF64Matmul(leftData, rightData, 1, m, k, n),
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedF32OpenBlasArrayCopyingMatmulReturnsExpectedResultsWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int batchCount = 2;
        int m = 2;
        int k = 3;
        int n = 2;
        float[] leftData = new float[batchCount * m * k];
        float[] rightData = new float[batchCount * k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 2.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.5f;
        }
        Tensor left = new Tensor(leftData, new int[]{batchCount, m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{batchCount, k, n}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.vectorParallel(4)
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), artifact.preparedMatmulUnit().launchConfig());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedBatchedF32Matmul(leftData, rightData, batchCount, m, k, n),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-4f
        );
    }

    @Test
    void preparedF64OpenBlasArrayCopyingMatmulReturnsExpectedResultsWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat64GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int batchCount = 2;
        int m = 2;
        int k = 3;
        int n = 2;
        double[] leftData = new double[batchCount * m * k];
        double[] rightData = new double[batchCount * k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 2.0d;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.25d;
        }
        Tensor left = new Tensor(leftData, new int[]{batchCount, m, k}, null, "left", DataType.FLOAT64);
        Tensor right = new Tensor(rightData, new int[]{batchCount, k, n}, null, "right", DataType.FLOAT64);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.vectorParallel(4)
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_ARRAY_COPYING);
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedMatmulUnit().vectorizationKind());
        assertEquals(Cpu1LaunchConfig.singleThread(), artifact.preparedMatmulUnit().launchConfig());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedBatchedF64Matmul(leftData, rightData, batchCount, m, k, n),
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedF32OpenBlasNativeSegmentMatmulReturnsExpectedNativeResultsWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat32GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int batchCount = 2;
        int m = 2;
        int k = 3;
        int n = 2;
        float[] leftData = new float[batchCount * m * k];
        float[] rightData = new float[batchCount * k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 2.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.5f;
        }
        Tensor left = new Tensor(leftData, new int[]{batchCount, m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{batchCount, k, n}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().get(0), leftData);
        attachNativeF32Input(context, fixture.node().inputIds().get(1), rightData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        assertInstanceOf(NativeFloat32Storage.class, output);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        NativeTensorStorage secondOutput = context.nativeStorageForNodeId(fixture.node().id());
        assertSame(output, secondOutput);
        assertArrayEquals(
                expectedBatchedF32Matmul(leftData, rightData, batchCount, m, k, n),
                nativeF32Values(assertInstanceOf(NativeFloat32Storage.class, secondOutput)),
                1.0e-4f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF64OpenBlasNativeSegmentMatmulReturnsExpectedNativeResultsWhenAvailable() {
        Assumptions.assumeTrue(OpenBlasRuntime.isFloat64GemmAvailable(), OpenBlasRuntime.unavailableReason());
        int batchCount = 2;
        int m = 2;
        int k = 3;
        int n = 2;
        double[] leftData = new double[batchCount * m * k];
        double[] rightData = new double[batchCount * k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = (i % 7) - 2.0d;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = (i % 5) + 0.25d;
        }
        Tensor left = new Tensor(leftData, new int[]{batchCount, m, k}, null, "left", DataType.FLOAT64);
        Tensor right = new Tensor(rightData, new int[]{batchCount, k, n}, null, "right", DataType.FLOAT64);
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_OPENBLAS_NATIVE_SEGMENT);
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT, artifact.preparedMatmulUnit().storageKind());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF64Input(context, fixture.node().inputIds().get(0), leftData);
        attachNativeF64Input(context, fixture.node().inputIds().get(1), rightData);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeTensorStorage output = context.nativeStorageForNodeId(fixture.node().id());
        NativeFloat64Storage f64 = assertInstanceOf(NativeFloat64Storage.class, output);
        assertArrayEquals(
                expectedBatchedF64Matmul(leftData, rightData, batchCount, m, k, n),
                nativeF64Values(f64),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedBf16MatmulAccumulatesInF32AndStoresBf16() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.BFLOAT16
        );
        Tensor right = new Tensor(
                new float[]{
                        5.0f, 6.0f,
                        7.0f, 8.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{19.0f, 22.0f, 43.0f, 50.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-6f
        );
    }

    @Test
    void matmulTraceReportsPreparedRoute() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR.name(), trace.kernel());
        assertEquals("JAVA_SCALAR", trace.metadata().matMul().route());
        assertEquals("JAVA_SCALAR", trace.metadata().attributes().get("cpu1MatmulRoute"));
        assertEquals(1, trace.metadata().attributes().get("cpu1MatmulLaunchWorkers"));
        assertEquals(0, trace.metadata().attributes().get("cpu1MatmulLaunchChunkSize"));
    }

    @Test
    void matmulTraceReportsOpenBlasArrayCopyingRouteAndKernel() {
        Fixture fixture = simpleF32MatmulFixture();
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_ARRAY_COPYING);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );

        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING.name(), trace.kernel());
        assertEquals("OPENBLAS_ARRAY_COPYING", trace.metadata().matMul().route());
        assertTrue(trace.metadata().matMul().useBlas());
        assertEquals(false, trace.metadata().matMul().useBatchedBlas());
        assertEquals("OPENBLAS_FFM", trace.metadata().matMul().blasProvider());
        assertEquals("cblas_sgemm", trace.metadata().matMul().blasSymbol());
        assertEquals("OPENBLAS_ARRAY_COPYING", trace.metadata().matMul().blasRoute());
        assertEquals(48L, trace.metadata().matMul().copyInBytes());
        assertEquals(16L, trace.metadata().matMul().copyOutBytes());
        assertEquals("OPENBLAS_ARRAY_COPYING", trace.metadata().attributes().get("cpu1MatmulRoute"));
        assertEquals("OPENBLAS_FFM", trace.metadata().attributes().get("blasProvider"));
        assertEquals("cblas_sgemm", trace.metadata().attributes().get("blasSymbol"));
        assertEquals(OpenBlasRuntime.isFloat32GemmAvailable(), trace.metadata().attributes().get("openblasSgemmAvailable"));
        assertEquals(
                Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_ARRAY_COPYING.name(),
                trace.metadata().attributes().get("cpu1MatmulKernelId")
        );
    }

    @Test
    void matmulTraceReportsOpenBlasNativeSegmentRouteAndKernel() {
        Fixture fixture = simpleF32MatmulFixture();
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
                .withMatmulRoute(Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );

        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT.name(), trace.kernel());
        assertEquals("OPENBLAS_NATIVE_SEGMENT", trace.metadata().matMul().route());
        assertTrue(trace.metadata().matMul().useBlas());
        assertEquals(false, trace.metadata().matMul().useBatchedBlas());
        assertEquals("OPENBLAS_FFM", trace.metadata().matMul().blasProvider());
        assertEquals("cblas_sgemm", trace.metadata().matMul().blasSymbol());
        assertEquals("OPENBLAS_NATIVE_SEGMENT", trace.metadata().matMul().blasRoute());
        assertEquals(0L, trace.metadata().matMul().copyInBytes());
        assertEquals(0L, trace.metadata().matMul().copyOutBytes());
        assertEquals(Cpu1StorageKind.MEMORY_SEGMENT.name(), trace.metadata().matMul().actualCpuStorage());
        assertEquals("OPENBLAS_NATIVE_SEGMENT", trace.metadata().attributes().get("cpu1MatmulRoute"));
        assertEquals("OPENBLAS_FFM", trace.metadata().attributes().get("blasProvider"));
        assertEquals("cblas_sgemm", trace.metadata().attributes().get("blasSymbol"));
        assertEquals(0L, trace.metadata().attributes().get("matMulCopyInBytes"));
        assertEquals(0L, trace.metadata().attributes().get("matMulCopyOutBytes"));
        assertEquals(
                Cpu1MatmulKernelId.MATMUL_F32_OPENBLAS_NATIVE_SEGMENT.name(),
                trace.metadata().attributes().get("cpu1MatmulKernelId")
        );
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static Cpu1PreparedArtifact prepareRoot(
            Fixture fixture,
            Cpu1PrepareConfig config,
            Cpu1MatmulPostOp postOp
    ) {
        return new Cpu1MatmulPreparer().prepare(fixture.node(), fixture.descriptorIndex(), config, postOp);
    }

    private static void assertMatmulKernel(Cpu1PreparedArtifact artifact, Cpu1MatmulKernelId expected) {
        Cpu1MatmulExecutableUnit executable = assertInstanceOf(Cpu1MatmulExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedMatmulUnit().kernelId());
        assertSame(artifact.preparedMatmulUnit(), executable.preparedUnit());
    }

    private static Fixture simpleF32MatmulFixture() {
        return f32MatmulFixture(2, 3, 2);
    }

    private static Fixture f32MatmulFixture(int m, int k, int n) {
        float[] leftData = new float[m * k];
        float[] rightData = new float[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = i + 1.0f;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = i + 1.0f;
        }
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT32);
        return fixture(left.matmul(right));
    }

    private static Fixture f64MatmulFixture(int m, int k, int n) {
        double[] leftData = new double[m * k];
        double[] rightData = new double[k * n];
        for (int i = 0; i < leftData.length; i++) {
            leftData[i] = i + 1.0d;
        }
        for (int i = 0; i < rightData.length; i++) {
            rightData[i] = i + 1.0d;
        }
        Tensor left = new Tensor(leftData, new int[]{m, k}, null, "left", DataType.FLOAT64);
        Tensor right = new Tensor(rightData, new int[]{k, n}, null, "right", DataType.FLOAT64);
        return fixture(left.matmul(right));
    }

    private static CpuKernelConfig matmulLaunchCpuKernelConfig(
            int matMulParallelMinSize,
            int highCostTargetChunksPerWorker
    ) {
        return new CpuKernelConfig(
                4,
                32,
                32,
                32,
                1,
                1,
                1,
                1,
                1,
                1,
                1_000_000,
                4,
                2,
                highCostTargetChunksPerWorker,
                1,
                1,
                1,
                1,
                0,
                SumAccuracyMode.FAST,
                matMulParallelMinSize,
                AttentionMatMulPolicy.AUTO
        );
    }

    private static CpuKernelConfig matmulVectorCpuKernelConfig(int cheapVectorMinSize) {
        return new CpuKernelConfig(4, 32, 32, 32, cheapVectorMinSize, Integer.MAX_VALUE);
    }

    private static int matmulWork(int m, int k, int n) {
        return Math.multiplyExact(Math.multiplyExact(m, k), n);
    }

    private static BlasConfig openBlasConfig(long matmulMinWork, BlasStorageMode storageMode) {
        return new BlasConfig(
                BlasProvider.OPENBLAS_FFM,
                matmulMinWork,
                false,
                1_000.0d,
                false,
                1_000.0d,
                storageMode,
                false
        );
    }

    private static float[] expectedBatchedF32Matmul(
            float[] left,
            float[] right,
            int batchCount,
            int m,
            int k,
            int n
    ) {
        float[] output = new float[batchCount * m * n];
        int leftBatchSize = m * k;
        int rightBatchSize = k * n;
        int outputBatchSize = m * n;
        for (int batch = 0; batch < batchCount; batch++) {
            int leftBatchBase = batch * leftBatchSize;
            int rightBatchBase = batch * rightBatchSize;
            int outputBatchBase = batch * outputBatchSize;
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < n; col++) {
                    float sum = 0.0f;
                    for (int index = 0; index < k; index++) {
                        sum += left[leftBatchBase + row * k + index]
                                * right[rightBatchBase + index * n + col];
                    }
                    output[outputBatchBase + row * n + col] = sum;
                }
            }
        }
        return output;
    }

    private static double[] expectedBatchedF64Matmul(
            double[] left,
            double[] right,
            int batchCount,
            int m,
            int k,
            int n
    ) {
        double[] output = new double[batchCount * m * n];
        int leftBatchSize = m * k;
        int rightBatchSize = k * n;
        int outputBatchSize = m * n;
        for (int batch = 0; batch < batchCount; batch++) {
            int leftBatchBase = batch * leftBatchSize;
            int rightBatchBase = batch * rightBatchSize;
            int outputBatchBase = batch * outputBatchSize;
            for (int row = 0; row < m; row++) {
                for (int col = 0; col < n; col++) {
                    double sum = 0.0d;
                    for (int index = 0; index < k; index++) {
                        sum += left[leftBatchBase + row * k + index]
                                * right[rightBatchBase + index * n + col];
                    }
                    output[outputBatchBase + row * n + col] = sum;
                }
            }
        }
        return output;
    }

    private static float[] relu(float[] values) {
        float[] output = values.clone();
        for (int i = 0; i < output.length; i++) {
            if (output[i] < 0.0f) {
                output[i] = 0.0f;
            }
        }
        return output;
    }

    private static float[] addColumnBias(float[] values, float[] bias, int m, int n) {
        float[] output = values.clone();
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                output[row * n + col] += bias[col];
            }
        }
        return output;
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static CompiledNode node(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow();
    }

    private static int nodeId(List<CompiledNode> nodes, String label) {
        return nodes.stream()
                .filter(node -> label.equals(node.label()))
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static ExecutionContext context(
            Fixture fixture,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex
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

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] source = TensorInternalAccess.bfloat16Data(tensor);
        float[] out = new float[source.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(source[i]);
        }
        return out;
    }

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1-test-f32-input-" + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1-test-f64-input-" + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 test native F64 input");
    }

    private static float[] nativeF32Values(NativeFloat32Storage storage) {
        float[] out = new float[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat32At(i);
        }
        return out;
    }

    private static double[] nativeF64Values(NativeFloat64Storage storage) {
        double[] out = new double[storage.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = storage.getFloat64At(i);
        }
        return out;
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
