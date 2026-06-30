package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1AttentionBackwardPreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.execution.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.compile.CompiledProgram;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;
import runtime.execution.ExecutionState;
import planning.region.specialization.RegionSpecializationCandidate;
import planning.region.specialization.RegionSpecializationKind;
import planning.region.specialization.SdpaBackwardOutputKind;
import planning.region.specialization.SdpaBackwardSpecializationPayload;
import planning.value.GraphValueRef;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.options.AttentionOptions;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.Arrays;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Tag("benchmark")
final class Cpu1AttentionBackwardBenchmarkTest {
    private static final int WARMUP_ITERATIONS = 3;
    private static final int MEASURE_ITERATIONS = 7;

    @Test
    void benchmarkDenseSdpaBackwardArrayScalarVectorAndSegmentScalarVector() {
        BenchmarkShape[] shapes = {
                new BenchmarkShape("medium-b2-h4-t64-d32", 8, 64, 64, 32),
                new BenchmarkShape("large-b4-h8-t64-d32", 32, 64, 64, 32)
        };
        Map<ResultKey, BenchmarkResult> arrayScalarResults = new HashMap<>();
        System.out.println("shape,dtype,outputKind,storageKind,vectorizationKind,kernelId,workers,chunkSize,rowCount,"
                + "itemCount,medianMs,p90Ms,minMs,maxMs,medianVsArrayScalar,medianVsSegmentScalar");
        for (BenchmarkShape shape : shapes) {
            AttentionFixture fixture = fixture(shape);
            for (DataType dataType : new DataType[]{DataType.FLOAT32, DataType.FLOAT64}) {
                for (SdpaBackwardOutputKind outputKind : SdpaBackwardOutputKind.values()) {
                    BenchmarkResult arrayScalar = runCase(
                            fixture,
                            dataType,
                            outputKind,
                            Cpu1StorageKind.JAVA_ARRAY,
                            Cpu1VectorizationKind.SCALAR
                    );
                    arrayScalarResults.put(new ResultKey(shape.name(), dataType, outputKind), arrayScalar);
                    printResult(arrayScalar, 1.0d, null);

                    BenchmarkResult arrayVector = runCase(
                            fixture,
                            dataType,
                            outputKind,
                            Cpu1StorageKind.JAVA_ARRAY,
                            Cpu1VectorizationKind.VECTOR
                    );
                    double vectorRatio = arrayVector.stats().medianMs() / arrayScalar.stats().medianMs();
                    printResult(arrayVector, vectorRatio, null);

                    BenchmarkResult segmentScalar = runCase(
                            fixture,
                            dataType,
                            outputKind,
                            Cpu1StorageKind.MEMORY_SEGMENT,
                            Cpu1VectorizationKind.SCALAR
                    );
                    BenchmarkResult arrayBaseline = arrayScalarResults.get(new ResultKey(shape.name(), dataType, outputKind));
                    double segmentScalarArrayRatio = segmentScalar.stats().medianMs() / arrayBaseline.stats().medianMs();
                    printResult(segmentScalar, segmentScalarArrayRatio, 1.0d);

                    BenchmarkResult segmentVector = runCase(
                            fixture,
                            dataType,
                            outputKind,
                            Cpu1StorageKind.MEMORY_SEGMENT,
                            Cpu1VectorizationKind.VECTOR
                    );
                    double segmentVectorArrayRatio = segmentVector.stats().medianMs() / arrayBaseline.stats().medianMs();
                    double segmentVectorScalarRatio = segmentVector.stats().medianMs() / segmentScalar.stats().medianMs();
                    printResult(segmentVector, segmentVectorArrayRatio, segmentVectorScalarRatio);
                }
            }
        }
    }

    private static BenchmarkResult runCase(
            AttentionFixture fixture,
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        PreparedBenchmark prepared = prepareBenchmark(fixture, dataType, outputKind, storageKind, vectorizationKind);
        Cpu1Backend backend = new Cpu1Backend();
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            backend.execute(prepared.step().compiledNode(), prepared.step().metadata(), prepared.context());
        }
        double[] samples = new double[MEASURE_ITERATIONS];
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.nanoTime();
            backend.execute(prepared.step().compiledNode(), prepared.step().metadata(), prepared.context());
            samples[i] = (System.nanoTime() - start) / 1_000_000.0d;
        }
        return new BenchmarkResult(fixture.shape(), dataType, outputKind, storageKind, prepared.unit(), stats(samples));
    }

    private static PreparedBenchmark prepareBenchmark(
            AttentionFixture fixture,
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        RuntimeConfig runtimeConfig = runtimeConfig().withCpuStorageProfile(storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                ? CpuStorageProfile.CPU_NATIVE
                : CpuStorageProfile.CPU_ARRAY);
        PreparedCase prepared = prepareCase(fixture, dataType, outputKind, runtimeConfig);
        PreparedExecutionStep originalStep = requireAttentionBackwardStep(prepared.execution(), outputKind);
        PreparedExecutionStep step = reprepareStep(prepared, originalStep, storageKind, vectorizationKind);
        Cpu1PreparedAttentionBackwardUnit unit = ((Cpu1PreparedArtifact) step.metadata().executable())
                .preparedAttentionBackwardUnit();
        ExecutionContext context = isolatedContext(prepared, step);
        double[] weights = attentionWeights(fixture);
        double[] outGrad = outGradValues(fixture);
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            attachNative(context, unit.weightsNodeId(), weights, dataType);
            attachNative(context, unit.outGradNodeId(), outGrad, dataType);
            if (unit.queryNodeId() >= 0) {
                attachNative(context, unit.queryNodeId(), fixture.query(), dataType);
            }
            if (unit.keyNodeId() >= 0) {
                attachNative(context, unit.keyNodeId(), fixture.key(), dataType);
            }
            if (unit.valueNodeId() >= 0) {
                attachNative(context, unit.valueNodeId(), fixture.value(), dataType);
            }
        } else {
            attachArray(context, unit.weightsNodeId(), weights, dataType);
            attachArray(context, unit.outGradNodeId(), outGrad, dataType);
        }
        return new PreparedBenchmark(prepared, step, unit, context);
    }

    private static PreparedExecutionStep reprepareStep(
            PreparedCase prepared,
            PreparedExecutionStep originalStep,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        Cpu1PreparedAttentionBackwardUnit original = ((Cpu1PreparedArtifact) originalStep.metadata().executable())
                .preparedAttentionBackwardUnit();
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                vectorizationKind,
                Cpu1LaunchConfig.parallel(Runtime.getRuntime().availableProcessors()),
                storageKind,
                false,
                false,
                false,
                true,
                prepared.execution().runtimeConfig().cpuKernelConfig()
        );
        RegionSpecializationCandidate candidate = candidateFromUnit(original, originalStep.orderedNodeIds());
        Cpu1PreparedArtifact artifact = new Cpu1AttentionBackwardPreparer().prepare(
                originalStep.compiledNode(),
                candidate,
                prepared.compiledGraph().program().descriptorIndex(),
                config
        );
        List<Integer> inputNodeIds = candidate.inputValueRefs().stream()
                .map(GraphValueRef::nodeId)
                .toList();
        PreparedStepMetadata metadata = new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                inputNodeIds,
                artifact,
                storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? InputResidencyRequirement.none()
                        : InputResidencyRequirement.cpuReadableAll(),
                storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? OutputResidencyEffect.none()
                        : OutputResidencyEffect.cpuCurrentPreserveNative()
        );
        return new PreparedExecutionStep(
                originalStep.compiledNode(),
                metadata,
                originalStep.orderedNodeIds(),
                originalStep.boundaryOutputNodeIds()
        );
    }

    private static RegionSpecializationCandidate candidateFromUnit(
            Cpu1PreparedAttentionBackwardUnit unit,
            List<Integer> orderedNodeIds
    ) {
        SdpaBackwardSpecializationPayload payload = new SdpaBackwardSpecializationPayload(
                unit.outputKind(),
                unit.scale(),
                unit.hasMask(),
                unit.weightsNodeId(),
                unit.outGradNodeId(),
                unit.queryNodeId(),
                unit.keyNodeId(),
                unit.valueNodeId(),
                unit.maskNodeId()
        );
        return new RegionSpecializationCandidate(
                RegionSpecializationKind.SDPA_BACKWARD,
                orderedNodeIds,
                inputRefs(unit),
                GraphValueRef.node(unit.nodeId()),
                unit.nodeId(),
                "benchmark SDPA backward " + unit.outputKind(),
                payload
        );
    }

    private static List<GraphValueRef> inputRefs(Cpu1PreparedAttentionBackwardUnit unit) {
        LinkedHashSet<Integer> nodeIds = new LinkedHashSet<>();
        nodeIds.add(unit.weightsNodeId());
        nodeIds.add(unit.outGradNodeId());
        if (unit.queryNodeId() >= 0) {
            nodeIds.add(unit.queryNodeId());
        }
        if (unit.keyNodeId() >= 0) {
            nodeIds.add(unit.keyNodeId());
        }
        if (unit.valueNodeId() >= 0) {
            nodeIds.add(unit.valueNodeId());
        }
        if (unit.maskNodeId() >= 0) {
            nodeIds.add(unit.maskNodeId());
        }
        ArrayList<GraphValueRef> refs = new ArrayList<>();
        for (int nodeId : nodeIds) {
            refs.add(GraphValueRef.node(nodeId));
        }
        return refs;
    }

    private static PreparedCase prepareCase(
            AttentionFixture fixture,
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            RuntimeConfig runtimeConfig
    ) {
        Tensor q = tensor(fixture.query(), fixture.queryShape(), "benchmarkQ", dataType);
        Tensor k = tensor(fixture.key(), fixture.keyShape(), "benchmarkK", dataType);
        Tensor v = tensor(fixture.value(), fixture.valueShape(), "benchmarkV", dataType);
        Tensor target = switch (outputKind) {
            case QUERY -> q;
            case KEY -> k;
            case VALUE -> v;
        };
        target.setRequiresGrad(true);
        Tensor loss = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(fixture.scale())).sum();
        CompiledGraph graph = CompiledGraph.compile(loss, CompileConfig.training());
        return new PreparedCase(loss, graph, graph.prepare(runtimeConfig));
    }

    private static Tensor tensor(double[] values, int[] shape, String label, DataType dataType) {
        return new Tensor(values.clone(), shape.clone(), null, label, dataType);
    }

    private static PreparedExecutionStep requireAttentionBackwardStep(
            PreparedExecution execution,
            SdpaBackwardOutputKind outputKind
    ) {
        for (PreparedExecutionStep step : execution.backwardSteps()) {
            if (step.metadata().executable() instanceof Cpu1PreparedArtifact artifact
                    && artifact.preparedAttentionBackwardUnit().outputKind() == outputKind) {
                return step;
            }
        }
        throw new AssertionError("Missing cpu1 SDPA backward step for " + outputKind);
    }

    private static ExecutionContext isolatedContext(PreparedCase prepared, PreparedExecutionStep overrideStep) {
        CompiledProgram program = prepared.compiledGraph().program();
        Map<Integer, PreparedStepMetadata> metadataIndex = new HashMap<>();
        for (PreparedExecutionStep step : prepared.execution().executionSteps()) {
            metadataIndex.put(step.compiledNode().id(), step.metadata());
        }
        metadataIndex.put(overrideStep.compiledNode().id(), overrideStep.metadata());
        ExecutionState state = ExecutionState.create(
                program.compiledNodes(),
                program.descriptorIndex(),
                metadataIndex,
                program.forwardBoundaryNodeId(),
                testsupport.PublicationPlans.forRoot(
                        prepared.loss(),
                        program.compiledNodes(),
                        program.forwardOutputNodeId()
                )
        );
        return ExecutionContext.fromRuntimeConfig(
                prepared.execution().runtimeConfig(),
                ExecutionMode.FORWARD_BACKWARD,
                metadataIndex,
                state
        );
    }

    private static void attachArray(
            ExecutionContext context,
            int nodeId,
            double[] values,
            DataType dataType
    ) {
        Tensor tensor = context.runtimeTensorForNodeId(nodeId);
        if (dataType == DataType.FLOAT32) {
            float[] target = TensorInternalAccess.float32Data(tensor);
            if (target.length == 1) {
                target[0] = (float) values[0];
            } else {
                for (int i = 0; i < values.length; i++) {
                    target[i] = (float) values[i];
                }
            }
        } else if (dataType == DataType.FLOAT64) {
            double[] target = TensorInternalAccess.float64Data(tensor);
            if (target.length == 1) {
                target[0] = values[0];
            } else {
                System.arraycopy(values, 0, target, 0, values.length);
            }
        } else {
            throw new IllegalArgumentException("Unsupported benchmark dtype " + dataType);
        }
        context.markCpuCurrent(nodeId, "cpu1 SDPA backward benchmark array input");
    }

    private static void attachNative(
            ExecutionContext context,
            int nodeId,
            double[] values,
            DataType dataType
    ) {
        NativeTensorStorage storage = context.allocateNativeStorage(
                dataType,
                values.length,
                "cpu1-sdpa-backward-benchmark-native-" + nodeId
        );
        if (dataType == DataType.FLOAT32) {
            NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
            for (int i = 0; i < values.length; i++) {
                f32.setFloat32At(i, (float) values[i]);
            }
        } else if (dataType == DataType.FLOAT64) {
            NativeFloat64Storage f64 = (NativeFloat64Storage) storage;
            for (int i = 0; i < values.length; i++) {
                f64.setFloat64At(i, values[i]);
            }
        } else {
            throw new IllegalArgumentException("Unsupported benchmark dtype " + dataType);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 SDPA backward benchmark native input");
    }

    private static RuntimeConfig runtimeConfig() {
        return new RuntimeConfig(CpuKernelConfig.defaultsTraining(), ApproximationConfig.defaults(), BlasConfig.disabled());
    }

    private static AttentionFixture fixture(BenchmarkShape shape) {
        int valueDim = shape.depth();
        double[] query = patternedValues(shape.batchCount() * shape.queryLen() * shape.depth(), 0.017d, 0.003d);
        double[] key = patternedValues(shape.batchCount() * shape.keyLen() * shape.depth(), -0.013d, 0.002d);
        double[] value = patternedValues(shape.batchCount() * shape.keyLen() * valueDim, 0.011d, -0.004d);
        return new AttentionFixture(
                shape,
                shape.batchCount(),
                shape.queryLen(),
                shape.keyLen(),
                shape.depth(),
                valueDim,
                new int[]{shape.batchCount(), shape.queryLen(), shape.depth()},
                new int[]{shape.batchCount(), shape.keyLen(), shape.depth()},
                new int[]{shape.batchCount(), shape.keyLen(), valueDim},
                query,
                key,
                value,
                1.0d / Math.sqrt(shape.depth())
        );
    }

    private static double[] patternedValues(int size, double angleScale, double offsetScale) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin((i + 1) * angleScale) * 0.5d + ((i % 11) - 5) * offsetScale;
        }
        return out;
    }

    private static double[] attentionWeights(AttentionFixture fixture) {
        double[] weights = new double[fixture.batchCount() * fixture.queryLen() * fixture.keyLen()];
        for (int batch = 0; batch < fixture.batchCount(); batch++) {
            int queryBatchBase = batch * fixture.queryLen() * fixture.depth();
            int keyBatchBase = batch * fixture.keyLen() * fixture.depth();
            int weightsBatchBase = batch * fixture.queryLen() * fixture.keyLen();
            for (int queryIndex = 0; queryIndex < fixture.queryLen(); queryIndex++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    max = Math.max(max, dot(
                            fixture.query(),
                            queryBatchBase + queryIndex * fixture.depth(),
                            fixture.key(),
                            keyBatchBase + keyIndex * fixture.depth(),
                            fixture.depth()
                    ) * fixture.scale());
                }
                double sum = 0.0d;
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    double exp = Math.exp(dot(
                            fixture.query(),
                            queryBatchBase + queryIndex * fixture.depth(),
                            fixture.key(),
                            keyBatchBase + keyIndex * fixture.depth(),
                            fixture.depth()
                    ) * fixture.scale() - max);
                    weights[weightsBatchBase + queryIndex * fixture.keyLen() + keyIndex] = exp;
                    sum += exp;
                }
                for (int keyIndex = 0; keyIndex < fixture.keyLen(); keyIndex++) {
                    weights[weightsBatchBase + queryIndex * fixture.keyLen() + keyIndex] /= sum;
                }
            }
        }
        return weights;
    }

    private static double dot(double[] left, int leftBase, double[] right, int rightBase, int depth) {
        double sum = 0.0d;
        for (int i = 0; i < depth; i++) {
            sum += left[leftBase + i] * right[rightBase + i];
        }
        return sum;
    }

    private static double[] outGradValues(AttentionFixture fixture) {
        double[] out = new double[fixture.batchCount() * fixture.queryLen() * fixture.valueDim()];
        Arrays.fill(out, 1.0d);
        return out;
    }

    private static Stats stats(double[] samples) {
        double[] sorted = samples.clone();
        Arrays.sort(sorted);
        return new Stats(
                percentile(sorted, 50.0d),
                percentile(sorted, 90.0d),
                sorted[0],
                sorted[sorted.length - 1]
        );
    }

    private static double percentile(double[] sorted, double percentile) {
        if (sorted.length == 1) {
            return sorted[0];
        }
        double index = (percentile / 100.0d) * (sorted.length - 1);
        int lower = (int) Math.floor(index);
        int upper = (int) Math.ceil(index);
        if (lower == upper) {
            return sorted[lower];
        }
        double weight = index - lower;
        return sorted[lower] * (1.0d - weight) + sorted[upper] * weight;
    }

    private static void printResult(
            BenchmarkResult result,
            Double medianVsArrayScalar,
            Double medianVsSegmentScalar
    ) {
        Cpu1PreparedAttentionBackwardUnit unit = result.unit();
        Stats stats = result.stats();
        System.out.printf(
                Locale.US,
                "%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%s,%s%n",
                result.shape().name(),
                result.dataType(),
                result.outputKind(),
                result.storageKind(),
                unit.vectorizationKind(),
                unit.kernelId(),
                unit.launchConfig().workerCount(),
                unit.launchConfig().chunkSize(),
                unit.rowCount(),
                unit.rowCount(),
                stats.medianMs(),
                stats.p90Ms(),
                stats.minMs(),
                stats.maxMs(),
                ratio(medianVsArrayScalar),
                ratio(medianVsSegmentScalar)
        );
    }

    private static String ratio(Double value) {
        return value == null ? "NA" : String.format(Locale.US, "%.4f", value);
    }

    private record BenchmarkShape(String name, int batchCount, int queryLen, int keyLen, int depth) {
    }

    private record AttentionFixture(
            BenchmarkShape shape,
            int batchCount,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim,
            int[] queryShape,
            int[] keyShape,
            int[] valueShape,
            double[] query,
            double[] key,
            double[] value,
            double scale
    ) {
    }

    private record PreparedCase(Tensor loss, CompiledGraph compiledGraph, PreparedExecution execution) {
    }

    private record PreparedBenchmark(
            PreparedCase prepared,
            PreparedExecutionStep step,
            Cpu1PreparedAttentionBackwardUnit unit,
            ExecutionContext context
    ) {
    }

    private record ResultKey(String shape, DataType dataType, SdpaBackwardOutputKind outputKind) {
    }

    private record BenchmarkResult(
            BenchmarkShape shape,
            DataType dataType,
            SdpaBackwardOutputKind outputKind,
            Cpu1StorageKind storageKind,
            Cpu1PreparedAttentionBackwardUnit unit,
            Stats stats
    ) {
    }

    private record Stats(double medianMs, double p90Ms, double minMs, double maxMs) {
    }
}
