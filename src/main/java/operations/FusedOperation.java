package operations;

import backend.ComputeBackend;
import backend.kernels.cpu.FusedExecutionProfiler;
import graph.codegen.FusedDTypeOps;
import graph.codegen.FusedOperationGeneratorRouter;
import tensor.Tensor;
import tensor.DataType;
import utils.CustomClassLoader;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class FusedOperation implements Operation {

    private static final AtomicInteger CLASS_COUNTER = new AtomicInteger();

    private final String expression;
    private final Operation compiledInstance;
    private final int precisionMode;
    private final boolean lowCostHint;
    private final String schedulerSignature;
    private final int clusterSize;
    private final int externalInputCount;
    private final int dispatchComplexity;
    private final int dispatchScale;

    public FusedOperation(List<Tensor> cluster, Tensor root) {
        this(cluster, root, findExternalInputs(cluster));
    }

    public FusedOperation(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        if (cluster == null || cluster.isEmpty()) {
            throw new IllegalArgumentException("Fused cluster cannot be null/empty.");
        }
        if (root == null) {
            throw new IllegalArgumentException("Fused root cannot be null.");
        }

        this.expression = "fused(" + cluster.size() + ")";
        this.precisionMode = resolvePrecisionMode(cluster, root, externalInputsInOrder);
        this.lowCostHint = resolveLowCostHint(cluster);
        this.schedulerSignature = buildSchedulerSignature(cluster, this.precisionMode);
        this.clusterSize = cluster.size();
        this.externalInputCount = externalInputsInOrder == null ? 0 : externalInputsInOrder.size();
        this.dispatchComplexity = estimateDispatchComplexity(cluster);
        this.dispatchScale = resolveDispatchScale(this.dispatchComplexity);

        try {
            long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
            int id = CLASS_COUNTER.incrementAndGet();
            String binaryName = "operations.fused.GeneratedFusedOp" + id;
            String internalName = binaryName.replace('.', '/');

            byte[] bytecode = FusedOperationGeneratorRouter.generate(
                    internalName,
                    cluster,
                    root,
                    externalInputsInOrder,
                    this.precisionMode
            );

            CustomClassLoader loader = new CustomClassLoader();
            Class<?> generatedClass = loader.define(binaryName, bytecode);
            Constructor<?> ctor = generatedClass.getConstructor(List.class, String.class, int.class);
            this.compiledInstance = (Operation) ctor.newInstance(cluster, this.expression, this.precisionMode);
            if (FusedExecutionProfiler.enabled()) {
                FusedExecutionProfiler.recordCompile(
                        this.schedulerSignature,
                        this.expression,
                        this.clusterSize,
                        this.externalInputCount,
                        this.precisionMode,
                        this.lowCostHint,
                        System.nanoTime() - t0
                );
            }
        } catch (Throwable t) {
            throw new RuntimeException("Failed to generate fused operation class", t);
        }
    }

    @Override
    public OpType opType() {
        return OpType.FUSED;
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public void apply(List<Tensor> inputs, Tensor out) {
        compiledInstance.apply(inputs, out);
    }

    public Operation getCompiledInstance() {
        return compiledInstance;
    }

    public int getPrecisionMode() {
        return precisionMode;
    }

    public boolean isLowCostHint() {
        return lowCostHint;
    }

    public String getSchedulerSignature() {
        return schedulerSignature;
    }

    public int getClusterSize() {
        return clusterSize;
    }

    public int getExternalInputCount() {
        return externalInputCount;
    }

    public int getDispatchComplexity() {
        return dispatchComplexity;
    }

    public int getDispatchScale() {
        return dispatchScale;
    }

    @Override
    public void gradient(List<Tensor> inputs, Tensor out) {
        // Backward je v tomto projektu realizovaný explicitními uzly v grafu.
    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return expression;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return false;
    }

    private static List<Tensor> findExternalInputs(List<Tensor> cluster) {
        Set<Tensor> clusterSet = new LinkedHashSet<>(cluster);
        Set<Tensor> external = new LinkedHashSet<>();

        for (Tensor t : cluster) {
            List<Tensor> parents = t.getPrevTensors();
            if (parents == null) {
                continue;
            }
            for (Tensor p : parents) {
                if (!clusterSet.contains(p)) {
                    external.add(p);
                }
            }
        }

        return new ArrayList<>(external);
    }

    private static int resolvePrecisionMode(List<Tensor> cluster, Tensor root, List<Tensor> externalInputsInOrder) {
        DataType target = root != null ? root.getDataType() : DataType.FLOAT64;
        if (target == null) {
            target = DataType.FLOAT64;
        }

        List<Tensor> all = new ArrayList<>();
        if (cluster != null) all.addAll(cluster);
        if (externalInputsInOrder != null) all.addAll(externalInputsInOrder);
        if (root != null) all.add(root);

        for (Tensor t : all) {
            if (t == null) continue;
            DataType dt = t.getDataType();
            if (dt == DataType.FLOAT64) {
                target = DataType.FLOAT64;
                break;
            }
            if (dt == DataType.FLOAT32 && target == DataType.FLOAT16) {
                target = DataType.FLOAT32;
            }
        }

        return switch (target) {
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case FLOAT16 -> FusedDTypeOps.MODE_F16;
        };
    }

    private static boolean resolveLowCostHint(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return false;
        }
        for (Tensor t : cluster) {
            if (t == null || t.getOperation() == null) {
                continue;
            }
            OpType type = t.getOperation().opType();
            if (type == null) {
                return false;
            }
            switch (type) {
                case ADD, SUB, MUL, MIN, MAX, NEG, MUL_SCALAR, RELU, NOOP -> {
                    // keep scanning
                }
                default -> {
                    return false;
                }
            }
        }
        return true;
    }

    private static int estimateDispatchComplexity(List<Tensor> cluster) {
        if (cluster == null || cluster.isEmpty()) {
            return 1;
        }
        int total = 0;
        for (Tensor t : cluster) {
            if (t == null || t.getOperation() == null) {
                continue;
            }
            total += t.getOperation().isCheap() ? 1 : 4;
        }
        return Math.max(1, total);
    }

    private static int resolveDispatchScale(int dispatchComplexity) {
        int normalized = (Math.max(1, dispatchComplexity) + 7) / 8;
        return Math.max(1, Math.min(8, normalized));
    }

    private static String buildSchedulerSignature(List<Tensor> cluster, int precisionMode) {
        StringBuilder sb = new StringBuilder(128);
        sb.append("fused:pm=").append(precisionMode).append('|');
        if (cluster != null) {
            for (Tensor t : cluster) {
                if (t == null || t.getOperation() == null) {
                    continue;
                }
                sb.append(t.getOperation().opType()).append(',');
            }
        }
        return sb.toString();
    }
}
