package Operations;

import Backend.ComputeBackend;
import Graph.codegen.DFusedOperationGenerator;
import Graph.codegen.FusedDTypeOps;
import Tensor.Tensor;
import Tensor.DataType;
import Utils.CustomClassLoader;

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

        try {
            int id = CLASS_COUNTER.incrementAndGet();
            String binaryName = "Operations.fused.GeneratedFusedOp" + id;
            String internalName = binaryName.replace('.', '/');

            byte[] bytecode = DFusedOperationGenerator.generate(
                    internalName,
                    cluster,
                    root,
                    externalInputsInOrder
            );

            CustomClassLoader loader = new CustomClassLoader();
            Class<?> generatedClass = loader.define(binaryName, bytecode);
            Constructor<?> ctor = generatedClass.getConstructor(List.class, String.class, int.class);
            this.compiledInstance = (Operation) ctor.newInstance(cluster, this.expression, this.precisionMode);
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
}
