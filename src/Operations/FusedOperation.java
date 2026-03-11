package Operations;

import Backend.ComputeBackend;
import Graph.codegen.DFusedOperationGenerator;
import Tensor.Tensor;
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
            Constructor<?> ctor = generatedClass.getConstructor(List.class, String.class);
            this.compiledInstance = (Operation) ctor.newInstance(cluster, this.expression);
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
}
