package Graph;

import Backend.ComputeEngine;
import Tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CompiledGraph {
    private Tensor rootTensor; // Kořenový tensor grafu
    List<Tensor> forwardGraph = new ArrayList<>();
    List<Tensor> backwardGraph = new ArrayList<>();

    GraphOptimizer forwardOptimizer;
    GraphOptimizer backwardOptimizer;

    public CompiledGraph(Tensor rootTensor, GraphOptimizer forwardOptimizer) {
        this.rootTensor = rootTensor;

        this.forwardOptimizer=forwardOptimizer;
    }

    // Spuštění zkompilovaného grafu (dopředný průchod)
    public void forward() {
        if (rootTensor == null) {
            throw new IllegalStateException("Backend is not set.");
        }

        // First pass - perform optimization
        if (forwardGraph.isEmpty()) {
            forwardGraph.addAll(forwardOptimizer.optimize(rootTensor));
        }

        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() != null) {
                ComputeEngine.compute(tensor);
            }
        }
    }

    // Backward pass
    public void backward() {
        if (rootTensor == null) {
            throw new IllegalStateException("Root tensor is not set.");
        }
        // reset gradients
        for (Tensor t : forwardGraph) {
            boolean isOutput = forwardGraph.getLast() == t;
            double fillValue = isOutput ? 1.0 : 0.0;

            if (t.getGradient() == null) {
                double[] arr = new double[t.getFlatDataSize()];
                Arrays.fill(arr, fillValue);
                t.setGradient(new Tensor(arr, t.getShape(), t.getStrides(), new ArrayList<>(), "gradient"));
            } else {
                Arrays.fill(t.getGradient().getData(), fillValue);
            }

        }

        for (Tensor tensor : forwardGraph.reversed()) {
            if (tensor.getOperation() != null) {
                ComputeEngine.backward(tensor);
            }
        }

    }

    public Tensor getRootTensor() {
        return rootTensor;
    }
}
