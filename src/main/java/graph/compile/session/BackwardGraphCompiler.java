package graph.compile.session;

import operations.layout.noop;
import tensor.CompileMode;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.autograd.DifferentiableDTypePolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Decides whether a compile needs backward support and builds the training closure when required.
 */
final class BackwardGraphCompiler {
    private BackwardGraphCompiler() {
    }

    record Result(
            List<Tensor> workingGraph,
            boolean supportsBackward
    ) {
        public Result {
            workingGraph = List.copyOf(workingGraph == null ? List.of() : workingGraph);
        }
    }

    static Result compile(
            Tensor rootTensor,
            List<Tensor> forwardGraph,
            Tensor forwardOutput,
            CompileMode compileMode
    ) {
        Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        List<Tensor> graph = List.copyOf(forwardGraph == null ? List.of() : forwardGraph);
        Tensor actualForwardRoot = requireForwardRoot(forwardOutput);
        resetAutogradBuildState(rootTensor, graph);

        boolean supportsBackward = shouldCompileBackward(compileMode, hasTrainableLeafInputs(graph));
        if (!supportsBackward) {
            return new Result(graph, false);
        }

        DifferentiableDTypePolicy.requireGradientSupported(rootTensor.getDataType(), "Backward execution");
        BackwardGraphBuilder.Result backward = BackwardGraphBuilder.build(graph, actualForwardRoot);

        List<Tensor> targetsToSave = new ArrayList<>();
        targetsToSave.add(forwardOutput);
        targetsToSave.addAll(backward.backwardTargets());
        Tensor superRoot = new Tensor(new int[]{1}, targetsToSave, new noop(), "System_Super_Root");

        List<Tensor> workingGraph = new ArrayList<>(superRoot.topologicalSort());
        workingGraph.remove(superRoot);
        return new Result(workingGraph, true);
    }

    private static boolean hasTrainableLeafInputs(List<Tensor> forwardGraph) {
        for (Tensor tensor : forwardGraph) {
            if (tensor.getOperation() == null && tensor.getRequiresGrad()) {
                return true;
            }
        }
        return false;
    }

    private static boolean shouldCompileBackward(CompileMode compileMode, boolean trainableLeafInputs) {
        CompileMode mode = compileMode == null ? CompileMode.AUTO : compileMode;
        return switch (mode) {
            case INFERENCE_ONLY -> false;
            case TRAINING, AUTO -> trainableLeafInputs;
        };
    }

    private static void resetAutogradBuildState(Tensor rootTensor, List<Tensor> forwardGraph) {
        for (Tensor tensor : forwardGraph) {
            TensorInternalAccess.clearGradient(tensor);
            TensorInternalAccess.setBackward(tensor, false);
        }
        TensorInternalAccess.clearGradient(rootTensor);
    }

    static Tensor requireForwardRoot(Tensor forwardOutput) {
        List<Tensor> inputs = forwardOutput == null ? null : forwardOutput.getPrevTensors();
        if (inputs == null || inputs.size() != 1 || inputs.get(0) == null) {
            throw new IllegalStateException("System forward output must have exactly one input.");
        }
        return inputs.get(0);
    }
}
