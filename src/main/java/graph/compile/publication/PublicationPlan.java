package graph.compile.publication;

import graph.model.CompiledGradientBinding;
import graph.compile.GraphStructureContract;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Compile-time plan for runtime input seeding and user-visible publication.
 */
public record PublicationPlan(
        Tensor rootTensor,
        GraphStructureContract graphContract,
        List<RuntimeInputBinding> runtimeInputBindings,
        ForwardPublicationBinding rootOutput,
        List<ForwardPublicationBinding> forwardValuePublications,
        List<GradientPublicationBinding> gradientPublications,
        List<Tensor> gradientClearTargets,
        CompiledGradientBinding forwardSeedGradient,
        List<TrainableParameterBinding> trainableParameters
) {
    public PublicationPlan {
        rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        graphContract = graphContract == null ? GraphStructureContract.unchecked() : graphContract;
        runtimeInputBindings = List.copyOf(runtimeInputBindings == null ? List.of() : runtimeInputBindings);
        rootOutput = Objects.requireNonNull(rootOutput, "rootOutput cannot be null");
        forwardValuePublications = List.copyOf(forwardValuePublications == null ? List.of() : forwardValuePublications);
        gradientPublications = List.copyOf(gradientPublications == null ? List.of() : gradientPublications);
        gradientClearTargets = List.copyOf(gradientClearTargets == null ? List.of() : gradientClearTargets);
        trainableParameters = List.copyOf(trainableParameters == null ? List.of() : trainableParameters);
    }

    public Map<Integer, RuntimeInputBinding> runtimeInputsByNodeId() {
        java.util.HashMap<Integer, RuntimeInputBinding> out = new java.util.HashMap<>();
        for (RuntimeInputBinding binding : runtimeInputBindings) {
            out.put(binding.nodeId(), binding);
        }
        return Map.copyOf(out);
    }

    public Tensor publicationTargetForNodeId(int nodeId) {
        if (rootOutput.sourceNodeId() == nodeId) {
            return rootOutput.targetTensor();
        }
        for (ForwardPublicationBinding binding : forwardValuePublications) {
            if (binding.sourceNodeId() == nodeId) {
                return binding.targetTensor();
            }
        }
        return null;
    }

    public Map<Tensor, Integer> nodeIdsByPublicationTarget() {
        IdentityHashMap<Tensor, Integer> out = new IdentityHashMap<>();
        out.put(rootOutput.targetTensor(), rootOutput.sourceNodeId());
        for (ForwardPublicationBinding binding : forwardValuePublications) {
            out.put(binding.targetTensor(), binding.sourceNodeId());
        }
        return new IdentityHashMap<>(out);
    }

    public record RuntimeInputBinding(
            int nodeId,
            Tensor sourceTensor,
            RuntimeInputBindingKind kind
    ) {
        public RuntimeInputBinding {
            if (nodeId < 0) {
                throw new IllegalArgumentException("nodeId must be >= 0");
            }
            sourceTensor = Objects.requireNonNull(sourceTensor, "sourceTensor cannot be null");
            kind = kind == null ? RuntimeInputBindingKind.FORWARD_LEAF_ALIAS : kind;
        }
    }

    public enum RuntimeInputBindingKind {
        FORWARD_LEAF_ALIAS,
        BACKWARD_LEAF_COPY,
        STATIC_LEAF_COPY
    }

    public record ForwardPublicationBinding(
            Tensor targetTensor,
            int sourceNodeId,
            PublicationKind kind,
            List<AliasRepairStep> aliasRepairChain
    ) {
        public ForwardPublicationBinding {
            targetTensor = Objects.requireNonNull(targetTensor, "targetTensor cannot be null");
            if (sourceNodeId < 0) {
                throw new IllegalArgumentException("sourceNodeId must be >= 0");
            }
            kind = kind == null ? PublicationKind.FORWARD_VALUE : kind;
            aliasRepairChain = List.copyOf(aliasRepairChain == null ? List.of() : aliasRepairChain);
        }
    }

    public enum PublicationKind {
        ROOT_OUTPUT,
        FORWARD_VALUE,
        ACTUAL_FORWARD_ROOT_FOR_ALIAS
    }

    public record AliasRepairStep(
            Tensor aliasTensor,
            Tensor sourceTensor
    ) {
        public AliasRepairStep {
            aliasTensor = Objects.requireNonNull(aliasTensor, "aliasTensor cannot be null");
            sourceTensor = Objects.requireNonNull(sourceTensor, "sourceTensor cannot be null");
        }
    }

    public record GradientPublicationBinding(
            Tensor targetTensor,
            CompiledGradientBinding binding
    ) {
        public GradientPublicationBinding {
            targetTensor = Objects.requireNonNull(targetTensor, "targetTensor cannot be null");
            binding = Objects.requireNonNull(binding, "binding cannot be null");
        }
    }

    public record TrainableParameterBinding(
            Tensor parameterTensor,
            int parameterNodeId,
            CompiledGradientBinding gradientBinding
    ) {
        public TrainableParameterBinding {
            parameterTensor = Objects.requireNonNull(parameterTensor, "parameterTensor cannot be null");
            if (parameterNodeId < 0) {
                throw new IllegalArgumentException("parameterNodeId must be >= 0");
            }
            gradientBinding = Objects.requireNonNull(gradientBinding, "gradientBinding cannot be null");
        }
    }

    public static List<AliasRepairStep> aliasRepairChainFor(Tensor tensor) {
        if (tensor == null) {
            return List.of();
        }
        ArrayList<AliasRepairStep> out = new ArrayList<>();
        appendAliasRepairSteps(tensor, out);
        return List.copyOf(out);
    }

    private static void appendAliasRepairSteps(Tensor tensor, List<AliasRepairStep> out) {
        if (tensor == null || tensor.getOperation() == null
                || tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return;
        }
        if (!graph.model.AliasViewPolicy.aliasesInput0AtRuntime(tensor)) {
            return;
        }
        Tensor source = tensor.getPrevTensors().getFirst();
        appendAliasRepairSteps(source, out);
        out.add(new AliasRepairStep(tensor, source));
    }
}
