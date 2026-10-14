package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Conv1d;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.layers.Conv3d;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.LayerNorm;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * A closed Tensor-to-Tensor model reconstructed as a fully bound, descriptor-only candidate.
 *
 * <p>Root children are owned directly under numeric names: there is no wrapper path prefix.
 * Reconstruction starts in evaluation mode and creates no parameter values or storage. Load
 * values separately through inherited strict state loading before execution through a caller's
 * Engine. NN does not own execution, persistence, native resources or a training lifecycle.</p>
 *
 * <p>Aliases retain exact Tensor identity. Callers coordinate description, forward construction
 * and replacement; in particular bindings must not change during an open Training session.
 * Identity groups are recomputed after legal replacement or strict load, never cached as
 * creation-time authority. Sharing values or physical storage does not establish an alias.</p>
 */
public final class DeclarativeModel extends Model<Tensor, Tensor> {
    private final List<UnaryTensorModule> modules;

    private DeclarativeModel(List<UnaryTensorModule> modules) {
        this.modules = registerIndexedChildren(modules);
    }

    /**
     * Reconstructs a private, fully bound evaluation skeleton without running any initializer.
     *
     * <p>Exactly one storage-free, provenance-free Tensor is created per identity group or
     * singleton. The same reference is installed at every group member. Convolution reservations
     * are strictly loaded before any forward; no random generator or weight-sized array is used.
     * A failure returns no partially constructed model.</p>
     *
     * @param specification non-null validated immutable semantic configuration
     * @return non-null fully bound evaluation model with state available but no materialized values
     * @throws NullPointerException if specification is null
     * @throws IllegalArgumentException if a schema cannot be bound by its ordinary layer
     * @throws ArithmeticException if checked Model or layer arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static DeclarativeModel reconstruct(ModelSpecification specification) {
        Objects.requireNonNull(specification, "specification");
        Map<String, String> representatives = new HashMap<>();
        for (AliasGroup group : specification.aliases())
            for (String path : group.paths()) representatives.put(path, group.paths().getFirst());
        Map<String, Tensor> leaves = new HashMap<>();
        for (StateDescriptor state : specification.stateDescriptors()) {
            String representative = representatives.getOrDefault(state.path(), state.path());
            Tensor leaf = leaves.get(representative);
            if (leaf == null) {
                leaf = TensorFactory.create(new TensorDescriptor(state.dataType(), state.shape(),
                        Optional.empty(), state.requiresGrad()));
                leaves.put(representative, leaf);
            }
            leaves.put(state.path(), leaf);
        }
        DeclarativeModel model = new DeclarativeModel(buildChildren(specification.root(), "", leaves));
        model.eval();
        return model;
    }

    /**
     * Describes current configuration and current exact parameter identity groups.
     *
     * <p>The caller must keep the owned tree quiescent for this read. This changes neither
     * ownership, bindings nor mode and produces no detached parameter values.</p>
     *
     * @return non-null immutable snapshot; legal replacement joins and splits are observed anew
     * @throws IllegalArgumentException if a current configuration fails its closed schema
     * @throws IllegalStateException if state is unexpectedly unbound
     * @throws ArithmeticException if checked schema arithmetic overflows
     */
    public ModelSpecification describe() {
        return ModelSpecification.describeChildren(this);
    }

    /**
     * Passes the exact input through every root child once in numeric order.
     *
     * <p>Ordinary nested Sequential and layer operation/provenance order is unchanged.
     * No initialization, evaluation, storage allocation or execution occurs. Completed prefix
     * expressions are not rolled back if a later step rejects the composition.</p>
     *
     * @param input non-null Tensor satisfying the composed ordinary layer rules
     * @return exact input for an empty tree, otherwise the exact non-null final layer expression
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if a child rejects input, with root numeric step context
     * @throws ArithmeticException if ordinary Model arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor current = Objects.requireNonNull(input, "input");
        for (int i = 0; i < modules.size(); i++) {
            try {
                current = modules.get(i).forward(current);
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException("step " + i + ": " + failure.getMessage(), failure);
            }
        }
        return current;
    }

    // Recursion is safe only after ModelSpecification has admitted the bounded immutable tree.
    private static List<UnaryTensorModule> buildChildren(Sequence sequence, String path,
            Map<String, Tensor> leaves) {
        List<UnaryTensorModule> children = new ArrayList<>();
        for (int i = 0; i < sequence.steps().size(); i++) {
            String childPath = ModelSpecification.childPath(path, i);
            children.add(buildStep(sequence.steps().get(i), childPath, leaves));
        }
        return children;
    }

    private static UnaryTensorModule buildStep(Step step, String path, Map<String, Tensor> leaves) {
        Tensor weight = leaves.get(path + ".weight");
        Tensor bias = leaves.get(path + ".bias");
        return switch (step) {
            case Sequence sequence -> new Sequential(buildChildren(sequence, path, leaves));
            case LinearStep linear -> Linear.fromState(weight, Optional.ofNullable(bias),
                    linear.automaticInputGuards());
            case LayerNormStep norm -> new LayerNorm(leaves.get(path + ".scale"), bias, norm.epsilon());
            case EmbeddingStep _ -> new Embedding(weight);
            case ActivationStep activation -> new Activation(activation.kind());
            case Conv1dStep conv -> {
                var attrs = conv.attrs();
                Conv1d layer = new Conv1d(conv.outChannels(), conv.kernelWidth(), attrs.stride(),
                        attrs.padding(), attrs.dilation(), attrs.groups(), conv.bias(), conv.dataType(),
                        ParameterInitialization.zeros(), 0);
                bindConvolution(layer, weight, bias);
                yield layer;
            }
            case Conv2dStep conv -> {
                var attrs = conv.attrs();
                Conv2d layer = new Conv2d(conv.outChannels(), conv.kernelHeight(), conv.kernelWidth(),
                        attrs.strideHeight(), attrs.strideWidth(), attrs.paddingHeight(), attrs.paddingWidth(),
                        attrs.dilationHeight(), attrs.dilationWidth(), attrs.groups(), conv.bias(),
                        conv.dataType(), ParameterInitialization.zeros(), 0);
                bindConvolution(layer, weight, bias);
                yield layer;
            }
            case Conv3dStep conv -> {
                var attrs = conv.attrs();
                Conv3d layer = new Conv3d(conv.outChannels(), conv.kernelDepth(), conv.kernelHeight(),
                        conv.kernelWidth(), attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                        attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                        attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(), attrs.groups(),
                        conv.bias(), conv.dataType(), ParameterInitialization.zeros(), 0);
                bindConvolution(layer, weight, bias);
                yield layer;
            }
        };
    }

    private static void bindConvolution(Module layer, Tensor weight, Tensor bias) {
        List<StateEntry> entries = new ArrayList<>();
        entries.add(new StateEntry("weight", StateKind.PARAMETER, weight));
        if (bias != null) entries.add(new StateEntry("bias", StateKind.PARAMETER, bias));
        layer.loadStateDictionary(new StateDictionary(entries));
    }
}
