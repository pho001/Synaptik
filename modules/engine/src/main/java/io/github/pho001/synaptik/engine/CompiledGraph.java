package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GradientPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable ordinary compile handle owned by the exact {@link Engine} that created it.
 *
 * <p>The handle exposes only ordered logical caller-input metadata. It privately retains the
 * immutable Compiler recipe and ordered publication specifications needed by later preparation
 * and runs. It retains no Tensor, host storage, request list, Runtime coordinate, or backend
 * representation. Repeated use of one exact leaf has one input, while identity-distinct leaves
 * remain separate even when descriptors are equal. Compiler constants have no input entry.
 * Metadata remains readable after Engine closure, but the closed owner rejects preparation.</p>
 */
public final class CompiledGraph {
    private final Engine owner;
    private final CompileArtifacts artifacts;
    private final List<Input> inputs;
    private final List<PublicationSpec> publications;

    /**
     * Constructs one owner-bound handle from final Compiler artifacts.
     *
     * @param owner non-null exact ordinary Engine owner
     * @param artifacts non-null immutable final Compiler recipe
     * @throws NullPointerException if an argument or resolved graph descriptor is {@code null}
     * @throws IllegalArgumentException if an artifact binding references no final graph value
     */
    CompiledGraph(Engine owner, CompileArtifacts artifacts) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.artifacts = Objects.requireNonNull(artifacts, "artifacts");
        Map<ValueId, TensorDescriptor> descriptors = new HashMap<>();
        for (GraphValue value : artifacts.graph().values()) {
            descriptors.put(value.id(), value.descriptor());
        }

        var inputSnapshot = new ArrayList<Input>();
        artifacts.constants().bindableInputBindings().forEach(binding -> inputSnapshot.add(
                new Input(binding.tensorId(), requireDescriptor(descriptors, binding.valueId()))));
        inputs = List.copyOf(inputSnapshot);

        var publicationSnapshot = new ArrayList<PublicationSpec>();
        artifacts.publication().forwardBindings().forEach(binding -> publicationSnapshot.add(
                new PublicationSpec(
                        binding.tensorId(), requireDescriptor(descriptors, binding.valueId()),
                        RunResult.Role.FORWARD, 0, -1)));
        for (GradientPublicationBinding binding : artifacts.publication().gradientBindings()) {
            publicationSnapshot.add(new PublicationSpec(
                    binding.target(), requireDescriptor(descriptors, binding.valueId()),
                    RunResult.Role.GRADIENT, binding.derivativeOrder(), binding.targetIndex()));
        }
        publications = List.copyOf(publicationSnapshot);
    }

    /**
     * Returns caller-bindable logical inputs in final Compiler binding order.
     * Callers may later supply matching Tensors to {@link Engine#run} in any list order; this
     * descriptive list carries no storage or execution authority.
     *
     * @return the same non-null immutable ordered metadata snapshot on every call
     */
    public List<Input> inputs() {
        return inputs;
    }

    Engine owner() {
        return owner;
    }

    CompileArtifacts artifacts() {
        return artifacts;
    }

    List<PublicationSpec> publicationSpecs() {
        return publications;
    }

    private static TensorDescriptor requireDescriptor(
            Map<ValueId, TensorDescriptor> descriptors, ValueId valueId) {
        TensorDescriptor descriptor = descriptors.get(valueId);
        if (descriptor == null) {
            throw new IllegalArgumentException("artifact binding references unknown " + valueId);
        }
        return descriptor;
    }

    /**
     * Describes one logical caller input without carrying storage or execution authority.
     *
     * @param tensorId non-null immutable caller Tensor identity
     * @param descriptor non-null exact final logical descriptor required for binding
     */
    public record Input(TensorId tensorId, TensorDescriptor descriptor) {
        /**
         * Validates one standalone metadata value.
         *
         * @param tensorId non-null immutable caller Tensor identity
         * @param descriptor non-null exact final logical descriptor
         * @throws NullPointerException if either component is {@code null}, in component order
         */
        public Input {
            Objects.requireNonNull(tensorId, "tensorId");
            Objects.requireNonNull(descriptor, "descriptor");
        }
    }

    static final class PublicationSpec {
        final TensorId tensorId;
        final TensorDescriptor descriptor;
        final RunResult.Role role;
        final int derivativeOrder;
        final int targetIndex;

        PublicationSpec(
                TensorId tensorId,
                TensorDescriptor descriptor,
                RunResult.Role role,
                int derivativeOrder,
                int targetIndex) {
            this.tensorId = Objects.requireNonNull(tensorId, "tensorId");
            this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
            this.role = Objects.requireNonNull(role, "role");
            this.derivativeOrder = derivativeOrder;
            this.targetIndex = targetIndex;
        }
    }
}
