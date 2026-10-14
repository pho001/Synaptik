package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Conv1d;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.layers.Conv3d;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.LayerNorm;
import io.github.pho001.synaptik.nn.layers.Linear;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable semantic configuration and parameter-identity topology of a closed Sequential tree.
 *
 * <p>Describe a quiescent bound tree, reconstruct a descriptor-only candidate, then load values
 * through the existing strict state dictionary boundary. This value owns no Tensor, storage,
 * callback, wire format or resource. Numeric child order determines paths and state order.
 * Equal records reused in a tree describe independent module occurrences.</p>
 *
 * <p>Each tree permits at most 4096 step occurrences, including its root, and depth 64 with
 * root depth one. Aliases describe exact Tensor identity, not equal values or physical overlap.
 * There is no promise of backend admission or execution for a valid semantic specification.</p>
 *
 * <p>The closed matrix includes nested sequences, six parameterized layer families and
 * RELU/TANH/SIGMOID. All four floating parameter types retain ordinary Model promotion and
 * guards; Embedding inputs are INT32 or INT64. No batch example fixes compatible leading
 * dimensions. Description rejects callbacks, custom modules, contextual and recurrent layers.
 * State values and modes load separately; this API does not reconstruct models from files.</p>
 */
public final class ModelSpecification {
    private final Sequence root;
    private final List<AliasGroup> aliases;
    private final List<StateDescriptor> stateDescriptors;

    /**
     * Validates and snapshots a complete configuration without allocating Tensor state.
     *
     * @param root non-null closed ordered root; retained as an immutable value
     * @param aliases non-null canonical groups, defensively copied; unlisted paths are singletons
     * @throws NullPointerException if root, aliases or any list element is null
     * @throws IllegalArgumentException if tree bounds, schema or alias topology is invalid
     * @throws ArithmeticException if checked element or byte arithmetic overflows
     */
    public ModelSpecification(Sequence root, List<AliasGroup> aliases) {
        this.root = Objects.requireNonNull(root, "root");
        validateTree(root);
        this.stateDescriptors = descriptors(root);
        this.aliases = List.copyOf(Objects.requireNonNull(aliases, "aliases"));
        validateAliases(this.aliases, stateDescriptors);
    }

    /**
     * Borrows a fully bound ordinary Sequential without changing its ownership, mode or state.
     *
     * @param source non-null supported tree; caller ensures quiescence through the snapshot
     * @return non-null immutable configuration with current exact Tensor-identity groups
     * @throws NullPointerException if source is null
     * @throws IllegalArgumentException if a module is unsupported or tree limits are exceeded
     * @throws IllegalStateException if any parameter remains unbound
     * @throws ArithmeticException if checked schema arithmetic overflows
     */
    public static ModelSpecification describe(Sequential source) {
        return describeChildren(Objects.requireNonNull(source, "source"));
    }

    /**
     * Returns the immutable root value retained at construction.
     *
     * @return the non-null closed Sequence in numeric ownership and forward order
     */
    public Sequence root() { return root; }

    /**
     * Returns the snapshot's canonical parameter-identity groups, not live source bindings.
     *
     * @return the immutable non-null groups ordered by first state member; unlisted paths are singletons
     */
    public List<AliasGroup> aliases() { return aliases; }

    /**
     * Returns the parameter schema derived solely from the retained configuration.
     *
     * @return immutable non-null parameter entries in Module traversal order, with no Tensor references
     */
    public List<StateDescriptor> stateDescriptors() { return stateDescriptors; }

    /**
     * Checks input-specific composition using ordinary storage-free Model operations.
     *
     * <p>No values are evaluated or allocated; temporary descriptor leaves and expression metadata
     * may consume Tensor identifiers. Empty trees return the exact supplied descriptor, including
     * layout. Other steps retain their ordinary Model metadata and runtime-check ownership.</p>
     *
     * <p>This checks only the supplied input. Successful schema construction is neither proof
     * that any particular input is accepted nor a general existence/shape-solving guarantee.</p>
     *
     * @param input non-null Model input metadata; no saved batch signature constrains leading axes
     * @return non-null output metadata, without backend admission or materialized values
     * @throws NullPointerException if input is null
     * @throws IllegalArgumentException if a step rejects input composition, with numeric step context
     * @throws ArithmeticException if checked Model arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public TensorDescriptor outputDescriptor(TensorDescriptor input) {
        Objects.requireNonNull(input, "input");
        if (root.steps().isEmpty()) return input;
        return DeclarativeModel.reconstruct(this).forward(
                io.github.pho001.synaptik.model.tensor.TensorFactory.create(input)).descriptor();
    }

    /** Closed semantic steps; no callback, custom module or arbitrary connection is representable. */
    public sealed interface Step permits Sequence, LinearStep, LayerNormStep, EmbeddingStep,
            Conv1dStep, Conv2dStep, Conv3dStep, ActivationStep { }

    /**
     * An ordered immutable composition; empty is the identity.
     *
     * @param steps non-null ordered children, defensively copied; each occurrence is independent
     */
    public record Sequence(List<Step> steps) implements Step {
        /**
         * Creates a validated immutable Sequence configuration.
         *
         * @param steps non-null ordered children, defensively copied; each occurrence is independent
         * @throws NullPointerException if {@code steps} or a child is null
         * @throws IllegalArgumentException if the subtree exceeds 4096 occurrences or depth 64,
         *     counting this sequence as depth one and as one occurrence
         */
        public Sequence {
            Objects.requireNonNull(steps, "steps");
            if (steps.size() >= 4096) throw new IllegalArgumentException("root step count exceeds 4096");
            steps = List.copyOf(steps);
            validateTreeChildren(steps);
        }
    }

    /**
     * A bound linear projection, preserving supplied promotion or automatic input guards.
     *
     * @param inFeatures positive resolved input-feature width
     * @param outFeatures positive output-feature width
     * @param bias whether a rank-one bias is declared
     * @param dataType non-null floating parameter type
     * @param automaticInputGuards true for exact input-type and positive static feature/count guards
     */
    public record LinearStep(long inFeatures, long outFeatures, boolean bias,
            DataType dataType, boolean automaticInputGuards) implements Step {
        /**
         * Creates a validated immutable LinearStep configuration.
         *
         * @param inFeatures positive resolved input-feature width
         * @param outFeatures positive output-feature width
         * @param bias whether a rank-one bias is declared
         * @param dataType non-null floating parameter type
         * @param automaticInputGuards true for exact input-type and positive static feature/count guards
         * @throws NullPointerException if {@code dataType} is null
         * @throws IllegalArgumentException if a width is non-positive, the type is not floating,
         *     or automatic parameter counts exceed the Java-array limit
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public LinearStep {
            positive(inFeatures, "LinearStep.inFeatures");
            positive(outFeatures, "LinearStep.outFeatures");
            parameterShape(dataType, Shape.of(outFeatures, inFeatures), automaticInputGuards);
            if (bias) parameterShape(dataType, Shape.of(outFeatures), automaticInputGuards);
        }
    }

    /**
     * Affine trailing normalization with exact retained epsilon type and bits.
     *
     * @param normalizedShape non-null positive fully static trailing Shape, retained unchanged
     * @param dataType non-null floating affine parameter type
     * @param epsilon non-null exact finite positive arithmetic-typed epsilon, retained without conversion
     */
    public record LayerNormStep(Shape normalizedShape, DataType dataType, ScalarValue epsilon) implements Step {
        /**
         * Creates a validated immutable LayerNormStep configuration.
         *
         * @param normalizedShape non-null positive fully static trailing Shape, retained unchanged
         * @param dataType non-null floating affine parameter type
         * @param epsilon non-null exact finite positive arithmetic-typed epsilon, retained without conversion
         * @throws NullPointerException if {@code normalizedShape}, {@code dataType} or {@code epsilon} is null
         * @throws IllegalArgumentException if the Shape is not positive fully static with positive
         *     rank, the type is not floating, or epsilon is not finite, positive and arithmetic-typed
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public LayerNormStep {
            parameterShape(dataType, normalizedShape, false);
            new AffineLayerNormAttrs(normalizedShape, Objects.requireNonNull(epsilon, "epsilon"));
            DataType expected = dataType == DataType.FLOAT16 ? DataType.FLOAT32 : dataType;
            if (epsilon.dataType() != expected)
                throw new IllegalArgumentException("LayerNormStep epsilon must match parameter arithmetic type");
        }
    }

    /**
     * An ordinary trainable table accepting INT32 or INT64 indices of any rank.
     *
     * @param vocabularySize positive number of trainable rows
     * @param embeddingSize positive appended output width
     * @param dataType non-null floating table type
     */
    public record EmbeddingStep(long vocabularySize, long embeddingSize, DataType dataType) implements Step {
        /**
         * Creates a validated immutable EmbeddingStep configuration.
         *
         * @param vocabularySize positive number of trainable rows
         * @param embeddingSize positive appended output width
         * @param dataType non-null floating table type
         * @throws NullPointerException if {@code dataType} is null
         * @throws IllegalArgumentException if a size is non-positive or the table type is not floating
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public EmbeddingStep {
            positive(vocabularySize, "EmbeddingStep.vocabularySize");
            positive(embeddingSize, "EmbeddingStep.embeddingSize");
            parameterShape(dataType, Shape.of(vocabularySize, embeddingSize), false);
        }
    }

    /**
     * A bound channels-first rank-3 convolution with complete Model geometry.
     *
     * @param inChannels positive resolved input channels divisible by groups
     * @param outChannels positive output channels divisible by groups
     * @param kernelWidth positive kernel width
     * @param bias whether a rank-one bias is declared
     * @param dataType non-null exact floating parameter and input type
     * @param attrs non-null complete Model stride, padding, dilation and group geometry
     */
    public record Conv1dStep(long inChannels, long outChannels, long kernelWidth,
            boolean bias, DataType dataType, Conv1dAttrs attrs) implements Step {
        /**
         * Creates a validated immutable Conv1dStep configuration.
         *
         * @param inChannels positive resolved input channels divisible by groups
         * @param outChannels positive output channels divisible by groups
         * @param kernelWidth positive kernel width
         * @param bias whether a rank-one bias is declared
         * @param dataType non-null exact floating parameter and input type
         * @param attrs non-null complete Model stride, padding, dilation and group geometry
         * @throws NullPointerException if {@code dataType} or {@code attrs} is null
         * @throws IllegalArgumentException if channels/kernel width are non-positive, channels
         *     are not divisible by groups, the type is not floating, or parameter counts exceed
         *     the Java-array limit
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public Conv1dStep {
            Objects.requireNonNull(attrs, "attrs");
            convolution(dataType, inChannels, outChannels, attrs.groups(), bias,
                    new long[] {kernelWidth});
            geometry(kernelWidth, attrs.padding(), attrs.dilation());
        }
    }

    /**
     * A bound channels-first rank-4 convolution with complete Model geometry.
     *
     * @param inChannels positive resolved input channels divisible by groups
     * @param outChannels positive output channels divisible by groups
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param bias whether a rank-one bias is declared
     * @param dataType non-null exact floating parameter and input type
     * @param attrs non-null complete Model stride, padding, dilation and group geometry
     */
    public record Conv2dStep(long inChannels, long outChannels, long kernelHeight, long kernelWidth,
            boolean bias, DataType dataType, Conv2dAttrs attrs) implements Step {
        /**
         * Creates a validated immutable Conv2dStep configuration.
         *
         * @param inChannels positive resolved input channels divisible by groups
         * @param outChannels positive output channels divisible by groups
         * @param kernelHeight positive kernel height
         * @param kernelWidth positive kernel width
         * @param bias whether a rank-one bias is declared
         * @param dataType non-null exact floating parameter and input type
         * @param attrs non-null complete Model stride, padding, dilation and group geometry
         * @throws NullPointerException if {@code dataType} or {@code attrs} is null
         * @throws IllegalArgumentException if channels/kernel extents are non-positive, channels
         *     are not divisible by groups, the type is not floating, or parameter counts exceed
         *     the Java-array limit
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public Conv2dStep {
            Objects.requireNonNull(attrs, "attrs");
            convolution(dataType, inChannels, outChannels, attrs.groups(), bias,
                    new long[] {kernelHeight, kernelWidth});
            geometry(kernelHeight, attrs.paddingHeight(), attrs.dilationHeight());
            geometry(kernelWidth, attrs.paddingWidth(), attrs.dilationWidth());
        }
    }

    /**
     * A bound channels-first rank-5 convolution with complete Model geometry.
     *
     * @param inChannels positive resolved input channels divisible by groups
     * @param outChannels positive output channels divisible by groups
     * @param kernelDepth positive kernel depth
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param bias whether a rank-one bias is declared
     * @param dataType non-null exact floating parameter and input type
     * @param attrs non-null complete Model stride, padding, dilation and group geometry
     */
    public record Conv3dStep(long inChannels, long outChannels, long kernelDepth, long kernelHeight,
            long kernelWidth, boolean bias, DataType dataType, Conv3dAttrs attrs) implements Step {
        /**
         * Creates a validated immutable Conv3dStep configuration.
         *
         * @param inChannels positive resolved input channels divisible by groups
         * @param outChannels positive output channels divisible by groups
         * @param kernelDepth positive kernel depth
         * @param kernelHeight positive kernel height
         * @param kernelWidth positive kernel width
         * @param bias whether a rank-one bias is declared
         * @param dataType non-null exact floating parameter and input type
         * @param attrs non-null complete Model stride, padding, dilation and group geometry
         * @throws NullPointerException if {@code dataType} or {@code attrs} is null
         * @throws IllegalArgumentException if channels/kernel extents are non-positive, channels
         *     are not divisible by groups, the type is not floating, or parameter counts exceed
         *     the Java-array limit
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public Conv3dStep {
            Objects.requireNonNull(attrs, "attrs");
            convolution(dataType, inChannels, outChannels, attrs.groups(), bias,
                    new long[] {kernelDepth, kernelHeight, kernelWidth});
            geometry(kernelDepth, attrs.paddingDepth(), attrs.dilationDepth());
            geometry(kernelHeight, attrs.paddingHeight(), attrs.dilationHeight());
            geometry(kernelWidth, attrs.paddingWidth(), attrs.dilationWidth());
        }
    }

    /** Closed mode-insensitive floating activations, with ordinary Model numerical semantics. */
    public enum ActivationKind {
        /** Model rectified linear activation. */ RELU,
        /** Model hyperbolic tangent activation. */ TANH,
        /** Model logistic sigmoid activation. */ SIGMOID
    }

    /**
     * A stateless activation delegating to the matching Model operation.
     *
     * @param kind non-null closed floating activation kind
     */
    public record ActivationStep(ActivationKind kind) implements Step {
        /**
         * Creates a validated immutable ActivationStep configuration.
         *
         * @param kind non-null closed floating activation kind
         * @throws NullPointerException if {@code kind} is null
         */
        public ActivationStep {
            Objects.requireNonNull(kind, "kind");
        }
    }

    /**
     * An exact parameter Tensor-identity group; equal values alone never imply sharing.
     *
     * @param paths non-null canonical state-order list of at least two distinct parameter paths, defensively copied
     */
    public record AliasGroup(List<String> paths) {
        /**
         * Snapshots distinct nonblank dot-segment paths; the enclosing specification validates
         * their existence, matching schema, traversal order and disjoint group membership.
         *
         * @param paths non-null canonical state-order list of at least two distinct parameter paths, defensively copied
         * @throws NullPointerException if {@code paths} or a member is null
         * @throws IllegalArgumentException if fewer than two distinct paths are supplied,
         *     a path repeats, or a dot-separated segment is blank
         */
        public AliasGroup {
            Objects.requireNonNull(paths, "paths");
            paths = List.copyOf(paths);
            if (paths.size() < 2 || new HashSet<>(paths).size() != paths.size())
                throw new IllegalArgumentException("AliasGroup requires at least two distinct paths");
            for (String path : paths) validatePath(path);
        }
    }

    /**
     * Detached parameter schema, ordered by declaration rather than lexical path sorting.
     *
     * @param path non-null numeric child path followed by the declared parameter name
     * @param kind non-null PARAMETER role; buffers are outside this closed matrix
     * @param dataType non-null floating parameter type
     * @param shape non-null positive fully static parameter Shape
     * @param requiresGrad true for every trainable state entry
     */
    public record StateDescriptor(String path, StateKind kind, DataType dataType,
            Shape shape, boolean requiresGrad) {
        /**
         * Creates a validated immutable StateDescriptor configuration.
         *
         * @param path non-null numeric child path followed by the declared parameter name
         * @param kind non-null PARAMETER role; buffers are outside this closed matrix
         * @param dataType non-null floating parameter type
         * @param shape non-null positive fully static parameter Shape
         * @param requiresGrad true for every trainable state entry
         * @throws NullPointerException if {@code path}, {@code kind}, {@code dataType} or {@code shape} is null
         * @throws IllegalArgumentException if a path segment is blank, kind is not PARAMETER,
         *     gradient eligibility is false, the type is not floating, or Shape is not positive
         *     fully static with positive rank
         * @throws ArithmeticException if checked schema arithmetic overflows
         */
        public StateDescriptor {
            validatePath(path);
            Objects.requireNonNull(kind, "kind");
            if (kind != StateKind.PARAMETER || !requiresGrad)
                throw new IllegalArgumentException("StateDescriptor requires gradient-eligible PARAMETER");
            parameterShape(dataType, shape, false);
        }
    }

    private static void parameterShape(DataType type, Shape shape, boolean arrayLimit) {
        Objects.requireNonNull(type, "dataType");
        Objects.requireNonNull(shape, "shape");
        if (!type.isFloating()) throw new IllegalArgumentException("parameter dataType must be floating");
        if (shape.rank() == 0 || !shape.isFullyStatic())
            throw new IllegalArgumentException("parameter Shape must have positive static rank: " + shape);
        for (int axis = 0; axis < shape.rank(); axis++)
            positive(((StaticDimension) shape.dimension(axis)).size(), "parameter axis " + axis);
        long count = shape.knownElementCount().orElseThrow();
        Math.multiplyExact(count, type.byteWidth());
        if (arrayLimit && count > Integer.MAX_VALUE)
            throw new IllegalArgumentException("parameter count exceeds Java array limit");
    }

    private static void positive(long value, String role) {
        if (value <= 0) throw new IllegalArgumentException(role + " must be positive: " + value);
    }

    private static void geometry(long kernel, long padding, long dilation) {
        Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
        Math.multiplyExact(padding, 2);
    }

    private static void convolution(DataType type, long in, long out, long groups,
            boolean bias, long[] kernels) {
        positive(in, "convolution inChannels");
        positive(out, "convolution outChannels");
        if (in % groups != 0 || out % groups != 0)
            throw new IllegalArgumentException("convolution channels must be divisible by groups");
        long[] extents = new long[kernels.length + 2];
        extents[0] = out;
        extents[1] = in / groups;
        long volume = 1;
        for (int i = 0; i < kernels.length; i++) {
            positive(kernels[i], "convolution kernel axis " + i);
            volume = Math.multiplyExact(volume, kernels[i]);
            extents[i + 2] = kernels[i];
        }
        Math.multiplyExact(in / groups, volume);
        Math.multiplyExact(out / groups, volume);
        parameterShape(type, Shape.of(extents), true);
        if (bias) parameterShape(type, Shape.of(out), true);
    }

    private static void validatePath(String path) {
        Objects.requireNonNull(path, "path");
        for (String segment : path.split("\\.", -1))
            if (segment.isBlank()) throw new IllegalArgumentException("invalid state path: " + path);
    }

    private static void validateTree(Sequence root) {
        validateTreeChildren(root.steps());
    }

    // Only bounded frames are pushed; no recursion is used to admit a tree.
    private static void validateTreeChildren(List<Step> steps) {
        ArrayDeque<TreeVisit> pending = new ArrayDeque<>();
        pushSteps(pending, steps, "", 2);
        int count = 1;
        while (!pending.isEmpty()) {
            TreeVisit visit = pending.pop();
            if (++count > 4096 || visit.depth > 64)
                throw new IllegalArgumentException("tree exceeds 4096 steps or depth 64 at " + visit.path);
            if (visit.step instanceof Sequence sequence)
                pushSteps(pending, sequence.steps(), visit.path, visit.depth + 1);
        }
    }

    private static void pushSteps(ArrayDeque<TreeVisit> pending, List<Step> steps, String path, int depth) {
        if (steps.size() + pending.size() > 4096)
            throw new IllegalArgumentException("tree step count exceeds 4096 at " + path);
        for (int i = steps.size() - 1; i >= 0; i--)
            pending.push(new TreeVisit(Objects.requireNonNull(steps.get(i), "step " + i),
                    childPath(path, i), depth));
    }

    static String childPath(String parent, int index) {
        return parent.isEmpty() ? Integer.toString(index) : parent + "." + index;
    }

    private record TreeVisit(Step step, String path, int depth) { }

    private static List<StateDescriptor> descriptors(Sequence root) {
        List<StateDescriptor> result = new ArrayList<>();
        ArrayDeque<TreeVisit> pending = new ArrayDeque<>();
        pushSteps(pending, root.steps(), "", 2);
        while (!pending.isEmpty()) {
            TreeVisit visit = pending.pop();
            String path = visit.path;
            switch (visit.step) {
                case Sequence sequence -> pushSteps(pending, sequence.steps(), path, visit.depth + 1);
                case LinearStep step -> addWeight(result, path, step.dataType(),
                        Shape.of(step.outFeatures(), step.inFeatures()), step.bias(), step.outFeatures());
                case EmbeddingStep step -> addWeight(result, path, step.dataType(),
                        Shape.of(step.vocabularySize(), step.embeddingSize()), false, 0);
                case LayerNormStep step -> {
                    addState(result, path + ".scale", step.dataType(), step.normalizedShape());
                    addState(result, path + ".bias", step.dataType(), step.normalizedShape());
                }
                case Conv1dStep step -> addWeight(result, path, step.dataType(),
                        Shape.of(step.outChannels(), step.inChannels() / step.attrs().groups(),
                                step.kernelWidth()), step.bias(), step.outChannels());
                case Conv2dStep step -> addWeight(result, path, step.dataType(),
                        Shape.of(step.outChannels(), step.inChannels() / step.attrs().groups(),
                                step.kernelHeight(), step.kernelWidth()), step.bias(), step.outChannels());
                case Conv3dStep step -> addWeight(result, path, step.dataType(),
                        Shape.of(step.outChannels(), step.inChannels() / step.attrs().groups(),
                                step.kernelDepth(), step.kernelHeight(), step.kernelWidth()),
                        step.bias(), step.outChannels());
                case ActivationStep _ -> { }
            }
        }
        return List.copyOf(result);
    }

    private static void addWeight(List<StateDescriptor> result, String path, DataType type,
            Shape shape, boolean bias, long out) {
        addState(result, path + ".weight", type, shape);
        if (bias) addState(result, path + ".bias", type, Shape.of(out));
    }

    private static void addState(List<StateDescriptor> result, String path, DataType type, Shape shape) {
        result.add(new StateDescriptor(path, StateKind.PARAMETER, type, shape, true));
    }

    private static void validateAliases(List<AliasGroup> aliases, List<StateDescriptor> states) {
        Map<String, Integer> order = new HashMap<>();
        for (int i = 0; i < states.size(); i++) order.put(states.get(i).path(), i);
        HashSet<String> used = new HashSet<>();
        int previousFirst = -1;
        for (AliasGroup group : aliases) {
            int previous = -1;
            StateDescriptor first = null;
            for (String path : group.paths()) {
                Integer index = order.get(path);
                if (index == null || index <= previous || !used.add(path))
                    throw new IllegalArgumentException("unknown, repeated or noncanonical alias path: " + path);
                StateDescriptor current = states.get(index);
                if (first == null) {
                    if (index <= previousFirst)
                        throw new IllegalArgumentException("noncanonical alias group at " + path);
                    previousFirst = index;
                    first = current;
                } else if (first.kind() != current.kind() || first.dataType() != current.dataType()
                        || !first.shape().equals(current.shape()) || first.requiresGrad() != current.requiresGrad())
                    throw new IllegalArgumentException("alias schema mismatch at " + path);
                previous = index;
            }
        }
    }

    static ModelSpecification describeChildren(Module source) {
        // Admit the source structure stack-safely before recursive immutable copying.
        ArrayDeque<SourceVisit> pending = new ArrayDeque<>();
        pending.push(new SourceVisit(source, "", 1));
        int count = 0;
        while (!pending.isEmpty()) {
            SourceVisit visit = pending.pop();
            if (++count > 4096 || visit.depth > 64)
                throw new IllegalArgumentException("source tree exceeds limits at " + visit.path);
            if (visit.module == source || visit.module instanceof Sequential) {
                List<Module> children = List.copyOf(visit.module.children().values());
                if (pending.size() + children.size() > 4096)
                    throw new IllegalArgumentException("source tree exceeds count at " + visit.path);
                for (int i = children.size() - 1; i >= 0; i--)
                    pending.push(new SourceVisit(children.get(i), childPath(visit.path, i), visit.depth + 1));
            } else {
                snapshotStep(visit.module, visit.path);
            }
        }
        Sequence root = snapshotSequence(source, "");
        List<StateEntry> entries = source.stateDictionary().entries();
        IdentityHashMap<Tensor, List<String>> groups = new IdentityHashMap<>();
        List<List<String>> ordered = new ArrayList<>();
        for (StateEntry entry : entries) {
            List<String> group = groups.get(entry.value());
            if (group == null) {
                group = new ArrayList<>();
                groups.put(entry.value(), group);
                ordered.add(group);
            }
            group.add(entry.path());
        }
        List<AliasGroup> aliases = ordered.stream().filter(group -> group.size() > 1)
                .map(AliasGroup::new).toList();
        return new ModelSpecification(root, aliases);
    }

    private record SourceVisit(Module module, String path, int depth) { }

    private static Sequence snapshotSequence(Module source, String path) {
        List<Step> steps = new ArrayList<>();
        int index = 0;
        for (Module child : source.children().values()) {
            String childPath = childPath(path, index++);
            steps.add(child instanceof Sequential ? snapshotSequence(child, childPath)
                    : snapshotStep(child, childPath));
        }
        return new Sequence(steps);
    }

    private static Step snapshotStep(Module module, String path) {
        try {
            return switch (module) {
                case Linear layer -> layer.specification();
                case LayerNorm layer -> layer.specification();
                case Embedding layer -> layer.specification();
                case Conv1d layer -> layer.specification();
                case Conv2d layer -> layer.specification();
                case Conv3d layer -> layer.specification();
                case Activation layer -> layer.specification();
                default -> throw new IllegalArgumentException("unsupported module at " + path + ": "
                        + module.getClass().getName());
            };
        } catch (IllegalStateException failure) {
            throw new IllegalStateException("unbound module at " + path, failure);
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("invalid step at " + path + ": " + failure.getMessage(), failure);
        }
    }
}

