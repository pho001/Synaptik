package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;

/**
 * Owns bounded cold source admission and reconstruction, separate from ordinary compilation.
 *
 * <p>Model construction already proves structural closure, signature counts, attributes classes
 * and descriptor layout geometry. This boundary adds static checked counts, FORWARD-only closed
 * family admission and existing exact descriptor inference. No Tensor is constructed until all
 * stored facts, every independently scoped body and positional substitutions pass. ValueId-to-Tensor
 * tables are call-local and separate for each scope after preflight; nothing caches them or attaches
 * them to returned expressions.</p>
 * <p>The closed slice includes the narrow default CNN/static recurrent compositions, exact typed
 * maximum/average Pool3d geometry and fixed flat scans, plus general scans with a bounded body-only
 * constructor closure. All-valid bodies retain applicable Model dtypes; length bodies require
 * static per-value floating types, including OFP8, and the retained rank/role/attribute intersection.
 * Cold headers still require nonempty carry and, in length mode, at least one ROW carry;
 * ordinary compilation and generated reverse scans have independent, broader admission.
 * Reconstruction creates fresh canonical formals for every declaration, invokes each public
 * family constructor once per occurrence and retains all canonical
 * ports, without time unrolling. New substitution ties are checked per occurrence before IDs.
 * Index and length values are never read; raw unresolved layouts stay unresolved. Admission is
 * independent of backend dtype qualification and persistence eligibility.</p>
 */
final class SourceGraphReconstruction {
    private SourceGraphReconstruction() {}

    /**
     * Verifies all stored source facts without rewriting descriptors or constructing Tensors.
     *
     * @param graph non-null immutable, structurally validated graph
     * @return non-null call-local descriptor index for subsequent positional association checks
     * @throws NullPointerException if graph is null
     * @throws IllegalArgumentException if static geometry, phase, family admission, inference or
     *     exact descriptor validation fails; checked count overflow is wrapped with value context
     */
    static Map<ValueId, TensorDescriptor> preflight(CompiledGraphModel graph) {
        return preflight(graph, false);
    }

    /**
     * Checks every value and occurrence in one lexical scope before any Tensor allocation.
     * General bodies recurse only from the outer scope; their broader inventory stays local.
     * Static floating length bodies intersect that same constructor inventory with independently
     * inferred per-role dtypes and row rules, including dead/zero-time scans.
     *
     * @param graph non-null structurally validated raw graph
     * @param bodyScope whether this graph is a contained general-scan body
     * @return complete scope-local descriptor index, never null
     * @throws NullPointerException if graph is null
     * @throws IllegalArgumentException if cold inventory, static geometry, raw roles/demand,
     *     phases, exact inference or absence of deferred obligations fails
     */
    private static Map<ValueId, TensorDescriptor> preflight(CompiledGraphModel graph, boolean bodyScope) {
        Objects.requireNonNull(graph, "graph");
        var descriptors = new HashMap<ValueId, TensorDescriptor>();
        for (GraphValue value : graph.values()) {
            TensorDescriptor descriptor = value.descriptor();
            if (!descriptor.shape().isFullyStatic()) {
                throw new IllegalArgumentException(value.id() + ": source Shape must be fully static");
            }
            try {
                descriptor.shape().knownElementCount();
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(value.id() + ": source element count overflow", failure);
            }
            descriptors.put(value.id(), descriptor);
        }
        for (CompiledNode node : graph.nodes()) {
            if (graph.nodePhases().get(node.id()) != GraphPhase.FORWARD) {
                throw new IllegalArgumentException(node.id() + ": source phase must be FORWARD");
            }
            var kind = node.operation().kind();
            if (kind == ScanKind.SCAN) {
                var attrs = (ScanAttrs) node.operation().attrs();
                if (attrs.carryCount() == 0 || attrs.lengthAware() && attrs.rowCarryCount() == 0) {
                    throw new IllegalArgumentException(node.id()
                            + ": cold scan requires nonempty carry and positive ROW carry in length mode");
                }
                if (bodyScope) throw new IllegalArgumentException("nested source scan");
                var body = node.scanBody().orElseThrow();
                var roles = new LinkedHashSet<>(body.nextCarry());
                roles.addAll(body.emissions());
                roles.addAll(body.residuals());
                if (!body.demandedOutputIndices().isEmpty()
                        || !body.graph().outputs().equals(List.copyOf(roles))) {
                    throw new IllegalArgumentException(node.id()
                            + ": cold scan requires empty demand and first-encounter unique role outputs");
                }
                if (attrs.lengthAware()) {
                    int formalCount = Math.addExact(Math.addExact(attrs.carryCount(),
                            attrs.sequenceCount()), attrs.invariantCount());
                    if (node.inputs().subList(0, formalCount).stream()
                                    .anyMatch(id -> !descriptors.get(id).dataType().isFloating())
                            || body.graph().values().stream()
                                    .anyMatch(value -> !value.descriptor().dataType().isFloating())) {
                        throw new IllegalArgumentException(node.id()
                                + ": cold length scan requires floating role and body values");
                    }
                }
                Map<ValueId, TensorDescriptor> bodyDescriptors = preflight(body.graph(), true);
                if (attrs.lengthAware()) {
                    try {
                        validateColdLengthIntersection(node, bodyDescriptors, descriptors);
                    } catch (ArithmeticException failure) {
                        throw new IllegalArgumentException(
                                node.id() + ": cold length Scan geometry overflows", failure);
                    }
                }
                continue;
            }
            if (bodyScope && kind instanceof RecurrentScanKind) {
                throw new IllegalArgumentException(node.id() + ": nested fixed source scan");
            }
            if (!bodyScope && kind == BinaryArithmeticKind.SUB && node.inputs().stream()
                    .anyMatch(id -> !descriptors.get(id).dataType().isFloating()))
                throw new IllegalArgumentException(node.id() + ": source SUB requires floating operands");
            boolean parameterless = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL
                    || kind == UnaryElementwiseKind.NEG
                    || kind == UnaryElementwiseKind.RELU
                    || kind == UnaryElementwiseKind.TANH
                    || kind == UnaryElementwiseKind.SIGMOID
                    || kind == MatmulKind.MATMUL;
            boolean transpose = kind == AxisTransformKind.PERMUTE
                    && node.operation().attrs() instanceof PermutationAttrs attrs
                    && attrs.axes().equals(List.of(1, 0));
            if (!(parameterless && node.operation().attrs() == NoOperationAttrs.INSTANCE)
                    && !transpose && !(bodyScope && bodyAdditional(node, descriptors))
                    && !additional(node, descriptors)) {
                throw new IllegalArgumentException(node.id() + ": unsupported source operation "
                        + node.operation());
            }
        }
        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);
        if (!validated.constraints().isEmpty()) {
            throw new IllegalArgumentException("source has deferred obligations: " + validated.constraints());
        }
        return descriptors;
    }

    /**
     * Retains the pre-coordinate-cutover length cold inventory plus exact typed Pool3d constructors.
     * Shared semantic facts are recomputed for the independently preflighted body; this predicate restricts
     * declarations and representability, not contributor mathematics. In particular, unused
     * sequence declarations still require ROW(0) or rank-two ROW(1), independently of the
     * broader source header. The containing occurrence is validated once by the existing
     * whole-graph inference boundary below. Entirely SHARED RELU and batched MATMUL remain
     * cold-unsupported even though forward metadata now admits them.
     * @param scan stored static length occurrence whose body has passed independent preflight
     * @param outerDescriptors exact containing-scope descriptors, including external lengths
     * @param descriptors exact independently inferred body-scope descriptors
     * @throws IllegalArgumentException if old operation/rank/role/attribute limits are exceeded
     */
    private static void validateColdLengthIntersection(CompiledNode scan,
            Map<ValueId, TensorDescriptor> descriptors, Map<ValueId, TensorDescriptor> outerDescriptors) {
        ScanAttrs header = (ScanAttrs) scan.operation().attrs();
        var body = scan.scanBody().orElseThrow();
        var lengths = outerDescriptors.get(scan.inputs().getLast()).shape();
        if (lengths.rank() != 1) throw coldLengthFailure();
        var batch = lengths.dimension(0);
        Map<ValueId, Scan.ValueSemantics> facts = new HashMap<>();
        for (int i = 0; i < body.graph().inputs().size(); i++) {
            int row = i < header.rowCarryCount() ? 0 : i >= header.carryCount()
                    && i < header.carryCount() + header.sequenceCount()
                    ? header.sequenceRowAxes().get(i - header.carryCount()).orElse(-1) : -1;
            if (row > 1 || row == 1
                    && descriptors.get(body.graph().inputs().get(i)).shape().rank() != 2)
                throw coldLengthFailure();
            facts.put(body.graph().inputs().get(i), Scan.declaredSemantics(
                    descriptors.get(body.graph().inputs().get(i)), row < 0 ? OptionalInt.empty() : OptionalInt.of(row), batch));
        }
        for (CompiledNode node : body.graph().nodes()) {
            var kind = node.operation().kind();
            var attrs = node.operation().attrs();
            boolean supported = kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV
                    || kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB
                    || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV
                    || kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH
                    || kind == UnaryElementwiseKind.SIGMOID || kind == ContiguousKind.CONTIGUOUS
                    || kind == CastKind.CAST || kind == AggregateReductionKind.SUM
                    || kind instanceof ShapeTransformKind || kind instanceof AxisTransformKind
                    || kind == Pool3dKind.MAX_POOL3D || kind == Pool3dKind.AVERAGE_POOL3D;
            if (kind == MatmulKind.MATMUL)
                supported = node.inputs().stream().allMatch(id -> descriptors.get(id).shape().rank() == 2);
            if (!supported || attrs instanceof MaskedReductionAttrs) throw coldLengthFailure();
            List<Scan.ValueSemantics> operands = node.inputs().stream().map(facts::get).toList();
            List<Scan.ValueSemantics> results = Scan.ordinarySemantics(node.operation(),
                    node.inputs().stream().map(descriptors::get).toList(), operands,
                    node.outputs().stream().map(descriptors::get).toList(), batch);
            boolean hadRow = operands.stream().anyMatch(f -> f.rowAxis().isPresent());
            if (hadRow && operands.stream().anyMatch(Scan.ValueSemantics::terminal)) throw coldLengthFailure();
            if (hadRow && (kind instanceof ShapeTransformKind
                    || kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE)) {
                if (operands.getFirst().rowAxis().orElse(-1) != 0
                        || results.getFirst().rowAxis().orElse(-1) != 0) throw coldLengthFailure();
                if (kind == ShapeTransformKind.EXPAND
                        && descriptors.get(node.inputs().getFirst()).shape().rank()
                        != descriptors.get(node.outputs().getFirst()).shape().rank()) throw coldLengthFailure();
                if (attrs instanceof AxisTransformAttrs a && a.axis() == 0) throw coldLengthFailure();
            }
            for (int slot = 0; slot < results.size(); slot++) {
                int row = results.get(slot).rowAxis().orElse(-1);
                if (row > 1 || row == 1 && descriptors.get(node.outputs().get(slot)).shape().rank() != 2)
                    throw coldLengthFailure();
                facts.put(node.outputs().get(slot), results.get(slot));
            }
        }
    }

    /** @return a fresh diagnostic for the retained cold constructor/coordinate intersection */
    private static IllegalArgumentException coldLengthFailure() {
        return new IllegalArgumentException("unsupported retained length-scan cold operation/rank/role/attributes");
    }

    /**
     * Selects the body-only constructor closure; raw inference owns family and dtype guards.
     *
     * @param node structurally validated ordinary local occurrence
     * @param descriptors complete local descriptor index, never an outer-scope table
     * @return whether the exact kind/attributes belong to the additional body inventory
     */
    private static boolean bodyAdditional(CompiledNode node, Map<ValueId, TensorDescriptor> descriptors) {
        var kind = node.operation().kind();
        var attrs = node.operation().attrs();
        if (kind == BinaryArithmeticKind.DIV || kind == ContiguousKind.CONTIGUOUS)
            return attrs == NoOperationAttrs.INSTANCE;
        if (kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB
                || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV)
            return attrs instanceof ScalarValueAttrs;
        if (kind == AggregateReductionKind.SUM)
            return attrs == NoOperationAttrs.INSTANCE || attrs instanceof AxisReductionAttrs
                    || attrs instanceof MultiAxisReductionAttrs || attrs instanceof SumToShapeAttrs;
        if (kind == ShapeTransformKind.RESHAPE || kind == ShapeTransformKind.EXPAND)
            return attrs instanceof TargetShapeAttrs target && target.targetShape().isFullyStatic();
        if (kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE)
            return attrs instanceof AxisTransformAttrs;
        if (kind != CastKind.CAST || !(attrs instanceof CastAttrs cast)) return false;
        DataType inputType = descriptors.get(node.inputs().getFirst()).dataType();
        return cast.targetDataType() == inputType
                || inputType.isFloating() && cast.targetDataType().isFloating();
    }

    /**
     * Admits selected builder geometry, full typed Pool3d geometry and fixed scans without storage access.
     * @param node structurally validated occurrence
     * @param descriptors complete static descriptor index
     * @return whether the occurrence belongs to the additional closed source slice
     * @throws IllegalArgumentException if a target attribute's checked element count overflows
     */
    private static boolean additional(CompiledNode node, Map<ValueId, TensorDescriptor> descriptors) {
        if (node.inputs().isEmpty()) return false;
        var kind = node.operation().kind();
        var attrs = node.operation().attrs();
        var first = descriptors.get(node.inputs().getFirst());
        if (kind == ShapeTransformKind.RESHAPE && attrs instanceof TargetShapeAttrs target) {
            try {
                return target.targetShape().isFullyStatic()
                    && java.util.Arrays.stream(target.targetShape().toLongArray()).allMatch(n -> n > 0)
                    && target.targetShape().knownElementCount().orElseThrow()
                        == first.shape().knownElementCount().orElseThrow();
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(node.id() + ": source target element count overflow", failure);
            }
        }
        if ((kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE)
                && attrs instanceof AxisTransformAttrs axis) return axis.axis() == 2;
        if (kind == SelectKind.SELECT && attrs instanceof SelectAttrs select)
            return select.axis() == 0 && first.shape().rank() == 3;
        if (kind == AxisGatherKind.GATHER && attrs instanceof IndexAxisAttrs axis) {
            var indices = descriptors.get(node.inputs().get(1));
            return axis.axis() == 0 && first.shape().rank() == 2 && indices.shape().rank() == 1
                    && indices.dataType() == DataType.INT64 && !indices.requiresGrad();
        }
        if (kind == SliceKind.SLICE && attrs instanceof SliceAttrs slice)
            return first.shape().rank() == 2 && slice.axes().equals(List.of(1))
                    && slice.lengths().getFirst() > 0 && slice.steps().equals(List.of(1L));
        if (kind == TensorCompositionKind.CONCAT && attrs instanceof CompositionAxisAttrs axis)
            return axis.axis() == 0 && node.inputs().stream()
                    .allMatch(id -> descriptors.get(id).shape().rank() == 3);
        if (kind == Conv2dKind.CONV2D) return node.inputs().size() == 3 && attrs.equals(Conv2dAttrs.defaults());
        if (kind == Conv3dKind.CONV3D) return node.inputs().size() == 3 && attrs.equals(Conv3dAttrs.defaults());
        if (kind == Pool2dKind.MAX_POOL2D && attrs instanceof MaxPool2dAttrs pool)
            return pool.strideHeight() == pool.kernelHeight() && pool.strideWidth() == pool.kernelWidth()
                    && pool.paddingHeight() == 0 && pool.paddingWidth() == 0
                    && pool.dilationHeight() == 1 && pool.dilationWidth() == 1 && !pool.ceilMode();
        if (kind == Pool3dKind.MAX_POOL3D) return attrs instanceof MaxPool3dAttrs;
        if (kind == Pool3dKind.AVERAGE_POOL3D) return attrs instanceof AveragePool3dAttrs;
        return kind instanceof RecurrentScanKind && attrs instanceof RecurrentDirection;
    }

    /**
     * Checks a snapshotted positional association without inspecting storage or upstream DAGs.
     *
     * @param graph non-null preflighted source graph
     * @param descriptors non-null complete descriptor index for that graph
     * @param leaves non-null snapshot without null elements
     * @param requireSourceLeaves whether to require provenance-free unique exact references;
     *     reconstruction instead permits expressions and repeated substitutions
     * @throws IllegalArgumentException if count, full descriptors, required leaf identity or
     *     per-occurrence length-role compatibility of tied substitutions fails
     */
    static void validateLeaves(
            CompiledGraphModel graph,
            Map<ValueId, TensorDescriptor> descriptors,
            List<Tensor> leaves,
            boolean requireSourceLeaves) {
        if (leaves.size() != graph.inputs().size()) {
            throw new IllegalArgumentException("leaves count " + leaves.size()
                    + " must match graph inputs count " + graph.inputs().size());
        }
        var seen = requireSourceLeaves ? new IdentityHashMap<Tensor, Boolean>() : null;
        for (int index = 0; index < leaves.size(); index++) {
            Tensor tensor = leaves.get(index);
            ValueId input = graph.inputs().get(index);
            String context = "leaves[" + index + "] for " + input + ": ";
            if (!descriptors.get(input).equals(tensor.descriptor())) {
                throw new IllegalArgumentException(context + "descriptor mismatch; expected="
                        + descriptors.get(input) + ", supplied=" + tensor.descriptor());
            }
            if (requireSourceLeaves && (tensor.provenance().isPresent()
                    || seen.put(tensor, Boolean.TRUE) != null)) {
                throw new IllegalArgumentException(context + "requires unique provenance-free leaves");
            }
        }
        if (!requireSourceLeaves) validateSubstitutionRoles(graph, leaves);
    }

    /**
     * Checks new input ties per length occurrence before creating even an earlier producer.
     * Only exact replacement identity matters: produced replacements remain opaque, while
     * source-produced actuals will receive fresh identities during reconstruction.
     *
     * @param graph completely preflighted raw outer graph
     * @param leaves descriptor-compatible positional substitutions, including retained named state
     * @throws IllegalArgumentException if a tie combines SHARED and ROW or different outer row axes
     */
    private static void validateSubstitutionRoles(CompiledGraphModel graph, List<Tensor> leaves) {
        var replacements = new HashMap<ValueId, Tensor>();
        for (int i = 0; i < leaves.size(); i++) replacements.put(graph.inputs().get(i), leaves.get(i));
        for (CompiledNode node : graph.nodes()) {
            if (node.operation().kind() != ScanKind.SCAN) continue;
            var attrs = (ScanAttrs) node.operation().attrs();
            if (!attrs.lengthAware()) continue;
            var roles = new IdentityHashMap<Tensor, Integer>();
            int sequenceEnd = attrs.carryCount() + attrs.sequenceCount();
            int formalEnd = sequenceEnd + attrs.invariantCount();
            for (int i = 0; i < formalEnd; i++) {
                Tensor replacement = replacements.get(node.inputs().get(i));
                if (replacement == null) continue;
                int role = -1;
                if (i < attrs.rowCarryCount()) role = 0;
                else if (i >= attrs.carryCount() && i < sequenceEnd) {
                    int stepRole = attrs.sequenceRowAxes().get(i - attrs.carryCount()).orElse(-1);
                    if (stepRole >= 0) role = stepRole + 1;
                }
                Integer previous = roles.put(replacement, role);
                if (previous != null && previous != role) {
                    throw new IllegalArgumentException(node.id()
                            + ": tied substitutions require compatible outer ROW/SHARED roles");
                }
            }
        }
    }

    /**
     * Preflights a complete source and substitutions before rebuilding each occurrence once.
     *
     * @param graph non-null raw source graph; never mutated
     * @param leaves non-null input-aligned substitutions without nulls; snapshotted and borrowed
     * @return non-null immutable ordered ordinary outputs, retaining substitutions by provenance
     * @throws NullPointerException if an argument or substitution is null
     * @throws IllegalArgumentException if source preflight or positional association fails
     * @throws IllegalStateException if a family constructor exhausts Tensor identity allocation
     */
    static List<Tensor> reconstruct(CompiledGraphModel graph, List<Tensor> leaves) {
        Objects.requireNonNull(graph, "graph");
        List<Tensor> snapshot = List.copyOf(Objects.requireNonNull(leaves, "leaves"));
        var descriptors = preflight(graph);
        validateLeaves(graph, descriptors, snapshot, false);

        var tensors = new HashMap<ValueId, Tensor>();
        for (int index = 0; index < snapshot.size(); index++) {
            tensors.put(graph.inputs().get(index), snapshot.get(index));
        }
        reconstructNodes(graph, tensors);
        return graph.outputs().stream().map(tensors::get).toList();
    }

    /**
     * Rebuilds each preflighted occurrence once into its owning scope's call-local table.
     * The table starts with every declared input; all canonical outputs are bound, not just roots.
     *
     * @param graph complete preflighted graph in topological order
     * @param tensors mutable scope-local table, initialized with input substitutions or fresh formals
     * @throws IllegalStateException if Tensor identity allocation is exhausted
     */
    private static void reconstructNodes(CompiledGraphModel graph, Map<ValueId, Tensor> tensors) {
        for (CompiledNode node : graph.nodes()) {
            Tensor first = tensors.get(node.inputs().getFirst());
            var kind = node.operation().kind();
            Tensor output;
            if (kind == BinaryArithmeticKind.ADD) {
                output = first.add(tensors.get(node.inputs().get(1)));
            } else if (kind == BinaryArithmeticKind.SUB) {
                output = first.sub(tensors.get(node.inputs().get(1)));
            } else if (kind == BinaryArithmeticKind.MUL) {
                output = first.mul(tensors.get(node.inputs().get(1)));
            } else if (kind == BinaryArithmeticKind.DIV) {
                output = first.div(tensors.get(node.inputs().get(1)));
            } else if (kind instanceof ScalarElementwiseKind scalar) {
                var value = ((ScalarValueAttrs) node.operation().attrs()).value();
                output = switch (scalar) {
                    case ADD -> first.add(value);
                    case SUB -> first.sub(value);
                    case MUL -> first.mul(value);
                    case DIV -> first.div(value);
                    default -> throw new IllegalArgumentException("Unsupported preflighted scalar: " + scalar);
                };
            } else if (kind == AggregateReductionKind.SUM) {
                output = reconstructSum(node, first);
            } else if (kind == UnaryElementwiseKind.NEG) {
                output = first.neg();
            } else if (kind == MatmulKind.MATMUL) {
                output = first.matmul(tensors.get(node.inputs().get(1)));
            } else if (kind == UnaryElementwiseKind.RELU) {
                output = first.relu();
            } else if (kind == UnaryElementwiseKind.TANH) {
                output = first.tanh();
            } else if (kind == UnaryElementwiseKind.SIGMOID) {
                output = first.sigmoid();
            } else if (kind == AxisTransformKind.PERMUTE) {
                output = first.permute(1, 0);
            } else if (kind == ShapeTransformKind.RESHAPE) {
                output = first.reshape(((TargetShapeAttrs) node.operation().attrs()).targetShape());
            } else if (kind == ShapeTransformKind.EXPAND) {
                output = first.expand(((TargetShapeAttrs) node.operation().attrs()).targetShape());
            } else if (kind == AxisTransformKind.EXPAND_DIMS) {
                output = first.expandDims(((AxisTransformAttrs) node.operation().attrs()).axis());
            } else if (kind == AxisTransformKind.SQUEEZE) {
                output = first.squeeze(((AxisTransformAttrs) node.operation().attrs()).axis());
            } else if (kind == ContiguousKind.CONTIGUOUS) {
                output = first.contiguous();
            } else if (kind == CastKind.CAST) {
                output = first.cast(((CastAttrs) node.operation().attrs()).targetDataType());
            } else if (kind == SelectKind.SELECT) {
                var attrs = (SelectAttrs) node.operation().attrs();
                output = first.select(attrs.axis(), attrs.index());
            } else if (kind == AxisGatherKind.GATHER) {
                output = first.gather(tensors.get(node.inputs().get(1)), 0);
            } else if (kind == SliceKind.SLICE) {
                var attrs = (SliceAttrs) node.operation().attrs();
                output = first.sliceByLength(new long[] {attrs.starts().getFirst()},
                        new long[] {attrs.lengths().getFirst()}, new int[] {1}, new long[] {1});
            } else if (kind == TensorCompositionKind.CONCAT) {
                output = Tensor.concat(0, node.inputs().stream().map(tensors::get).toArray(Tensor[]::new));
            } else if (kind == Conv2dKind.CONV2D) {
                output = first.conv2d(tensors.get(node.inputs().get(1)), tensors.get(node.inputs().get(2)),
                        (Conv2dAttrs) node.operation().attrs());
            } else if (kind == Conv3dKind.CONV3D) {
                output = first.conv3d(tensors.get(node.inputs().get(1)), tensors.get(node.inputs().get(2)),
                        (Conv3dAttrs) node.operation().attrs());
            } else if (kind == Pool2dKind.MAX_POOL2D) {
                output = first.maxPool2d((MaxPool2dAttrs) node.operation().attrs());
            } else if (kind == Pool3dKind.MAX_POOL3D) {
                output = first.maxPool3d((MaxPool3dAttrs) node.operation().attrs());
            } else if (kind == Pool3dKind.AVERAGE_POOL3D) {
                output = first.averagePool3d((AveragePool3dAttrs) node.operation().attrs());
            } else if (kind == ScanKind.SCAN) {
                List<Tensor> ports = reconstructScan(node, tensors);
                for (int j = 0; j < ports.size(); j++) tensors.put(node.outputs().get(j), ports.get(j));
                continue;
            } else if (kind instanceof RecurrentScanKind scan) {
                List<Tensor> args = node.inputs().stream().map(tensors::get).toList();
                var direction = (RecurrentDirection) node.operation().attrs();
                List<Tensor> ports;
                if (scan == RecurrentScanKind.LSTM) {
                    var result = args.size() == 6
                            ? RecurrentScan.lstm(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction)
                            : RecurrentScan.lstm(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), args.get(6), direction);
                    ports = List.of(result.outputs(), result.finalHidden(), result.finalCell());
                } else {
                    var result = scan == RecurrentScanKind.RNN_TANH
                            ? (args.size() == 5
                                ? RecurrentScan.rnn(first, args.get(1), args.get(2), args.get(3), args.get(4), direction)
                                : RecurrentScan.rnn(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction))
                            : (args.size() == 5
                                ? RecurrentScan.gru(first, args.get(1), args.get(2), args.get(3), args.get(4), direction)
                                : RecurrentScan.gru(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction));
                    ports = List.of(result.outputs(), result.finalHidden());
                }
                for (int j = 0; j < ports.size(); j++) tensors.put(node.outputs().get(j), ports.get(j));
                continue;
            } else {
                throw new IllegalArgumentException("Unsupported preflighted source operation: " + kind);
            }
            tensors.put(node.outputs().getFirst(), output);
        }
    }

    /**
     * Preserves the raw reduction variant, ordered axes, empty axes and exact Shape target.
     * @param node preflighted SUM occurrence with one input and selected attributes
     * @param input ordinary reconstructed input retained by the resulting producer
     * @return fresh canonical storage-free reduction result
     * @throws IllegalStateException if Tensor identity allocation is exhausted
     */
    private static Tensor reconstructSum(CompiledNode node, Tensor input) {
        return switch (node.operation().attrs()) {
            case NoOperationAttrs ignored -> input.sum();
            case AxisReductionAttrs axis -> input.sum(axis.axis(), axis.keepDimensions());
            case MultiAxisReductionAttrs axes -> input.sum(
                    axes.axes().stream().mapToInt(Integer::intValue).toArray(), axes.keepDimensions());
            case SumToShapeAttrs target -> input.sumToShape(target.targetShape());
            default -> throw new IllegalArgumentException("Unsupported preflighted SUM attributes");
        };
    }

    /**
     * Constructs an independent body and occurrence with no original template/formal references.
     * Every declaration receives a fresh canonical leaf, even when unused or bound to repeated
     * actuals. Local IDs resolve only in the new local table, discarded after construction.
     *
     * @param node preflighted body-bearing outer scan occurrence
     * @param outer call-local outer table containing opaque positional actual substitutions
     * @return all canonical final-carry, emission and residual ports in header order
     * @throws IllegalStateException if Tensor identity allocation is exhausted
     */
    private static List<Tensor> reconstructScan(CompiledNode node, Map<ValueId, Tensor> outer) {
        var captured = node.scanBody().orElseThrow();
        var graph = captured.graph();
        var attrs = (ScanAttrs) node.operation().attrs();
        var descriptors = new HashMap<ValueId, TensorDescriptor>();
        graph.values().forEach(value -> descriptors.put(value.id(), value.descriptor()));
        var local = new HashMap<ValueId, Tensor>();
        var formals = new ArrayList<Tensor>();
        for (ValueId input : graph.inputs()) {
            Tensor formal = TensorFactory.create(descriptors.get(input));
            local.put(input, formal);
            formals.add(formal);
        }
        reconstructNodes(graph, local);
        int c = attrs.carryCount();
        int s = c + attrs.sequenceCount();
        var body = new ScanBody(formals.subList(0, c), formals.subList(c, s),
                formals.subList(s, formals.size()), captured.nextCarry().stream().map(local::get).toList(),
                captured.emissions().stream().map(local::get).toList());
        var actuals = node.inputs().stream().map(outer::get).toList();
        int formalEnd = s + attrs.invariantCount();
        var result = !attrs.lengthAware()
                ? Scan.apply(actuals.subList(0, c), actuals.subList(c, s),
                        actuals.subList(s, formalEnd), body, attrs.reverse())
                : Scan.apply(actuals.subList(0, c), actuals.subList(c, s),
                        actuals.subList(s, formalEnd), actuals.get(formalEnd), attrs.rowCarryCount(),
                        attrs.sequenceRowAxes(), body, attrs.reverse());
        var ports = new ArrayList<>(result.finalCarry());
        ports.addAll(result.outputs());
        ports.addAll(result.residuals());
        return List.copyOf(ports);
    }
}
