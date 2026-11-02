package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.*;
import io.github.pho001.synaptik.model.operation.elementwise.logical.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Source-only construction of fixed-count pure scans from reusable ordinary Tensor bodies.
 *
 * <p>Each application creates one producer with distinct canonical final-carry, emission and
 * residual wrappers. Descriptors express stacked histories without allocating history storage.
 * This namespace performs no capture, differentiation, liveness analysis, preparation or
 * execution and owns no runner. Source admission is independent of backend qualification.</p>
 *
 * <p>Carry and final-carry groups may be empty when emissions supply semantic roots. Explicit
 * length mode also permits zero ROW carries, including terminal SHARED accumulators only.
 * These source forms remain independently rejected by current Compiler, Prepare, CPU and cold/wire consumers.</p>
 *
 * <p>All-valid source accepts static or bounded named-batch shapes. Compiler independently supports
 * bounded symbolic inference, FLOAT32/FLOAT64 first-order AD and all-port metadata binding.
 * Prepare/CPU/Engine implement bounded homogeneous FLOAT32 symbolic forward and actual first-order
 * reverse with one reusable prepared batch template, fixed time/features/state widths and generated
 * scalar single-range execution. Model PURE/dtype source validity does not grant either AD or CPU
 * support; each complete forward and generated reverse body qualifies independently. Symbolic cold
 * reconstruction, persistence and NN/training remain unavailable. Length-aware source also accepts
 * one named B. Compiler supports its bounded FLOAT32/FLOAT64 inference, actual first-order sibling
 * AD and fresh scoped all-port metadata binding. Prepare/CPU/Engine additionally implement the
 * bounded symbolic-length combination for homogeneous FLOAT32/FLOAT64 forward and actual reverse,
 * with fixed time/features, separate INT64 control and generated scalar single-range execution.
 * This does not admit all-valid symbolic FLOAT64, cold/persistence or NN/training workflows.</p>
 *
 * @see ScanBody
 * @see ScanResult
 */
public final class Scan {
    private Scan() { }

    /**
     * Binds a reusable step template and reserves every canonical result port before returning.
     * Inputs retain exact positional identities, including repeats. Carry/invariant descriptors
     * match formal shape, dtype and gradient eligibility; sequence formals remove axis zero and
     * have canonical geometry when static. All sequences have one static time count, including zero.
     * Shapes may instead contain one consistently named batch dimension as admitted by
     * {@link ScanBody}. Actuals match that exact named shape, never a sample numeric batch size.
     * Symbolic ports retain unresolved layouts; fully static ports retain canonical layouts.
     *
     * <p>Every differentiable outer port receives the OR of all actual input gradient flags,
     * including unused inputs and zero steps; nondifferentiable ports receive false. Source flags
     * are unchanged. This is conservative eligibility, not a derivative or execution guarantee.
     * Reverse traversal still describes histories in original-time order. Zero steps retain
     * initial carry values/raw bits but do not alias input wrappers. No body executes here.</p>
     *
     * <p>For matching canonical source leaves {@code h} and {@code step}, a recurrence that adds
     * each slice and emits the previous carry can be described without a callback:</p>
     * <pre>{@code
     * ScanBody body = new ScanBody(List.of(h), List.of(step), List.of(),
     *         List.of(h.add(step)), List.of(h));
     * ScanResult result = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
     * }</pre>
     * <p>The result describes final carry and original-time histories only. It grants no
     * Compiler, automatic-differentiation, Prepare or backend support for the occurrence.</p>
     *
     * <p>Argument lists are snapshotted without mutating their elements. Ordered producer inputs
     * are carry, sequence and invariant positions, without identity deduplication. Body storage
     * checks observe current associations; they do not freeze Tensor storage or make concurrent
     * mutations across several Tensors atomic. All binding and descriptor failures precede the
     * first canonical output identity.</p>
     *
     * @param initialCarry non-null ordered non-null actual carried values, possibly empty; each has
     *     exact static or bounded-symbolic shape, dtype and eligibility matching its carry formal
     * @param sequences non-null, non-empty ordered non-null axis-zero sequences with common static
     *     time count; each slice matches its step formal's shape, dtype and eligibility
     * @param invariants non-null ordered non-null actual invariants, possibly empty; each matches
     *     its formal's exact named or static shape, dtype and eligibility
     * @param body non-null reusable validated source body whose declared formals are disjoint
     *     from all actual input identities; the body itself is retained by exact reference
     * @param reverse whether traversal is decreasing in original time
     * @return immutable groups of exact canonical final-carry, emission and latent residual wrappers
     * @throws NullPointerException if a list, element or body is null
     * @throws IllegalArgumentException if role counts, exact shapes, dtypes, eligibility or static time
     *     counts mismatch, an actual is a declared formal, or a contained value has host storage
     * @throws ArithmeticException if descriptor or cardinality arithmetic overflows
     * @throws IllegalStateException if identifiers are exhausted; prior IDs stay consumed but no
     *     partial occurrence is returned
     */
    public static ScanResult apply(List<Tensor> initialCarry, List<Tensor> sequences,
            List<Tensor> invariants, ScanBody body, boolean reverse) {
        initialCarry = List.copyOf(initialCarry);
        sequences = List.copyOf(sequences);
        invariants = List.copyOf(invariants);
        Objects.requireNonNull(body, "body");
        List<Tensor> inputs = new ArrayList<>(initialCarry);
        inputs.addAll(sequences);
        inputs.addAll(invariants);
        long time = validateBindings(initialCarry, sequences, invariants, body);
        ScanAttrs attrs = new ScanAttrs(time, initialCarry.size(), sequences.size(), invariants.size(),
                body.emissions().size(), body.retainedValues().size(), reverse);
        List<TensorDescriptor> descriptors = descriptors(inputs, body, time);
        List<Tensor> ports = TensorFactory.createDerivedOutputs(
                new Operation(ScanKind.SCAN, attrs), inputs, descriptors, Optional.of(body));
        int carryEnd = attrs.carryCount();
        int emissionEnd = Math.addExact(carryEnd, attrs.emissionCount());
        return new ScanResult(ports.subList(0, carryEnd), ports.subList(carryEnd, emissionEnd),
                ports.subList(emissionEnd, ports.size()));
    }

    /**
     * Describes a length-aware whole-batch scan without reading lengths or executing its body.
     * The first {@code rowCarryCount} carries and every emission are ROW(0): they depend on an
     * explicitly identified leading batch coordinate B. Remaining carries are terminal SHARED
     * additive accumulators, with no extra B axis. Sequence roles address the step shape after
     * removing time. Absent axes mean SHARED, not a row guessed from equal extents.
     *
     * <p>For example, ROW(0) {@code h[B,H]} plus SHARED bias {@code bias[H]} preserves row identity;
     * reducing h over B may feed only a terminal shared contribution, never row feedback.
     * Bodies retain ordinary per-value dtypes and every known PURE family's contributor mapping.
     * Intermediate and declared sequence row coordinates may have any proved rank/axis;
     * next-carry and emission roots remain ROW(0). All declarations and canonical siblings
     * are checked even for zero T/B.
     * Ordinary family guards remain in force. The all-valid overload is unchanged.</p>
     * <p>Scan inserts no conversion or common working dtype. Each actual/formal pair and each
     * carry/next-carry pair keeps its exact shape and dtype. For low-precision h and step, for
     * example, {@code h.cast(DataType.FLOAT32).add(step.cast(DataType.FLOAT32))} describes an
     * F32 intermediate/emission; the next carry requires an explicit cast back to h's dtype.
     * A terminal SHARED update must be {@code old.add(contribution)} with contribution shape
     * and dtype exactly matching old, even when ordinary ADD promotion would preserve old's dtype.
     * Casts preserve row roles and terminal/shared-state dependencies, not just shape. This
     * source admission grants no wider Compiler, AD, Prepare, CPU, cold or wire capability.</p>
     * <p>Batch B comes from the INT64[B] lengths descriptor, either static or one named Dimension,
     * compared by normalized name rather than object identity. Every explicit ROW declaration is
     * checked against it. Time and every feature/SHARED extent remain static. Symbolic formals,
     * actuals and ports keep unresolved layouts. Fixed-feature products are checked even at T/B=0.</p>
     *
     * <p>Inputs are snapshotted in carry/sequence/invariant/lengths order; lengths has no formal.
     * Repeated actual identities must have the same explicit outer row axis (step axes gain one
     * for time), or all be SHARED; a tie cannot reinterpret one coordinate across positions.
     * Canonical ports and conservative gradient eligibility are as in the all-valid overload,
     * excluding discrete lengths from the OR. No storage, history or runtime state is allocated.
     * All validation precedes output Tensor identities. Compiler independently supports static or
     * one-named-B inference, first-order reverse construction, role/demand queries and fresh all-port
     * metadata binding. Compile/bind reads descriptors only, never length values. Prepare/CPU/Engine
     * support the bounded static-or-one-named-B FLOAT32/FLOAT64 length-mode execution domain;
     * each complete forward and actual generated reverse occurrence qualifies independently.
     * Original-source AD requires every carry and every original sequence to be ROW(0), even unused/non-gradient
     * positions; generated reverse histories/seeds may have explicit mixed roles. Prepare projects
     * those static or one-B bindings and demand without reading lengths. The generated scalar,
     * single-range CPU path supports fixed geometry and the bounded one-B combination, with
     * complete length-value validation after producers and before any scan-local writes, including
     * zero-feature and zero-time domains. Activity dominates padded loads and arithmetic.
     * Cold source capture/reconstruction separately admits only the fully static homogeneous
     * FLOAT32/FLOAT64 constructor/row-rule intersection; symbolic or wider per-value typed cold
     * bodies and length persistence remain unsupported. Construction and Compiler success alone
     * do not validate length values or execute skipping.</p>
     *
     * @param initialCarry non-null ordered actuals matching carry formals exactly, possibly empty;
     *     leading row carries contain the same static or named B at axis zero
     * @param sequences non-null non-empty axis-zero actuals with common static T and exact
     *     formal shapes after time removal, including each declared static or named row axis
     * @param invariants non-null ordered fully static SHARED actuals matching invariant formals
     * @param validLengths non-null non-gradient INT64[B] actual, disjoint from body formals;
     *     values are not inspected and storage is borrowed, not frozen
     * @param rowCarryCount non-negative number of leading ROW(0) carries, at most carry count
     * @param sequenceRowAxes non-null snapshotted non-null optional step axes, one per sequence;
     *     a present nonnegative axis denotes the exact B coordinate within that step's rank,
     *     absent denotes SHARED
     * @param body non-null reusable source template, retained by exact reference
     * @param reverse decreasing traversal in original-time coordinates when true
     * @return immutable groups of distinct final-carry, emission and residual wrappers, canonical
     *     when static and numerically unresolved when symbolic
     * @throws NullPointerException if any required reference or list element is null
     * @throws IllegalArgumentException if bindings, lengths metadata, roles, bounded ordinary algebra,
     *     terminal accumulator use, body storage or static/named geometry are invalid/unproved
     * @throws ArithmeticException if fixed-feature, static shape/layout or cardinality arithmetic overflows
     * @throws IllegalStateException if output identities are exhausted; no partial result returns
     */
    public static ScanResult apply(List<Tensor> initialCarry, List<Tensor> sequences,
            List<Tensor> invariants, Tensor validLengths, int rowCarryCount,
            List<OptionalInt> sequenceRowAxes, ScanBody body, boolean reverse) {
        initialCarry = List.copyOf(initialCarry);
        sequences = List.copyOf(sequences);
        invariants = List.copyOf(invariants);
        Objects.requireNonNull(body, "body");
        Objects.requireNonNull(validLengths, "validLengths");
        long time = validateBindings(initialCarry, sequences, invariants, body);
        ScanAttrs attrs = new ScanAttrs(time, initialCarry.size(), sequences.size(), invariants.size(),
                body.emissions().size(), body.retainedValues().size(), reverse, true, rowCarryCount, sequenceRowAxes);
        List<Tensor> inputs = new ArrayList<>(initialCarry);
        inputs.addAll(sequences);
        inputs.addAll(invariants);
        inputs.add(validLengths);
        validateRows(inputs, body, attrs);
        List<Tensor> ports = TensorFactory.createDerivedOutputs(new Operation(ScanKind.SCAN, attrs),
                inputs, descriptors(inputs.subList(0, inputs.size() - 1), body, time), Optional.of(body));
        int carryEnd = attrs.carryCount();
        int emissionEnd = Math.addExact(carryEnd, attrs.emissionCount());
        return new ScanResult(ports.subList(0, carryEnd), ports.subList(carryEnd, emissionEnd),
                ports.subList(emissionEnd, ports.size()));
    }

    /**
     * Validates source containment at the producer seam, including direct package-owned callers.
     * No Tensor identity is requested; ordinary/fixed operations require an empty attachment.
     *
     * @param operation non-null operation whose signature has accepted the occurrence counts
     * @param inputs non-null snapshotted ordered actual inputs with non-null elements
     * @param outputs non-null snapshotted output descriptors with non-null elements
     * @param attachment non-null optional source body, present exactly for general scans
     * @throws NullPointerException if {@code attachment} is null
     * @throws IllegalArgumentException if the kind, header, bindings, body storage or output
     *     descriptors disagree
     * @throws ArithmeticException if checked shape, layout or cardinality arithmetic overflows
     */
    static void validateAttachment(Operation operation, List<Tensor> inputs,
            List<TensorDescriptor> outputs, Optional<ScanBody> attachment) {
        Objects.requireNonNull(attachment, "scanBody");
        if (operation.kind() != ScanKind.SCAN) {
            if (attachment.isPresent()) {
                throw new IllegalArgumentException("ordinary/fixed producers cannot contain a scan body");
            }
            return;
        }
        ScanBody body = attachment.orElseThrow(
                () -> new IllegalArgumentException("scan producer requires source body"));
        ScanAttrs attrs = (ScanAttrs) operation.attrs();
        if (inputs.size() != attrs.inputCount() || outputs.size() != attrs.outputCount()
                || attrs.carryCount() != body.carry().size()
                || attrs.sequenceCount() != body.step().size()
                || attrs.invariantCount() != body.invariants().size()
                || attrs.emissionCount() != body.emissions().size()
                || attrs.residualCount() != body.retainedValues().size()) {
            throw new IllegalArgumentException("scan header/body cardinality mismatch");
        }
        int c = attrs.carryCount();
        int s = Math.addExact(c, attrs.sequenceCount());
        int formalCount = Math.addExact(s, attrs.invariantCount());
        long time = validateBindings(inputs.subList(0, c), inputs.subList(c, s),
                inputs.subList(s, formalCount), body);
        if (attrs.lengthAware()) validateRows(inputs, body, attrs);
        if (time != attrs.timeCount()) {
            throw new IllegalArgumentException("scan header/output descriptor mismatch");
        }
        boolean grad = inputs.subList(0, formalCount).stream().anyMatch(t -> t.descriptor().requiresGrad());
        int index = 0;
        for (Tensor value : body.nextCarry()) {
            validatePort(value, outputs.get(index++), grad, time, false);
        }
        for (Tensor value : body.emissions()) {
            validatePort(value, outputs.get(index++), grad, time, true);
        }
        for (Tensor value : body.retainedValues()) {
            validatePort(value, outputs.get(index++), grad, time, true);
        }
    }

    /**
     * Checks one supplied port without constructing a replacement descriptor or Tensor.
     * @param source exact body role supplying shape and dtype
     * @param port supplied canonical output descriptor to validate
     * @param grad conservative OR over all actual formal-bound input flags
     * @param time common static sequence length, including zero
     * @param stacked whether the port prepends the time axis to the body shape
     * @throws IllegalArgumentException if shape, dtype, eligibility or layout differs
     * @throws ArithmeticException if canonical static geometry or rank arithmetic overflows
     */
    private static void validatePort(Tensor source, TensorDescriptor port, boolean grad,
            long time, boolean stacked) {
        TensorDescriptor d = source.descriptor();
        Shape shape = port.shape();
        int prefix = stacked ? 1 : 0;
        if (port.dataType() != d.dataType()
                || port.requiresGrad() != (grad && d.dataType().isDifferentiable())
                || shape.rank() != Math.addExact(d.shape().rank(), prefix)
                || (stacked && !shape.dimension(0).equals(new StaticDimension(time)))) {
            throw new IllegalArgumentException("scan output descriptor mismatch");
        }
        for (int i = 0; i < d.shape().rank(); i++) {
            if (!shape.dimension(i + prefix).equals(d.shape().dimension(i))) {
                throw new IllegalArgumentException("scan output shape mismatch");
            }
        }
        if (!port.layout().equals(shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty())) {
            throw new IllegalArgumentException("scan outputs require canonical static or unresolved symbolic geometry");
        }
    }

    /**
     * Matches all source positions exactly, including unused declarations, without binding B.
     * @param carry ordered actual carry values, disjoint from the formal identities
     * @param sequences ordered sequences with a common static axis-zero length
     * @param invariants ordered actual invariant values
     * @param body validated reusable template, whose mutable storage is rechecked
     * @return common non-negative static time length
     * @throws IllegalArgumentException if arity, identity, exact shape/type/flags, time or body storage fails
     * @throws ArithmeticException if a static actual element count overflows
     */
    private static long validateBindings(List<Tensor> carry, List<Tensor> sequences,
            List<Tensor> invariants, ScanBody body) {
        body.validateStorage();
        if (carry.size() != body.carry().size() || sequences.size() != body.step().size()
                || invariants.size() != body.invariants().size()) {
            throw new IllegalArgumentException("scan actual/formal arity mismatch");
        }
        Set<Tensor> formals = Collections.newSetFromMap(new IdentityHashMap<>());
        formals.addAll(body.formals());
        long time = -1;
        for (int group = 0; group < 3; group++) {
            List<Tensor> actuals = group == 0 ? carry : group == 1 ? sequences : invariants;
            List<Tensor> declared = group == 0 ? body.carry() : group == 1 ? body.step() : body.invariants();
            for (int i = 0; i < actuals.size(); i++) {
                Tensor actual = actuals.get(i);
                TensorDescriptor d = actual.descriptor();
                if (formals.contains(actual) || !d.shape().isFullyStatic() && d.layout().isPresent()) {
                    throw new IllegalArgumentException("actual inputs must have unresolved symbolic geometry and be disjoint from formals");
                }
                Shape shape = d.shape();
                if (group == 1) {
                    if (shape.rank() == 0 || !shape.dimension(0).isStatic()) {
                        throw new IllegalArgumentException("sequence requires static time axis zero");
                    }
                    long t = shape.dimension(0).staticSize().orElseThrow();
                    if (time >= 0 && time != t) {
                        throw new IllegalArgumentException("sequence time counts differ");
                    }
                    time = t;
                    shape = Shape.ofDimensions(shape.dimensions().subList(1, shape.rank()).toArray(Dimension[]::new));
                }
                TensorDescriptor f = declared.get(i).descriptor();
                if (d.dataType() != f.dataType() || !shape.equals(f.shape())
                        || d.requiresGrad() != f.requiresGrad()) {
                    throw new IllegalArgumentException("actual/formal descriptor mismatch at role " + group + " slot " + i);
                }
                d.shape().knownElementCount();
            }
        }
        return time;
    }

    /**
     * Proves explicit row roles over the complete immutable source inventory without Tensor IDs.
     * Ordinary descriptor guards were applied by ScanBody; this adds only length-mode semantics.
     * Dtypes remain per value; terminal SHARED contributions match their own carry exactly.
     * Facts are per canonical output, including output-specific value/control routes and terminal
     * taint. A static or named B is structural provenance, not a coincident feature extent.
     * @param inputs complete ordered actuals, ending in lengths
     * @param body validated ordinary body whose storage has just been rechecked
     * @param attrs validated length-mode header
     * @throws IllegalArgumentException if any declaration, role propagation or carry use is unproved
     * @throws ArithmeticException if any declaration or canonical sibling's fixed-feature product overflows
     */
    private static void validateRows(List<Tensor> inputs, ScanBody body, ScanAttrs attrs) {
        Tensor lengths = inputs.getLast();
        TensorDescriptor ld = lengths.descriptor();
        if (ld.dataType() != DataType.INT64 || ld.requiresGrad() || ld.shape().rank() != 1
                || !ld.shape().isFullyStatic() && ld.layout().isPresent()
                || body.formals().stream().anyMatch(f -> f == lengths)) throw rowFailure();
        Dimension batch = ld.shape().dimension(0);
        if (!(batch instanceof StaticDimension) && !(batch instanceof DynamicDimension)) throw rowFailure();
        ld.shape().knownElementCount();
        var facts = new IdentityHashMap<Tensor, ValueSemantics>();
        var actualRoles = new IdentityHashMap<Tensor, Integer>();
        var sharedRoots = new IdentityHashMap<Tensor, Tensor>();
        List<Tensor> formals = body.formals();
        for (int i = 0; i < formals.size(); i++) {
            Tensor formal = formals.get(i);
            int role = -1;
            int actualRole = -1;
            if (i < attrs.carryCount()) {
                if (i < attrs.rowCarryCount()) role = actualRole = 0;
                else sharedRoots.put(formal, body.nextCarry().get(i));
            } else if (i < attrs.carryCount() + attrs.sequenceCount()) {
                role = attrs.sequenceRowAxes().get(i - attrs.carryCount()).orElse(-1);
                actualRole = role < 0 ? -1 : role + 1;
            }
            facts.put(formal, declaredSemantics(formal.descriptor(),
                    role < 0 ? OptionalInt.empty() : OptionalInt.of(role), batch));
            Integer previous = actualRoles.put(inputs.get(i), actualRole);
            if (previous != null && previous != actualRole) throw rowFailure();
        }
        Set<Tensor> carryDependent = Collections.newSetFromMap(new IdentityHashMap<>());
        carryDependent.addAll(sharedRoots.keySet());
        Set<TensorProducer> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor value : body.retainedValues()) {
            if (value.provenance().isEmpty()) continue;
            TensorProducer p = value.provenance().orElseThrow().producer();
            if (!seen.add(p)) continue;
            // Isolation is syntactic, before per-output routes: a numerically unused input
            // must not hide a forbidden shared-carry use.
            for (int i = 0; i < p.inputs().size(); i++) {
                Tensor input = p.inputs().get(i);
                if (!facts.containsKey(input)) throw rowFailure();
                if (carryDependent.contains(input)
                        && (i != 0 || p.operation().kind() != BinaryArithmeticKind.ADD
                        || sharedRoots.get(input) != p.output(0))) throw rowFailure();
            }
            List<ValueSemantics> results = ordinarySemantics(p.operation(),
                    p.inputs().stream().map(Tensor::descriptor).toList(),
                    p.inputs().stream().map(facts::get).toList(), p.outputDescriptors(), batch);
            for (int slot = 0; slot < p.outputCount(); slot++) {
                Tensor sibling = p.output(slot);
                ValueSemantics fact = results.get(slot);
                facts.put(sibling, fact);
                if (fact.directInputs().stream().anyMatch(i -> carryDependent.contains(p.inputs().get(i))))
                    carryDependent.add(sibling);
            }
        }
        for (int i = 0; i < attrs.carryCount(); i++) {
            Tensor root = body.nextCarry().get(i);
            if (i < attrs.rowCarryCount()) {
                if (facts.get(root).rowAxis().orElse(-1) != 0 || facts.get(root).terminal()) throw rowFailure();
            } else {
                TensorProducer p = root.provenance().orElseThrow(Scan::rowFailure).producer();
                if (p.operation().kind() != BinaryArithmeticKind.ADD || p.inputs().size() != 2
                        || p.inputs().getFirst() != body.carry().get(i)
                        || facts.get(root).rowAxis().isPresent() || facts.get(p.inputs().get(1)).rowAxis().isPresent()
                        || !p.inputs().get(1).descriptor().shape().equals(body.carry().get(i).descriptor().shape())
                        || p.inputs().get(1).descriptor().dataType() != body.carry().get(i).descriptor().dataType()
                        || carryDependent.contains(p.inputs().get(1))) throw rowFailure();
            }
        }
        for (Tensor root : body.emissions()) {
            if (facts.get(root).rowAxis().orElse(-1) != 0 || facts.get(root).terminal()) throw rowFailure();
        }
    }

    /**
     * Descriptor-only coordinate and value/control routes for one canonical value. Empty rowAxis
     * denotes SHARED; terminal denotes dependence on an established active row aggregation.
     * Input positions describe this value's producing occurrence, not derivative eligibility.
     * @param rowAxis non-null identified original batch coordinate, or empty for SHARED
     * @param terminal whether a SHARED value depends on active row aggregation
     * @param directInputs non-null increasing distinct nonnegative producing-input positions
     */
    public record ValueSemantics(OptionalInt rowAxis, boolean terminal, List<Integer> directInputs) {
        /**
         * Snapshots producing-input routes and checks the carrier's structural invariants.
         * @param rowAxis non-null identified axis, or empty for SHARED
         * @param terminal true only for SHARED active-aggregation dependencies
         * @param directInputs non-null increasing distinct nonnegative input positions
         * @throws NullPointerException if an argument or route element is null
         * @throws IllegalArgumentException if a coordinate, terminal combination or route is invalid
         */
        public ValueSemantics {
            Objects.requireNonNull(rowAxis, "rowAxis");
            directInputs = List.copyOf(directInputs);
            if (rowAxis.isPresent() && (rowAxis.getAsInt() < 0 || terminal)) throw rowFailure();
            int previous = -1;
            for (int input : directInputs) {
                if (input <= previous) throw rowFailure();
                previous = input;
            }
        }
    }

    /**
     * Validates a declaration without inferring a role from sizes or allocating Tensor identities.
     * Fixed-feature arithmetic is checked independently of empty time/batch dimensions.
     * @param descriptor non-null independently validated ordinary descriptor
     * @param rowAxis non-null explicit row coordinate, or empty for fully static SHARED
     * @param batch non-null exact static or canonical named batch Dimension
     * @return nonterminal declaration semantics with no producing-input routes
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if batch provenance or fixed geometry is invalid
     * @throws ArithmeticException if nonempty fixed-feature geometry overflows
     */
    public static ValueSemantics declaredSemantics(
            TensorDescriptor descriptor, OptionalInt rowAxis, Dimension batch) {
        Objects.requireNonNull(descriptor, "descriptor");
        Objects.requireNonNull(rowAxis, "rowAxis");
        Objects.requireNonNull(batch, "batch");
        if (!(batch instanceof StaticDimension) && !(batch instanceof DynamicDimension)) throw rowFailure();
        checkRow(descriptor.shape(), rowAxis.orElse(-1), batch);
        return new ValueSemantics(rowAxis, false, List.of());
    }

    /**
     * Recomputes all canonical output coordinate facts after ordinary family descriptor validation.
     * This routine performs no shape inference, promotion, Tensor construction, storage access,
     * differentiation or backend admission. Shared PURE work has no coordinate kind whitelist.
     * Each ROW contributor must access the same original batch coordinate; terminal dependencies
     * follow actual per-output value/control routes, separately from shape-domain provenance.
     * @param operation non-null known PURE ordinary occurrence with normalized attributes
     * @param inputs non-null ordered independently checked ordinary input descriptors
     * @param inputSemantics non-null corresponding previously proved facts
     * @param outputs non-null complete independently checked canonical output descriptors
     * @param batch non-null exact static or canonical named batch Dimension
     * @return immutable canonical-slot-ordered semantics, one for every output
     * @throws NullPointerException if any argument or list element is null
     * @throws IllegalArgumentException if purity, cardinality, provenance or contributor mapping is unproved
     * @throws ArithmeticException if fixed-feature or reshape-boundary arithmetic overflows
     */
    public static List<ValueSemantics> ordinarySemantics(Operation operation,
            List<TensorDescriptor> inputs, List<ValueSemantics> inputSemantics,
            List<TensorDescriptor> outputs, Dimension batch) {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(inputs);
        inputSemantics = List.copyOf(inputSemantics);
        outputs = List.copyOf(outputs);
        Objects.requireNonNull(batch, "batch");
        operation.signature().validateOccurrence(inputs.size(), outputs.size());
        if (inputs.size() != inputSemantics.size() || !knownPure(operation.kind())) throw rowFailure();
        for (int i = 0; i < inputs.size(); i++)
            declaredSemantics(inputs.get(i), inputSemantics.get(i).rowAxis(), batch);
        validateAttributeGeometry(operation.attrs(), batch);
        List<ValueSemantics> result = new ArrayList<>();
        for (int slot = 0; slot < outputs.size(); slot++) {
            List<Integer> routes = directInputs(operation.kind(), inputs.size(), slot);
            int row = -1;
            boolean terminal = false, eliminated = false;
            boolean anyRow = inputSemantics.stream().anyMatch(f -> f.rowAxis().isPresent());
            if (anyRow) row = outputRow(operation, inputs, inputSemantics, outputs, slot);
            for (int input : routes) {
                ValueSemantics fact = inputSemantics.get(input);
                terminal |= fact.terminal();
                eliminated |= fact.rowAxis().isPresent() && row < 0;
            }
            if (row >= 0 && terminal) throw rowFailure();
            declaredSemantics(outputs.get(slot), row < 0 ? OptionalInt.empty() : OptionalInt.of(row), batch);
            result.add(new ValueSemantics(row < 0 ? OptionalInt.empty() : OptionalInt.of(row),
                    terminal || eliminated, routes));
        }
        return List.copyOf(result);
    }

    /**
     * Recognizes semantic owners, never backend support or derivative eligibility.
     * @param kind ordinary operation kind
     * @return whether the kind has known PURE ordinary contributor semantics
     */
    private static boolean knownPure(OperationKind kind) {
        return switch (kind) {
            case UnaryElementwiseKind ignored -> true;
            case BinaryArithmeticKind ignored -> true;
            case BinaryComparisonKind ignored -> true;
            case FloatingClassificationKind ignored -> true;
            case BooleanLogicalKind ignored -> true;
            case WhereSelectionKind ignored -> true;
            case CastKind ignored -> true;
            case ScalarElementwiseKind ignored -> true;
            case MatmulKind ignored -> true;
            case ContiguousKind ignored -> true;
            case ShapeTransformKind ignored -> true;
            case AxisTransformKind ignored -> true;
            case TensorCompositionKind ignored -> true;
            case SliceKind ignored -> true;
            case SelectKind ignored -> true;
            case PadKind ignored -> true;
            case TileKind ignored -> true;
            case AxisGatherKind ignored -> true;
            case GatherNdKind ignored -> true;
            case OneHotKind ignored -> true;
            case AxisScatterKind ignored -> true;
            case ScatterNdKind ignored -> true;
            case CumulativeScanKind ignored -> true;
            case AggregateReductionKind ignored -> true;
            case SoftmaxKind ignored -> true;
            case LayerNormKind ignored -> true;
            case RmsNormKind ignored -> true;
            case Conv2dKind ignored -> true;
            case Conv3dKind ignored -> true;
            case WindowTransformKind ignored -> true;
            case LossKind ignored -> true;
            case TopKKind ignored -> true;
            case OrderingKind ignored -> true;
            case BatchNormKind ignored -> true;
            case Pool2dKind ignored -> true;
            case Pool3dKind ignored -> true;
            case ScaledDotProductAttentionKind ignored -> true;
            default -> false;
        };
    }

    /**
     * Keeps producing-input positions distinct even when exact Tensor operands are tied.
     * @param kind known PURE family kind
     * @param count validated ordinary input count
     * @param slot canonical output slot
     * @return immutable ordered numerical/control dependency positions
     * @throws IllegalArgumentException if a BatchNorm canonical slot is invalid
     */
    private static List<Integer> directInputs(OperationKind kind, int count, int slot) {
        if (kind == BatchNormKind.BATCH_NORM_TRAINING) return switch (slot) {
            case 0 -> List.of(0, 1, 2);
            case 1 -> List.of(0, 3);
            case 2 -> List.of(0, 4);
            case 3, 4 -> List.of(0);
            default -> throw rowFailure();
        };
        if (kind instanceof ScaledDotProductAttentionKind && slot == 1)
            return count == 4 ? List.of(0, 1, 3) : List.of(0, 1);
        List<Integer> positions = new ArrayList<>(count);
        for (int i = 0; i < count; i++) positions.add(i);
        return List.copyOf(positions);
    }

    /**
     * Checks shape-bearing attributes without deriving ordinary result descriptors.
     * @param attrs normalized ordinary attributes
     * @param batch exact containing static or named batch Dimension
     * @throws IllegalArgumentException if an attribute contains foreign/repeated named geometry
     * @throws ArithmeticException if fixed-feature products overflow
     */
    private static void validateAttributeGeometry(OperationAttrs attrs, Dimension batch) {
        List<Shape> shapes = switch (attrs) {
            case TargetShapeAttrs a -> List.of(a.targetShape());
            case SumToShapeAttrs a -> List.of(a.targetShape());
            case CropToShapeAttrs a -> List.of(a.targetShape(), a.prefixShape());
            case LayerNormAttrs a -> List.of(a.normalizedShape());
            case AffineLayerNormAttrs a -> List.of(a.normalizedShape());
            case RmsNormAttrs a -> List.of(a.normalizedShape());
            case Fold2dAttrs a -> List.of(a.outputShape());
            case Fold3dAttrs a -> List.of(a.outputShape());
            default -> List.of();
        };
        for (Shape shape : shapes) {
            int named = -1;
            for (int axis = 0; axis < shape.rank(); axis++) if (!shape.dimension(axis).isStatic()) {
                if (named >= 0 || !shape.dimension(axis).equals(batch)) throw rowFailure();
                named = axis;
            }
            checkRow(shape, named, batch);
        }
    }

    /**
     * Merges independently identified coordinates without conflating two free row domains.
     * @param left first output row axis, or -1 for SHARED
     * @param right second output row axis, or -1 for SHARED
     * @return the common identified axis, or -1 when neither identifies a row
     * @throws IllegalArgumentException if two distinct axes are identified
     */
    private static int mergeRow(int left, int right) {
        if (left >= 0 && right >= 0 && left != right) throw rowFailure();
        return Math.max(left, right);
    }

    /**
     * Enforces right-aligned same-row access and SHARED singleton/omitted alignment.
     * @param inputs independently checked ordered operand descriptors
     * @param facts corresponding proved operand facts
     * @param out independently checked ordinary output shape
     * @return mapped output row axis, or -1 for entirely SHARED work
     * @throws IllegalArgumentException if alignment mixes or invents row coordinates
     */
    private static int pointwise(List<TensorDescriptor> inputs, List<ValueSemantics> facts, Shape out) {
        int row = -1;
        for (int i = 0; i < inputs.size(); i++) {
            int axis = facts.get(i).rowAxis().orElse(-1);
            if (axis >= 0) row = mergeRow(row, axis + out.rank() - inputs.get(i).shape().rank());
        }
        if (row >= 0) for (int i = 0; i < inputs.size(); i++) {
            if (facts.get(i).rowAxis().isPresent()) continue;
            Shape shape = inputs.get(i).shape();
            int aligned = row - out.rank() + shape.rank();
            if (aligned >= 0 && !singleton(shape, aligned)) throw rowFailure();
        }
        return row;
    }

    /**
     * Tests an exact fixed singleton, not an extent that might bind to one later.
     * @param shape independently checked operand shape
     * @param axis valid logical axis
     * @return whether the axis is structurally the static extent one
     */
    private static boolean singleton(Shape shape, int axis) {
        return shape.dimension(axis).equals(new StaticDimension(1));
    }

    /**
     * Preserves the explicit reshape boundary, including static B=0/1. Exact identity retains its
     * axis; changed shapes require a complete matching pair of fixed feature products. If several
     * axes independently prove that same contributor identity, the lowest is selected.
     * @param input independently checked input shape with identified row coordinate
     * @param row exact nonnegative input row axis
     * @param output independently checked ordinary reshape result
     * @return original exact-shape axis, otherwise the lowest fully proved output row axis
     * @throws IllegalArgumentException if no output axis preserves the exact coordinate boundary
     * @throws ArithmeticException if either fixed boundary product overflows
     */
    static int reshapeRow(Shape input, int row, Shape output) {
        if (input.equals(output)) return row;
        long before = fixedProduct(input, 0, row), after = fixedProduct(input, row + 1, input.rank());
        int result = -1;
        for (int axis = 0; axis < output.rank(); axis++) {
            if (!input.dimension(row).equals(output.dimension(axis))) continue;
            boolean fixed = true;
            for (int i = 0; i < output.rank(); i++) if (i != axis && !output.dimension(i).isStatic()) fixed = false;
            if (fixed && before == fixedProduct(output, 0, axis)
                    && after == fixedProduct(output, axis + 1, output.rank())) {
                if (result < 0) result = axis;
            }
        }
        if (result < 0) throw rowFailure();
        return result;
    }

    /**
     * Counts fixed coordinates on one side of an identified row boundary.
     * @param shape shape whose selected interval is fully static
     * @param start inclusive first fixed axis
     * @param end exclusive last fixed axis
     * @return checked product, one for no axes and zero for an empty feature interval
     * @throws ArithmeticException if a nonempty fixed product overflows
     */
    private static long fixedProduct(Shape shape, int start, int end) {
        for (int i = start; i < end; i++) if (shape.dimension(i).equals(new StaticDimension(0))) return 0;
        long count = 1;
        for (int i = start; i < end; i++)
            count = Math.multiplyExact(count, shape.dimension(i).staticSize().orElseThrow());
        return count;
    }

    /**
     * Maps actual contributor axes, independently of ordinary shape inference.
     * @param operation normalized known PURE occurrence
     * @param inputs independently validated ordinary inputs
     * @param facts corresponding proved input facts
     * @param outputs complete independently validated canonical siblings
     * @param slot canonical output whose coordinate is being proved
     * @return identified output row axis, or -1 for SHARED
     * @throws IllegalArgumentException if any contributor mixes or loses the identified row
     * @throws ArithmeticException if checked coordinate-boundary arithmetic overflows
     */
    private static int outputRow(Operation operation, List<TensorDescriptor> inputs,
            List<ValueSemantics> facts, List<TensorDescriptor> outputs, int slot) {
        OperationKind kind = operation.kind();
        OperationAttrs attrs = operation.attrs();
        Shape in = inputs.getFirst().shape(), out = outputs.get(slot).shape();
        int row = facts.getFirst().rowAxis().orElse(-1);
        if (kind instanceof BinaryArithmeticKind || kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind || kind instanceof WhereSelectionKind)
            return pointwise(inputs, facts, out);
        if (kind instanceof UnaryElementwiseKind || kind instanceof ScalarElementwiseKind
                || kind instanceof FloatingClassificationKind || kind instanceof CastKind
                || kind instanceof ContiguousKind || kind instanceof OneHotKind) return row;
        if (kind == ShapeTransformKind.RESHAPE) return row < 0 ? -1 : reshapeRow(in, row, out);
        if (kind == ShapeTransformKind.EXPAND) return pointwise(inputs, facts, out);
        if (kind == AxisTransformKind.PERMUTE) return row < 0 ? -1 : ((PermutationAttrs) attrs).axes().indexOf(row);
        if (kind == AxisTransformKind.EXPAND_DIMS) {
            int axis = ((AxisTransformAttrs) attrs).axis();
            return row < 0 ? -1 : row + (axis <= row ? 1 : 0);
        }
        if (kind == AxisTransformKind.SQUEEZE || kind instanceof SelectKind) {
            int axis = attrs instanceof SelectAttrs a ? a.axis() : ((AxisTransformAttrs) attrs).axis();
            if (row == axis) throw rowFailure();
            return row < 0 ? -1 : row - (axis < row ? 1 : 0);
        }
        if (kind instanceof TensorCompositionKind) {
            int axis = ((CompositionAxisAttrs) attrs).axis();
            boolean stack = kind == TensorCompositionKind.STACK;
            int result = -1;
            for (int i = 0; i < facts.size(); i++) {
                int r = facts.get(i).rowAxis().orElse(-1);
                if (r < 0) continue;
                if (!stack && axis == r && inputs.size() != 1) throw rowFailure();
                result = mergeRow(result, r + (stack && axis <= r ? 1 : 0));
            }
            if (result >= 0) for (int i = 0; i < facts.size(); i++) {
                if (facts.get(i).rowAxis().isPresent()) continue;
                int mapped = result - (stack && axis < result ? 1 : 0);
                if (!singleton(inputs.get(i).shape(), mapped)) throw rowFailure();
            }
            return result;
        }
        if (kind instanceof SliceKind) return sliceRow(inputs, facts, attrs);
        if (kind instanceof PadKind) {
            PadAttrs a = (PadAttrs) attrs;
            if (row >= 0 && (a.before().get(row) != 0 || a.after().get(row) != 0)) throw rowFailure();
            return row;
        }
        if (kind instanceof TileKind) {
            if (row >= 0 && ((TileAttrs) attrs).repeats().get(row) != 1) throw rowFailure();
            return row;
        }
        if (kind instanceof AxisGatherKind || kind instanceof GatherNdKind) return gatherRow(operation, inputs, facts);
        if (kind instanceof AxisScatterKind || kind instanceof ScatterNdKind) return scatterRow(operation, inputs, facts);
        if (kind instanceof AggregateReductionKind) return reductionRow(operation, inputs, facts, out);
        if (kind instanceof CumulativeScanKind) {
            if (row == ((CumulativeScanAttrs) attrs).axis()) throw rowFailure();
            return row;
        }
        if (kind instanceof SoftmaxKind || kind instanceof OrderingKind || kind instanceof TopKKind) {
            int axis = switch (attrs) {
                case SoftmaxAttrs a -> a.axis();
                case SortAttrs a -> a.axis();
                case TopKAttrs a -> a.axis();
                default -> throw rowFailure();
            };
            if (row == axis) throw rowFailure();
            return row;
        }
        if (kind instanceof LayerNormKind || kind instanceof RmsNormKind) {
            Shape normalized = switch (attrs) {
                case LayerNormAttrs a -> a.normalizedShape();
                case AffineLayerNormAttrs a -> a.normalizedShape();
                case RmsNormAttrs a -> a.normalizedShape();
                default -> throw rowFailure();
            };
            if (row >= in.rank() - normalized.rank()) throw rowFailure();
            return pointwise(inputs, facts, out);
        }
        if (kind instanceof BatchNormKind) return batchNormRow(operation, inputs, facts, slot);
        if (kind instanceof MatmulKind) return matmulRow(inputs, facts, out);
        if (kind instanceof ScaledDotProductAttentionKind) return attentionRow(inputs, facts, outputs, slot);
        if (kind instanceof Conv2dKind || kind instanceof Conv3dKind) return convolutionRow(inputs, facts, attrs);
        if (kind instanceof Pool2dKind || kind instanceof Pool3dKind) return poolRow(in, row, attrs);
        if (kind instanceof WindowTransformKind) return windowRow(in, out, row, attrs);
        if (kind instanceof LossKind) return lossRow(inputs, facts, attrs, out);
        throw rowFailure();
    }

    /**
     * Proves addressed-row identity for slices, crops and functional updates.
     * @param inputs validated base and optional update descriptors
     * @param facts corresponding proved value facts
     * @param attrs normalized slice or crop/update region attributes
     * @return identified unchanged row axis, or -1 for SHARED
     * @throws IllegalArgumentException if a region selects, shifts or reverses the row
     */
    private static int sliceRow(List<TensorDescriptor> inputs, List<ValueSemantics> facts, OperationAttrs attrs) {
        int row = facts.getFirst().rowAxis().orElse(-1);
        Shape shape = inputs.getFirst().shape();
        if (inputs.size() == 2) {
            row = mergeRow(row, facts.get(1).rowAxis().orElse(-1));
            if (row >= 0 && (!sameCoordinate(facts.get(0), shape, row)
                    || !sameCoordinate(facts.get(1), inputs.get(1).shape(), row))) throw rowFailure();
        }
        if (attrs instanceof SliceAttrs a) {
            int position = a.axes().indexOf(row);
            if (position >= 0 && (a.starts().get(position) != 0 || a.steps().get(position) != 1
                    || !shape.dimension(row).equals(new StaticDimension(a.lengths().get(position))))) throw rowFailure();
        } else if (attrs instanceof CropToShapeAttrs a) {
            if (row >= 0 && (!a.targetShape().dimension(row).equals(shape.dimension(row))
                    || !a.prefixShape().dimension(row).equals(new StaticDimension(0)))) throw rowFailure();
            if (row >= 0) {
                checkRow(a.targetShape(), row, shape.dimension(row));
                checkRow(a.prefixShape(), row, new StaticDimension(0));
            }
        } else throw rowFailure();
        return row;
    }

    /**
     * Traces surviving data axes and inserted index-position axes separately.
     * @param operation ordinary axis-gather or Gather-ND occurrence
     * @param inputs validated data and indices descriptors
     * @param facts corresponding proved data and indices facts
     * @return mapped output row axis, or -1 for SHARED
     * @throws IllegalArgumentException if tuple/axis indexing addresses arbitrary original rows
     */
    private static int gatherRow(Operation operation, List<TensorDescriptor> inputs, List<ValueSemantics> facts) {
        Shape data = inputs.get(0).shape(), indices = inputs.get(1).shape();
        int dataRow = facts.get(0).rowAxis().orElse(-1), indexRow = facts.get(1).rowAxis().orElse(-1);
        if (operation.attrs() instanceof GatherNdAttrs a) {
            int prefix = a.batchDimensions(), depth = Math.toIntExact(indices.dimension(-1).staticSize().orElseThrow());
            if (indexRow == indices.rank() - 1 || dataRow >= prefix && dataRow < prefix + depth) throw rowFailure();
            int result = dataRow < 0 ? -1 : dataRow < prefix ? dataRow : indices.rank() - 1 + dataRow - prefix - depth;
            if (indexRow >= 0) result = mergeRow(result, indexRow);
            if (result >= 0 && result < prefix
                    && (dataRow < 0 && !singleton(data, result) || indexRow < 0 && !singleton(indices, result))) throw rowFailure();
            return result;
        }
        int axis = ((IndexAxisAttrs) operation.attrs()).axis();
        if (dataRow == axis) throw rowFailure();
        if (operation.kind() == AxisGatherKind.GATHER_ELEMENTS) {
            int result = mergeRow(dataRow, indexRow);
            if (result >= 0 && (dataRow < 0 && result != axis && !singleton(data, result)
                    || indexRow < 0 && !singleton(indices, result))) throw rowFailure();
            return result;
        }
        int result = dataRow < 0 ? -1 : dataRow < axis ? dataRow : dataRow + indices.rank() - 1;
        return indexRow < 0 ? result : mergeRow(result, axis + indexRow);
    }

    /**
     * Inverts ordinary gather placement; only ADD establishes an active shared-target terminal.
     * @param operation normalized ordinary functional scatter occurrence
     * @param inputs validated base, indices and updates descriptors
     * @param facts corresponding proved value facts
     * @return preserved base row axis, or -1 for shared targets
     * @throws IllegalArgumentException if contributor rows disagree, row destinations vary, or
     *     an active terminal is not ADD or uses terminal-derived indices/updates
     */
    private static int scatterRow(Operation operation, List<TensorDescriptor> inputs, List<ValueSemantics> facts) {
        int base = facts.get(0).rowAxis().orElse(-1);
        int index = facts.get(1).rowAxis().orElse(-1), update = facts.get(2).rowAxis().orElse(-1);
        Shape indices = inputs.get(1).shape(), updates = inputs.get(2).shape();
        boolean add = operation.kind() == AxisScatterKind.SCATTER_ADD
                || operation.attrs() instanceof ScatterElementsAttrs elements && elements.reduction() == ScatterReduction.ADD
                || operation.attrs() instanceof ScatterNdAttrs nd && nd.reduction() == ScatterReduction.ADD;
        if (base < 0 && (!add || facts.get(1).terminal() || facts.get(2).terminal())) throw rowFailure();
        if (operation.attrs() instanceof ScatterNdAttrs a) {
            int prefix = a.batchDimensions(), depth = Math.toIntExact(indices.dimension(-1).staticSize().orElseThrow());
            if (index == indices.rank() - 1) throw rowFailure();
            if (base < 0) {
                if (update >= indices.rank() - 1) {
                    // Untouched data suffix coordinates do not occur in the indices tuple prefix.
                    if (index >= 0) throw rowFailure();
                    return -1;
                }
                int contributor = mergeRow(index, update);
                if (contributor < 0 || contributor >= indices.rank() - 1
                        || !sameCoordinate(facts.get(1), indices, contributor)
                        || !sameCoordinate(facts.get(2), updates, contributor)) throw rowFailure();
                return -1;
            }
            if (base >= prefix && base < prefix + depth) throw rowFailure();
            int mapped = base < prefix ? base : indices.rank() - 1 + base - prefix - depth;
            if (!sameCoordinate(facts.get(2), updates, mapped)
                    || base < prefix && !sameCoordinate(facts.get(1), indices, base)
                    || base >= prefix && index >= 0) throw rowFailure();
            return base;
        }
        int axis = operation.attrs() instanceof IndexAxisAttrs a ? a.axis()
                : ((ScatterElementsAttrs) operation.attrs()).axis();
        if (operation.kind() == AxisScatterKind.SCATTER_ELEMENTS) {
            if (base < 0) {
                int contributor = mergeRow(index, update);
                if (contributor < 0 || !sameCoordinate(facts.get(1), indices, contributor)
                        || !sameCoordinate(facts.get(2), updates, contributor)) throw rowFailure();
            } else if (base == axis || !sameCoordinate(facts.get(1), indices, base)
                    || !sameCoordinate(facts.get(2), updates, base)) throw rowFailure();
            return base;
        }
        if (base < 0) {
            if (update >= 0 && (update < axis || update >= axis + indices.rank())) {
                // The contributor is an untouched data axis; SHARED indices omit that coordinate.
                if (index >= 0) throw rowFailure();
                return -1;
            }
            int contributor = mergeRow(index, update < 0 ? -1 : update - axis);
            if (contributor < 0 || contributor >= indices.rank()
                    || !sameCoordinate(facts.get(1), indices, contributor)
                    || !sameCoordinate(facts.get(2), updates, axis + contributor)) throw rowFailure();
            return -1;
        }
        if (base == axis || index >= 0) throw rowFailure();
        int mapped = base < axis ? base : base + indices.rank() - 1;
        if (!sameCoordinate(facts.get(2), updates, mapped)) throw rowFailure();
        return base;
    }

    /**
     * Requires singleton SHARED alignment without inferring provenance from equal extents.
     * @param fact explicit operand coordinate facts
     * @param shape independently checked operand shape
     * @param axis logical input axis accessed at the identified row coordinate
     * @return whether the operand reads exactly that coordinate or a fixed singleton
     */
    private static boolean sameCoordinate(ValueSemantics fact, Shape shape, int axis) {
        return fact.rowAxis().isPresent() ? fact.rowAxis().getAsInt() == axis : singleton(shape, axis);
    }

    /**
     * Retains rows outside feature reductions; only represented SUM may actively eliminate one.
     * @param operation normalized ordinary reduction occurrence
     * @param inputs independently checked data and optional mask descriptors
     * @param facts corresponding proved data/control facts
     * @param out independently checked canonical result shape
     * @return retained/shifted row axis, or -1 for a terminal/shared result
     * @throws IllegalArgumentException if an unqualified aggregate traverses the row
     */
    private static int reductionRow(Operation operation, List<TensorDescriptor> inputs,
            List<ValueSemantics> facts, Shape out) {
        Shape in = inputs.getFirst().shape();
        int row = facts.getFirst().rowAxis().orElse(-1);
        if (inputs.size() > 1) row = pointwise(inputs, facts, in);
        if (row < 0) return -1;
        if (operation.attrs() instanceof SumToShapeAttrs) {
            int removed = in.rank() - out.rank();
            return row < removed || !in.dimension(row).equals(out.dimension(row - removed)) ? -1 : row - removed;
        }
        boolean[] axes = new boolean[in.rank()];
        boolean keep = false;
        switch (operation.attrs()) {
            case NoOperationAttrs ignored -> java.util.Arrays.fill(axes, true);
            case AxisReductionAttrs a -> { axes[a.axis()] = true; keep = a.keepDimensions(); }
            case MultiAxisReductionAttrs a -> {
                for (int axis : a.axes()) axes[axis] = true;
                keep = a.keepDimensions();
            }
            case StatisticalReductionAttrs a -> {
                for (int axis : a.axes()) axes[axis] = true;
                keep = a.keepDimensions();
            }
            case ArgExtremaAttrs a -> { axes[a.axis()] = true; keep = a.keepDimensions(); }
            case MaskedReductionAttrs a -> axes[a.axis()] = true;
            default -> throw rowFailure();
        }
        if (axes[row]) {
            if (operation.kind() != AggregateReductionKind.SUM) throw rowFailure();
            return -1;
        }
        int result = row;
        if (!keep) for (int axis = 0; axis < row; axis++) if (axes[axis]) result--;
        return result;
    }

    /**
     * Traces broadcast prefixes, free M/N and the matched active K contraction.
     * @param inputs independently checked left and right matrix/vector descriptors
     * @param facts corresponding proved operand coordinates
     * @param out independently checked ordinary output shape
     * @return mapped row axis, or -1 for SHARED or matched terminal contraction
     * @throws IllegalArgumentException if free rows conflict or only one K operand identifies B
     */
    private static int matmulRow(List<TensorDescriptor> inputs, List<ValueSemantics> facts, Shape out) {
        Shape left = inputs.get(0).shape(), right = inputs.get(1).shape();
        int l = facts.get(0).rowAxis().orElse(-1), r = facts.get(1).rowAxis().orElse(-1);
        int lk = left.rank() - 1, rk = right.rank() == 1 ? 0 : right.rank() - 2;
        if (l == lk || r == rk) {
            if (l != lk || r != rk) throw rowFailure();
            return -1;
        }
        int prefix = Math.max(Math.max(0, left.rank() - 2), Math.max(0, right.rank() - 2));
        int lm = l < 0 ? -1 : l < left.rank() - 2 ? l + prefix - (left.rank() - 2) : prefix;
        int rm = r < 0 ? -1 : r < right.rank() - 2 ? r + prefix - (right.rank() - 2)
                : prefix + (left.rank() == 1 ? 0 : 1);
        int result = mergeRow(lm, rm);
        if (result >= 0 && result < prefix) {
            for (int input = 0; input < 2; input++) {
                if (facts.get(input).rowAxis().isPresent()) continue;
                Shape shape = inputs.get(input).shape();
                int axis = result - prefix + Math.max(0, shape.rank() - 2);
                if (axis >= 0 && !singleton(shape, axis)) throw rowFailure();
            }
        }
        return result;
    }

    /**
     * Applies channel-indexed affine routes and training's nonchannel contributor traversal.
     * @param operation ordinary inference or five-output training occurrence
     * @param inputs independently checked data, affine and running-state descriptors
     * @param facts corresponding per-value coordinate/terminal facts
     * @param slot complete canonical output slot
     * @return normalized/statistic output row axis, or -1 for SHARED
     * @throws IllegalArgumentException if training traverses ROW or channel alignment is unproved
     */
    private static int batchNormRow(Operation operation, List<TensorDescriptor> inputs,
            List<ValueSemantics> facts, int slot) {
        int channel = operation.attrs() instanceof BatchNormInferenceAttrs a ? a.channelAxis()
                : ((BatchNormTrainingAttrs) operation.attrs()).channelAxis();
        boolean training = operation.kind() == BatchNormKind.BATCH_NORM_TRAINING;
        int result = -1;
        for (int input : directInputs(operation.kind(), inputs.size(), slot)) {
            int row = facts.get(input).rowAxis().orElse(-1);
            if (row < 0) continue;
            if (input == 0) {
                if (training && row != channel) throw rowFailure();
                result = mergeRow(result, slot == 0 ? row : 0);
            } else result = mergeRow(result, slot == 0 ? channel : 0);
        }
        if (result >= 0 && (slot != 0 || result == channel)) {
            for (int input : directInputs(operation.kind(), inputs.size(), slot)) {
                if (facts.get(input).rowAxis().isPresent()) continue;
                if (!singleton(inputs.get(input).shape(), input == 0 ? channel : 0)) throw rowFailure();
            }
        }
        return result;
    }

    /**
     * Separates batch-domain broadcasting from Q/K/mask-only weights value dependence.
     * @param inputs independently checked Q/K/V and optional mask descriptors
     * @param facts corresponding explicit operand coordinates
     * @param outputs complete independently checked ordinary siblings
     * @param slot canonical output or weights slot
     * @return per-output identified row axis, or -1 for SHARED
     * @throws IllegalArgumentException if embedding/key traversal or batch alignment mixes rows
     */
    private static int attentionRow(List<TensorDescriptor> inputs, List<ValueSemantics> facts,
            List<TensorDescriptor> outputs, int slot) {
        int prefix = outputs.getFirst().shape().rank() - 2;
        int result = -1;
        for (int input = 0; input < 3; input++) {
            Shape shape = inputs.get(input).shape();
            int row = facts.get(input).rowAxis().orElse(-1);
            if (row < 0) continue;
            if (row < shape.rank() - 2) result = mergeRow(result, row + prefix - (shape.rank() - 2));
            else if (input == 0 && row == shape.rank() - 2) result = mergeRow(result, prefix);
            else if (input == 2 && row == shape.rank() - 1) {
                if (slot == 0) result = mergeRow(result, prefix + 1);
            } else throw rowFailure();
        }
        if (inputs.size() == 4) {
            int row = facts.get(3).rowAxis().orElse(-1);
            if (row >= 0) {
                int mapped = row + prefix + 2 - inputs.get(3).shape().rank();
                if (mapped == prefix + 1) throw rowFailure();
                result = mergeRow(result, mapped);
            }
        }
        if (result >= 0 && result < prefix) {
            for (int input = 0; input < inputs.size(); input++) {
                if (facts.get(input).rowAxis().isPresent()) continue;
                Shape shape = inputs.get(input).shape();
                int axis = input == 3 ? result - prefix - 2 + shape.rank() : result - prefix + shape.rank() - 2;
                if (axis >= 0 && !singleton(shape, axis)) throw rowFailure();
            }
        }
        if (result == prefix) {
            if (facts.get(0).rowAxis().isEmpty() && !singleton(inputs.get(0).shape(), inputs.get(0).shape().rank() - 2))
                throw rowFailure();
            if (inputs.size() == 4 && facts.get(3).rowAxis().isEmpty()) {
                Shape mask = inputs.get(3).shape();
                if (mask.rank() >= 2 && !singleton(mask, mask.rank() - 2)) throw rowFailure();
            }
        }
        return result;
    }

    /**
     * Retains NONE-loss group coordinates without inventing active loss/divisor semantics.
     * @param inputs independently checked prediction and dense/integral target descriptors
     * @param facts corresponding proved value/control coordinates
     * @param attrs normalized ordinary loss attributes
     * @param out independently checked ordinary output shape
     * @return group-coordinate row axis
     * @throws IllegalArgumentException if class traversal or loss reduction consumes the row
     */
    private static int lossRow(List<TensorDescriptor> inputs, List<ValueSemantics> facts, OperationAttrs attrs, Shape out) {
        LossReduction reduction = switch (attrs) {
            case MeanSquaredErrorAttrs a -> a.reduction();
            case DenseCategoricalCrossEntropyWithLogitsAttrs a -> a.reduction();
            case IndexCategoricalCrossEntropyWithLogitsAttrs a -> a.reduction();
            default -> throw rowFailure();
        };
        if (reduction != LossReduction.NONE) throw rowFailure();
        int axis = attrs instanceof DenseCategoricalCrossEntropyWithLogitsAttrs a ? a.axis()
                : attrs instanceof IndexCategoricalCrossEntropyWithLogitsAttrs a ? a.axis() : -1;
        int left = facts.get(0).rowAxis().orElse(-1), right = facts.get(1).rowAxis().orElse(-1);
        if (left == axis && axis >= 0) throw rowFailure();
        boolean indexed = attrs instanceof IndexCategoricalCrossEntropyWithLogitsAttrs;
        if (!indexed && right == axis && axis >= 0) throw rowFailure();
        int mappedLeft = left < 0 ? -1 : left - (axis >= 0 && axis < left ? 1 : 0);
        int mappedRight = indexed ? right : right < 0 ? -1 : right - (axis >= 0 && axis < right ? 1 : 0);
        int result = mergeRow(mappedLeft, mappedRight);
        int logitsAxis = result + (axis >= 0 && axis <= result ? 1 : 0);
        if (mappedLeft < 0 && !singleton(inputs.get(0).shape(), logitsAxis)) throw rowFailure();
        if (mappedRight < 0 && !singleton(inputs.get(1).shape(), indexed ? result : logitsAxis)) throw rowFailure();
        return result;
    }

    /**
     * Traces convolution's free N/O, grouped C and spatial contributor equations.
     * @param inputs independently checked input, kernel and optional bias descriptors
     * @param facts corresponding explicit row coordinates
     * @param attrs normalized ordinary Conv2d/Conv3d geometry
     * @return the unique output row coordinate
     * @throws IllegalArgumentException if channel contraction or spatial sampling mixes rows
     */
    private static int convolutionRow(List<TensorDescriptor> inputs, List<ValueSemantics> facts, OperationAttrs attrs) {
        Shape data = inputs.get(0).shape(), weight = inputs.get(1).shape();
        int dataRow = facts.get(0).rowAxis().orElse(-1), weightRow = facts.get(1).rowAxis().orElse(-1);
        long groups = attrs instanceof Conv2dAttrs a ? a.groups() : ((Conv3dAttrs) attrs).groups();
        int result = -1;
        if (dataRow == 0) result = 0;
        else if (dataRow == 1) {
            // inputChannel = outputGroup*channelsPerGroup + kernelChannel.
            // It is exactly outputChannel only for one input and one output per group.
            if (!singleton(weight, 1) || !weight.dimension(0).equals(data.dimension(1))
                    || !data.dimension(1).equals(new StaticDimension(groups))) throw rowFailure();
            result = 1;
        } else if (dataRow >= 2) {
            long[][] geometry = spatialGeometry(attrs, weight);
            if (!identityWindow(geometry, dataRow - 2)) throw rowFailure();
            result = dataRow;
        }
        if (weightRow >= 0) {
            if (weightRow != 0) throw rowFailure();
            result = mergeRow(result, 1);
        }
        if (inputs.size() == 3) {
            int bias = facts.get(2).rowAxis().orElse(-1);
            if (bias >= 0) result = mergeRow(result, 1);
            else if (result == 1 && !singleton(inputs.get(2).shape(), 0)) throw rowFailure();
        }
        if (result == 1 && weightRow < 0 && !singleton(weight, 0)) throw rowFailure();
        return result;
    }

    /**
     * Preserves N/C rows and proves exact identity sampling for a spatial row.
     * @param input independently checked ordinary input shape
     * @param row explicit row axis, or -1 for SHARED
     * @param attrs normalized ordinary pooling geometry
     * @return unchanged row axis
     * @throws IllegalArgumentException if a window traverses another original row
     */
    private static int poolRow(Shape input, int row, OperationAttrs attrs) {
        if (row >= 2 && !identityWindow(spatialGeometry(attrs, input), row - 2)) throw rowFailure();
        return row;
    }

    /**
     * Extracts ordinary kernel/stride/padding fields only, never result geometry.
     * @param attrs normalized convolution, pooling or window attributes
     * @param weight checked kernel shape for convolution; otherwise unused input geometry
     * @return kernel, stride and padding arrays in ordinary spatial-axis order
     * @throws IllegalArgumentException if attributes have no supported spatial interpretation
     */
    private static long[][] spatialGeometry(OperationAttrs attrs, Shape weight) {
        return switch (attrs) {
            case Conv2dAttrs a -> new long[][] {
                {weight.dimension(2).staticSize().orElseThrow(), weight.dimension(3).staticSize().orElseThrow()},
                {a.strideHeight(), a.strideWidth()}, {a.paddingHeight(), a.paddingWidth()}};
            case Conv3dAttrs a -> new long[][] {
                {weight.dimension(2).staticSize().orElseThrow(), weight.dimension(3).staticSize().orElseThrow(), weight.dimension(4).staticSize().orElseThrow()},
                {a.strideDepth(), a.strideHeight(), a.strideWidth()}, {a.paddingDepth(), a.paddingHeight(), a.paddingWidth()}};
            case MaxPool2dAttrs a -> new long[][] {{a.kernelHeight(), a.kernelWidth()},
                {a.strideHeight(), a.strideWidth()}, {a.paddingHeight(), a.paddingWidth()}};
            case AveragePool2dAttrs a -> new long[][] {{a.kernelHeight(), a.kernelWidth()},
                {a.strideHeight(), a.strideWidth()}, {a.paddingHeight(), a.paddingWidth()}};
            case MaxPool3dAttrs a -> new long[][] {{a.kernelDepth(), a.kernelHeight(), a.kernelWidth()},
                {a.strideDepth(), a.strideHeight(), a.strideWidth()}, {a.paddingDepth(), a.paddingHeight(), a.paddingWidth()}};
            case AveragePool3dAttrs a -> new long[][] {{a.kernelDepth(), a.kernelHeight(), a.kernelWidth()},
                {a.strideDepth(), a.strideHeight(), a.strideWidth()}, {a.paddingDepth(), a.paddingHeight(), a.paddingWidth()}};
            case Window2dAttrs a -> new long[][] {{a.kernelHeight(), a.kernelWidth()},
                {a.strideHeight(), a.strideWidth()}, {a.paddingHeight(), a.paddingWidth()}};
            case Window3dAttrs a -> new long[][] {{a.kernelDepth(), a.kernelHeight(), a.kernelWidth()},
                {a.strideDepth(), a.strideHeight(), a.strideWidth()}, {a.paddingDepth(), a.paddingHeight(), a.paddingWidth()}};
            default -> throw rowFailure();
        };
    }

    /**
     * Proves the exact equation inputRow=outputRow for every kernel contributor.
     * @param geometry kernel/stride/padding arrays in spatial-axis order
     * @param axis selected spatial axis, excluding N/C
     * @return whether the selected window has one sample, unit stride and no padding
     */
    private static boolean identityWindow(long[][] geometry, int axis) {
        return geometry[0][axis] == 1 && geometry[1][axis] == 1 && geometry[2][axis] == 0;
    }

    /**
     * Maps unflattened window axes and proves fixed grid boundaries before accepting flattening.
     * @param in independently checked ordinary input shape
     * @param out independently checked ordinary output/target shape
     * @param row explicitly identified input row axis
     * @param attrs normalized ordinary axis/2D/3D window attributes
     * @return lowest fully proved output row axis, or -1 for SHARED
     * @throws IllegalArgumentException if flattened/window contributors cannot preserve B
     * @throws ArithmeticException if ordinary checked grid/kernel arithmetic overflows
     */
    private static int windowRow(Shape in, Shape out, int row, OperationAttrs attrs) {
        if (row < 0) return -1;
        if (attrs instanceof UnfoldAxisAttrs a) {
            if (row != a.axis()) return row;
            if (a.size() == 1 && a.step() == 1) return row;
            if (singleton(out, a.axis()) && in.dimension(row).equals(new StaticDimension(a.size())))
                return out.rank() - 1; // One complete window: inputRow = windowComponent.
            throw rowFailure();
        }
        if (attrs instanceof FoldAxisAttrs a) {
            if (row == in.rank() - 1) {
                if (!singleton(in, a.axis()) || !in.dimension(row).equals(out.dimension(a.axis()))) throw rowFailure();
                return a.axis(); // One complete window, with no overlap or row shift.
            }
            if (row == a.axis() && (!singleton(in, in.rank() - 1) || a.step() != 1)) throw rowFailure();
            return row;
        }
        boolean fold = attrs instanceof Fold2dAttrs || attrs instanceof Fold3dAttrs;
        OperationAttrs window = switch (attrs) {
            case Unfold2dAttrs a -> a.window();
            case Fold2dAttrs a -> a.window();
            case Unfold3dAttrs a -> a.window();
            case Fold3dAttrs a -> a.window();
            case Window2dAttrs a -> a;
            case Window3dAttrs a -> a;
            default -> throw rowFailure();
        };
        long[][] geometry = spatialGeometry(window, in);
        if (row == 0) return 0;
        long kernel = 1;
        for (long size : geometry[0]) kernel = Math.multiplyExact(kernel, size);
        if (row == 1) {
            if (kernel == 1) return 1;
            if (!fold) throw rowFailure();
            for (int axis = 2; axis < out.rank(); axis++)
                if (in.dimension(row).equals(out.dimension(axis)) && fullWindowRow(out, window, geometry, axis)) return axis;
            throw rowFailure();
        }
        if (!fold) {
            // Channel-window flattening can carry a complete row-sized kernel component only
            // when every other flattened factor is one and there is one row-axis window.
            if (fullWindowRow(in, window, geometry, row)) return 1;
            if (!identityWindow(geometry, row - 2)) throw rowFailure();
            for (int axis = 2; axis < in.rank(); axis++) {
                if (axis == row) continue;
                if (!windowGrid(in, window, axis).equals(new StaticDimension(1))) throw rowFailure();
            }
            return 2;
        }
        // Each candidate independently proves the complete grid contributor equation. More than
        // one valid witness does not invalidate either; retain the lowest fully proved axis.
        int result = -1;
        for (int axis = 2; axis < out.rank(); axis++) {
            if (!in.dimension(row).equals(out.dimension(axis)) || !identityWindow(geometry, axis - 2)) continue;
            boolean othersSingleton = true;
            for (int other = 2; other < out.rank(); other++)
                if (other != axis && !windowGrid(out, window, other).equals(new StaticDimension(1))) othersSingleton = false;
            if (othersSingleton) {
                if (result < 0) result = axis;
            }
        }
        if (result < 0) throw rowFailure();
        return result;
    }

    /**
     * Proves a row-sized full-window component is the sole varying channel-window factor.
     * The exact contributor equation is inputRow = 0*stride + component*1 - 0 = component.
     * @param shape independently checked NCHW/NCDHW input or fold target shape
     * @param window normalized ordinary window attributes
     * @param geometry extracted ordinary kernel/stride/padding parameters
     * @param row selected logical spatial axis whose exact extent is the identified B
     * @return whether channel-window flattening preserves that complete row component
     * @throws IllegalArgumentException if ordinary window geometry is invalid
     * @throws ArithmeticException if ordinary checked grid arithmetic overflows
     */
    private static boolean fullWindowRow(Shape shape, OperationAttrs window, long[][] geometry, int row) {
        if (!singleton(shape, 1) || !shape.dimension(row).equals(new StaticDimension(geometry[0][row - 2]))
                || geometry[2][row - 2] != 0) return false;
        long dilation = window instanceof Window2dAttrs a
                ? row == 2 ? a.dilationHeight() : a.dilationWidth()
                : window instanceof Window3dAttrs a
                ? row == 2 ? a.dilationDepth() : row == 3 ? a.dilationHeight() : a.dilationWidth()
                : 0;
        if (dilation != 1 || !windowGrid(shape, window, row).equals(new StaticDimension(1))) return false;
        for (int axis = 2; axis < shape.rank(); axis++)
            if (axis != row && geometry[0][axis - 2] != 1) return false;
        return true;
    }

    /**
     * Reuses the ordinary window owner's checked ceil/fit/symbolic geometry.
     * @param shape validated NCHW/NCDHW input or fold target shape
     * @param window normalized ordinary 2D/3D window attributes
     * @param axis selected logical spatial axis
     * @return exact ordinary grid Dimension on that axis
     * @throws IllegalArgumentException if ordinary window fit fails
     * @throws ArithmeticException if ordinary grid arithmetic overflows
     */
    private static Dimension windowGrid(Shape shape, OperationAttrs window, int axis) {
        if (window instanceof Window2dAttrs a) return TensorWindowExpressions.windowOutputDimension(
                shape.dimension(axis), axis == 2 ? a.kernelHeight() : a.kernelWidth(),
                axis == 2 ? a.paddingHeight() : a.paddingWidth(),
                axis == 2 ? a.strideHeight() : a.strideWidth(),
                axis == 2 ? a.dilationHeight() : a.dilationWidth(), a.ceilMode(), "scan window", "feature grid");
        Window3dAttrs a = (Window3dAttrs) window;
        return TensorWindowExpressions.windowOutputDimension(shape.dimension(axis),
                axis == 2 ? a.kernelDepth() : axis == 3 ? a.kernelHeight() : a.kernelWidth(),
                axis == 2 ? a.paddingDepth() : axis == 3 ? a.paddingHeight() : a.paddingWidth(),
                axis == 2 ? a.strideDepth() : axis == 3 ? a.strideHeight() : a.strideWidth(),
                axis == 2 ? a.dilationDepth() : axis == 3 ? a.dilationHeight() : a.dilationWidth(),
                a.ceilMode(), "scan window", "feature grid");
    }

    /**
     * Checks every declared or propagated axis without using equal sizes to infer its role.
     * Only the identified row axis may be named. Fixed-feature geometry is checked even for an
     * unused declaration or canonical sibling and independently of zero time/batch extents.
     * @param shape static or one-named-B local shape
     * @param role SHARED (-1) or explicit row axis
     * @param batch common static or normalized named batch Dimension, never a sampled extent
     * @throws IllegalArgumentException if a row axis/rank/Dimension is invalid or any other axis is not static
     * @throws ArithmeticException if the nonzero fixed-feature product overflows
     */
    private static void checkRow(Shape shape, int role, Dimension batch) {
        if (role < -1 || role >= shape.rank()
                || role >= 0 && !shape.dimension(role).equals(batch)) throw rowFailure();
        for (int axis = 0; axis < shape.rank(); axis++) {
            if (axis != role && !(shape.dimension(axis) instanceof StaticDimension)) throw rowFailure();
        }
        fixedFeatures(shape, role);
    }

    /**
     * Counts fixed features independently of a static or named row Dimension. A legitimate zero
     * feature domain stays empty even when earlier nonzero axes would overflow their product.
     * @param shape admitted shape whose non-row axes have already been proved static
     * @param row explicit local row axis to omit, or -1 for an entirely SHARED shape
     * @return checked fixed product, one for no feature axes or zero for an empty feature domain
     * @throws ArithmeticException if the nonzero fixed-feature product overflows
     */
    private static long fixedFeatures(Shape shape, int row) {
        for (int i = 0; i < shape.rank(); i++) {
            if (i != row && shape.dimension(i) instanceof StaticDimension fixed && fixed.size() == 0) return 0;
        }
        long count = 1;
        for (int i = 0; i < shape.rank(); i++) {
            if (i != row) count = Math.multiplyExact(count, shape.dimension(i).staticSize().orElseThrow());
        }
        return count;
    }

    /** @return a fresh diagnostic for an unsupported or unproved length-mode source */
    private static IllegalArgumentException rowFailure() {
        return new IllegalArgumentException("invalid or unproved length-aware scan row semantics");
    }

    private static List<TensorDescriptor> descriptors(List<Tensor> inputs, ScanBody body, long time) {
        boolean grad = inputs.stream().anyMatch(t -> t.descriptor().requiresGrad());
        List<TensorDescriptor> result = new ArrayList<>();
        for (Tensor value : body.nextCarry()) {
            result.add(portDescriptor(value, grad, time, false));
        }
        for (Tensor value : body.emissions()) {
            result.add(portDescriptor(value, grad, time, true));
        }
        for (Tensor value : body.retainedValues()) {
            result.add(portDescriptor(value, grad, time, true));
        }
        return List.copyOf(result);
    }

    /**
     * Describes a result role without allocating storage or resolving a symbolic batch extent.
     * @param value body value supplying the exact named or static shape and dtype
     * @param grad conservative actual-input gradient OR
     * @param time common static time length
     * @param stacked whether to prepend time for an emission or residual
     * @return fresh descriptor with canonical static or absent symbolic layout and dtype-filtered eligibility
     * @throws ArithmeticException if static canonical layout arithmetic overflows
     */
    private static TensorDescriptor portDescriptor(Tensor value, boolean grad, long time, boolean stacked) {
        TensorDescriptor d = value.descriptor();
        Shape shape = d.shape();
        if (stacked) {
            List<Dimension> dims = new ArrayList<>();
            dims.add(new StaticDimension(time));
            dims.addAll(shape.dimensions());
            shape = Shape.ofDimensions(dims.toArray(Dimension[]::new));
        }
        return new TensorDescriptor(d.dataType(), shape, shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),
                grad && d.dataType().isDifferentiable());
    }
}
