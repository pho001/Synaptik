package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.*;
import java.util.function.Function;

/**
 * First-order joint scan orchestration, not another derivative algebra. Existing rule owners
 * construct one reverse body using exact original-primal to formal bindings. Original identities
 * remain contribution keys, while saved outputs are consumed without activation replay. Graph
 * size is independent of time count; this component neither executes nor allocates history.
 * Bodies retain each ordinary floating/discrete primal descriptor. Cotangent channels are
 * floating-only and retain their original target type; shared gradients use parameter-shaped
 * additive carries and actual named-axis contraction, including B=0/1.
 * Length reverse construction retains the same discrete lengths actual and derives seed/history
 * roles from checked declarations/dataflow. Each incoming SHARED-target term is row-reduced
 * before accumulation; invariant-shaped suffix carries retain distinct old.add(contribution)
 * sites. Row-shaped literal seeds use explicit typed sequences, including disconnected zeros.
 */
final class ScanGradientRules {
    /** Prevents instances of the stateless scan rule orchestrator. */
    private ScanGradientRules() {}

    /**
     * Validates the complete lexical source independently of selected derivative demand.
     * Ordinary Model and Compiler guards check every canonical sibling, including unused work.
     * Grouped 2D/3D convolution uses the ordinary shared formula; nested fixed-cell occurrences
     * remain outside Model's ordinary Scan-body algebra.
     * @param producer exact containing source occurrence
     * @return immutable compile-local source facts; no derivative Tensor is constructed
     * @throws IllegalArgumentException if the source or its lexical inventory is invalid
     */
    static SourcePlan preflightSource(TensorProducer producer) {
        ScanAttrs attrs = (ScanAttrs) producer.operation().attrs();
        ScanBody body = producer.scanBody().orElseThrow();
        if (attrs.lengthAware() && (attrs.rowCarryCount() != attrs.carryCount()
                || attrs.sequenceRowAxes().stream().anyMatch(a -> a.isEmpty() || a.getAsInt() != 0))) {
            throw new IllegalArgumentException("length-aware scan AD requires all original carries/sequences ROW(0)");
        }
        ScanBodyInference.capture(body);
        List<Tensor> formals = ScanBodyInference.formals(body);
        var scope = new ArrayList<>(formals);
        scope.addAll(producer.inputs());
        scope.addAll(body.nextCarry());
        scope.addAll(body.emissions());
        scope.addAll(body.retainedValues());
        DynamicDimension batch = null;
        for (Tensor value : scope) for (Dimension dimension : value.descriptor().shape().dimensions()) {
            if (dimension instanceof DynamicDimension named) {
                if (batch != null && !batch.equals(named)) {
                    throw new IllegalArgumentException("multiple scan batch symbols");
                }
                batch = named;
            }
        }
        Optional<DynamicDimension> namedBatch = Optional.ofNullable(batch);
        Set<TensorProducer> checked = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor value : body.retainedValues()) value.provenance().ifPresent(p -> {
            if (!checked.add(p.producer())) return;
            if (p.producer().operation().kind()
                    instanceof io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind) {
                throw new IllegalArgumentException("nested fixed recurrence is outside the Scan-body algebra");
            }
            List<TensorDescriptor> inputs = p.producer().inputs().stream().map(Tensor::descriptor).toList();
            List<TensorDescriptor> outputs = p.producer().outputDescriptors();
            ScanBodyInference.validateConstruction(p.producer().operation(), inputs, outputs,
                    CapturedGraphInference.checkOrdinary(p.producer().operation(), inputs, outputs), namedBatch);
        });
        var roles = new IdentityHashMap<Tensor, Scan.ValueSemantics>();
        if (attrs.lengthAware()) collectSourceRoles(body, attrs,
                producer.inputs().getLast().descriptor().shape().dimension(0), roles);
        var roots = new ArrayList<>(body.nextCarry());
        roots.addAll(body.emissions());
        var routes = new IdentityHashMap<Tensor, Route[]>();
        for (int i = 0; i < formals.size(); i++) {
            Route[] row = emptyRoutes(formals.size());
            row[i] = Route.identity();
            routes.put(formals.get(i), row);
        }
        Route[][] edges = new Route[roots.size()][formals.size()];
        for (int r = 0; r < roots.size(); r++) {
            Route[] row = sourceRoutes(roots.get(r), formals.size(), routes);
            for (int i = 0; i < row.length; i++) {
                BitSet uses = new BitSet();
                if (row[i].differentiable) uses.set(r * formals.size() + i);
                edges[r][i] = new Route(row[i].syntactic, row[i].differentiable,
                        row[i].failClosed, uses);
            }
        }
        // Histories observe the same finite execution as emissions, but never grant a VJP.
        // Keep derivative roots/edges public-only so their use-bit declaration stays unchanged.
        Route[][] observations = Arrays.copyOf(edges, roots.size() + body.retainedValues().size());
        for (int retained = 0; retained < body.retainedValues().size(); retained++) {
            Route[] row = sourceRoutes(body.retainedValues().get(retained), formals.size(), routes);
            Route[] primal = emptyRoutes(formals.size());
            for (int input = 0; input < row.length; input++) {
                if (row[input].syntactic) primal[input] = Route.nonDifferentiable();
            }
            observations[roots.size() + retained] = primal;
        }
        return new SourcePlan(producer, body, attrs, List.copyOf(formals),
                List.copyOf(roots), edges, roles, finitePrefixes(attrs, observations, formals.size()), namedBatch);
    }

    /**
     * Exact selected source output/input pair, not independent endpoint unions.
     * @param seedOutputPort original canonical public carry or emission cotangent port
     * @param actualInputPosition original positional actual declaration
     */
    record Pair(int seedOutputPort, int actualInputPosition) {}

    /**
     * Ordered selected pairs; tied actual Tensor identities do not collapse declaration slots.
     * @param pairs exact selected positional requests in retained encounter order
     */
    record Demand(List<Pair> pairs) {
        /**
         * Snapshots requests, retaining the first occurrence of each exact positional pair.
         * @param pairs non-null ordered requests with no null pair
         * @throws NullPointerException if the list or a pair is null
         */
        Demand { pairs = List.copyOf(new LinkedHashSet<>(pairs)); }
    }

    /**
     * Whole-source facts retained once per exact containing occurrence.
     * @param producer exact original Scan occurrence
     * @param body its validated closed lexical body
     * @param attrs independently checked containing Scan header
     * @param formals body carry, step-sequence and invariant declarations in that order
     * @param roots next-carry then emission roots, excluding primal-only histories
     * @param edges public-root-by-formal one-step S/D/FC facts and derivative-use bits
     * @param roles proved per-value length-mode coordinate facts, empty in all-valid mode
     * @param finitePrefixes shared finite public-route and primal-history observation timeline
     * @param namedBatch sole declared named dimension of the checked scope, or empty if absent
     */
    record SourcePlan(TensorProducer producer, ScanBody body, ScanAttrs attrs,
            List<Tensor> formals, List<Tensor> roots, Route[][] edges,
            IdentityHashMap<Tensor, Scan.ValueSemantics> roles, FinitePrefixes finitePrefixes,
            Optional<DynamicDimension> namedBatch) {}

    /**
     * Reads a length declaration's outer coordinate, including the stacked time prefix.
     * All-valid ports have no declared row, regardless of coincident extents.
     * @param source validated exact occurrence
     * @param outputPort canonical public or retained-history output position
     * @return declared/proved row axis, or empty for all-valid or SHARED ports
     * @throws IndexOutOfBoundsException if the port is outside the complete output inventory
     */
    static OptionalInt outerRowAxis(SourcePlan source, int outputPort) {
        Objects.checkIndex(outputPort, source.producer.outputCount());
        if (!source.attrs.lengthAware()) return OptionalInt.empty();
        if (outputPort < source.attrs.carryCount()) {
            return outputPort < source.attrs.rowCarryCount() ? OptionalInt.of(0) : OptionalInt.empty();
        }
        if (outputPort < source.attrs.carryCount() + source.attrs.emissionCount()) return OptionalInt.of(1);
        Tensor retained = source.body.retainedValues().get(
                outputPort - source.attrs.carryCount() - source.attrs.emissionCount());
        OptionalInt axis = source.roles.get(retained).rowAxis();
        return axis.isPresent() ? OptionalInt.of(axis.getAsInt() + 1) : OptionalInt.empty();
    }

    /**
     * Reads only explicit length-mode actual roles; lengths and invariants are not numeric rows.
     * @param source validated exact occurrence
     * @param actualInputPosition original positional actual, including the lengths position
     * @return declared outer row axis, including a sequence's leading time dimension
     * @throws IndexOutOfBoundsException if the position is outside the actual inventory
     */
    static OptionalInt actualRowAxis(SourcePlan source, int actualInputPosition) {
        Objects.checkIndex(actualInputPosition, source.producer.inputs().size());
        return declaredActualRowAxis(source.attrs, actualInputPosition);
    }

    /**
     * Interprets an already checked header without deriving roles from descriptor extents.
     * @param attrs exact source length/all-valid declaration
     * @param position checked actual declaration position
     * @return explicit outer ROW axis, or empty for an undeclared numeric row
     */
    private static OptionalInt declaredActualRowAxis(ScanAttrs attrs, int position) {
        if (!attrs.lengthAware()) return OptionalInt.empty();
        if (position < attrs.carryCount()) {
            return position < attrs.rowCarryCount() ? OptionalInt.of(0) : OptionalInt.empty();
        }
        int sequence = position - attrs.carryCount();
        if (sequence >= attrs.sequenceCount()) return OptionalInt.empty();
        OptionalInt axis = attrs.sequenceRowAxes().get(sequence);
        return axis.isPresent() ? OptionalInt.of(axis.getAsInt() + 1) : OptionalInt.empty();
    }

    /**
     * Optionally proves one exact original-row correspondence through ordinary semantic owners.
     * Unknown operands, mixing and terminal taint are correlation misses, not ordinary-math
     * rejections. Complete ordinary descriptor checks remain outside the optional-proof catch.
     * No payload, literal value or length value is inspected and no Tensor identity is allocated.
     * @param origin exact canonical starting Tensor
     * @param originAxis explicitly identified original-row axis on {@code origin}
     * @param value exact canonical destination Tensor
     * @param valueAxis required destination row axis
     * @param batch exact canonical batch Dimension, never a sampled extent
     * @return true only for a proved origin-dependent nonterminal matching row
     * @throws NullPointerException if a Tensor or the batch dimension is null
     * @throws IllegalArgumentException if independent ordinary descriptor validation fails
     */
    static boolean sameRow(Tensor origin, int originAxis, Tensor value, int valueAxis, Dimension batch) {
        Objects.requireNonNull(origin, "origin");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(batch, "batch");
        if (originAxis < 0 || valueAxis < 0) return false;
        Scan.ValueSemantics initial;
        try {
            initial = Scan.declaredSemantics(origin.descriptor(), OptionalInt.of(originAxis), batch);
        } catch (IllegalArgumentException | ArithmeticException miss) {
            return false;
        }
        var proof = new RowCorrelation(origin, batch);
        proof.facts.put(origin, Optional.of(new RowFact(initial, true)));
        Optional<RowFact> result = proof.resolve(value);
        return result.isPresent() && result.get().originDependent
                && !result.get().semantics.terminal()
                && result.get().semantics.rowAxis().orElse(-1) == valueAxis;
    }

    /** Coordinate facts never imply origin dependence merely from equal batch dimensions. */
    private record RowFact(Scan.ValueSemantics semantics, boolean originDependent) {}

    /** Compile-local optional proof; unknown leaves receive no invented ROW or SHARED fact. */
    private static final class RowCorrelation {
        final Tensor origin;
        final Dimension batch;
        final IdentityHashMap<Tensor, Optional<RowFact>> facts = new IdentityHashMap<>();
        final IdentityHashMap<Tensor, Boolean> dependencies = new IdentityHashMap<>();

        /**
         * Starts an isolated optional proof with no inferred leaf bindings.
         * @param origin exact canonical starting Tensor
         * @param batch declared canonical batch dimension
         */
        RowCorrelation(Tensor origin, Dimension batch) { this.origin = origin; this.batch = batch; }

        /**
         * Resolves complete canonical siblings using descriptor-only ordinary semantics.
         * @param value exact destination or operand Tensor
         * @return proved coordinate/dependency facts, or empty when correspondence is unknown
         * @throws IllegalArgumentException if ordinary descriptor validation fails
         */
        Optional<RowFact> resolve(Tensor value) {
            Optional<RowFact> known = facts.get(value);
            if (known != null) return known;
            if (value.provenance().isEmpty()) {
                facts.put(value, Optional.empty());
                return Optional.empty();
            }
            TensorProducer producer = value.provenance().orElseThrow().producer();
            if (producer.scanBody().isPresent()) {
                resolveScan(producer);
            } else {
                List<TensorDescriptor> inputs = producer.inputs().stream().map(Tensor::descriptor).toList();
                List<TensorDescriptor> outputs = producer.outputDescriptors();
                if (!ScanBody.describeOrdinary(producer.operation(), inputs, outputs.size()).equals(outputs)) {
                    throw new IllegalArgumentException("ordinary row-proof descriptors disagree with Model construction");
                }
                CapturedGraphInference.checkOrdinary(producer.operation(), inputs, outputs);
                List<Optional<RowFact>> operands = producer.inputs().stream().map(this::resolve).toList();
                List<Scan.ValueSemantics> results = null;
                if (operands.stream().allMatch(Optional::isPresent)) {
                    try {
                        results = Scan.ordinarySemantics(producer.operation(), inputs,
                                operands.stream().map(f -> f.orElseThrow().semantics).toList(), outputs, batch);
                    } catch (IllegalArgumentException | ArithmeticException miss) {
                        // Missing coordinate proof does not reject independently valid ordinary math.
                    }
                }
                for (int port = 0; port < producer.outputCount(); port++) {
                    Optional<RowFact> result = Optional.empty();
                    if (results != null) {
                        Scan.ValueSemantics semantics = results.get(port);
                        boolean dependent = semantics.directInputs().stream()
                                .anyMatch(i -> operands.get(i).orElseThrow().originDependent);
                        result = Optional.of(new RowFact(semantics, dependent));
                    }
                    facts.putIfAbsent(producer.output(port), result);
                }
            }
            return facts.get(value);
        }

        /**
         * Uses the validated length boundary as a declaration, not a shape-based fallback.
         * Every origin-bearing actual must independently match its declared row. Thus all
         * origin routes through the local body preserve the same canonical row, even when
         * recurrence or terminal contributions make individual prefix routes disappear.
         * @param producer exact body-bearing occurrence whose complete outputs are cached
         * @throws IllegalArgumentException if independent body descriptor validation fails
         */
        private void resolveScan(TensorProducer producer) {
            ScanBody body = producer.scanBody().orElseThrow();
            ScanAttrs attrs = (ScanAttrs) producer.operation().attrs();
            ScanBodyInference.capture(body);
            for (int port = 0; port < producer.outputCount(); port++)
                facts.putIfAbsent(producer.output(port), Optional.empty());
            if (!attrs.lengthAware()
                    || !producer.inputs().getLast().descriptor().shape().dimension(0).equals(batch)
                    || dependsOnOrigin(producer.inputs().getLast())) return;
            var roles = new IdentityHashMap<Tensor, Scan.ValueSemantics>();
            try {
                collectSourceRoles(body, attrs, batch, roles);
            } catch (IllegalArgumentException | ArithmeticException miss) {
                return;
            }
            List<Tensor> formals = ScanBodyInference.formals(body);
            BitSet incoming = new BitSet();
            for (int i = 0; i < formals.size(); i++) {
                Tensor actual = producer.inputs().get(i);
                Optional<RowFact> actualFact = resolve(actual);
                if (!dependsOnOrigin(actual)) continue;
                OptionalInt declared = declaredActualRowAxis(attrs, i);
                if (actualFact.isEmpty() || !actualFact.get().originDependent
                        || actualFact.get().semantics.terminal() || declared.isEmpty()
                        || !declared.equals(actualFact.get().semantics.rowAxis())) return;
                incoming.set(i);
            }
            var routes = new IdentityHashMap<Tensor, Route[]>();
            for (int i = 0; i < formals.size(); i++) {
                Route[] row = emptyRoutes(formals.size());
                row[i] = Route.identity();
                routes.put(formals.get(i), row);
            }
            var roots = new ArrayList<>(body.nextCarry());
            roots.addAll(body.emissions());
            roots.addAll(body.retainedValues());
            Route[][] edges = new Route[roots.size()][];
            for (int i = 0; i < roots.size(); i++)
                edges[i] = coordinateRoutes(roots.get(i), formals.size(), roles, routes);
            FinitePrefixes prefixes = finitePrefixes(attrs, edges, formals.size());
            for (int port = 0; port < roots.size(); port++) {
                Scan.ValueSemantics local = roles.get(roots.get(port));
                OptionalInt axis = local.rowAxis();
                if (port >= attrs.carryCount() && axis.isPresent()) axis = OptionalInt.of(axis.getAsInt() + 1);
                boolean dependent = false;
                for (FinitePrefix prefix : prefixes.states) for (int input = incoming.nextSetBit(0);
                        input >= 0; input = incoming.nextSetBit(input + 1))
                    dependent |= prefix.routes[port][input].syntactic;
                Scan.ValueSemantics outer = new Scan.ValueSemantics(axis, local.terminal(), List.of());
                // Replace only this occurrence's provisional unknown, never the exact origin fact.
                if (producer.output(port) != origin)
                    facts.put(producer.output(port), Optional.of(new RowFact(outer, dependent)));
            }
        }

        /**
         * Conservatively detects possible origin use before trusting a boundary declaration.
         * @param value exact actual Tensor to inspect structurally
         * @return whether any producing-input path reaches the exact origin
         */
        private boolean dependsOnOrigin(Tensor value) {
            if (value == origin) return true;
            Boolean known = dependencies.get(value);
            if (known != null) return known;
            boolean result = value.provenance().isPresent()
                    && value.provenance().orElseThrow().inputs().stream().anyMatch(this::dependsOnOrigin);
            dependencies.put(value, result);
            return result;
        }
    }

    /**
     * Reuses finite route composition with the coordinate owner's canonical value/control routes.
     * Producer work is iterative, preserving operand and sibling order at arbitrary body depth.
     * @param value exact canonical body root or operand
     * @param count complete formal declaration count
     * @param roles complete validated per-canonical-output coordinate facts
     * @param cache caller-owned route cache seeded by exact formal positions
     * @return positional syntactic routes; no derivative eligibility is inferred
     */
    private static Route[] coordinateRoutes(Tensor value, int count,
            IdentityHashMap<Tensor, Scan.ValueSemantics> roles, IdentityHashMap<Tensor, Route[]> cache) {
        Route[] known = cache.get(value);
        if (known != null) return known;
        var pending = new ArrayDeque<RouteVisit>();
        pending.addLast(new RouteVisit(value, false));
        while (!pending.isEmpty()) {
            RouteVisit visit = pending.removeLast();
            if (!visit.complete() && cache.containsKey(visit.value())) continue;
            TensorProducer producer = visit.value().provenance().orElseThrow().producer();
            if (!visit.complete()) {
                pending.addLast(new RouteVisit(visit.value(), true));
                for (int i = producer.inputs().size() - 1; i >= 0; i--) {
                    pending.addLast(new RouteVisit(producer.inputs().get(i), false));
                }
                continue;
            }
            for (int port = 0; port < producer.outputCount(); port++) {
                Route[] row = emptyRoutes(count);
                for (int input : roles.get(producer.output(port)).directInputs()) {
                    Route[] operand = cache.get(producer.inputs().get(input));
                    for (int formal = 0; formal < count; formal++) row[formal] = row[formal].or(operand[formal]);
                }
                cache.put(producer.output(port), row);
            }
        }
        return cache.get(value);
    }

    /**
     * A finite selected route result. Syntactic-only routes are distinct from disconnection.
     * Its pair-local ND-prefix fact is diagnostic only: ordinary admission must first combine
     * every selected route to the same requested target at each common legal prefix.
     * @param syntacticRoute whether an original route exists for a permitted prefix
     * @param differentiableRoute whether a fully differentiable finite route exists
     * @param failClosed whether a required route crosses an unsupported ordinary role
     * @param nonDifferentiablePrefix whether a permitted prefix has only intentional ND routes
     * @param bodyPairs exact one-step root/formal pairs used by differentiable finite routes
     */
    record PairQualification(boolean syntacticRoute, boolean differentiableRoute,
            boolean failClosed, boolean nonDifferentiablePrefix, BitSet bodyPairs) {}

    /**
     * Qualifies the exact finite time count and, in length mode, every legal active prefix.
     * Repeated finite states are compressed; no Tensor graph or host length value is involved.
     * @param source complete validated source
     * @param output canonical public seed port
     * @param input original actual declaration position
     * @return independent syntactic, D and required-FC route facts
     */
    static PairQualification qualify(SourcePlan source, int output, int input) {
        requirePair(source, output, input);
        boolean syntactic = false, d = false, fc = false, nd = false;
        BitSet used = new BitSet();
        List<FinitePrefix> prefixes = source.attrs.lengthAware()
                ? source.finitePrefixes.states : List.of(source.finitePrefixes.at(source.attrs.timeCount()));
        for (FinitePrefix prefix : prefixes) {
            Route result = route(source, prefix, output, input);
            syntactic |= result.syntactic;
            d |= result.differentiable;
            fc |= result.failClosed;
            nd |= result.syntactic && !result.differentiable && !result.failClosed;
            used.or(result.uses);
        }
        return new PairQualification(syntactic, d, fc, nd, used);
    }

    /**
     * Reads one exact prefix relation before reducing different prefixes or selected roots.
     * Discrete endpoints suppress cotangents, never their actual finite syntactic relation.
     * The separate lengths control has no body formal and retains its intentional ND role.
     * @param source complete validated occurrence facts
     * @param prefix one of this source's finite relation states
     * @param output canonical public result declaration
     * @param input original actual declaration, including the separate lengths position
     * @return syntactic, differentiable and required-FC facts at exactly that prefix
     */
    static Route route(SourcePlan source, FinitePrefix prefix, int output, int input) {
        requirePair(source, output, input);
        if (input == source.formals.size()) return Route.nonDifferentiable();
        Route result = prefix.routes[output][input];
        if (!source.producer.inputs().get(input).descriptor().dataType().isFloating()
                || !source.producer.output(output).descriptor().dataType().isFloating()) {
            return new Route(result.syntactic, false, false, new BitSet());
        }
        return result;
    }

    /**
     * Reads only original primal ancestry for a public result or retained-history observation.
     * Histories share the public finite recurrence, including empty prefixes and carry timing.
     * No differentiability, fail-closed requirement or derivative use bits are granted.
     * @param source complete validated occurrence facts
     * @param prefix one of this source's finite relation states
     * @param canonicalOutputPort original public or retained-history canonical port
     * @param actualInputPosition original actual declaration, including separate lengths control
     * @return exactly the syntactic relation at this common prefix, with no VJP facts
     */
    static Route primalRoute(SourcePlan source, FinitePrefix prefix,
            int canonicalOutputPort, int actualInputPosition) {
        if (canonicalOutputPort < 0 || canonicalOutputPort >= source.producer.outputCount()
                || actualInputPosition < 0 || actualInputPosition >= source.producer.inputs().size()) {
            throw new IllegalArgumentException("scan primal port outside declaration inventory");
        }
        boolean syntactic = actualInputPosition == source.formals.size()
                || prefix.routes[canonicalOutputPort][actualInputPosition].syntactic;
        return syntactic ? Route.nonDifferentiable() : Route.empty();
    }

    /**
     * Checks whether an actual syntactic route exists at any permitted source prefix.
     * All-valid sources use exactly T; length-mode sources include every legal active prefix.
     * @param source complete validated occurrence facts
     * @param canonicalOutputPort original public or retained-history canonical port
     * @param actualInputPosition original positional actual, including lengths control
     * @return whether a primal route exists, without granting derivative eligibility
     */
    static boolean hasPrimalRoute(SourcePlan source,
            int canonicalOutputPort, int actualInputPosition) {
        List<FinitePrefix> prefixes = source.attrs.lengthAware()
                ? source.finitePrefixes.states : List.of(source.finitePrefixes.at(source.attrs.timeCount()));
        for (FinitePrefix prefix : prefixes) {
            if (primalRoute(source, prefix, canonicalOutputPort, actualInputPosition).syntactic) return true;
        }
        return false;
    }

    private static void requirePair(SourcePlan source, int output, int input) {
        if (output < 0 || output >= source.roots.size() || input < 0
                || input >= source.producer.inputs().size()) {
            throw new IllegalArgumentException("scan selected port outside declaration inventory");
        }
    }

    /**
     * One exact common-prefix relation; public roots and retained observations share that prefix.
     * @param routes canonical-output-by-formal S/D/FC facts, never mutated after publication
     */
    record FinitePrefix(Route[][] routes) {}

    /**
     * Stores the transient and one complete period of the existing finite recurrence.
     * The retained states are compile-local route facts, not time-expanded derivative work.
     * A negative cycleStart means the states end at the source's exact fixed T.
     * @param states immutable transient and one-cycle states in increasing representative-prefix order
     * @param cycleStart first periodic state, or negative when the list ends at fixed T
     */
    record FinitePrefixes(List<FinitePrefix> states, int cycleStart) {
        /**
         * Snapshots the transient and complete retained period supplied by the finite owner.
         * @param states non-null ordered states with no null entry
         * @param cycleStart first periodic state index, or negative for an exact finite end
         * @throws NullPointerException if the state list or an entry is null
         */
        FinitePrefixes { states = List.copyOf(states); }
        /**
         * Resolves a nonnegative checked prefix without unrolling a repeated period.
         * @param prefix legal prefix in the source's fixed time domain
         * @return retained representative index with exactly the same endpoint facts
         * @throws IllegalArgumentException if a nonperiodic checked domain is exceeded
         */
        int indexAt(long prefix) {
            if (prefix < states.size()) return Math.toIntExact(prefix);
            if (cycleStart < 0) throw new IllegalArgumentException("prefix exceeds the checked finite domain");
            return cycleStart + (int) ((prefix - cycleStart) % (states.size() - cycleStart));
        }
        /**
         * Reads exact endpoint facts for a checked prefix.
         * @param prefix nonnegative legal prefix
         * @return retained common-prefix relation, without copying or constructing derivative work
         */
        FinitePrefix at(long prefix) { return states.get(indexAt(prefix)); }
    }

    /**
     * Computes all endpoint facts once, retaining exact common-prefix correlation.
     * @param attrs validated fixed-time and declaration header
     * @param edges public roots and syntactic-only retained observations by formal position
     * @param formals declaration count, including discrete and unused positions
     * @return transient and period facts bounded by the exact source time domain
     */
    private static FinitePrefixes finitePrefixes(ScanAttrs attrs, Route[][] edges, int formals) {
        int carries = attrs.carryCount(), roots = edges.length;
        Route[][] current = new Route[roots][];
        for (int root = 0; root < roots; root++) {
            current[root] = emptyRoutes(formals);
            if (root < carries) current[root][root] = Route.identity();
        }
        var states = new ArrayList<FinitePrefix>();
        var seen = new HashMap<List<Route>, Integer>();
        for (long prefix = 0; ; prefix++) {
            var key = new ArrayList<Route>();
            for (Route[] row : current) Collections.addAll(key, row);
            Integer repeated = seen.putIfAbsent(List.copyOf(key), states.size());
            if (repeated != null) return new FinitePrefixes(states, repeated);
            states.add(new FinitePrefix(current));
            if (prefix == attrs.timeCount()) return new FinitePrefixes(states, -1);
            Route[][] next = new Route[roots][];
            for (int root = 0; root < roots; root++) {
                next[root] = emptyRoutes(formals);
                for (int input = carries; input < formals; input++) next[root][input] = current[root][input];
                for (int carry = 0; carry < carries; carry++) {
                    for (int input = 0; input < formals; input++) {
                        next[root][input] = next[root][input].or(current[root][carry].then(edges[carry][input]));
                    }
                }
                if (root >= carries) for (int input = 0; input < formals; input++) {
                    next[root][input] = next[root][input].or(edges[root][input]);
                }
            }
            current = next;
        }
    }

    /**
     * Computes ordinary D/ND/FC source paths in iterative producer postorder, without selecting
     * or constructing a derivative. All siblings are cached in canonical output order.
     * @param value exact source root or residual
     * @param count complete formal declaration count
     * @param cache identity-based route cache seeded by the exact formal positions
     * @return complete positional source routes for the requested identity
     */
    private static Route[] sourceRoutes(Tensor value, int count,
            IdentityHashMap<Tensor, Route[]> cache) {
        Route[] known = cache.get(value);
        if (known != null) return known;
        var pending = new ArrayDeque<RouteVisit>();
        pending.addLast(new RouteVisit(value, false));
        while (!pending.isEmpty()) {
            RouteVisit visit = pending.removeLast();
            if (!visit.complete() && cache.containsKey(visit.value())) continue;
            TensorProducer p = visit.value().provenance().orElseThrow(
                    () -> new IllegalArgumentException("undeclared scan lexical leaf")).producer();
            if (!visit.complete()) {
                pending.addLast(new RouteVisit(visit.value(), true));
                for (int i = p.inputs().size() - 1; i >= 0; i--) {
                    pending.addLast(new RouteVisit(p.inputs().get(i), false));
                }
                continue;
            }
            for (int output = 0; output < p.outputCount(); output++) {
                Route[] row = emptyRoutes(count);
                for (int input = 0; input < p.inputs().size(); input++) {
                    var decision = FirstOrderGradientCoverage.classify(p, output, input);
                    Route edge = Route.ordinary(decision);
                    Route[] dependencies = cache.get(p.inputs().get(input));
                    for (int formal = 0; formal < count; formal++) {
                        row[formal] = row[formal].or(edge.then(dependencies[formal]));
                    }
                }
                cache.put(p.output(output), row);
            }
        }
        return cache.get(value);
    }

    /**
     * Source-identity entry/completion work for both ordinary and coordinate route composition.
     * @param value canonical source root or operand
     * @param complete whether ordered producer operands have completed
     */
    private record RouteVisit(Tensor value, boolean complete) {}

    private static Route[] emptyRoutes(int count) {
        Route[] routes = new Route[count];
        Arrays.fill(routes, Route.empty());
        return routes;
    }

    /**
     * Finite S/D/FC path algebra; use bits belong only to fully differentiable paths.
     * @param syntactic whether a value/control dependency exists
     * @param differentiable whether at least one complete ordinary D path exists
     * @param failClosed whether a required path crosses an unsupported ordinary derivative role
     * @param uses one-step derivative-pair use bits, read-only by convention
     */
    record Route(boolean syntactic, boolean differentiable, boolean failClosed, BitSet uses) {
        /**
         * Creates disconnected facts with no derivative uses.
         * @return a fresh empty route
         */
        static Route empty() { return new Route(false, false, false, new BitSet()); }
        /**
         * Creates the differentiable identity path used at a formal or requested target.
         * @return a fresh syntactic D route with no one-step use bits
         */
        static Route identity() { return new Route(true, true, false, new BitSet()); }
        /**
         * Creates an intentional syntactic-only dependency, distinct from disconnection.
         * @return a fresh ND route with no derivative uses
         */
        static Route nonDifferentiable() { return new Route(true, false, false, new BitSet()); }
        /**
         * Adapts the existing ordinary coverage decision without introducing another rule table.
         * @param decision checked ordinary input-role classification
         * @return corresponding S/D/FC facts with no one-step use bits
         * @throws NullPointerException if the decision is null
         */
        static Route ordinary(FirstOrderGradientCoverage.Decision decision) {
            return switch (decision.disposition()) {
                case D -> identity();
                case ND -> nonDifferentiable();
                case FC -> new Route(true, false, true, new BitSet());
            };
        }
        /**
         * Keeps eligibility facts while discarding body-specific derivative-pair provenance.
         * @return a fresh route with the same S/D/FC flags and empty use bits
         */
        Route withoutUses() { return new Route(syntactic, differentiable, failClosed, new BitSet()); }
        /**
         * Unions alternative paths before target-level ordinary policy is applied.
         * @param other alternative route with its own read-only use bits
         * @return fresh unioned flags and derivative-use bits
         */
        Route or(Route other) {
            BitSet union = (BitSet) uses.clone(); union.or(other.uses);
            return new Route(syntactic || other.syntactic, differentiable || other.differentiable,
                    failClosed || other.failClosed, union);
        }
        /**
         * Composes consecutive path segments; ND cuts derivative and unsupported-path traversal.
         * @param other following segment
         * @return fresh composed facts, retaining use bits only for a complete D path
         */
        Route then(Route other) {
            boolean d = differentiable && other.differentiable;
            BitSet union = new BitSet();
            if (d) { union.or(uses); union.or(other.uses); }
            boolean fc = failClosed && (other.differentiable || other.failClosed)
                    || differentiable && other.failClosed;
            return new Route(syntactic && other.syntactic, d, fc, union);
        }
    }

    /**
     * Retains exact finite demand and the checked ordinary construction before materialization.
     * All position lists name original declarations, not deduplicated actual identities.
     * @param source complete validated original occurrence facts
     * @param demand exact selected public-output/original-input pairs
     * @param selectedSeedPorts ordered union of public cotangent ports used by joint dispatch
     * @param bodies checked construction groups with identical selected seed-port sets
     */
    record BackwardPlan(SourcePlan source, Demand demand, List<Integer> selectedSeedPorts,
            List<BodyPlan> bodies) {}

    /**
     * Targets share one contribution map only when their selected seed-port sets are identical.
     * Otherwise ordinary accumulation would silently replace exact pairs by a Cartesian product.
     */
    private record BodyPlan(SourcePlan source, Demand demand, List<Integer> selectedSeedPorts,
            List<Integer> carryPositions, List<Integer> sequenceTargets, List<Integer> invariantTargets,
            List<Integer> localRootPorts, List<Integer> localTargetPositions,
            AutogradPreflight.StagePlan bodyPlan, ConstructionProof proof) {}

    /**
     * Checks only the finite selected routes and runs their actual ordinary construction on metadata.
     * @param source complete lexical source facts
     * @param demand exact selected canonical-output/original-input pairs
     * @return checked compact plan consumed by real joint dispatch
     * @throws IllegalArgumentException if any selected prefix or created operation is unsupported
     */
    static BackwardPlan plan(SourcePlan source, Demand demand) {
        if (demand.pairs.isEmpty()) throw new IllegalArgumentException("selected Scan demand must not be empty");
        var byTarget = new TreeMap<Integer, TreeSet<Integer>>();
        for (Pair pair : demand.pairs) byTarget.computeIfAbsent(
                pair.actualInputPosition, ignored -> new TreeSet<>()).add(pair.seedOutputPort);
        var groups = new LinkedHashMap<List<Integer>, List<Pair>>();
        byTarget.forEach((target, seeds) -> {
            List<Pair> pairs = groups.computeIfAbsent(List.copyOf(seeds), ignored -> new ArrayList<>());
            for (int seed : seeds) pairs.add(new Pair(seed, target));
        });
        var bodies = new ArrayList<BodyPlan>();
        var selected = new TreeSet<Integer>();
        for (var group : groups.entrySet()) {
            selected.addAll(group.getKey());
            bodies.add(planBody(source, new Demand(group.getValue())));
        }
        return new BackwardPlan(source, demand, List.copyOf(selected), List.copyOf(bodies));
    }

    private static BodyPlan planBody(SourcePlan source, Demand demand) {
        BitSet localPairs = new BitSet();
        var seeds = new TreeSet<Integer>();
        for (Pair pair : demand.pairs) {
            PairQualification q = qualify(source, pair.seedOutputPort, pair.actualInputPosition);
            // Ordinary target/prefix policy belongs to AutogradPreflight. A pair-local ND
            // prefix cannot veto another selected D root or tied actual route to that target.
            if (q.failClosed || !q.differentiableRoute) {
                throw new IllegalArgumentException("selected finite Scan route is not constructible: " + pair);
            }
            seeds.add(pair.seedOutputPort);
            localPairs.or(q.bodyPairs);
        }
        int c = source.attrs.carryCount(), s = source.attrs.sequenceCount(), n = source.formals.size();
        var roots = new TreeSet<Integer>();
        var targets = new TreeSet<Integer>();
        var carries = new TreeSet<Integer>();
        var sequences = new TreeSet<Integer>();
        var invariants = new TreeSet<Integer>();
        var pairs = new ArrayList<Pair>();
        for (int bit = localPairs.nextSetBit(0); bit >= 0; bit = localPairs.nextSetBit(bit + 1)) {
            int root = bit / n, target = bit % n;
            roots.add(root); targets.add(target); pairs.add(new Pair(root, target));
            if (root < c) carries.add(root);
            if (target < c) carries.add(target);
        }
        for (Pair pair : demand.pairs) {
            int target = pair.actualInputPosition;
            if (target < c) carries.add(target);
            else if (target < c + s) sequences.add(target);
            else invariants.add(target);
            if (pair.seedOutputPort < c) carries.add(pair.seedOutputPort);
        }
        var bodyPlan = AutogradPreflight.preflightBody(source.body, pairs,
                List.copyOf(roots), List.copyOf(targets));
        var unchecked = new BodyPlan(source, demand, List.copyOf(seeds), List.copyOf(carries),
                List.copyOf(sequences), List.copyOf(invariants), List.copyOf(roots),
                List.copyOf(targets), bodyPlan, null);
        ConstructionProof proof = source.attrs.timeCount() == 0 ? ConstructionProof.empty()
                : new MetadataBody(unchecked).check();
        return new BodyPlan(source, demand, unchecked.selectedSeedPorts, unchecked.carryPositions,
                unchecked.sequenceTargets, unchecked.invariantTargets, unchecked.localRootPorts,
                unchecked.localTargetPositions, bodyPlan, proof);
    }

    /**
     * Materializes the checked selected construction with original-input-aligned nullable results.
     * @param plan successful metadata-first finite plan
     * @param selectedSeeds cotangents in the plan's selected canonical-port order
     * @param constants existing outer registered exact-splat owner
     * @return derivatives only for requested floating declaration positions; lengths remain null
     */
    static Tensor[] apply(BackwardPlan plan, List<Tensor> selectedSeeds,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (selectedSeeds.size() != plan.selectedSeedPorts.size()) {
            throw new IllegalArgumentException("Scan selected seed cardinality mismatch");
        }
        if (plan.source.attrs.timeCount() == 0) {
            Tensor[] result = new Tensor[plan.source.producer.inputs().size()];
            for (Pair pair : plan.demand.pairs) {
                if (pair.seedOutputPort == pair.actualInputPosition
                        && pair.actualInputPosition < plan.source.attrs.carryCount()) {
                    result[pair.actualInputPosition] = selectedSeeds.get(
                            plan.selectedSeedPorts.indexOf(pair.seedOutputPort));
                }
            }
            return result;
        }
        var parts = new ArrayList<RealBody>();
        for (BodyPlan body : plan.bodies) {
            List<Tensor> seeds = body.selectedSeedPorts.stream()
                    .map(port -> selectedSeeds.get(plan.selectedSeedPorts.indexOf(port))).toList();
            RealBody part = new RealBody(body, seeds, constants);
            part.prepare();
            parts.add(part);
        }
        return materialize(plan, parts, constants);
    }

    /**
     * Composes independent selected-seed-set bodies inside one ordinary lexical reverse sibling.
     * Cotangent channels precede temporal accumulators, preserving the length header distinction.
     */
    private static Tensor[] materialize(BackwardPlan plan, List<RealBody> parts,
            FirstOrderAutograd.DerivativeConstants constants) {
        var carry = new ArrayList<Tensor>();
        var next = new ArrayList<Tensor>();
        var initial = new ArrayList<Tensor>();
        var step = new ArrayList<Tensor>();
        var sequences = new ArrayList<Tensor>();
        var axes = new ArrayList<OptionalInt>();
        var invariants = new ArrayList<Tensor>();
        var actualInvariants = new ArrayList<Tensor>();
        var emissions = new ArrayList<Tensor>();
        int channels = 0;
        for (RealBody part : parts) {
            int count = part.plan.carryPositions.size();
            channels = Math.addExact(channels, count);
            carry.addAll(part.carry.subList(0, count));
            next.addAll(part.next.subList(0, count));
            initial.addAll(part.actualsFor(BindingKind.CARRY));
            for (int position : part.usedStepPositions) {
                step.add(part.step.get(position));
                int binding = part.stepBindings.get(position);
                sequences.add(part.actuals.get(binding));
                axes.add(part.bindings.get(binding).rowAxis);
            }
            invariants.addAll(part.invariants);
            for (int binding : part.invariantBindings) actualInvariants.add(part.actuals.get(binding));
            emissions.addAll(part.emissions);
        }
        for (RealBody part : parts) {
            int count = part.plan.carryPositions.size();
            carry.addAll(part.carry.subList(count, part.carry.size()));
            next.addAll(part.next.subList(count, part.next.size()));
            initial.addAll(part.actualsFor(BindingKind.ACCUMULATOR));
        }
        SourcePlan source = plan.source;
        ScanBody body = new ScanBody(carry, step, invariants, next, emissions);
        ScanResult reverse = source.attrs.lengthAware()
                ? Scan.apply(initial, sequences, actualInvariants, source.producer.inputs().getLast(),
                        channels, axes, body, !source.attrs.reverse())
                : Scan.apply(initial, sequences, actualInvariants, body, !source.attrs.reverse());
        TensorProducer producer = (!reverse.finalCarry().isEmpty() ? reverse.finalCarry().getFirst()
                : reverse.outputs().getFirst()).provenance().orElseThrow().producer();
        constants.recordScanActivity(producer, combineActivity(parts, channels));
        Tensor[] result = new Tensor[source.producer.inputs().size()];
        int carryOffset = 0, sequenceOffset = 0, accumulatorOffset = channels;
        int originalCarries = source.attrs.carryCount(), originalSequences = source.attrs.sequenceCount();
        for (RealBody part : parts) {
            for (Pair pair : part.plan.demand.pairs) {
                int target = pair.actualInputPosition;
                if (target < originalCarries) result[target] = reverse.finalCarry().get(
                        carryOffset + part.plan.carryPositions.indexOf(target));
                else if (target < originalCarries + originalSequences) result[target] = reverse.outputs().get(
                        sequenceOffset + part.plan.sequenceTargets.indexOf(target));
                else result[target] = reverse.finalCarry().get(
                        accumulatorOffset + part.plan.invariantTargets.indexOf(target));
            }
            carryOffset += part.plan.carryPositions.size();
            sequenceOffset += part.plan.sequenceTargets.size();
            accumulatorOffset += part.plan.invariantTargets.size();
        }
        return result;
    }

    private static ActivityDraft combineActivity(List<RealBody> parts, int channels) {
        var nodes = new IdentityHashMap<TensorProducer, List<Integer>>();
        var merges = new IdentityHashMap<TensorProducer, ScopeMerge>();
        var carries = new ArrayList<DraftCarry>();
        var seeds = new ArrayList<Tensor>();
        var accumulators = new ArrayList<DraftAccumulator>();
        int carryOffset = 0, seedOffset = 0, accumulatorOffset = channels;
        for (RealBody part : parts) {
            ActivityDraft draft = part.activityDraft();
            int localCarries = draft.carries.size();
            final int carriedBase = carryOffset, seededBase = seedOffset;
            Function<List<Integer>, List<Integer>> rebase = scope -> scope.stream().map(root ->
                    root < localCarries ? carriedBase + root : channels + seededBase + root - localCarries).toList();
            draft.sourceScopes.forEach((producer, scope) -> nodes.put(producer, rebase.apply(scope)));
            draft.sourceMerges.forEach((producer, merge) -> merges.put(producer,
                    new ScopeMerge(rebase.apply(merge.left), rebase.apply(merge.right))));
            for (DraftCarry carried : draft.carries) carries.add(new DraftCarry(
                    carryOffset + carried.position, carried.initialPresence, rebase.apply(carried.nextScope)));
            seeds.addAll(draft.seeds);
            for (DraftAccumulator accumulator : draft.accumulators) accumulators.add(new DraftAccumulator(
                    accumulatorOffset + accumulator.position - localCarries,
                    rebase.apply(accumulator.contributionScope)));
            carryOffset += localCarries;
            seedOffset += draft.seeds.size();
            accumulatorOffset += draft.accumulators.size();
        }
        return new ActivityDraft(nodes, merges, carries, seeds, accumulators);
    }

    /** Explicit binding role used by both checked and real construction. */
    private enum BindingKind { CARRY, ACCUMULATOR, SEQUENCE, SEED, INVARIANT, HISTORY, LITERAL, ROW_LITERAL }

    private record Binding(BindingKind kind, int position, TensorDescriptor descriptor,
            Optional<ScalarValue> literal, OptionalInt rowAxis) {}

    /**
     * Checked numerical descriptor/literal facts, not an executable derivative recipe.
     * Request-local cotangent gradient flags may inherit enclosing primal operands; ordinary
     * formulas never branch on those flags. Every metadata operation still undergoes complete
     * Model and independent Compiler descriptor checking.
     */
    private record OperationProof(Operation operation, List<TensorDescriptor> inputs,
            List<Optional<ScalarValue>> inputLiterals, List<TensorDescriptor> outputs,
            List<Integer> scope) {}

    private record ConstructionProof(List<Binding> bindings, List<OperationProof> operations) {
        ConstructionProof { bindings = List.copyOf(bindings); operations = List.copyOf(operations); }
        static ConstructionProof empty() { return new ConstructionProof(List.of(), List.of()); }
    }

    private static TensorDescriptor numericalDescriptor(TensorDescriptor d) {
        return new TensorDescriptor(d.dataType(), d.shape(), d.layout(), false);
    }

    private static Binding numericalBinding(Binding binding) {
        return new Binding(binding.kind, binding.position, numericalDescriptor(binding.descriptor),
                binding.literal, binding.rowAxis);
    }

    /**
     * One compile-local ordinary construction. Direct operands are never wrapped, and only this
     * controlled route retains scopes. The same sequence of bindings, formulas, normalizations,
     * merges and temporal ADDs is used for metadata proof and real materialization.
     * @param <V> metadata operand or direct Tensor
     */
    private abstract static class BodyConstruction<V>
            implements OrdinaryConstruction.Observer<V>, FirstOrderAutograd.ContributionObserver<V> {
        final BodyPlan plan;
        final SourcePlan source;
        final int c, s;
        final Dimension batch;
        final IdentityHashMap<Tensor, V> sourceValues = new IdentityHashMap<>();
        final IdentityHashMap<V, V> primalBindings = new IdentityHashMap<>();
        final IdentityHashMap<V, Scan.ValueSemantics> roles = new IdentityHashMap<>();
        final IdentityHashMap<V, BitSet> scopes = new IdentityHashMap<>();
        final IdentityHashMap<V, List<V>> operands = new IdentityHashMap<>();
        final IdentityHashMap<V, Optional<ScalarValue>> literals = new IdentityHashMap<>();
        final IdentityHashMap<V, ScopeMerge> merges = new IdentityHashMap<>();
        final List<Binding> bindings = new ArrayList<>();
        final List<OperationProof> operations = new ArrayList<>();
        final List<V> carry = new ArrayList<>(), step = new ArrayList<>(), invariants = new ArrayList<>();
        final List<V> next = new ArrayList<>(), emissions = new ArrayList<>();
        final List<V> accumulatorContributions = new ArrayList<>();
        final List<Integer> stepBindings = new ArrayList<>(), invariantBindings = new ArrayList<>();
        final List<Integer> seedStepPositions = new ArrayList<>();
        final List<Integer> usedStepPositions = new ArrayList<>();
        OrdinaryConstruction<V> construction;
        BitSet owner = new BitSet();

        BodyConstruction(BodyPlan plan) {
            this.plan = plan; source = plan.source;
            c = source.attrs.carryCount(); s = source.attrs.sequenceCount();
            batch = source.attrs.lengthAware()
                    ? source.producer.inputs().getLast().descriptor().shape().dimension(0) : null;
        }

        abstract V sourceValue(Tensor original);
        abstract V newFormal(Binding binding);
        abstract TensorDescriptor descriptor(V value);
        abstract boolean seedRequiresGrad(int originalPort);

        final V role(Tensor original) {
            return sourceValues.computeIfAbsent(original, this::sourceValue);
        }

        final OptionalInt sourceAxis(Tensor original) {
            return batch == null ? OptionalInt.empty() : source.roles.get(original).rowAxis();
        }

        final V bind(BindingKind kind, int position, TensorDescriptor descriptor,
                Optional<ScalarValue> literal, OptionalInt axis) {
            Binding binding = new Binding(kind, position, descriptor, literal, axis);
            V value = newFormal(binding);
            bindings.add(binding);
            literals.put(value, literal);
            scopes.put(value, new BitSet());
            if (batch != null) roles.put(value, Scan.declaredSemantics(descriptor, axis, batch));
            return value;
        }

        final TensorDescriptor formalDescriptor(Tensor original, boolean grad) {
            TensorDescriptor d = original.descriptor();
            return new TensorDescriptor(d.dataType(), d.shape(), d.shape().isFullyStatic()
                    ? Optional.of(LayoutDescriptor.contiguous(d.shape())) : Optional.empty(), grad);
        }

        final V literal(ScalarValue value) {
            var descriptor = new TensorDescriptor(value.dataType(), Shape.scalar(),
                    Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false);
            V formal = bind(BindingKind.LITERAL, -1, descriptor, Optional.of(value), OptionalInt.empty());
            invariants.add(formal); invariantBindings.add(bindings.size() - 1);
            return formal;
        }

        final int rowRole(V value) {
            if (batch == null) return -1;
            Scan.ValueSemantics fact = roles.get(value);
            if (fact != null) return fact.rowAxis().orElse(-1);
            for (var entry : sourceValues.entrySet()) if (entry.getValue() == value) {
                return sourceAxis(entry.getKey()).orElse(-1);
            }
            throw new IllegalStateException("missing exact source or generated coordinate fact");
        }

        final V literalLike(ScalarValue value, V original) {
            int axis = rowRole(original);
            if (axis < 0) return null;
            TensorDescriptor d = descriptor(original);
            var formalDescriptor = new TensorDescriptor(value.dataType(), d.shape(),
                    d.shape().isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(d.shape()))
                            : Optional.empty(), false);
            V formal = bind(BindingKind.ROW_LITERAL, -1, formalDescriptor,
                    Optional.of(value), OptionalInt.of(axis));
            step.add(formal); stepBindings.add(bindings.size() - 1);
            return formal;
        }

        final V addSequence(BindingKind kind, int position, Tensor original,
                boolean grad, OptionalInt axis) {
            V formal = bind(kind, position, formalDescriptor(original, grad), Optional.empty(), axis);
            step.add(formal); stepBindings.add(bindings.size() - 1);
            return formal;
        }

        final void construct() {
            for (int original : plan.carryPositions) {
                V formal = bind(BindingKind.CARRY, original,
                        formalDescriptor(source.body.carry().get(original), seedRequiresGrad(original)),
                        Optional.empty(), batch == null ? OptionalInt.empty() : OptionalInt.of(0));
                BitSet root = new BitSet(); root.set(carry.size()); scopes.put(formal, root);
                carry.add(formal);
            }
            for (int original : plan.invariantTargets) {
                V formal = bind(BindingKind.ACCUMULATOR, original,
                        formalDescriptor(source.formals.get(original), false),
                        Optional.empty(), OptionalInt.empty());
                carry.add(formal);
            }
            // An existing declared sequence binds the fixed time domain even when no primal is
            // consumed. It is not a fabricated carry, history, coordinate or cotangent.
            for (int i = 0; i < s; i++) {
                V formal = addSequence(BindingKind.SEQUENCE, c + i, source.body.step().get(i),
                        source.producer.inputs().get(c + i).descriptor().requiresGrad(),
                        sourceAxis(source.body.step().get(i)));
                primalBindings.put(role(source.body.step().get(i)), formal);
            }
            int rootNumber = plan.carryPositions.size();
            for (int port : plan.selectedSeedPorts) if (port >= c) {
                V formal = addSequence(BindingKind.SEED, port, source.roots.get(port),
                        seedRequiresGrad(port), batch == null ? OptionalInt.empty() : OptionalInt.of(0));
                BitSet scope = new BitSet(); scope.set(rootNumber++); scopes.put(formal, scope);
                seedStepPositions.add(step.size() - 1);
            }
            for (int i = 0; i < source.body.invariants().size(); i++) {
                Tensor original = source.body.invariants().get(i);
                V formal = bind(BindingKind.INVARIANT, c + s + i,
                        formalDescriptor(original, source.producer.inputs().get(c + s + i)
                                .descriptor().requiresGrad()), Optional.empty(), OptionalInt.empty());
                invariants.add(formal); invariantBindings.add(bindings.size() - 1);
                primalBindings.put(role(original), formal);
            }
            for (int i = 0; i < source.body.retainedValues().size(); i++) {
                Tensor original = source.body.retainedValues().get(i);
                int port = c + source.attrs.emissionCount() + i;
                V formal = addSequence(BindingKind.HISTORY, port, original,
                        source.producer.output(port).descriptor().requiresGrad(), sourceAxis(original));
                primalBindings.put(role(original), formal);
            }
            var localSeeds = new ArrayList<V>();
            for (int root : plan.localRootPorts) {
                if (root < c) localSeeds.add(carry.get(plan.carryPositions.indexOf(root)));
                else {
                    int selected = 0;
                    for (int port : plan.selectedSeedPorts) if (port >= c) {
                        if (port == root) break;
                        selected++;
                    }
                    localSeeds.add(step.get(seedStepPositions.get(selected)));
                }
            }
            List<V> gradients = FirstOrderAutograd.expandOrdinaryBody(plan.bodyPlan, localSeeds,
                    this::role, original -> {
                        V bound = primalBindings.get(original);
                        if (bound == null) throw new IllegalStateException("missing exact Scan primal binding");
                        return bound;
                    }, construction, (original, incoming) -> {
                        if (batch == null || sourceAxis(original).isPresent()) return incoming;
                        int row = rowRole(incoming);
                        return row < 0 ? incoming : construction.sum(incoming, row,
                                descriptor(incoming).shape().rank() == original.descriptor().shape().rank());
                    }, this);
            for (int original : plan.carryPositions) {
                int position = plan.localTargetPositions.indexOf(original);
                V value;
                if (position < 0) {
                    entering(null);
                    try { value = construction.zeroLike(role(source.formals.get(original))); }
                    finally { leaving(); }
                } else value = gradients.get(position);
                next.add(value);
            }
            for (int i = 0; i < plan.invariantTargets.size(); i++) {
                int original = plan.invariantTargets.get(i);
                V contribution = gradients.get(plan.localTargetPositions.indexOf(original));
                accumulatorContributions.add(contribution);
                entering(contribution);
                try {
                    // Temporal accumulation is deliberately NOT an optional list merge.
                    next.add(construction.add(carry.get(plan.carryPositions.size() + i), contribution));
                } finally { leaving(); }
            }
            for (int original : plan.sequenceTargets) {
                emissions.add(gradients.get(plan.localTargetPositions.indexOf(original)));
            }
            Set<V> used = Collections.newSetFromMap(new IdentityHashMap<>());
            var pending = new ArrayDeque<V>(); pending.addAll(next); pending.addAll(emissions);
            while (!pending.isEmpty()) {
                V value = pending.removeLast();
                if (used.add(value)) pending.addAll(operands.getOrDefault(value, List.of()));
            }
            for (int i = 0; i < step.size(); i++) {
                if (i == 0 || used.contains(step.get(i)) || seedStepPositions.contains(i)) usedStepPositions.add(i);
            }
            validateBoundary();
        }

        private void validateBoundary() {
            for (int i = 0; i < next.size(); i++) {
                TensorDescriptor before = descriptor(carry.get(i)), after = descriptor(next.get(i));
                if (before.dataType() != after.dataType() || !before.shape().equals(after.shape())) {
                    throw new IllegalArgumentException("generated Scan carry changes its own descriptor");
                }
                if (batch != null && i < plan.carryPositions.size()
                        && (rowRole(next.get(i)) != 0 || roles.get(next.get(i)).terminal())) {
                    throw new IllegalArgumentException("generated Scan cotangent carry is not nonterminal ROW(0)");
                }
                if (batch != null && i >= plan.carryPositions.size() && rowRole(next.get(i)) >= 0) {
                    throw new IllegalArgumentException("generated Scan accumulator is not SHARED");
                }
            }
            for (V emission : emissions) if (batch != null
                    && (rowRole(emission) != 0 || roles.get(emission).terminal())) {
                throw new IllegalArgumentException("generated sequence cotangent is not nonterminal ROW(0)");
            }
        }

        @Override public final void entering(V incoming) {
            owner = incoming == null ? new BitSet() : scope(incoming);
        }
        @Override public final void leaving() { owner = new BitSet(); }
        @Override public final void merging(V left, V right) {
            owner = scope(left); owner.or(scope(right));
        }
        @Override public final void merge(V left, V right, V result) {
            merges.put(result, new ScopeMerge(indices(scope(left)), indices(scope(right))));
        }
        final BitSet scope(V value) {
            BitSet scope = scopes.get(value);
            if (scope == null) throw new IllegalStateException("generated cotangent has no activity owner");
            return (BitSet) scope.clone();
        }
        @Override public final void created(Operation operation, List<V> inputs, List<V> outputs,
                List<CapturedGraphInference.ConstraintRequest> deferred) {
            ScanBodyInference.validateConstruction(operation, inputs.stream().map(this::descriptor).toList(),
                    outputs.stream().map(this::descriptor).toList(), deferred, source.namedBatch);
            if (batch != null) {
                List<Scan.ValueSemantics> facts = Scan.ordinarySemantics(operation,
                        inputs.stream().map(this::descriptor).toList(),
                        inputs.stream().map(roles::get).toList(),
                        outputs.stream().map(this::descriptor).toList(), batch);
                for (int i = 0; i < outputs.size(); i++) roles.put(outputs.get(i), facts.get(i));
            }
            for (V output : outputs) {
                scopes.put(output, (BitSet) owner.clone());
                operands.put(output, List.copyOf(inputs));
                literals.put(output, Optional.empty());
            }
            var proof = new OperationProof(operation,
                    inputs.stream().map(this::descriptor).map(ScanGradientRules::numericalDescriptor).toList(),
                    inputs.stream().map(v -> literals.getOrDefault(v, Optional.empty())).toList(),
                    outputs.stream().map(this::descriptor).map(ScanGradientRules::numericalDescriptor).toList(),
                    indices(owner));
            if (plan.proof != null && (operations.size() >= plan.proof.operations.size()
                    || !proof.equals(plan.proof.operations.get(operations.size())))) {
                throw new IllegalStateException("real Scan operation differs from checked ordinary construction");
            }
            operations.add(proof);
        }
    }

    private record ScopeMerge(List<Integer> left, List<Integer> right) {}

    private static List<Integer> indices(BitSet scope) { return scope.stream().boxed().toList(); }

    /** Descriptor-only execution of the same eager ordinary construction, before derivative IDs. */
    private static final class MetadataBody extends BodyConstruction<OrdinaryConstruction.Value> {
        MetadataBody(BodyPlan plan) {
            super(plan);
            construction = OrdinaryConstruction.metadata(this::literal, this::literalLike, this::rowRole, this);
        }
        @Override OrdinaryConstruction.Value sourceValue(Tensor original) {
            return new OrdinaryConstruction.Value(original.descriptor());
        }
        @Override OrdinaryConstruction.Value newFormal(Binding binding) {
            if (binding.kind == BindingKind.ROW_LITERAL || binding.kind == BindingKind.ACCUMULATOR
                    || binding.kind == BindingKind.CARRY
                            && !plan.selectedSeedPorts.contains(binding.position)) {
                var outer = OrdinaryConstruction.metadata(null, null, null,
                        (operation, inputs, outputs, deferred) -> ScanBodyInference.validateConstruction(
                                operation, inputs.stream().map(OrdinaryConstruction.Value::descriptor).toList(),
                                outputs.stream().map(OrdinaryConstruction.Value::descriptor).toList(),
                                deferred, source.namedBatch));
                ScalarValue value = binding.literal.orElseGet(() ->
                        FirstOrderAutograd.DerivativeConstants.scalar(binding.descriptor.dataType(), false));
                Shape shape = binding.kind == BindingKind.ROW_LITERAL
                        ? stackedShape(binding.descriptor.shape(), source.attrs.timeCount())
                        : binding.descriptor.shape();
                outer.expand(outer.base(value), shape);
            }
            return new OrdinaryConstruction.Value(binding.descriptor, binding.literal);
        }
        @Override TensorDescriptor descriptor(OrdinaryConstruction.Value value) { return value.descriptor(); }
        @Override boolean seedRequiresGrad(int originalPort) { return false; }
        ConstructionProof check() {
            construct();
            return new ConstructionProof(bindings.stream().map(ScanGradientRules::numericalBinding).toList(),
                    operations);
        }
    }

    /** Real materialization consumes every checked binding/operation in exactly the proved order. */
    private static final class RealBody extends BodyConstruction<Tensor> {
        final List<Tensor> selectedSeeds;
        final FirstOrderAutograd.DerivativeConstants outer;
        final List<Tensor> actuals = new ArrayList<>();
        RealBody(BodyPlan plan, List<Tensor> selectedSeeds,
                FirstOrderAutograd.DerivativeConstants outer) {
            super(plan); this.selectedSeeds = selectedSeeds; this.outer = outer;
            var constants = new FirstOrderAutograd.DerivativeConstants(this::literal,
                    original -> literalLike(FirstOrderAutograd.DerivativeConstants.scalar(
                            original.descriptor().dataType(), false), original),
                    this::rowRole, this::literalLike);
            construction = OrdinaryConstruction.real(constants, this);
        }
        @Override Tensor sourceValue(Tensor original) { return original; }
        @Override TensorDescriptor descriptor(Tensor value) { return value.descriptor(); }
        @Override boolean seedRequiresGrad(int port) {
            int selected = plan.selectedSeedPorts.indexOf(port);
            return selected >= 0 && selectedSeeds.get(selected).descriptor().requiresGrad();
        }
        @Override Tensor newFormal(Binding binding) {
            int index = bindings.size();
            if (index >= plan.proof.bindings.size()
                    || !numericalBinding(binding).equals(plan.proof.bindings.get(index))) {
                throw new IllegalStateException("real Scan binding differs from checked construction");
            }
            Tensor actual = switch (binding.kind) {
                case CARRY -> {
                    int seed = plan.selectedSeedPorts.indexOf(binding.position);
                    yield seed >= 0 ? selectedSeeds.get(seed)
                            : outer.zeroLike(source.body.carry().get(binding.position));
                }
                case ACCUMULATOR -> outer.zeroLike(source.formals.get(binding.position));
                case SEQUENCE, INVARIANT -> source.producer.inputs().get(binding.position);
                case SEED -> selectedSeeds.get(plan.selectedSeedPorts.indexOf(binding.position));
                case HISTORY -> source.producer.output(binding.position);
                case LITERAL -> outer.base(binding.literal.orElseThrow());
                case ROW_LITERAL -> outer.base(binding.literal.orElseThrow())
                        .expand(stackedShape(binding.descriptor.shape(), source.attrs.timeCount()));
            };
            actuals.add(actual);
            return TensorFactory.create(binding.descriptor);
        }
        void prepare() {
            construct();
            if (bindings.size() != plan.proof.bindings.size()
                    || operations.size() != plan.proof.operations.size()) {
                throw new IllegalStateException("real Scan did not consume its complete construction proof");
            }
        }
        private List<Tensor> actualsFor(BindingKind kind) {
            var result = new ArrayList<Tensor>();
            for (int i = 0; i < bindings.size(); i++) {
                if (bindings.get(i).kind == kind) result.add(actuals.get(i));
            }
            return result;
        }
        private ActivityDraft activityDraft() {
            var nodeScopes = new IdentityHashMap<TensorProducer, List<Integer>>();
            var nodeMerges = new IdentityHashMap<TensorProducer, ScopeMerge>();
            for (var entry : scopes.entrySet()) entry.getKey().provenance().ifPresent(p -> {
                if (operands.containsKey(entry.getKey())) nodeScopes.put(p.producer(), indices(entry.getValue()));
            });
            for (var entry : merges.entrySet()) {
                nodeMerges.put(entry.getKey().provenance().orElseThrow().producer(), entry.getValue());
            }
            var carries = new ArrayList<DraftCarry>();
            for (int i = 0; i < plan.carryPositions.size(); i++) carries.add(new DraftCarry(i,
                    plan.selectedSeedPorts.contains(plan.carryPositions.get(i)), indices(scope(next.get(i)))));
            var seedFormals = new ArrayList<Tensor>();
            for (int i : seedStepPositions) seedFormals.add(step.get(i));
            var accumulators = new ArrayList<DraftAccumulator>();
            for (int i = 0; i < accumulatorContributions.size(); i++) accumulators.add(new DraftAccumulator(
                    plan.carryPositions.size() + i, indices(scope(accumulatorContributions.get(i)))));
            return new ActivityDraft(nodeScopes, nodeMerges, carries, seedFormals, accumulators);
        }
    }

    private record DraftCarry(int position, boolean initialPresence, List<Integer> nextScope) {}
    private record DraftAccumulator(int position, List<Integer> contributionScope) {}

    /**
     * Ephemeral generated-occurrence draft, finished only at actual local ID assignment.
     * The compiled result contains no Tensor/producer or capture correspondence.
     */
    static final class ActivityDraft implements GraphCapture.AssignmentObserver {
        private final IdentityHashMap<TensorProducer, List<Integer>> sourceScopes;
        private final IdentityHashMap<TensorProducer, ScopeMerge> sourceMerges;
        private final List<DraftCarry> carries;
        private final List<Tensor> seeds;
        private final List<DraftAccumulator> accumulators;
        private final IdentityHashMap<Tensor, ValueId> formalIds = new IdentityHashMap<>();
        private final Map<NodeId, List<Integer>> assignedScopes = new LinkedHashMap<>();
        private final Map<NodeId, ScopeMerge> assignedMerges = new LinkedHashMap<>();

        /**
         * Snapshots generated activity bindings before capture assigns real local IDs.
         * @param sourceScopes exact generated occurrences mapped to proved root memberships
         * @param sourceMerges optional ordinary list-ADD occurrences and operand memberships
         * @param carries ordered compact numeric carried-cotangent declarations
         * @param seeds selected emission-seed formals in activity-root order
         * @param accumulators separate temporal shared-carry contribution declarations
         * @throws NullPointerException if a required container or list entry is null
         */
        ActivityDraft(IdentityHashMap<TensorProducer, List<Integer>> sourceScopes,
                IdentityHashMap<TensorProducer, ScopeMerge> sourceMerges,
                List<DraftCarry> carries, List<Tensor> seeds, List<DraftAccumulator> accumulators) {
            this.sourceScopes = new IdentityHashMap<>(sourceScopes);
            this.sourceMerges = new IdentityHashMap<>(sourceMerges);
            this.carries = List.copyOf(carries);
            this.seeds = List.copyOf(seeds);
            this.accumulators = List.copyOf(accumulators);
        }

        @Override public void formal(Tensor formal, ValueId value) { formalIds.put(formal, value); }
        @Override public void producer(TensorProducer producer, NodeId node, List<ValueId> outputs) {
            List<Integer> scope = sourceScopes.get(producer);
            if (scope == null) throw new IllegalArgumentException("generated Scan node has no construction scope");
            assignedScopes.put(node, scope);
            ScopeMerge merge = sourceMerges.get(producer);
            if (merge != null) assignedMerges.put(node, merge);
        }

        /**
         * Finishes neutral activity facts against the exact graph whose IDs capture assigned.
         * @param graph exact captured local body graph, not a structurally equal replacement
         * @return immutable activity plan associated with that graph reference
         * @throws IllegalArgumentException if captured membership, bindings or protocol are invalid
         */
        ScanActivityPlan finish(CompiledGraphModel graph) {
            var scopeTable = new ArrayList<List<Integer>>();
            var scopeIndices = new LinkedHashMap<List<Integer>, Integer>();
            Function<List<Integer>, Integer> intern = membership -> scopeIndices.computeIfAbsent(
                    List.copyOf(membership), key -> { scopeTable.add(key); return scopeTable.size() - 1; });
            intern.apply(List.of());
            var numeric = new ArrayList<ScanActivityPlan.Carry>();
            for (DraftCarry carry : carries) numeric.add(new ScanActivityPlan.Carry(
                    carry.position, carry.initialPresence, intern.apply(carry.nextScope)));
            var seedPositions = new ArrayList<Integer>();
            for (Tensor seed : seeds) {
                ValueId id = formalIds.get(seed);
                int position = graph.inputs().indexOf(id);
                if (position < 0) throw new IllegalArgumentException("selected seed formal was not captured");
                seedPositions.add(position);
            }
            var nodes = new LinkedHashMap<NodeId, Integer>();
            var merges = new LinkedHashMap<NodeId, ScanActivityPlan.Merge>();
            for (CompiledNode node : graph.nodes()) {
                List<Integer> scope = assignedScopes.get(node.id());
                if (scope == null) throw new IllegalArgumentException("captured activity membership is incomplete");
                nodes.put(node.id(), intern.apply(scope));
                ScopeMerge merge = assignedMerges.get(node.id());
                if (merge != null) merges.put(node.id(), new ScanActivityPlan.Merge(
                        intern.apply(merge.left), intern.apply(merge.right)));
            }
            var temporal = new ArrayList<ScanActivityPlan.Accumulator>();
            for (DraftAccumulator accumulator : accumulators) temporal.add(new ScanActivityPlan.Accumulator(
                    accumulator.position, intern.apply(accumulator.contributionScope)));
            return new ScanActivityPlan(graph, numeric, seedPositions, scopeTable, nodes, merges, temporal);
        }
    }

    private static Shape stackedShape(Shape shape, long time) {
        var dimensions = new ArrayList<Dimension>();
        dimensions.add(new StaticDimension(time)); dimensions.addAll(shape.dimensions());
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Recomputes ephemeral original facts per canonical output after ordinary source validation.
     * @param body independently checked per-value original template
     * @param attrs original length header
     * @param batch exact batch Dimension from the separate lengths descriptor
     * @param facts empty caller-owned destination; contains no IR identities or persisted proof
     */
    private static void collectSourceRoles(ScanBody body, ScanAttrs attrs, Dimension batch,
            IdentityHashMap<Tensor, Scan.ValueSemantics> facts) {
        for (int i = 0; i < body.carry().size(); i++) {
            Tensor formal = body.carry().get(i);
            facts.put(formal, Scan.declaredSemantics(formal.descriptor(),
                    i < attrs.rowCarryCount() ? OptionalInt.of(0) : OptionalInt.empty(), batch));
        }
        for (int i = 0; i < body.step().size(); i++) {
            Tensor formal = body.step().get(i);
            facts.put(formal, Scan.declaredSemantics(formal.descriptor(), attrs.sequenceRowAxes().get(i), batch));
        }
        body.invariants().forEach(t -> facts.put(t, Scan.declaredSemantics(t.descriptor(), OptionalInt.empty(), batch)));
        body.retainedValues().forEach(t -> valueSemantics(t, facts, batch));
    }

    /**
     * Resolves complete canonical occurrences, retaining sibling-specific routes and terminal
     * facts. Ordinary Model constructors already checked the generated descriptor metadata.
     * @param value exact source or generated canonical Tensor
     * @param facts compile-local identity cache seeded by explicit declarations
     * @param batch exact containing batch Dimension, never a sampled extent
     * @return the exact value's recomputed coordinate and terminal facts
     * @throws IllegalArgumentException if contributor mapping cannot preserve the declared row
     * @throws IllegalStateException if an undeclared leaf is encountered
     */
    private static Scan.ValueSemantics valueSemantics(Tensor value,
            IdentityHashMap<Tensor, Scan.ValueSemantics> facts, Dimension batch) {
        Scan.ValueSemantics known = facts.get(value);
        if (known != null) return known;
        TensorProducer p = value.provenance().orElseThrow(
                () -> new IllegalStateException("missing explicit scan row-role binding")).producer();
        List<Scan.ValueSemantics> operands = p.inputs().stream().map(t -> valueSemantics(t, facts, batch)).toList();
        List<Scan.ValueSemantics> results = Scan.ordinarySemantics(p.operation(),
                p.inputs().stream().map(Tensor::descriptor).toList(), operands, p.outputDescriptors(), batch);
        for (int slot = 0; slot < p.outputCount(); slot++) facts.put(p.output(slot), results.get(slot));
        return facts.get(value);
    }

}
