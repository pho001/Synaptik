package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/** Behavioral coverage of the shared enclosing contribution algorithm, not shadow formulas. */
final class FirstOrderContributionEngineTest {
    @Test
    void metadataAndRealShareNormalizationCastBackOrderedMergesAndDisconnectedPublication() {
        Tensor x = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor y = tensor(DataType.FLOAT64, Shape.of(2), true);
        Tensor unused = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor output = x.mul(y);
        TensorProducer producer = output.provenance().orElseThrow().producer();
        var occurrence = new AutogradPreflight.SelectedOccurrence(0, producer, 0,
                new boolean[] {true, true}, FirstOrderGradientCoverage.FamilyOwner.ELEMENTWISE);
        BitSet connected = new BitSet();
        connected.set(0, 2);
        var plan = plan(List.of(output, output, output), List.of(x, y, unused),
                List.of(producer), List.of(occurrence), connected);
        List<Tensor> seeds = List.of(tensor(DataType.FLOAT64, Shape.of(2), false),
                tensor(DataType.FLOAT64, Shape.of(2), false),
                tensor(DataType.FLOAT64, Shape.of(2), false));
        Trace<Tensor> realTrace = new Trace<>(Tensor::descriptor);
        realTrace.roots(seeds);
        var real = OrdinaryConstruction.real(new FirstOrderAutograd.DerivativeConstants(), realTrace);
        List<Tensor> realResults = FirstOrderAutograd.expandOrdinaryBody(plan, seeds,
                FirstOrderAutograd.IDENTITY, Function.identity(), real,
                (target, term) -> real.reshape(term, term.descriptor().shape()), realTrace);

        IdentityHashMap<Tensor, OrdinaryConstruction.Value> roles = new IdentityHashMap<>();
        Function<Tensor, OrdinaryConstruction.Value> role = original -> roles.computeIfAbsent(
                original, key -> new OrdinaryConstruction.Value(key.descriptor()));
        List<OrdinaryConstruction.Value> metadataSeeds = seeds.stream().map(role).toList();
        Trace<OrdinaryConstruction.Value> metadataTrace = new Trace<>(OrdinaryConstruction.Value::descriptor);
        metadataTrace.roots(metadataSeeds);
        var metadata = OrdinaryConstruction.metadata(null, null, null, metadataTrace);
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        List<OrdinaryConstruction.Value> metadataResults = FirstOrderAutograd.expandOrdinaryBody(
                plan, metadataSeeds, role, Function.identity(), metadata,
                (target, term) -> metadata.reshape(term, term.descriptor().shape()), metadataTrace);
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
        assertEquals(realTrace.sites, metadataTrace.sites);
        assertEquals(realTrace.merges, metadataTrace.merges);
        assertEquals(List.of(new Merge(Set.of(0), Set.of(1)),
                new Merge(Set.of(0, 1), Set.of(2))), metadataTrace.merges);
        assertEquals(realResults.stream().map(Tensor::descriptor).toList(),
                metadataResults.stream().map(OrdinaryConstruction.Value::descriptor).toList());
        assertEquals(DataType.FLOAT32, metadataResults.get(0).descriptor().dataType());
        assertTrue(metadataTrace.sites.stream().anyMatch(site -> site.operation().kind() == CastKind.CAST
                && site.outputs().getFirst().dataType() == DataType.FLOAT32));
        assertEquals(Set.of(), metadataTrace.scopes.get(metadataResults.get(2)));
        assertNull(realTrace.current);
        assertNull(metadataTrace.current);
    }

    @Test
    void eagerUnusedPrimalOnlyOperationsInheritTheSelectedIncomingScope() {
        Tensor query = tensor(DataType.FLOAT64, Shape.of(3, 4), true);
        Tensor key = tensor(DataType.FLOAT64, Shape.of(5, 4), true);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(5, 6), true);
        var attention = query.scaledDotProductAttentionWithWeights(key, value);
        TensorProducer producer = attention.output().provenance().orElseThrow().producer();
        var occurrence = new AutogradPreflight.SelectedOccurrence(0, producer, 1,
                new boolean[] {true, false, false}, FirstOrderGradientCoverage.FamilyOwner.ATTENTION);
        BitSet connected = new BitSet();
        connected.set(0);
        var plan = plan(List.of(attention.weights()), List.of(query), List.of(producer),
                List.of(occurrence), connected);
        IdentityHashMap<Tensor, OrdinaryConstruction.Value> roles = new IdentityHashMap<>();
        Function<Tensor, OrdinaryConstruction.Value> role = original -> roles.computeIfAbsent(
                original, tensor -> new OrdinaryConstruction.Value(tensor.descriptor()));
        var seed = new OrdinaryConstruction.Value(new TensorDescriptor(
                DataType.FLOAT64, attention.weights().descriptor().shape(), Optional.empty(), false));
        Trace<OrdinaryConstruction.Value> trace = new Trace<>(OrdinaryConstruction.Value::descriptor);
        trace.roots(List.of(seed));
        var metadata = OrdinaryConstruction.metadata(null, null, null, trace);
        FirstOrderAutograd.expandOrdinaryBody(plan, List.of(seed), role, Function.identity(),
                metadata, (target, term) -> term, trace);
        assertTrue(trace.sites.stream().anyMatch(site -> site.operation().kind() == CastKind.CAST
                && site.inputs().getFirst().equals(value.descriptor())
                && site.scope().equals(Set.of(0))));
        assertTrue(trace.sites.stream().allMatch(site -> site.scope().equals(Set.of(0))));
        assertNull(trace.current);
    }

    @Test
    void absentSeedsNeverBecomeOnesOrZerosAndSoleContributionRetainsExactIdentity() {
        Tensor target = tensor(DataType.FLOAT32, Shape.of(2), true);
        BitSet connected = new BitSet();
        connected.set(0);
        var plan = plan(List.of(target, target, target), List.of(target), List.of(), List.of(), connected);
        Tensor seed = tensor(DataType.FLOAT32, Shape.of(2), false);
        var real = OrdinaryConstruction.real(new FirstOrderAutograd.DerivativeConstants(),
                (operation, inputs, outputs, deferred) -> fail("absence and sole identity create no operation"));
        List<Tensor> one = FirstOrderAutograd.expandOrdinaryBody(plan, Arrays.asList(null, seed, null),
                FirstOrderAutograd.IDENTITY, original -> fail("identity flow needs no primal"),
                real, (targetRole, term) -> term, null);
        assertSame(seed, one.getFirst());
        List<Tensor> none = FirstOrderAutograd.expandOrdinaryBody(plan, Arrays.asList(null, null, null),
                FirstOrderAutograd.IDENTITY, original -> fail("absence needs no primal"),
                real, (targetRole, term) -> fail("absent seeds must not normalize"), null);
        assertNull(none.getFirst());
    }

    @Test
    void disconnectedPublicationPreservesScalarBaseAndDescriptorCacheWithoutPrimalReads() {
        Tensor scalar = tensor(DataType.FLOAT64, Shape.scalar(), true);
        Tensor first = tensor(DataType.FLOAT64, Shape.of(2), true);
        Tensor second = tensor(DataType.FLOAT64, Shape.of(2), true);
        var plan = plan(List.of(), List.of(scalar, first, second), List.of(), List.of(), new BitSet());
        var constants = new FirstOrderAutograd.DerivativeConstants();
        List<Operation> realSites = new ArrayList<>();
        var real = OrdinaryConstruction.real(constants,
                (operation, inputs, outputs, deferred) -> realSites.add(operation));
        List<Tensor> realResults = FirstOrderAutograd.expandOrdinaryBody(plan, List.of(),
                FirstOrderAutograd.IDENTITY, original -> fail("a disconnected target is not a primal"),
                real, (target, term) -> term, null);
        assertSame(constants.zeroBase(DataType.FLOAT64), realResults.getFirst());
        assertSame(realResults.get(1), realResults.get(2));
        List<Operation> metadataSites = new ArrayList<>();
        var metadata = OrdinaryConstruction.metadata(null, null, null,
                (operation, inputs, outputs, deferred) -> metadataSites.add(operation));
        IdentityHashMap<Tensor, OrdinaryConstruction.Value> roles = new IdentityHashMap<>();
        List<OrdinaryConstruction.Value> metadataResults = FirstOrderAutograd.expandOrdinaryBody(plan,
                List.of(), original -> roles.computeIfAbsent(original,
                        key -> new OrdinaryConstruction.Value(key.descriptor())),
                original -> fail("a disconnected metadata target is not a primal"),
                metadata, (target, term) -> term, null);
        assertSame(metadata.zeroBase(DataType.FLOAT64), metadataResults.getFirst());
        assertSame(metadataResults.get(1), metadataResults.get(2));
        assertEquals(realSites, metadataSites);
    }

    /**
     * Retains an already-selected local plan without inventing public request eligibility.
     * @param roots original selected roots, with repeated roles allowed
     * @param targets exact ordered floating targets
     * @param producers original producer postorder
     * @param selected checked ordinary occurrences
     * @param connected target-position connectivity
     * @return immutable local stage plan
     */
    private static AutogradPreflight.StagePlan plan(List<Tensor> roots, List<Tensor> targets,
            List<TensorProducer> producers, List<AutogradPreflight.SelectedOccurrence> selected,
            BitSet connected) {
        return new AutogradPreflight.StagePlan(0, roots,
                java.util.Collections.nCopies(roots.size(), Optional.empty()), targets,
                producers, selected, connected, FunctionalGradientRequest.DisconnectedPolicy.ZERO,
                Set.copyOf(producers), Map.of());
    }

    /**
     * Creates source metadata or a Tensor-allocation boundary marker.
     * @param type exact ordinary dtype
     * @param shape exact source shape
     * @param gradient original gradient eligibility
     * @return storage-free ordinary Tensor
     */
    private static Tensor tensor(DataType type, Shape shape, boolean gradient) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), gradient));
    }

    /**
     * Records operation semantics and synchronous owners without retaining generated graph IDs.
     * @param <V> direct Tensor or metadata operand type
     */
    private static final class Trace<V> implements OrdinaryConstruction.Observer<V>,
            FirstOrderAutograd.ContributionObserver<V> {
        private final Function<V, TensorDescriptor> descriptor;
        private final IdentityHashMap<V, Set<Integer>> scopes = new IdentityHashMap<>();
        private final List<Site> sites = new ArrayList<>();
        private final List<Merge> merges = new ArrayList<>();
        private Set<Integer> current;

        /**
         * Binds a descriptor reader, not a primal resolver.
         * @param descriptor direct value descriptor reader
         */
        Trace(Function<V, TensorDescriptor> descriptor) { this.descriptor = descriptor; }

        /**
         * Assigns distinct structural roots even when their source output identity is repeated.
         * @param roots exact ordered seed operands
         */
        void roots(List<V> roots) {
            for (int index = 0; index < roots.size(); index++) scopes.put(roots.get(index), Set.of(index));
        }

        @Override public void entering(V incoming) {
            assertNull(current);
            current = incoming == null ? Set.of() : scopes.get(incoming);
            assertNotNull(current, "incoming cotangent must retain its structural scope");
        }

        @Override public void merging(V left, V right) {
            assertNull(current);
            TreeSet<Integer> union = new TreeSet<>(scopes.get(left));
            union.addAll(scopes.get(right));
            current = Set.copyOf(union);
        }

        @Override public void merge(V left, V right, V result) {
            merges.add(new Merge(scopes.get(left), scopes.get(right)));
            assertEquals(current, scopes.get(result));
        }

        @Override public void leaving() {
            assertNotNull(current);
            current = null;
        }

        @Override public void created(Operation operation, List<V> inputs, List<V> outputs,
                List<CapturedGraphInference.ConstraintRequest> deferred) {
            assertNotNull(current, "every enclosing and eager operation needs an owner");
            sites.add(new Site(operation, inputs.stream().map(descriptor).toList(),
                    outputs.stream().map(descriptor).toList(), current));
            outputs.forEach(output -> scopes.put(output, current));
        }
    }

    /**
     * Exact observable ordinary construction site including its incoming owner scope.
     * @param operation canonical operation and attributes
     * @param inputs positional operand descriptors
     * @param outputs complete canonical sibling descriptors
     * @param scope structural cotangent roots owning the site
     */
    private record Site(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, Set<Integer> scope) {}

    /**
     * Two-sided optional ADD activity, deliberately not just its union.
     * @param left exact accumulated-left roots
     * @param right exact next-contribution roots
     */
    private record Merge(Set<Integer> left, Set<Integer> right) {}
}
