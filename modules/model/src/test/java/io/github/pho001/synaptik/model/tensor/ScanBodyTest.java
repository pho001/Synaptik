package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class ScanBodyTest {
    static Tensor leaf(boolean grad, long... shape) {
        return leaf(DataType.FLOAT32, grad, shape);
    }

    static Tensor leaf(DataType type, boolean grad, long... dims) {
        Shape shape = Shape.of(dims);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    static ScanBody body(Tensor h, Tensor x, List<Tensor> invariants, Tensor next, Tensor... emit) {
        return new ScanBody(List.of(h), List.of(x), invariants, List.of(next), List.of(emit));
    }

    static long nextId() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
    }

    @Test
    void snapshotsListsAndKeepsExactRootsAndUnusedFormals() {
        Tensor h = leaf(false, 2), x = leaf(false, 2), unused = leaf(false);
        Tensor next = h.add(x);
        List<Tensor> carry = new ArrayList<>(List.of(h));
        ScanBody b = new ScanBody(carry, List.of(x), List.of(unused), List.of(next), List.of(h, next, next));
        carry.clear();
        assertSame(h, b.carry().getFirst());
        assertSame(next, b.nextCarry().getFirst());
        assertEquals(List.of(h, next), b.retainedValues());
        assertEquals(List.of(h, next, next), b.emissions());
        assertThrows(UnsupportedOperationException.class, () -> b.retainedValues().clear());
        assertThrows(UnsupportedOperationException.class, () -> b.carry().clear());
    }

    @Test
    void inventoriesPostorderAndUnusedCanonicalSiblingsExactlyOnce() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        Tensor sum = h.add(x);
        TopKResult top = sum.topK(2, 0, true, true);
        ScanBody b = body(h, x, List.of(), top.values(), top.values(), sum);
        assertEquals(List.of(h, sum, top.values(), top.indices()), b.retainedValues());
    }

    @Test
    void includesUnreferencedPreviousCarryButExcludesStepAndInvariantLeaves() {
        Tensor h = leaf(false, 2), x = leaf(false, 2), p = leaf(false, 2);
        ScanBody b = body(h, x, List.of(p), x, p, x);
        assertEquals(List.of(h), b.retainedValues());
    }

    @Test
    void validatesWithoutAllocatingAnyTensorIdentity() throws Exception {
        Tensor h = leaf(true, 2), x = leaf(false, 2);
        TopKResult top = h.add(x).topK(2, 0, true, false);
        long before = nextId();
        ScanBody b = body(h, x, List.of(), top.values(), top.indices());
        assertEquals(before, nextId());
        assertEquals(4, b.retainedValues().size());
    }

    @Test
    void rejectsInvalidUnusedSiblingWithoutConsumingIdsEvenWhenAllocatorExhausted() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        TensorDescriptor values = new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
        TensorProducer malformed = new TensorProducer(new Operation(TopKKind.TOP_K,
                new TopKAttrs(0, 2, true, true)), List.of(h), List.of(values, values));
        TopKResult valid = h.topK(2, 0, true, true);
        long before = nextId();
        assertThrows(IllegalArgumentException.class,
                () -> body(h, x, List.of(), malformed.output(0)));
        assertEquals(before, nextId());
        Field exhausted = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        Field next = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        exhausted.setAccessible(true);
        next.setAccessible(true);
        var claimed = (java.util.concurrent.atomic.AtomicBoolean) exhausted.get(null);
        var counter = (AtomicLong) next.get(null);
        boolean oldClaimed = claimed.get();
        try {
            counter.set(Long.MAX_VALUE);
            claimed.set(true);
            assertDoesNotThrow(() -> body(h, x, List.of(), valid.values()));
            assertThrows(IllegalArgumentException.class,
                    () -> body(h, x, List.of(), malformed.output(0)));
            assertEquals(Long.MAX_VALUE, counter.get());
            assertTrue(claimed.get());
        } finally {
            counter.set(before);
            claimed.set(oldClaimed);
        }
    }

    @Test
    void rejectsMalformedSingleOutputDescriptorsAndInvalidTypeGuards() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        Tensor wrong = TensorFactory.createDerived(leaf(false, 3).descriptor(), Optional.empty(),
                new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE), List.of(h));
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), h, wrong));
        assertEquals(before, nextId());
        Tensor integral = leaf(DataType.INT32, false, 2);
        Tensor bad = TensorFactory.createDerived(integral.descriptor(), Optional.empty(),
                new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE), List.of(integral));
        assertThrows(IllegalArgumentException.class,
                () -> body(h, x, List.of(integral), h, bad));
    }

    @Test
    void rejectsFreeAndHostBackedLeavesAndDerivedStorage() {
        Tensor h = leaf(false, 2), x = leaf(false, 2), free = leaf(false, 2);
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), h.add(free)));
        Tensor stored = TensorFactory.zeros(Shape.of(2), DataType.FLOAT32, Optional.empty(), false);
        assertThrows(IllegalArgumentException.class, () -> body(stored, x, List.of(), stored));
        Tensor next = h.add(x);
        next.replaceHostStorage(stored.hostStorage().orElseThrow());
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), next));
    }

    @Test
    void rejectsDuplicateNoncanonicalAndDerivedFormalsAndCarryRoleMismatches() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        assertThrows(IllegalArgumentException.class, () -> body(h, h, List.of(), h));
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(x), h));
        assertThrows(IllegalArgumentException.class, () -> body(h.tanh(), x, List.of(), h));
        Tensor unresolved = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.of(2), Optional.empty(), false));
        assertThrows(IllegalArgumentException.class, () -> body(unresolved, x, List.of(), unresolved));
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), h.sum()));
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), h.cast(DataType.FLOAT64)));
        assertThrows(IllegalArgumentException.class,
                () -> new ScanBody(List.of(h), List.of(x), List.of(), List.of(), List.of()));
        for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor lh = leaf(low, false, 0, 2), lx = leaf(low, false, 0, 2);
            Tensor work = lh.cast(DataType.FLOAT32).add(lx.cast(DataType.FLOAT32)).mul(lx.cast(DataType.FLOAT32));
            assertThrows(IllegalArgumentException.class, () -> body(lh, lx, List.of(), work));
            assertDoesNotThrow(() -> body(lh, lx, List.of(), work.cast(low), work));
        }
    }

    @Test
    void perValueTypingDoesNotBypassOrdinaryBoolArithmeticOrIntegralTanhGuards() throws Exception {
        Tensor bool = leaf(DataType.BOOL, false, 0, 2), integral = leaf(DataType.INT32, false, 0, 2);
        Tensor badAdd = TensorFactory.createDerived(bool.descriptor(), Optional.empty(),
                new Operation(io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD,
                        NoOperationAttrs.INSTANCE), List.of(bool, bool));
        Tensor badTanh = TensorFactory.createDerived(integral.descriptor(), Optional.empty(),
                new Operation(UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE), List.of(integral));
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> body(bool, integral, List.of(), badAdd));
        assertThrows(IllegalArgumentException.class, () -> body(integral, bool, List.of(), badTanh));
        assertEquals(before, nextId());
    }

    @Test
    void admitsNamedDynamicFormalsButRejectsNoncanonicalWrapperAliases() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        Tensor dynamic = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.DynamicDimension("n")),
                Optional.empty(), false));
        assertDoesNotThrow(() -> body(h, dynamic, List.of(), h));
        Tensor next = h.tanh();
        Tensor alias = new Tensor(new TensorId(900000), next.descriptor(), Optional.empty(),
                next.provenance(), Optional.empty());
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), alias));
    }

    @Test
    void rejectsUnknownKindsRegardlessOfNameAndStructuralSignature() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        OperationKind unknown = new OperationKind() {
            public String name() { return "TANH"; }
            public List<OperationSignature> signatures() {
                return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
            }
        };
        Tensor bad = TensorFactory.createDerived(h.descriptor(), Optional.empty(),
                new Operation(unknown, NoOperationAttrs.INSTANCE), List.of(h));
        assertThrows(IllegalArgumentException.class, () -> body(h, x, List.of(), bad));
    }
}
