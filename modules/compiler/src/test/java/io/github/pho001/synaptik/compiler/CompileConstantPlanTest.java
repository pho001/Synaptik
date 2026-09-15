package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

final class CompileConstantPlanTest {
    @Test
    void snapshotsMembershipAndRetainsExactIdsSourcesAndScalarValues() {
        ValueId bindable = new ValueId(1);
        ValueId fixed = new ValueId(2);
        ScalarValue scalar = ScalarValue.float32(-0.0f);
        CompileConstantPlan.ConstantSource source =
                new CompileConstantPlan.ConstantSource(fixed, scalar);
        TensorId tensorId = new TensorId(11);
        CompileConstantPlan.BindableInput binding =
                new CompileConstantPlan.BindableInput(tensorId, bindable);
        List<CompileConstantPlan.BindableInput> bindableSource =
                new ArrayList<>(List.of(binding));

        CompileConstantPlan plan =
                new CompileConstantPlan(bindableSource, List.of(source));
        bindableSource.clear();

        assertSame(binding, plan.bindableInputBindings().getFirst());
        assertSame(tensorId, plan.bindableInputBindings().getFirst().tensorId());
        assertSame(bindable, plan.bindableInputs().getFirst());
        assertSame(source, plan.constantSources().getFirst());
        assertSame(fixed, plan.constantSources().getFirst().valueId());
        assertSame(scalar, plan.constantSources().getFirst().value());
        assertThrows(
                UnsupportedOperationException.class,
                () -> plan.constantSources().clear());
        assertThrows(
                UnsupportedOperationException.class,
                () -> plan.bindableInputBindings().clear());
        assertThrows(
                UnsupportedOperationException.class,
                () -> plan.bindableInputs().clear());
    }

    @Test
    void validatesListsElementsDuplicatesAndCrossRoleOverlapInOrder() {
        ValueId first = new ValueId(1);
        ValueId second = new ValueId(2);
        assertEquals(
                "bindableInputBindings",
                assertThrows(
                        NullPointerException.class,
                        () -> new CompileConstantPlan(null, null))
                        .getMessage());
        assertEquals(
                "constantSources",
                assertThrows(
                        NullPointerException.class,
                        () -> new CompileConstantPlan(List.of(), null))
                        .getMessage());
        assertEquals(
                "bindableInputBindings[1] repeats ValueId[value=1]",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileConstantPlan(
                                List.of(binding(1, first), binding(2, first)),
                                List.of()))
                        .getMessage());
        assertEquals(
                "bindableInputBindings[1] repeats TensorId[value=1]",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileConstantPlan(
                                List.of(binding(1, first), binding(1, second)),
                                List.of()))
                        .getMessage());
        assertEquals(
                "constantSources[0] overlaps or duplicates ValueId[value=1]",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileConstantPlan(
                                List.of(binding(1, first)),
                                List.of(new CompileConstantPlan.ConstantSource(
                                        first, ScalarValue.float32(1.0f)))))
                        .getMessage());
        new CompileConstantPlan(
                List.of(binding(1, first)),
                List.of(new CompileConstantPlan.ConstantSource(
                        second, ScalarValue.float32(1.0f))));
        assertEquals("tensorId", assertThrows(NullPointerException.class,
                () -> new CompileConstantPlan.BindableInput(null, first)).getMessage());
        assertEquals("valueId", assertThrows(NullPointerException.class,
                () -> new CompileConstantPlan.BindableInput(new TensorId(1), null)).getMessage());
    }

    private static CompileConstantPlan.BindableInput binding(long tensorId, ValueId valueId) {
        return new CompileConstantPlan.BindableInput(new TensorId(tensorId), valueId);
    }
}
