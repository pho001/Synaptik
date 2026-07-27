package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
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
        List<ValueId> bindableSource = new ArrayList<>(List.of(bindable));

        CompileConstantPlan plan =
                new CompileConstantPlan(bindableSource, List.of(source));
        bindableSource.clear();

        assertSame(bindable, plan.bindableInputs().getFirst());
        assertSame(source, plan.constantSources().getFirst());
        assertSame(fixed, plan.constantSources().getFirst().valueId());
        assertSame(scalar, plan.constantSources().getFirst().value());
        assertThrows(
                UnsupportedOperationException.class,
                () -> plan.constantSources().clear());
    }

    @Test
    void validatesListsElementsDuplicatesAndCrossRoleOverlapInOrder() {
        ValueId first = new ValueId(1);
        ValueId second = new ValueId(2);
        assertEquals(
                "bindableInputs",
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
                "bindableInputs[1] duplicates ValueId[value=1]",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileConstantPlan(
                                List.of(first, first),
                                List.of()))
                        .getMessage());
        assertEquals(
                "constantSources[0] overlaps or duplicates ValueId[value=1]",
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new CompileConstantPlan(
                                List.of(first),
                                List.of(new CompileConstantPlan.ConstantSource(
                                        first, ScalarValue.float32(1.0f)))))
                        .getMessage());
        new CompileConstantPlan(
                List.of(first),
                List.of(new CompileConstantPlan.ConstantSource(
                        second, ScalarValue.float32(1.0f))));
    }
}
