package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.resource.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class RunMemoryGeometryTest {
    @Test void runBoundCreationRollsBackAttemptAllReverseAndRetriesWithFreshActualSizedObjects() {
        var plan=new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0),recipe(new ArrayList<>(),"a",4),4),
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(1),recipe(new ArrayList<>(),"b",8),8)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(0),recipe(new ArrayList<>(),"s",48),8)),true);
        var geometry=RunMemoryGeometry.resolve(plan,OptionalLong.of(40));
        List<String> events=new ArrayList<>();List<BufferRepresentation> acquired=new ArrayList<>();
        var primary=new IllegalStateException("allocation");var cleanupA=new IllegalStateException("closeA");var cleanupB=new IllegalStateException("closeB");
        AtomicInteger mode=new AtomicInteger(1);
        var a=new PreparedRepresentationPlan.CreatedBuffer((bytes,alignment)->{
            assertEquals(160,bytes);assertEquals(4,alignment);
            BufferRepresentation r=()->{events.add("a");if(mode.get()==1)throw cleanupA;};acquired.add(r);return r;
        });
        var b=new PreparedRepresentationPlan.CreatedBuffer((bytes,alignment)->{
            assertEquals(320,bytes);assertEquals(8,alignment);
            if(mode.get()==2)return null;
            if(mode.get()==3)return acquired.getLast();
            BufferRepresentation r=()->{events.add("b");if(mode.get()==1)throw cleanupB;};acquired.add(r);return r;
        });
        var representations=new PreparedRepresentationPlan(plan,List.of(List.of(a),List.of(b)),List.of((bytes,alignment)->{
            assertEquals(1920,bytes);assertEquals(8,alignment);if(mode.get()==1)throw primary;return ()->events.add("s");
        }));
        assertSame(primary,assertThrows(IllegalStateException.class,()->RunStateCreation.create(representations,List.of(),geometry)));
        assertEquals(List.of("b","a"),events);assertArrayEquals(new Throwable[]{cleanupB,cleanupA},primary.getSuppressed());
        mode.set(2);events.clear();assertThrows(NullPointerException.class,()->RunStateCreation.create(representations,List.of(),geometry));assertEquals(List.of("a"),events);
        mode.set(3);events.clear();assertThrows(IllegalArgumentException.class,()->RunStateCreation.create(representations,List.of(),geometry));assertEquals(List.of("a"),events);
        mode.set(0);events.clear();
        try(var first=RunStateCreation.create(representations,List.of(),geometry);var second=RunStateCreation.create(representations,List.of(),geometry)) {
            assertNotSame(first.bufferRepresentation(0,0),second.bufferRepresentation(0,0));
            assertSame(geometry,first.geometry());
        }
        assertEquals(List.of("s","b","a","s","b","a"),events);
    }

    @Test void legacyCompatibilityHooksFailClosedInConstantByteBatchMode() {
        var plan=new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0),0,4)),List.of(),true);
        var geometry=RunMemoryGeometry.resolve(plan,OptionalLong.of(0));
        BufferRepresentation resource=()->{};
        AtomicInteger oldHooks=new AtomicInteger();
        var executable=new io.github.pho001.synaptik.runtime.execution.PreparedExecutable(plan,
                List.of(new io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferSelection(0,0)),List.of()) {
            protected boolean acceptsBufferRepresentation(int i,BufferRepresentation b){oldHooks.incrementAndGet();return true;}
            protected boolean acceptsWorkspaceRepresentation(int i,WorkspaceRepresentation w){return true;}
            protected io.github.pho001.synaptik.runtime.execution.BoundInvocation bindCompatible(RunState s,BufferRepresentation[] b,WorkspaceRepresentation[] w){throw new AssertionError();}
        };
        try(var state=new RunState(plan,List.of(List.of(new BufferRepresentationBinding(resource,RunResourceOwnership.BORROWED))),List.of(),geometry)) {
            assertThrows(IllegalArgumentException.class,()->executable.bind(state));assertEquals(0,oldHooks.get());
        }
    }

    @Test void explicitModeDoesNotDependOnByteIndependenceAndRejectsBeforeCreatorsOrOwnership() {
        var plan = new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 0, 4)), List.of(), true);
        assertThrows(IllegalArgumentException.class, () -> RunMemoryGeometry.resolve(plan, OptionalLong.empty()));
        var geometry = RunMemoryGeometry.resolve(plan, OptionalLong.of(0));
        assertSame(plan, geometry.memoryPlan()); assertEquals(0, geometry.bufferByteSize(0));
        AtomicInteger calls = new AtomicInteger();
        var representations = new PreparedRepresentationPlan(plan,
                List.of(List.of(new PreparedRepresentationPlan.CreatedBuffer((bytes, alignment) -> { calls.incrementAndGet(); return () -> {}; }))), List.of());
        assertThrows(IllegalArgumentException.class, () -> RunStateCreation.create(representations, List.of()));
        assertThrows(IllegalArgumentException.class, () -> new RunState(plan, List.of(), List.of()));
        assertEquals(0, calls.get());
        try (var state = RunStateCreation.create(representations, List.of(), geometry)) { assertSame(geometry, state.geometry()); }
        assertEquals(1, calls.get());
    }

    @Test void resolvesEveryEntryInOrderOnceAndKeepsActualArraysPrivate() {
        List<String> calls = new ArrayList<>();
        PreparedMemoryPlan.ByteSize buffer = recipe(calls, "b", 16);
        PreparedMemoryPlan.ByteSize scratch = recipe(calls, "w", 48);
        var entry = new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), buffer, 4);
        assertThrows(IllegalStateException.class, entry::byteSize);
        assertThrows(IllegalArgumentException.class, () -> new PreparedMemoryPlan(List.of(entry), List.of()));
        var plan = new PreparedMemoryPlan(List.of(entry), List.of(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(0), scratch, 8)), true);
        var a = RunMemoryGeometry.resolve(plan, OptionalLong.of(64));
        var b = RunMemoryGeometry.resolve(plan, OptionalLong.of(40));
        assertEquals(List.of("b64", "w64", "b40", "w40"), calls);
        assertEquals(1024, a.bufferByteSize(0)); assertEquals(640, b.bufferByteSize(0));
        assertEquals(3072, a.workspaceByteSize(0)); assertEquals(4, calls.size());
        assertThrows(IndexOutOfBoundsException.class, () -> a.bufferByteSize(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> a.workspaceByteSize(1));
        assertThrows(IllegalArgumentException.class, () -> RunMemoryGeometry.resolve(plan, OptionalLong.of(-1)));
        assertThrows(ArithmeticException.class, () -> RunMemoryGeometry.resolve(plan, OptionalLong.of(Long.MAX_VALUE)));
    }

    @Test void foreignEqualPlansAndPresentAbsentModeMismatchReject() {
        var a = new PreparedMemoryPlan(List.of(), List.of(), true);
        var b = new PreparedMemoryPlan(List.of(), List.of(), true);
        var g = RunMemoryGeometry.resolve(a, OptionalLong.of(40));
        assertEquals(a, b); assertThrows(IllegalArgumentException.class, () -> new RunState(b, List.of(), List.of(), g));
        try (var state = new RunState(a, List.of(), List.of(), g)) { assertSame(g, state.geometry()); }
        var fixed = new PreparedMemoryPlan.FixedByteSize(0);
        assertEquals(OptionalLong.of(0), fixed.fixedByteSize()); assertEquals(0, fixed.byteSize(64));
        assertThrows(IllegalArgumentException.class, () -> fixed.byteSize(-1));
        assertThrows(IllegalArgumentException.class, () -> RunMemoryGeometry.resolve(new PreparedMemoryPlan(List.of(), List.of()), OptionalLong.of(0)));
    }

    private static PreparedMemoryPlan.ByteSize recipe(List<String> calls, String name, long factor) {
        return new PreparedMemoryPlan.ByteSize() {
            public OptionalLong fixedByteSize() { return OptionalLong.empty(); }
            public long byteSize(long batch) { calls.add(name + batch); return Math.multiplyExact(factor, batch); }
        };
    }
}
