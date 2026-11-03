package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class CpuConcurrencyBudgetTest {
    /** Real length validation and retry do not depend on a private selected-dispatch layout. */
    @Test void selectedScanInvalidLengthsFailBeforeWritesAndAllowSameInvocationRetry() {
        var type = io.github.pho001.synaptik.model.datatype.DataType.FLOAT32;
        var compiled = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConditionalScanFixture.conditionalFamily(type,true);
        var config = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig(
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,4,4,2);
        var inputs = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs(false,java.util.List.of(),config);
        var analysis = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer().analyze(
                io.github.pho001.synaptik.prepare.GraphPreparation.project(compiled,compiled.partitions().getFirst(),inputs));
        var scans = analysis.plan().units().stream().filter(u -> u.runtimeFacts().generalScan().isPresent()).toList();
        var outputs = scans.stream().flatMap(unit -> unit.boundaryValues().subList(
                unit.boundaryValues().size()-unit.outputCount(),unit.boundaryValues().size()).stream()).distinct().toList();
        var lengths = scans.stream().map(unit -> unit.boundaryValues().get(
                unit.runtimeFacts().generalScan().orElseThrow().ir().lengthRead())).distinct().toList();
        for (long invalid : new long[]{-1,5,Long.MAX_VALUE}) {
            try (var reference = new CpuRecurrentScanExecutionTest.Fixture(compiled,inputs,analysis,1,true,true,java.util.Map.of());
                    var fixture = new CpuRecurrentScanExecutionTest.Fixture(compiled,inputs,analysis,1,true,true,java.util.Map.of())) {
                for (var run : java.util.List.of(reference,fixture))
                    for (var input : compiled.constants().bindableInputBindings()) {
                        var storage = run.storage.get(input.valueId());
                        var inputType = compiled.graph().values().stream().filter(v -> v.id().equals(input.valueId()))
                                .findFirst().orElseThrow().descriptor().dataType();
                        if (inputType == io.github.pho001.synaptik.model.datatype.DataType.INT64)
                            for (long cell = 0; cell < storage.byteSize()/8; cell++)
                                storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,cell,4);
                        else for (long cell = 0; cell < storage.byteSize()/4; cell++)
                            storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT,cell,.125f);
                    }
                reference.executable.bind(reference.state).execute();
                var expected = outputs.stream().map(id -> reference.storage.get(id)
                        .toArray(java.lang.foreign.ValueLayout.JAVA_BYTE)).toList();
                var bound = fixture.executable.bind(fixture.state);
                bound.execute();
                for (int port = 0; port < outputs.size(); port++) assertArrayEquals(expected.get(port),
                        fixture.storage.get(outputs.get(port)).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                var producedValues = compiled.graph().nodes().stream()
                        .flatMap(node -> node.outputs().stream()).collect(java.util.stream.Collectors.toSet());
                for (var output : outputs) {
                    if (producedValues.contains(output)) fixture.storage.get(output).fill((byte) 0xa5);
                }
                var canaries = outputs.stream().map(id -> fixture.storage.get(id)
                        .toArray(java.lang.foreign.ValueLayout.JAVA_BYTE)).toList();
                for (var id : lengths) {
                    var storage = fixture.storage.get(id);
                    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,storage.byteSize()/8-1,invalid);
                }
                assertThrows(IllegalArgumentException.class,bound::execute);
                for (int port = 0; port < outputs.size(); port++) assertArrayEquals(canaries.get(port),
                        fixture.storage.get(outputs.get(port)).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE),
                        "complete Scan validation precedes every current-unit write");
                for (var id : lengths) {
                    var storage = fixture.storage.get(id);
                    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_LONG,storage.byteSize()/8-1,4);
                }
                bound.execute();
                for (int port = 0; port < outputs.size(); port++) assertArrayEquals(expected.get(port),
                        fixture.storage.get(outputs.get(port)).toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
            }
        }
    }
    @Test void reservationCoverageChecksIdentityLivenessAndExactDemandOnly() throws Exception {
        var budget = new CpuConcurrencyBudget(3);
        try (var lease = budget.acquire(2)) {
            assertTrue(lease.covers(budget,1));
            assertTrue(lease.covers(budget,2));
            assertFalse(lease.covers(budget,0));
            assertFalse(lease.covers(budget,3));
            assertFalse(lease.covers(null,1));
            assertFalse(lease.covers(new CpuConcurrencyBudget(3),1));
            // The predicate is not a thread-affinity or additional lease-owner protocol.
            var result = new AtomicReference<Boolean>();
            var thread = Thread.ofPlatform().start(() -> result.set(lease.covers(budget,2)));
            thread.join(5_000);
            assertEquals(Boolean.TRUE,result.get());
        }
        var released = budget.acquire(2); released.close();
        assertFalse(released.covers(budget,1));
    }

    @Test void borrowedDispatchDoesNotAcquireOrReleaseOuterReservation() {
        var budget = new CpuConcurrencyBudget(2);
        try (var group = new CpuWorkerGroup(2,budget); var lease = budget.acquire(2)) {
            var count = new java.util.concurrent.atomic.AtomicInteger();
            group.execute(new CpuWorkerGroup.RangeCall[]{count::incrementAndGet,count::incrementAndGet},lease);
            assertEquals(2,count.get());
            assertTrue(lease.covers(budget,2));
            var failure = new IllegalStateException("borrowed worker failed");
            assertSame(failure,assertThrows(IllegalStateException.class,() -> group.execute(
                    new CpuWorkerGroup.RangeCall[]{() -> { throw failure; },() -> {}},lease)));
            assertTrue(lease.covers(budget,2));
        }
        assertTimeoutPreemptively(java.time.Duration.ofSeconds(5),() -> budget.acquire(2).close());
    }

    @Test void borrowedDispatchRejectsInvalidReservationsBeforeAnyCall() {
        var budget = new CpuConcurrencyBudget(2);
        var count = new java.util.concurrent.atomic.AtomicInteger();
        var calls = new CpuWorkerGroup.RangeCall[]{count::incrementAndGet,count::incrementAndGet};
        try (var group = new CpuWorkerGroup(2,budget); var shortLease = budget.acquire(1);
                var foreign = new CpuConcurrencyBudget(2).acquire(2)) {
            assertThrows(IllegalArgumentException.class,() -> group.execute(calls,null));
            assertThrows(IllegalArgumentException.class,() -> group.execute(calls,shortLease));
            assertThrows(IllegalArgumentException.class,() -> group.execute(calls,foreign));
        }
        var released = budget.acquire(2); released.close();
        try (var group = new CpuWorkerGroup(2,budget)) {
            assertThrows(IllegalArgumentException.class,() -> group.execute(calls,released));
        }
        assertEquals(0,count.get());
    }

    @Test void borrowedInterruptionJoinsBeforeCallerCanReleaseAndRetry() throws Exception {
        var budget = new CpuConcurrencyBudget(2);
        var started = new CountDownLatch(2); var finish = new CountDownLatch(1);
        var observed = new AtomicReference<Throwable>(); var interrupted = new AtomicReference<Boolean>();
        var callerFailure = new AtomicReference<Throwable>();
        var coveredAfterJoin = new AtomicReference<Boolean>();
        try (var group = new CpuWorkerGroup(2,budget)) {
            var caller = Thread.ofPlatform().start(() -> {
                try (var lease = budget.acquire(2)) {
                    try { group.execute(new CpuWorkerGroup.RangeCall[]{
                            () -> { started.countDown(); finish.await(); },
                            () -> { started.countDown(); finish.await(); }},lease); }
                    catch (Throwable failure) { observed.set(failure); }
                    interrupted.set(Thread.currentThread().isInterrupted());
                    coveredAfterJoin.set(lease.covers(budget,2));
                } catch (Throwable failure) { callerFailure.set(failure); }
            });
            try {
                assertTrue(started.await(5,TimeUnit.SECONDS));
                caller.interrupt();
                caller.join(100); assertTrue(caller.isAlive(),"started workers must quiesce first");
            } finally {
                finish.countDown();
                caller.join(5_000);
            }
            assertFalse(caller.isAlive());
            assertNull(callerFailure.get(),"unexpected child failure must reach the test thread");
            assertInstanceOf(CpuWorkerGroup.CpuParallelExecutionException.class,observed.get());
            assertEquals(Boolean.TRUE,interrupted.get());
            assertEquals(Boolean.TRUE,coveredAfterJoin.get());
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(5),() -> {
                try (var lease = budget.acquire(2)) {
                    group.execute(new CpuWorkerGroup.RangeCall[]{() -> {},() -> {}},lease);
                }
            });
        }
    }

    @Test void borrowedCloseJoinsStartedRangesBeforeOuterRelease() throws Exception {
        var budget = new CpuConcurrencyBudget(2);
        var group = new CpuWorkerGroup(2,budget);
        var started = new CountDownLatch(2); var finish = new CountDownLatch(1);
        var failure = new AtomicReference<Throwable>(); var callerFailure = new AtomicReference<Throwable>();
        var stillCovered = new AtomicReference<Boolean>();
        var calls = new CpuWorkerGroup.RangeCall[]{
                () -> { started.countDown(); finish.await(); },
                () -> { started.countDown(); finish.await(); },
                () -> { throw new AssertionError("unclaimed range must be cancelled"); }};
        var caller = Thread.ofPlatform().start(() -> {
            try (var lease = budget.acquire(2)) {
                try { group.execute(calls,lease); }
                catch (Throwable actual) { failure.set(actual); }
                stillCovered.set(lease.covers(budget,2));
            } catch (Throwable actual) { callerFailure.set(actual); }
        });
        Thread closer = null;
        try {
            assertTrue(started.await(5,TimeUnit.SECONDS));
            closer = Thread.ofPlatform().start(group::close);
            long deadline = System.nanoTime()+TimeUnit.SECONDS.toNanos(5);
            while (group.isOpen() && System.nanoTime() < deadline) Thread.onSpinWait();
            assertFalse(group.isOpen());
            caller.join(100); assertTrue(caller.isAlive());
        } finally {
            finish.countDown(); caller.join(5_000);
            if (closer != null) closer.join(5_000);
            group.close();
        }
        assertFalse(caller.isAlive()); assertNull(callerFailure.get());
        assertInstanceOf(CpuWorkerGroup.CpuParallelExecutionException.class,failure.get());
        assertEquals("CPU parallel execution cancelled by worker-group close",failure.get().getMessage());
        assertEquals(Boolean.TRUE,stillCovered.get());
        assertTimeoutPreemptively(java.time.Duration.ofSeconds(5),() -> budget.acquire(2).close());
    }

    @Test void validatesCapacityAndDemandAndLeaseReleasesOnce() {
        assertThrows(IllegalArgumentException.class, () -> new CpuConcurrencyBudget(0));
        var budget = new CpuConcurrencyBudget(2);
        assertAll(() -> assertEquals(2, budget.capacity()),
                () -> assertThrows(IllegalArgumentException.class, () -> budget.acquire(0)),
                () -> assertThrows(IllegalArgumentException.class, () -> budget.acquire(3)));
        var lease = budget.acquire(2);
        assertEquals(2, lease.demand());
        lease.close();
        lease.close();
        assertDoesNotThrow(() -> budget.acquire(2).close());
    }

    @Test void interruptionRestoresStatusAndDoesNotLeakPermits() throws Exception {
        var budget = new CpuConcurrencyBudget(1);
        var held = budget.acquire(1);
        var failure = new AtomicReference<Throwable>();
        var interrupted = new AtomicReference<Boolean>();
        Thread waiter = Thread.ofPlatform().start(() -> {
            try { budget.acquire(1); }
            catch (Throwable actual) { failure.set(actual); }
            interrupted.set(Thread.currentThread().isInterrupted());
        });
        waiter.interrupt();
        waiter.join(5_000);
        assertAll(() -> assertInstanceOf(CpuConcurrencyBudget.CpuCoordinationException.class,
                        failure.get()),
                () -> assertEquals(Boolean.TRUE, interrupted.get()));
        held.close();
        assertDoesNotThrow(() -> budget.acquire(1).close());
    }

    @Test void coordinatedWorkerSubmissionChargesActualParticipants() throws Exception {
        var budget = new CpuConcurrencyBudget(2);
        var started = new CountDownLatch(2);
        var release = new CountDownLatch(1);
        try (var group = new CpuWorkerGroup(4, budget)) {
            Thread submitter = Thread.ofPlatform().start(() -> group.execute(
                    new CpuWorkerGroup.RangeCall[] {
                            () -> { started.countDown(); release.await(); },
                            () -> { started.countDown(); release.await(); }
                    }));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            var acquired = new CountDownLatch(1);
            Thread waiter = Thread.ofPlatform().start(() -> {
                try (var ignored = budget.acquire(1)) { acquired.countDown(); }
            });
            assertFalse(acquired.await(100, TimeUnit.MILLISECONDS));
            release.countDown();
            submitter.join(5_000);
            assertTrue(acquired.await(5, TimeUnit.SECONDS));
            waiter.join(5_000);
        }
    }

    @Test void workerFailureReleasesItsWholeParticipantLease() {
        var budget = new CpuConcurrencyBudget(2);
        assertThrows(NullPointerException.class, () -> new CpuWorkerGroup(1, null));
        try (var group = new CpuWorkerGroup(2, budget)) {
            assertThrows(IllegalStateException.class, () -> group.execute(
                    new CpuWorkerGroup.RangeCall[] {
                            () -> { throw new IllegalStateException("worker failed"); },
                            () -> { }
                    }));
            assertDoesNotThrow(() -> budget.acquire(2).close());
        }
    }
}
