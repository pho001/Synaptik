package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Dropout;
import io.github.pho001.synaptik.nn.module.Buffer;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingParameterStorage;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Exercises caller-owned private training recovery without adding a production coordinator.
 * The exact comparison is one same-route CPU continuation with a known next input and typed
 * caller random/cursor state. A missing affecting external participant, changing per-run graph
 * random stream, training-mode BatchNorm without step-aligned buffer proof, or cross-route replay
 * is outside this claim. A fixed {@code GraphRngState.initial} value is supplied during objective
 * construction, but effective evaluation mode bypasses dropout; the session never advances it.
 */
final class PrivateTrainingRestoreIntegrationTest {
    private static final byte[] CONFIG = new byte[32];
    private static final ArtifactLimits LIMITS = new ArtifactLimits(16_384, 8_192, 16, 32,
            8_192, 8_192);
    private static final Sgd OPTIMIZER = new Sgd(0.1, 0.5, 0.0, false);
    private static final ForwardContext EFFECTIVE = new ForwardContext(ForwardMode.EVALUATION);
    private static final float BUFFER_VALUE = 0.5f;

    @TempDir Path directory;

    @Test
    void privateRestoreContinuesLikeUninterruptedCpuSession() throws Exception {
        try (Fixture fixture = new Fixture()) {
            RecoveryPoint point = fixture.capture(directory.resolve("boundary.model"));
            RunHolder old = fixture.published;
            fixture.step();
            TrainingState uninterrupted = fixture.published.session.state();
            CallerCursor uninterruptedCursor = fixture.published.cursor;
            float uninterruptedBuffer = storedFloat(fixture.published.tree.offset.value());

            fixture.restore(point, Fault.NONE);
            assertNotSame(old, fixture.published);
            assertFalse(old.arena.scope().isAlive());
            assertSame(fixture.published, fixture.lastCandidate);
            assertEquals(BUFFER_VALUE, storedFloat(fixture.published.tree.offset.value()));
            assertEquals(point.parentMode(), fixture.published.tree.mode());
            assertEquals(point.branchMode(), fixture.published.tree.branch.mode());
            assertEquals(point.dropMode(), fixture.published.tree.branch.drop.mode());
            assertEquals(point.effective(), fixture.published.effective);
            assertEquals(ForwardMode.TRAINING, fixture.published.tree.branch.drop.mode());
            assertEquals(ForwardMode.EVALUATION, fixture.published.effective.mode());
            assertStateEquals(point.training(), fixture.published.session.state());
            assertEquals(1, fixture.published.session.state().optimizerSteps());
            assertEquals(1, fixture.published.session.state().accumulatedGradientRuns());
            assertTrue(fixture.published.session.state().parameters().getFirst()
                    .momentumBytes().getFloat(0) != 0.0f);
            assertTrue(fixture.published.session.state().parameters().getFirst()
                    .accumulatedGradientBytes().getFloat(0) != 0.0f);

            fixture.step();
            assertStateEquals(uninterrupted, fixture.published.session.state());
            assertEquals(uninterruptedCursor, fixture.published.cursor);
            assertEquals(uninterruptedBuffer, storedFloat(fixture.published.tree.offset.value()));
            assertEquals(point.parentMode(), fixture.published.tree.mode());
            assertEquals(point.branchMode(), fixture.published.tree.branch.mode());
            assertEquals(point.dropMode(), fixture.published.tree.branch.drop.mode());
            assertEquals(point.effective(), fixture.published.effective);
        }
    }

    @Test
    void ordinaryFailuresDiscardCandidateWithoutPublishingCallerState() throws Exception {
        for (Fault fault : Fault.values()) {
            if (fault == Fault.NONE) continue;
            try (Fixture fixture = new Fixture()) {
                RecoveryPoint point = fixture.capture(directory.resolve(fault + ".model"));
                if (fault == Fault.PARAMETER_BYTES) {
                    Path other = directory.resolve("different.model");
                    ModelCheckpoint.save(other, new Tree(9.0f), fixture.engine,
                            CONFIG, null, LIMITS);
                    point = point.withArtifact(other);
                }
                if (fault == Fault.MISSING_CURSOR) point = point.withCursor(null);
                if (fault == Fault.LATE_CURSOR) {
                    CallerCursor cursor = point.cursor();
                    point = point.withCursor(new CallerCursor(cursor.index(), cursor.randomState(),
                            cursor.expectedExecutions(), cursor.expectedExternalInputs() + 1));
                }
                RunHolder published = fixture.published;
                CallerCursor cursor = published.cursor;
                RecoveryPoint attempted = point;
                RuntimeException failure = assertThrows(RuntimeException.class,
                        () -> fixture.restore(attempted, fault),
                        fault.name());
                assertTrue(failure.getMessage().contains(expectedFailure(fault)), fault.name());
                assertSame(published, fixture.published, fault.name());
                assertSame(cursor, fixture.published.cursor, fault.name());
                assertTrue(published.arena.scope().isAlive(), fault.name());
                if (fixture.lastCandidate != null) {
                    assertNotSame(published.tree, fixture.lastCandidate.tree, fault.name());
                    assertFalse(fixture.lastCandidate.arena.scope().isAlive(), fault.name());
                    if (fixture.lastCandidate.session != null) {
                        assertThrows(IllegalStateException.class,
                                fixture.lastCandidate.session::state, fault.name());
                    }
                }
            }
        }
    }

    /** Test-owned failure points; owner APIs still perform their own validation. */
    private enum Fault {
        NONE, MISSING_CURSOR, BEFORE_LOAD, INCOMPATIBLE_LOAD, AFTER_LOAD, AFTER_COPY,
        INVALID_OBJECTIVE, AFTER_OPEN, PARAMETER_BYTES, RESTORE_MISMATCH, LATE_CURSOR
    }

    /**
     * One concrete next-input and external-random handoff, independent of Training counters.
     *
     * @param index position of the next input in this bounded sequence
     * @param randomState caller-owned random state affecting that input
     * @param expectedExecutions boundary count checked against detached Training state
     * @param expectedExternalInputs candidate-dependent input schema checked after open
     */
    private record CallerCursor(int index, long randomState, long expectedExecutions,
                                int expectedExternalInputs) {
        /**
         * Derives the next CPU input from both saved position and caller random state.
         *
         * @return a finite, deterministic input for this bounded fixture
         * @throws IllegalArgumentException if the saved position is outside the fixture sequence
         */
        float nextInput() {
            if (index < 0 || index >= 3) throw new IllegalArgumentException("cursor outside input sequence");
            return (index + 1) * 0.75f + (randomState & 3) * 0.125f;
        }

        /**
         * Advances the caller-owned position and random state after one successful run.
         *
         * @return the immutable next cursor, including the next expected Training execution count
         */
        CallerCursor advance() {
            return new CallerCursor(index + 1, randomState * 6364136223846793005L + 1,
                    expectedExecutions + 1, expectedExternalInputs);
        }
    }

    /** Small fully bound NN tree with a persistent buffer and a mode-sensitive child. */
    private static final class Tree extends Module {
        final Buffer offset;
        final Branch branch;

        /**
         * Creates independent heap-backed state; Checkpoint later replaces candidate bindings.
         *
         * @param initialWeight finite initial scalar for a control tree or private load target
         */
        Tree(float initialWeight) {
            this(initialWeight, BUFFER_VALUE);
        }

        /**
         * Creates a fully bound private tree with a selectable starting buffer value.
         *
         * @param initialWeight finite initial parameter value before any checkpoint load
         * @param offsetValue finite initial buffer value, deliberately different for a load target
         */
        Tree(float initialWeight, float offsetValue) {
            offset = buffer("offset", scalar(offsetValue, false));
            branch = child("branch", new Branch(initialWeight));
        }

        /**
         * Builds the fixed objective from current final bindings and the recorded effective mode.
         * The same input Tensor is supplied on each run with caller-updated storage; evaluation
         * dropout leaves its expression unchanged.
         *
         * @param input non-null scalar input expression for the fixed session
         * @param context non-null effective context recorded at capture
         * @return connected scalar objective using the parameter and persistent buffer
         */
        Tensor objective(Tensor input, ForwardContext context) {
            Tensor selected = branch.drop.forward(input, GraphRngState.initial(7, 11), context)
                    .output();
            Tensor residual = branch.weight.value().mul(selected).add(offset.value());
            return residual.mul(residual);
        }
    }

    /** Child owner separates local mode from the parent's mode and supplied context. */
    private static final class Branch extends Module {
        final Parameter weight;
        final Dropout drop;

        /**
         * Creates one trainable scalar and one mode-sensitive child.
         *
         * @param initialWeight finite initial weight before any checkpoint load
         */
        Branch(float initialWeight) {
            weight = parameter("weight", scalar(initialWeight, true));
            drop = child("drop", new Dropout(0.25));
        }
    }

    /**
     * Complete detached in-memory boundary; the artifact supplies NN values only.
     *
     * @param artifact bounded model-only file path
     * @param training detached Training snapshot
     * @param parentMode captured root local mode
     * @param branchMode captured branch local mode
     * @param dropMode captured dropout local mode
     * @param effective objective-construction context actually used
     * @param cursor caller-owned next-input and random handoff
     */
    private record RecoveryPoint(Path artifact, TrainingState training, ForwardMode parentMode,
                                 ForwardMode branchMode, ForwardMode dropMode,
                                 ForwardContext effective, CallerCursor cursor) {
        /**
         * Retains all non-model participants while selecting a different model artifact.
         *
         * @param other non-null model artifact path to attempt loading
         * @return an otherwise identical detached boundary
         */
        RecoveryPoint withArtifact(Path other) {
            return new RecoveryPoint(other, training, parentMode, branchMode, dropMode,
                    effective, cursor);
        }

        /**
         * Retains all other detached participants while replacing the caller cursor.
         *
         * @param other replacement cursor, possibly null to test missing-participant rejection
         * @return an otherwise identical boundary
         */
        RecoveryPoint withCursor(CallerCursor other) {
            return new RecoveryPoint(artifact, training, parentMode, branchMode, dropMode,
                    effective, other);
        }
    }

    /** Caller-owned live view; only one holder reference is published under fixture exclusion. */
    private static final class RunHolder implements AutoCloseable {
        final Tree tree;
        final Arena arena;
        final ForwardContext effective;
        final Tensor input;
        TrainingSession session;
        CallerCursor cursor;

        /**
         * Owns the private or published native lifetime for one fully bound tree.
         *
         * @param tree non-null distinct NN tree
         * @param arena non-null shareable native Arena owned by this holder
         * @param effective non-null objective-construction context
         * @param cursor typed caller state, possibly null only before staging validation
         */
        RunHolder(Tree tree, Arena arena, ForwardContext effective, CallerCursor cursor) {
            this.tree = tree;
            this.arena = arena;
            this.effective = effective;
            this.cursor = cursor;
            this.input = scalar(1.0f, false);
        }

        /** Closes the fixed session before its parameter Arena, including on rejected candidates. */
        @Override public void close() {
            try {
                if (session != null) session.close();
            } finally {
                arena.close();
            }
        }
    }

    /** Serial caller exclusion, staging, and single-reference publication for this test only. */
    private static final class Fixture implements AutoCloseable {
        final Engine engine = Engine.standard();
        RunHolder published;
        RunHolder lastCandidate;

        /** Opens the control route with a mixed local-mode tree and a fixed effective context. */
        Fixture() {
            RunHolder initial = new RunHolder(new Tree(2.0f), Arena.ofShared(), EFFECTIVE,
                    new CallerCursor(0, 19L, 0, 2));
            try {
                initial.tree.eval();
                initial.tree.branch.train();
                TrainingParameterStorage.copyToSharedNative(initial.tree.trainingParameters(), initial.arena);
                initial.session = TrainingSession.open(engine, initial.tree.trainingParameters(),
                        initial.tree.objective(initial.input, EFFECTIVE), OPTIMIZER);
                published = initial;
            } catch (RuntimeException | Error failure) {
                initial.close();
                engine.close();
                throw failure;
            }
        }

        /**
         * Executes the cursor-selected next input and advances typed state only on success.
         *
         * @throws RuntimeException if execution or caller input validation fails
         */
        synchronized void step() {
            CallerCursor cursor = published.cursor;
            GradientMode mode = cursor.index() == 0 ? GradientMode.RESET_AND_STEP
                    : cursor.index() == 1 ? GradientMode.ACCUMULATE
                    : GradientMode.ACCUMULATE_AND_STEP;
            published.input.hostStorage().orElseThrow().segment().set(
                    java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, cursor.nextInput());
            published.session.run(List.of(published.input, published.tree.offset.value()), mode);
            published.cursor = cursor.advance();
        }

        /**
         * Captures one quiescent successful-run boundary after a step and pending accumulation.
         * Saves the actual trained tree's native parameter and heap buffer leaves through
         * ordinary Model Checkpoint, without a private expression view or binding changes.
         *
         * @param artifact non-null bounded model-artifact destination in the temporary directory
         * @return complete detached in-memory recovery participants
         * @throws IOException if model artifact publication fails
         */
        synchronized RecoveryPoint capture(Path artifact) throws IOException {
            step();
            step();
            ModelCheckpoint.save(artifact, published.tree, engine,
                    CONFIG, null, LIMITS);
            return new RecoveryPoint(artifact, published.session.state(),
                    published.tree.mode(),
                    published.tree.branch.mode(), published.tree.branch.drop.mode(),
                    published.effective, published.cursor);
        }

        /**
         * Stages and validates a distinct candidate, then retires the old holder and publishes
         * exactly one replacement reference under caller exclusion. Test faults never mutate the
         * published holder; ordinary failure closes the private session and Arena.
         * Loaded heap leaves are saved and loaded again before native parameter copying;
         * complete artifact equality verifies that this intermediate round trip preserves bits.
         *
         * @param point non-null complete in-memory boundary
         * @param fault test-owned injection or publicly reachable owner rejection to exercise
         * @throws IOException if Checkpoint rejects the model artifact
         * @throws RuntimeException if a participant or owner validation rejects the candidate
         */
        synchronized void restore(RecoveryPoint point, Fault fault) throws IOException {
            Objects.requireNonNull(point, "point");
            Objects.requireNonNull(point.artifact(), "model artifact");
            Objects.requireNonNull(point.training(), "TrainingState");
            Objects.requireNonNull(point.parentMode(), "parent mode");
            Objects.requireNonNull(point.branchMode(), "branch mode");
            Objects.requireNonNull(point.dropMode(), "dropout mode");
            Objects.requireNonNull(point.effective(), "effective context");
            Objects.requireNonNull(point.cursor(), "caller cursor");
            point.cursor().nextInput();
            if (point.cursor().expectedExecutions() != point.training().executions()) {
                throw new IllegalArgumentException("caller cursor boundary differs");
            }
            if (fault == Fault.BEFORE_LOAD) throw new IllegalStateException("before load");

            RunHolder candidate = new RunHolder(new Tree(2.0f, -3.0f), Arena.ofShared(),
                    point.effective(), point.cursor());
            lastCandidate = candidate;
            try {
                assertEquals(-3.0f, storedFloat(candidate.tree.offset.value()));
                byte[] expectedDigest = CONFIG.clone();
                if (fault == Fault.INCOMPATIBLE_LOAD) expectedDigest[0] = 1;
                ModelCheckpoint.load(point.artifact(), candidate.tree, expectedDigest, null, LIMITS);
                assertEquals(BUFFER_VALUE, storedFloat(candidate.tree.offset.value()));
                if (fault == Fault.AFTER_LOAD) throw new IllegalStateException("after load");
                Path resaved = point.artifact().resolveSibling(point.artifact().getFileName() + ".resaved");
                ModelCheckpoint.save(resaved, candidate.tree, engine, CONFIG, null, LIMITS);
                assertArrayEquals(Files.readAllBytes(point.artifact()), Files.readAllBytes(resaved));
                ModelCheckpoint.load(resaved, candidate.tree, CONFIG, null, LIMITS);
                candidate.tree.eval();
                if (point.parentMode() == ForwardMode.TRAINING) candidate.tree.train();
                if (point.branchMode() == ForwardMode.TRAINING) candidate.tree.branch.train();
                else candidate.tree.branch.eval();
                if (point.dropMode() == ForwardMode.TRAINING) candidate.tree.branch.drop.train();
                else candidate.tree.branch.drop.eval();
                TrainingParameterStorage.copyToSharedNative(candidate.tree.trainingParameters(), candidate.arena);
                if (fault == Fault.AFTER_COPY) throw new IllegalStateException("after copy");

                Tensor objective = fault == Fault.INVALID_OBJECTIVE
                        ? scalar(1.0f, false)
                        : candidate.tree.objective(candidate.input, candidate.effective);
                Sgd optimizer = fault == Fault.RESTORE_MISMATCH
                        ? new Sgd(0.2, 0.5, 0.0, false) : OPTIMIZER;
                candidate.session = TrainingSession.open(engine, candidate.tree.trainingParameters(), objective, optimizer);
                if (fault == Fault.AFTER_OPEN) throw new IllegalStateException("after open");
                assertCanonicalParameterBytes(point.training(), candidate.session.state());
                candidate.session.restore(point.training());
                if (candidate.cursor.expectedExternalInputs() != candidate.session.inputs().size()) {
                    throw new IllegalArgumentException("caller input schema differs");
                }

                RunHolder old = published;
                old.close();
                published = candidate;
            } catch (IOException | RuntimeException | Error failure) {
                candidate.close();
                throw failure;
            }
        }

        /** Closes the currently published holder before the Engine it borrows. */
        @Override public void close() {
            try {
                if (published != null) published.close();
            } finally {
                engine.close();
            }
        }
    }

    /**
     * Compares exact stable-path sets and canonical big-endian logical bytes before restore.
     *
     * @param supplied non-null detached boundary state
     * @param candidate non-null fresh state from the private final-binding session
     * @throws IllegalArgumentException if a path or represented logical value differs
     */
    private static void assertCanonicalParameterBytes(TrainingState supplied, TrainingState candidate) {
        Map<String, ByteBuffer> expected = new HashMap<>();
        for (var parameter : supplied.parameters()) {
            if (expected.putIfAbsent(parameter.path(), parameter.parameterBytes()) != null) {
                throw new IllegalArgumentException("repeated supplied parameter path");
            }
        }
        Map<String, ByteBuffer> actual = new HashMap<>();
        for (var parameter : candidate.parameters()) {
            if (actual.putIfAbsent(parameter.path(), parameter.parameterBytes()) != null) {
                throw new IllegalArgumentException("repeated candidate parameter path");
            }
        }
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException("candidate logical parameter bytes differ");
        }
    }

    /**
     * Compares every detached Training field after the same-route continuation.
     *
     * @param expected uninterrupted CPU state
     * @param actual restored CPU state after the same next input
     */
    private static void assertStateEquals(TrainingState expected, TrainingState actual) {
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int i = 0; i < expected.parameters().size(); i++) {
            var left = expected.parameters().get(i);
            var right = actual.parameters().get(i);
            assertEquals(left.path(), right.path());
            assertEquals(left.dataType(), right.dataType());
            assertEquals(left.shape(), right.shape());
            assertEquals(left.parameterBytes(), right.parameterBytes());
            assertEquals(left.masterParameterBytes(), right.masterParameterBytes());
            assertEquals(left.firstMomentBytes(), right.firstMomentBytes());
            assertEquals(left.secondMomentBytes(), right.secondMomentBytes());
            assertEquals(left.accumulatedGradientBytes(), right.accumulatedGradientBytes());
        }
    }

    /**
     * Names the owner or fixture check expected to reject each injected ordinary failure.
     *
     * @param fault non-success failure stage under test
     * @return distinctive diagnostic fragment identifying the reached stage
     * @throws IllegalArgumentException if the success sentinel is supplied
     */
    private static String expectedFailure(Fault fault) {
        return switch (fault) {
            case MISSING_CURSOR -> "caller cursor";
            case BEFORE_LOAD -> "before load";
            case INCOMPATIBLE_LOAD -> "Model checkpoint fingerprint mismatch";
            case AFTER_LOAD -> "after load";
            case AFTER_COPY -> "after copy";
            case INVALID_OBJECTIVE -> "training objective must require gradients";
            case AFTER_OPEN -> "after open";
            case PARAMETER_BYTES -> "candidate logical parameter bytes differ";
            case RESTORE_MISMATCH -> "training state optimizer configuration differs";
            case LATE_CURSOR -> "caller input schema differs";
            case NONE -> throw new IllegalArgumentException("success is not a failure stage");
        };
    }

    /**
     * Reads one caller-quiescent FLOAT32 scalar value from its current host association.
     *
     * @param tensor non-null initialized scalar with live host storage
     * @return its represented FLOAT32 value without mutating the Tensor
     */
    private static float storedFloat(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment()
                .get(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0);
    }

    /**
     * Allocates one independent heap scalar for model state, objective input, or run input.
     *
     * @param value exact FLOAT32 represented value
     * @param gradient whether the scalar participates as a trainable parameter
     * @return fresh dense scalar Tensor
     */
    private static Tensor scalar(float value, boolean gradient) {
        return TensorFactory.scalar(value, Optional.empty(), gradient);
    }
}
