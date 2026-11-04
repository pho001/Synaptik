package io.github.pho001.synaptik.examples;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Model;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingStep;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * One fixed-shape regression journey using only supported public APIs.
 *
 * <p>Each mode starts independently with weight zero and no bias. The native-source copy checks
 * and detached-output checks are executable invariants, not a replacement for backend tests.
 * Run with JDK 26 and {@code --enable-native-access=ALL-UNNAMED}.</p>
 */
public final class LearningJourney {
    private static final List<String> MODES = List.of(
            "first", "tensors", "gradients", "train", "save-load", "reuse", "invalid-shape");

    private LearningJourney() {}

    /**
     * Runs an independent chapter or the complete sequence in tutorial order.
     *
     * @param args zero to two arguments: mode (default {@code all}) and optional graph destination
     *     (default {@code learning-journey.graph}); the destination is used only by
     *     {@code save-load}/{@code all}, its parent must exist, and an existing file is replaced
     * @throws IllegalArgumentException if the mode is unknown or more than two arguments are given
     * @throws Exception if model persistence or supported execution fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length > 2) throw new IllegalArgumentException("Usage: LearningJourney [mode] [graph-file]");
        String mode = args.length == 0 ? "all" : args[0];
        if (!mode.equals("all") && !MODES.contains(mode)) {
            throw new IllegalArgumentException("Modes: " + MODES + ", all");
        }
        Path file = Path.of(args.length < 2 ? "learning-journey.graph" : args[1]);
        for (String selected : mode.equals("all") ? MODES : List.of(mode)) {
            System.out.println("== " + selected + " ==");
            switch (selected) {
                case "first" -> first();
                case "tensors" -> tensors();
                case "gradients" -> gradients();
                case "train" -> train();
                case "save-load" -> saveLoad(file);
                case "reuse" -> reuse();
                case "invalid-shape" -> invalidShape();
                default -> throw new AssertionError("Unreachable mode: " + selected);
            }
        }
    }

    private static TensorDescriptor descriptor(boolean gradient, long... axes) {
        Shape shape = Shape.of(axes);
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    private static Tensor matrix(boolean gradient, long rows, long columns, float... values) {
        return TensorFactory.fromFlatArray(descriptor(gradient, rows, columns), Optional.empty(), values);
    }

    private static Tensor input() { return matrix(false, 2, 1, 1, 3); }
    private static Tensor target() { return matrix(false, 2, 1, 2, 6); }

    private static Tensor squaredErrorSum(Tensor prediction, Tensor target) {
        Tensor error = prediction.sub(target);
        return error.mul(error).sum();
    }

    private static void configure(Model<Tensor, Tensor> model) {
        model.configureTraining(descriptor(false, 2, 1), LearningJourney::squaredErrorSum, new Sgd(0.05));
    }

    private static Tensor weight(Model<Tensor, Tensor> model) {
        var entries = model.stateDictionary().entries();
        require(entries.size() == 1, "Expected exactly one parameter and no bias");
        return entries.getFirst().value(); // Borrowed state, never a detached result.
    }

    private static float storedScalar(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT, 0);
    }

    private static float[] values(HostTensorValue value) {
        require(value.dataType() == DataType.FLOAT32, "Expected FLOAT32 publication");
        float[] result = new float[Math.toIntExact(value.elementCount())];
        ByteBuffer bytes = value.bytes(); // Detached bytes are big-endian, unlike native host storage.
        for (int i = 0; i < result.length; i++) result[i] = bytes.getFloat();
        return result;
    }

    private static void expect(HostTensorValue actual, float... expected) {
        float[] actualValues = values(actual);
        if (!Arrays.equals(actualValues, expected)) {
            throw new AssertionError("Unexpected publication: " + Arrays.toString(actualValues));
        }
    }

    private static void print(String label, HostTensorValue value) {
        System.out.println(label + "=" + Arrays.toString(values(value)));
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message); // Active without the JVM -ea flag.
    }

    /** Builds while caller storage is live, then proves it is neither retained nor aliased. */
    private static Model<Tensor, Tensor> copiedModel() {
        try (Arena suppliedOwner = Arena.ofConfined()) {
            var memory = suppliedOwner.allocate(Float.BYTES, Float.BYTES);
            memory.set(ValueLayout.JAVA_FLOAT, 0, 0);
            Tensor supplied = TensorFactory.create(descriptor(true, 1, 1), Optional.empty(),
                    Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, memory)));
            var model = Model.builder().input(DataType.FLOAT32, 2, 1).linear(supplied).build();
            try {
                require(weight(model) != supplied, "Model retained caller Tensor identity");
                require(weight(model).hostStorage().orElseThrow() != supplied.hostStorage().orElseThrow(),
                        "Model retained caller storage identity");
                memory.set(ValueLayout.JAVA_FLOAT, 0, 99);
                require(storedScalar(weight(model)) == 0, "Caller mutation changed private weight");
                return model; // The supplied arena closes before any caller can compute/train.
            } catch (RuntimeException | Error failure) {
                try { model.close(); } catch (RuntimeException | Error cleanup) { failure.addSuppressed(cleanup); }
                throw failure;
            }
        }
    }

    private static void first() {
        HostTensorValue prediction;
        try (var model = copiedModel()) {
            prediction = model.compute(input());
            expect(prediction, 0, 0);
            System.out.println("supplied-weight.copy-and-close=ok");
            print("first.prediction", prediction);
        }
        expect(prediction, 0, 0);
        print("first.after-model-close", prediction);
    }

    private static void tensors() {
        float[] source = {1, 3};
        Tensor x = matrix(false, 2, 1, source);
        source[0] = 99; // fromFlatArray copied the Java array; x still contains [1,3].
        require(storedScalar(x) == 1, "Array import retained caller array");
        Tensor w = matrix(true, 1, 1, 0);
        Tensor prediction = x.linear(w);
        require(prediction.hostStorage().isEmpty(), "Expression unexpectedly owns evaluated data");
        System.out.println("tensors.input=FLOAT32 [2,1], values=[1.0, 3.0]");
        System.out.println("tensors.array-copy=ok");
        System.out.println("tensors.expression-has-storage=" + prediction.hostStorage().isPresent());
        try (Engine engine = Engine.standard()) {
            var actual = engine.compute(prediction);
            expect(actual, 0, 0);
            print("tensors.prediction", actual);
        }
        try (var model = copiedModel()) {
            Tensor prototype = TensorFactory.create(descriptor(false, 2, 1));
            Tensor composed = model.forward(prototype).relu();
            require(composed.hostStorage().isEmpty(), "Cold composition evaluated data");
            System.out.println("tensors.forward-composition-has-storage=" + composed.hostStorage().isPresent());
        }
    }

    private static void gradients() {
        Tensor w = matrix(true, 1, 1, 0);
        Tensor objective = squaredErrorSum(input().linear(w), target());
        HostTensorValue gradient;
        try (Engine engine = Engine.standard()) {
            var backward = engine.backward(objective, List.of(w), 8);
            expect(backward.objective(), 40);
            gradient = backward.gradients().getFirst();
            expect(gradient, -40);
            print("gradients.objective", backward.objective());
            print("gradients.weight", gradient);
            require(storedScalar(w) == 0, "Backward mutated the parameter");
            System.out.println("gradients.weight-without-optimizer=" + storedScalar(w));
        }
        expect(gradient, -40);
        print("gradients.after-engine-close", gradient);
    }

    private static void train() {
        Tensor supplied = matrix(true, 1, 1, 0);
        TrainingStep step;
        HostTensorValue prediction;
        try (var model = Model.builder().input(DataType.FLOAT32, 2, 1).linear(supplied).build()) {
            expect(model.compute(input()), 0, 0);
            configure(model);
            // Keep the acquired evaluation mode: inherited mode mutation invalidates owned sources.
            step = model.train(input(), target());
            expect(step.objective(), 40); // Objective belongs to the pre-update forward pass.
            require(step.optimizerStep().orElseThrow() == 1, "First optimizer step was not committed");
            require(storedScalar(weight(model)) == 2, "Expected private weight two");
            // Evaluation means compute without an optimizer update, not changing the frozen mode.
            prediction = model.compute(input());
            expect(prediction, 2, 6);
            print("train.objective-before-step", step.objective());
            System.out.println("train.private-weight=" + storedScalar(weight(model)));
            System.out.println("train.optimizer-step=" + step.optimizerStep().orElseThrow());
            print("train.prediction-after-step", prediction);
            require(storedScalar(supplied) == 0, "Training mutated caller-supplied weight");
            System.out.println("train.caller-weight=" + storedScalar(supplied));
        }
        expect(step.objective(), 40);
        expect(prediction, 2, 6);
        print("train.objective-after-model-close", step.objective());
        print("train.prediction-after-model-close", prediction);
    }

    private static void saveLoad(Path file) throws Exception {
        try (var original = copiedModel()) {
            configure(original);
            original.train(input(), target());
            original.save(file); // Replaces the destination; stores graph/current values, not this session.
        }
        HostTensorValue prediction;
        try (var restored = Model.load(file)) { // The original owner is already closed.
            prediction = restored.compute(matrix(false, 2, 1, 2, 4));
            expect(prediction, 4, 8);
            print("save-load.fresh-prediction", prediction);
            boolean rejected = false;
            try { restored.train(input(), target()); }
            catch (IllegalStateException expected) { rejected = true; }
            require(rejected, "Loaded model unexpectedly retained training configuration");
            System.out.println("save-load.unconfigured-train=rejected");
            configure(restored); // A fresh objective/optimizer, not checkpoint continuation.
            TrainingStep freshStep = restored.train(input(), target());
            expect(freshStep.objective(), 0);
            require(freshStep.optimizerStep().orElseThrow() == 1, "Loaded training was not freshly configured");
            require(storedScalar(weight(restored)) == 2, "Zero loss changed fitted weight");
            print("save-load.fresh-training-objective", freshStep.objective());
            System.out.println("save-load.fresh-optimizer-step=" + freshStep.optimizerStep().orElseThrow());
        }
        expect(prediction, 4, 8);
        print("save-load.after-model-close", prediction);
    }

    private static void reuse() {
        HostTensorValue detached;
        // Model borrows this Engine. Explicit results/preparation close before Model, then Engine.
        try (Engine engine = Engine.standard(); var model = copiedModel().withEngine(engine)) {
            configure(model);
            model.train(input(), target());
            print("reuse.model-first", model.compute(input()));
            print("reuse.model-fresh", model.compute(matrix(false, 2, 1, 2, 4)));

            Tensor prototype = TensorFactory.create(descriptor(false, 2, 1));
            Tensor parameter = weight(model);
            var compiled = engine.compile(List.of(model.forward(prototype)));
            try (var prepared = engine.prepare(compiled)) {
                detached = null;
                for (Tensor batch : List.of(input(), matrix(false, 2, 1, 2, 4))) {
                    // Compiler, not caller expression order, defines final positional input order.
                    var byId = Map.of(prototype.id(), batch, parameter.id(), parameter);
                    var bindings = compiled.inputs().stream().map(spec -> byId.get(spec.tensorId())).toList();
                    try (var result = engine.runValues(prepared, bindings)) {
                        detached = result.materialize(result.publications().getFirst(), 8);
                        print("reuse.prepared", detached);
                    }
                }
                // Do not train, mutate/rebind state or close Model while this preparation borrows it.
            }
        }
        expect(detached, 4, 8);
        print("reuse.after-all-owners-close", detached);
    }

    private static void invalidShape() {
        try (var model = copiedModel()) {
            // Same element count and dtype, but [1,2] is not the declared [2,1] descriptor.
            Tensor wrong = matrix(false, 1, 2, 1, 3);
            boolean rejected = false;
            try { model.compute(wrong); }
            catch (IllegalArgumentException expected) { rejected = true; }
            require(rejected, "Wrong shape was accepted");
            System.out.println("invalid-shape.compute=IllegalArgumentException");
            var valid = model.compute(input());
            expect(valid, 0, 0);
            print("invalid-shape.valid-after-rejection", valid);
        }
    }
}
