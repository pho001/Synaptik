package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Session-private stable parameter binding with active optimizer slots and staged primitive
 * update buffers.
 *
 * <p>Capture bounds both the declared-width and optimizer-width snapshot payloads to a Java
 * byte-array length before storage inspection or buffer allocation. This keeps later in-memory
 * snapshot and restore lengths representable without unchecked narrowing or wrapping. The active
 * accumulation and first optimizer slots, parameter candidates, and rollback arrays exist from
 * capture. An active second moment is allocated only for Adam/AdamW assignment at open; inactive
 * accumulation, first-slot, and second-slot candidates are allocated only before an operation
 * that stages them. Successful accumulation clearing releases its inactive candidate. A restored
 * negative-zero momentum slot under zero configured momentum is retained exactly until the next
 * successful update, when it is normalized without a pass on ordinary zero-momentum steps.</p>
 */
final class TrainingParameter {
    private final String path;
    private final Tensor tensor;
    private final HostTensorStorage storage;
    private final MemorySegment segment;
    private final DataType dataType;
    private final Shape shape;
    private final int elementCount;
    private Optimizer optimizer;
    private AdaptiveCoefficients adaptive;
    private float[] master32;
    private float[] masterCandidate32;
    private float[] masterRollback32;
    private short[] parameterCandidate16;
    private short[] parameterRollback16;

    private float[] gradient32;
    private float[] accumulated32;
    private float[] accumulatedCandidate32;
    private float[] parameterCandidate32;
    private float[] parameterRollback32;
    private float[] momentum32;
    private float[] momentumCandidate32;
    private float[] secondMoment32;
    private float[] secondMomentCandidate32;
    private boolean momentumNeedsZeroNormalization;
    private boolean stagedMomentumHasNegativeZero;

    private double[] gradient64;
    private double[] accumulated64;
    private double[] accumulatedCandidate64;
    private double[] parameterCandidate64;
    private double[] parameterRollback64;
    private double[] momentum64;
    private double[] momentumCandidate64;
    private double[] secondMoment64;
    private double[] secondMomentCandidate64;

    /**
     * Retains a validated exact binding and initializes active optimizer/rollback buffers.
     *
     * @param path non-null stable source path
     * @param tensor exact admitted Tensor with immutable descriptor
     * @param storage exact validated borrowed writable shareable native storage
     * @param elementCount positive Java-array-bounded element count
     */
    private TrainingParameter(
            String path,
            Tensor tensor,
            HostTensorStorage storage,
            int elementCount) {
        this.path = path;
        this.tensor = tensor;
        this.storage = storage;
        this.segment = storage.segment();
        this.dataType = tensor.descriptor().dataType();
        this.shape = tensor.descriptor().shape();
        this.elementCount = elementCount;
        if (dataType == DataType.FLOAT64) {
            gradient64 = new double[elementCount];
            accumulated64 = new double[elementCount];
            parameterCandidate64 = new double[elementCount];
            parameterRollback64 = new double[elementCount];
            momentum64 = new double[elementCount];
        } else {
            gradient32 = new float[elementCount];
            accumulated32 = new float[elementCount];
            momentum32 = new float[elementCount];
            if (dataType == DataType.FLOAT32) {
                parameterCandidate32 = new float[elementCount];
                parameterRollback32 = new float[elementCount];
            } else {
                master32 = new float[elementCount];
                masterCandidate32 = new float[elementCount];
                masterRollback32 = new float[elementCount];
                parameterCandidate16 = new short[elementCount];
                parameterRollback16 = new short[elementCount];
                initializeMaster();
            }
        }
    }

    /**
     * Resolves this path's immutable algorithm and dtype-narrowed coefficients at open.
     *
     * @param assigned non-null default or ordered-group optimizer for this path
     * @throws IllegalArgumentException if a coefficient cannot narrow to this parameter's domain
     */
    void configureOptimizer(Optimizer assigned) {
        this.optimizer = Objects.requireNonNull(assigned, "assigned");
        if (assigned instanceof Sgd sgd) {
            TrainingSession.validateSgdNarrowing(sgd, dataType);
            return;
        }
        adaptive = AdaptiveCoefficients.forParameter(assigned, dataType);
        if (dataType == DataType.FLOAT64) {
            secondMoment64 = new double[elementCount];
        } else {
            secondMoment32 = new float[elementCount];
        }
    }

    private void initializeMaster() {
        for (int index = 0; index < elementCount; index++) {
            short bits = segment.get(ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
            master32[index] = dataType == DataType.BFLOAT16
                    ? BFloat16Bits.toFloat(bits)
                    : Float16Bits.toFloat(bits);
        }
    }

    /**
     * Snapshots and validates complete generic membership before storage capture or allocation.
     *
     * @param source non-null caller-owned live parameter source
     * @return nonempty immutable entries in exact caller encounter order
     * @throws NullPointerException if the source, list or an entry is null
     * @throws IllegalArgumentException if paths repeat
     * @throws IllegalStateException if membership is empty or live validation fails
     */
    static List<TrainingParameters.Entry> captureEntries(TrainingParameters source) {
        Objects.requireNonNull(source, "parameters");
        List<TrainingParameters.Entry> entries = List.copyOf(
                Objects.requireNonNull(source.entries(), "parameters.entries()"));
        if (entries.isEmpty()) {
            throw new IllegalStateException("training source has no parameters");
        }
        var paths = new HashSet<String>();
        for (TrainingParameters.Entry entry : entries) {
            if (!paths.add(entry.path())) {
                throw new IllegalArgumentException("duplicate parameter path: " + entry.path());
            }
        }
        source.validateUnchanged();
        return entries;
    }

    /**
     * Captures one stable Tensor and exact supported native storage with private optimizer state.
     *
     * @param path non-null nonblank source path
     * @param tensor non-null exact Tensor from the admitted membership snapshot
     * @return non-null session-private binding with validated finite initial values
     * @throws NullPointerException if either input is null
     * @throws IllegalArgumentException if descriptor, capacity, sharing or snapshot size is unsupported
     * @throws IllegalStateException if host storage is absent, dead or inaccessible
     */
    static TrainingParameter capture(String path, Tensor tensor) {
        String stablePath = Objects.requireNonNull(path, "path");
        if (stablePath.isBlank()) {
            throw new IllegalArgumentException("parameter path must not be blank");
        }
        Tensor stableTensor = Objects.requireNonNull(tensor, "tensor");
        var descriptor = stableTensor.descriptor();
        DataType type = descriptor.dataType();
        if (!type.isFloating()) {
            throw new IllegalArgumentException(
                    "training parameter must use a floating type at path "
                            + stablePath + ": " + type);
        }
        if (!descriptor.requiresGrad()) {
            throw new IllegalArgumentException(
                    "training parameter must require gradients at path " + stablePath);
        }
        if (!descriptor.shape().isFullyStatic()) {
            throw new IllegalArgumentException(
                    "training parameter shape must be fully static at path " + stablePath);
        }
        var layout = descriptor.layout().orElseThrow(() -> new IllegalArgumentException(
                "training parameter layout must be resolved at path " + stablePath));
        if (layout.kind() != LayoutKind.DENSE_CONTIGUOUS
                || layout.storageOffset() != 0
                || layout.isView()) {
            throw new IllegalArgumentException(
                    "training parameter must use dense offset-zero non-view layout at path "
                            + stablePath);
        }
        long count = descriptor.shape().knownElementCount().orElseThrow();
        if (count <= 0) {
            throw new IllegalArgumentException(
                    "training parameter must contain at least one element at path " + stablePath);
        }
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "training parameter element count exceeds Java array limit at path "
                            + stablePath + ": " + count);
        }
        validateSnapshotByteSizes(stablePath, type, count);
        if (layout.referencedElementSpan() != count) {
            throw new IllegalArgumentException(
                    "training parameter layout span must equal element count at path "
                            + stablePath);
        }
        HostTensorStorage stableStorage = stableTensor.hostStorage().orElseThrow(
                () -> new IllegalStateException(
                        "training parameter has no host storage at path " + stablePath));
        validateStorage(stablePath, stableStorage, type, count, true, true);
        Thread accessProbe = Thread.ofVirtual().unstarted(() -> { });
        if (!stableStorage.segment().isAccessibleBy(accessProbe)) {
            throw new IllegalArgumentException(
                    "training parameter storage must be shareable across operation threads at path "
                            + stablePath);
        }
        validateFiniteStorage(stablePath, stableStorage.segment(), type, (int) count);
        return new TrainingParameter(
                stablePath, stableTensor, stableStorage, (int) count);
    }

    String path() {
        return path;
    }

    Tensor tensor() {
        return tensor;
    }

    HostTensorStorage storage() {
        return storage;
    }

    DataType dataType() {
        return dataType;
    }

    Shape shape() {
        return shape;
    }

    long byteSize() {
        return Math.multiplyExact((long) elementCount, dataType.byteWidth());
    }

    long masterByteSize() {
        return Math.multiplyExact(
                (long) elementCount,
                dataType == DataType.FLOAT64 ? Double.BYTES : Float.BYTES);
    }

    /**
     * Checks both payload widths required by an in-memory training-state snapshot.
     *
     * <p>The caller checks positive, int-representable element count first. The logical width is
     * checked before the optimizer width, so a low-precision parameter can report which payload
     * first exceeds the byte-array limit. Neither check allocates or inspects parameter storage.</p>
     *
     * @param path non-null diagnostic path of the parameter being captured
     * @param dataType non-null supported floating parameter type
     * @param elementCount positive element count already bounded by {@link Integer#MAX_VALUE}
     * @throws IllegalArgumentException if either checked byte size exceeds the Java array limit
     */
    static void validateSnapshotByteSizes(
            String path, DataType dataType, long elementCount) {
        long logicalBytes = Math.multiplyExact(elementCount, dataType.byteWidth());
        if (logicalBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "training parameter logical snapshot byte size exceeds Java array limit at path "
                            + path + ": " + logicalBytes);
        }
        long optimizerBytes = Math.multiplyExact(
                elementCount,
                dataType == DataType.FLOAT64 ? Double.BYTES : Float.BYTES);
        if (optimizerBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "training parameter optimizer snapshot byte size exceeds Java array limit at path "
                            + path + ": " + optimizerBytes);
        }
    }

    /**
     * Validates exact storage identity and native schema after the source checks live membership.
     *
     * @throws IllegalStateException if the association changed or storage is absent/dead/inaccessible
     * @throws IllegalArgumentException if the retained storage no longer satisfies its native schema
     */
    void validateBinding() {
        HostTensorStorage current = tensor.hostStorage().orElseThrow(
                () -> new IllegalStateException(
                        "parameter host storage is absent at path " + path));
        if (current != storage) {
            throw new IllegalStateException(
                    "parameter host storage changed while training session is open: " + path);
        }
        validateStorage(path, storage, dataType, elementCount, true, true);
    }

    boolean overlaps(HostTensorStorage other) {
        if (storage.byteSize() == 0 || other.byteSize() == 0) {
            return false;
        }
        return segment.asOverlappingSlice(other.segment()).isPresent();
    }

    void decodeGradient(HostTensorValue value) {
        Objects.requireNonNull(value, "gradient");
        if (value.dataType() != dataType
                || !value.shape().equals(shape)
                || value.elementCount() != elementCount
                || value.byteSize() != byteSize()) {
            throw new IllegalStateException(
                    "gradient schema does not match parameter at path " + path);
        }
        ByteBuffer bytes = value.bytes().order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                gradient64[index] = bytes.getDouble();
            }
        } else if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                gradient32[index] = bytes.getFloat();
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                gradient32[index] = decodeLow(bytes.getShort());
            }
        }
        for (int index = 0; index < elementCount; index++) {
            if (dataType == DataType.FLOAT64) {
                requireFinite(gradient64[index], "decoded gradient", path, index);
            } else {
                requireFinite(gradient32[index], "decoded gradient", path, index);
            }
        }
        if (bytes.hasRemaining()) {
            throw new IllegalStateException(
                    "gradient payload contains trailing bytes at path " + path);
        }
    }

    /** Stages a finite pending sum without mutating the active accumulation slot. */
    void stageAccumulation() {
        ensureAccumulationCandidate();
        if (dataType != DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                float candidate = accumulated32[index] + gradient32[index];
                requireFinite(candidate, "accumulated gradient candidate", path, index);
                accumulatedCandidate32[index] = candidate;
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                double candidate = accumulated64[index] + gradient64[index];
                requireFinite(candidate, "accumulated gradient candidate", path, index);
                accumulatedCandidate64[index] = candidate;
            }
        }
    }

    /**
     * Stages a parameter update and its configured optimizer-slot candidates before commit.
     *
     * <p>Adaptive updates use pre-narrowed coefficients and compute validated bias corrections
     * outside the element loop. Their subtraction is evaluated as
     * {@code (learningRate * correctedFirstMoment) / (sqrt(correctedSecondMoment) + epsilon)}:
     * the scaled numerator must be finite before division, and the update must be finite before
     * subtraction. This multiply-before-divide order applies in FLOAT64 and in the FLOAT32 path
     * shared by FLOAT32, BFLOAT16, and FLOAT16 parameters; reassociation can change rounding and
     * which updates overflow. A positive configured SGD momentum that narrows to binary32 zero
     * still stages a zero
     * candidate, preserving the existing swap behavior. Exactly zero configured momentum instead
     * leaves its active slot untouched until a successful commit.</p>
     *
     * @param completedSteps non-negative number of prior successful parameter updates
     * @param useAccumulatedCandidate whether a preceding accumulation stage supplies gradients
     * @throws ArithmeticException if an intermediate or candidate is non-finite, or an adaptive
     *     bias correction or denominator is not positive; active state remains unchanged
     */
    void stageUpdate(long completedSteps, boolean useAccumulatedCandidate) {
        if (adaptive != null) {
            ensureMomentumCandidate();
            ensureSecondMomentCandidate();
            if (dataType == DataType.FLOAT64) {
                stageAdaptive64(completedSteps + 1, useAccumulatedCandidate);
            } else {
                stageAdaptive32(completedSteps + 1, useAccumulatedCandidate);
            }
            return;
        }
        Sgd optimizer = (Sgd) this.optimizer;
        boolean configuredMomentum = optimizer.momentum() != 0.0d;
        if (configuredMomentum) {
            ensureMomentumCandidate();
        }
        if (dataType != DataType.FLOAT64) {
            stageUpdate32(optimizer, completedSteps, useAccumulatedCandidate, configuredMomentum);
        } else {
            stageUpdate64(optimizer, completedSteps, useAccumulatedCandidate);
        }
    }

    private void stageAdaptive32(long step, boolean useAccumulatedCandidate) {
        float lr = (float) adaptive.learningRate();
        float b1 = (float) adaptive.beta1();
        float b2 = (float) adaptive.beta2();
        float eps = (float) adaptive.epsilon();
        float decay = (float) adaptive.weightDecay();
        float correction1 = 1.0f - (float) Math.pow(b1, step);
        float correction2 = 1.0f - (float) Math.pow(b2, step);
        requirePositiveFinite(correction1, "first bias correction");
        requirePositiveFinite(correction2, "second bias correction");
        float decayFactor = 1.0f;
        if (adaptive.decoupled()) {
            float decayScale = lr * decay;
            requireFinite(decayScale, "decay scale", path, -1);
            decayFactor -= decayScale;
            requireFinite(decayFactor, "decay factor", path, -1);
        }
        float[] selected = useAccumulatedCandidate ? accumulatedCandidate32 : gradient32;
        if (adaptive.decoupled()) {
            for (int index = 0; index < elementCount; index++) {
                float parameterValue = readParameter32ForUpdate(index);
                float base = parameterValue * decayFactor;
                requireFinite(base, "decayed parameter", path, index);
                stageAdaptiveElement32(index, selected[index], base, b1, b2, correction1,
                        correction2, eps, lr);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                float parameterValue = readParameter32ForUpdate(index);
                float decayTerm = decay * parameterValue;
                requireFinite(decayTerm, "coupled decay term", path, index);
                float gradient = selected[index] + decayTerm;
                requireFinite(gradient, "adjusted gradient", path, index);
                stageAdaptiveElement32(index, gradient, parameterValue, b1, b2, correction1,
                        correction2, eps, lr);
            }
        }
    }

    private float readParameter32ForUpdate(int index) {
        float parameterValue;
        if (dataType == DataType.FLOAT32) {
            parameterValue = segment.get(ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES);
            parameterRollback32[index] = parameterValue;
        } else {
            short bits = segment.get(ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
            requireFinite(decodeLow(bits), "current logical parameter", path, index);
            parameterRollback16[index] = bits;
            parameterValue = master32[index];
            masterRollback32[index] = parameterValue;
        }
        requireFinite(parameterValue, "current parameter", path, index);
        return parameterValue;
    }

    private void stageAdaptiveElement32(int index, float gradient, float base, float b1,
            float b2, float correction1, float correction2, float eps, float lr) {
        float firstTerm = b1 * momentum32[index];
        requireFinite(firstTerm, "first moment history", path, index);
        float firstGradient = (1.0f - b1) * gradient;
        requireFinite(firstGradient, "first moment gradient", path, index);
        float first = firstTerm + firstGradient;
        requireFinite(first, "first moment candidate", path, index);
        float squared = gradient * gradient;
        requireFinite(squared, "squared gradient", path, index);
        float secondTerm = b2 * secondMoment32[index];
        requireFinite(secondTerm, "second moment history", path, index);
        float secondGradient = (1.0f - b2) * squared;
        requireFinite(secondGradient, "second moment gradient", path, index);
        float second = secondTerm + secondGradient;
        requireFinite(second, "second moment candidate", path, index);
        float correctedFirst = first / correction1;
        float correctedSecond = second / correction2;
        requireFinite(correctedFirst, "corrected first moment", path, index);
        requireFinite(correctedSecond, "corrected second moment", path, index);
        float root = (float) Math.sqrt(correctedSecond);
        requireFinite(root, "second moment root", path, index);
        float denominator = root + eps;
        requirePositiveFinite(denominator, "adaptive denominator");
        float numerator = lr * correctedFirst;
        requireFinite(numerator, "scaled adaptive numerator", path, index);
        float update = numerator / denominator;
        requireFinite(update, "adaptive update", path, index);
        float candidate = base - update;
        requireFinite(candidate, "parameter candidate", path, index);
        momentumCandidate32[index] = first;
        secondMomentCandidate32[index] = second;
        if (dataType == DataType.FLOAT32) {
            parameterCandidate32[index] = candidate;
        } else {
            masterCandidate32[index] = candidate;
        }
    }

    private void stageAdaptive64(long step, boolean useAccumulatedCandidate) {
        double lr = adaptive.learningRate();
        double b1 = adaptive.beta1();
        double b2 = adaptive.beta2();
        double eps = adaptive.epsilon();
        double decay = adaptive.weightDecay();
        double correction1 = 1.0d - Math.pow(b1, step);
        double correction2 = 1.0d - Math.pow(b2, step);
        requirePositiveFinite(correction1, "first bias correction");
        requirePositiveFinite(correction2, "second bias correction");
        double decayFactor = 1.0d;
        if (adaptive.decoupled()) {
            double decayScale = lr * decay;
            requireFinite(decayScale, "decay scale", path, -1);
            decayFactor -= decayScale;
            requireFinite(decayFactor, "decay factor", path, -1);
        }
        double[] selected = useAccumulatedCandidate ? accumulatedCandidate64 : gradient64;
        if (adaptive.decoupled()) {
            for (int index = 0; index < elementCount; index++) {
                double parameterValue = readParameter64ForUpdate(index);
                double base = parameterValue * decayFactor;
                requireFinite(base, "decayed parameter", path, index);
                stageAdaptiveElement64(index, selected[index], base, b1, b2, correction1,
                        correction2, eps, lr);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                double parameterValue = readParameter64ForUpdate(index);
                double decayTerm = decay * parameterValue;
                requireFinite(decayTerm, "coupled decay term", path, index);
                double gradient = selected[index] + decayTerm;
                requireFinite(gradient, "adjusted gradient", path, index);
                stageAdaptiveElement64(index, gradient, parameterValue, b1, b2, correction1,
                        correction2, eps, lr);
            }
        }
    }

    private double readParameter64ForUpdate(int index) {
        double parameterValue = segment.get(ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES);
        requireFinite(parameterValue, "current parameter", path, index);
        parameterRollback64[index] = parameterValue;
        return parameterValue;
    }

    private void stageAdaptiveElement64(int index, double gradient, double base, double b1,
            double b2, double correction1, double correction2, double eps, double lr) {
        double firstTerm = b1 * momentum64[index];
        requireFinite(firstTerm, "first moment history", path, index);
        double firstGradient = (1.0d - b1) * gradient;
        requireFinite(firstGradient, "first moment gradient", path, index);
        double first = firstTerm + firstGradient;
        requireFinite(first, "first moment candidate", path, index);
        double squared = gradient * gradient;
        requireFinite(squared, "squared gradient", path, index);
        double secondTerm = b2 * secondMoment64[index];
        requireFinite(secondTerm, "second moment history", path, index);
        double secondGradient = (1.0d - b2) * squared;
        requireFinite(secondGradient, "second moment gradient", path, index);
        double second = secondTerm + secondGradient;
        requireFinite(second, "second moment candidate", path, index);
        double correctedFirst = first / correction1;
        double correctedSecond = second / correction2;
        requireFinite(correctedFirst, "corrected first moment", path, index);
        requireFinite(correctedSecond, "corrected second moment", path, index);
        double root = Math.sqrt(correctedSecond);
        requireFinite(root, "second moment root", path, index);
        double denominator = root + eps;
        requirePositiveFinite(denominator, "adaptive denominator");
        double numerator = lr * correctedFirst;
        requireFinite(numerator, "scaled adaptive numerator", path, index);
        double update = numerator / denominator;
        requireFinite(update, "adaptive update", path, index);
        double candidate = base - update;
        requireFinite(candidate, "parameter candidate", path, index);
        momentumCandidate64[index] = first;
        secondMomentCandidate64[index] = second;
        parameterCandidate64[index] = candidate;
    }

    private void requirePositiveFinite(float value, String role) {
        requireFinite(value, role, path, -1);
        if (value <= 0.0f) {
            throw new ArithmeticException(role + " is not positive at path " + path);
        }
    }

    private void requirePositiveFinite(double value, String role) {
        requireFinite(value, role, path, -1);
        if (value <= 0.0d) {
            throw new ArithmeticException(role + " is not positive at path " + path);
        }
    }

    private void stageUpdate32(
            Sgd optimizer, long completedSteps, boolean useAccumulatedCandidate,
            boolean configuredMomentum) {
        float learningRate = (float) optimizer.learningRate();
        float momentum = (float) optimizer.momentum();
        float dampening = (float) optimizer.dampening();
        float weightDecay = (float) optimizer.weightDecay();
        float[] selected = useAccumulatedCandidate ? accumulatedCandidate32 : gradient32;
        boolean firstMomentumStep = completedSteps == 0;
        for (int index = 0; index < elementCount; index++) {
            float parameterValue;
            if (dataType == DataType.FLOAT32) {
                parameterValue = segment.get(
                        ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES);
                parameterRollback32[index] = parameterValue;
            } else {
                short logicalBits = segment.get(
                        ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
                requireFinite(
                        decodeLow(logicalBits), "current logical parameter", path, index);
                parameterRollback16[index] = logicalBits;
                parameterValue = master32[index];
                masterRollback32[index] = parameterValue;
            }
            requireFinite(parameterValue, "current parameter", path, index);
            float decayTerm = weightDecay * parameterValue;
            requireFinite(decayTerm, "weight-decay term", path, index);
            float adjusted = selected[index] + decayTerm;
            requireFinite(adjusted, "adjusted gradient", path, index);
            float selectedUpdate = adjusted;
            if (momentum != 0.0f) {
                float updatedMomentum;
                if (firstMomentumStep) {
                    updatedMomentum = adjusted;
                } else {
                    float momentumTerm = momentum * momentum32[index];
                    requireFinite(momentumTerm, "momentum term", path, index);
                    float dampenedGradient = (1.0f - dampening) * adjusted;
                    requireFinite(dampenedGradient, "dampened gradient", path, index);
                    updatedMomentum = momentumTerm + dampenedGradient;
                }
                requireFinite(updatedMomentum, "momentum candidate", path, index);
                momentumCandidate32[index] = updatedMomentum;
                if (optimizer.nesterov()) {
                    float lookAhead = momentum * updatedMomentum;
                    requireFinite(lookAhead, "Nesterov look-ahead", path, index);
                    selectedUpdate = adjusted + lookAhead;
                    requireFinite(selectedUpdate, "Nesterov update", path, index);
                } else {
                    selectedUpdate = updatedMomentum;
                }
            } else if (configuredMomentum) {
                momentumCandidate32[index] = 0.0f;
            }
            float scaledUpdate = learningRate * selectedUpdate;
            requireFinite(scaledUpdate, "scaled update", path, index);
            float candidate = parameterValue - scaledUpdate;
            requireFinite(candidate, "parameter candidate", path, index);
            if (dataType == DataType.FLOAT32) {
                parameterCandidate32[index] = candidate;
            } else {
                masterCandidate32[index] = candidate;
            }
        }
    }

    private void stageUpdate64(
            Sgd optimizer, long completedSteps, boolean useAccumulatedCandidate) {
        double learningRate = optimizer.learningRate();
        double momentum = optimizer.momentum();
        double dampening = optimizer.dampening();
        double weightDecay = optimizer.weightDecay();
        double[] selected = useAccumulatedCandidate ? accumulatedCandidate64 : gradient64;
        boolean firstMomentumStep = completedSteps == 0;
        for (int index = 0; index < elementCount; index++) {
            double parameterValue = segment.get(
                    ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES);
            requireFinite(parameterValue, "current parameter", path, index);
            parameterRollback64[index] = parameterValue;
            double decayTerm = weightDecay * parameterValue;
            requireFinite(decayTerm, "weight-decay term", path, index);
            double adjusted = selected[index] + decayTerm;
            requireFinite(adjusted, "adjusted gradient", path, index);
            double selectedUpdate = adjusted;
            if (momentum != 0.0d) {
                double updatedMomentum;
                if (firstMomentumStep) {
                    updatedMomentum = adjusted;
                } else {
                    double momentumTerm = momentum * momentum64[index];
                    requireFinite(momentumTerm, "momentum term", path, index);
                    double dampenedGradient = (1.0d - dampening) * adjusted;
                    requireFinite(dampenedGradient, "dampened gradient", path, index);
                    updatedMomentum = momentumTerm + dampenedGradient;
                }
                requireFinite(updatedMomentum, "momentum candidate", path, index);
                momentumCandidate64[index] = updatedMomentum;
                if (optimizer.nesterov()) {
                    double lookAhead = momentum * updatedMomentum;
                    requireFinite(lookAhead, "Nesterov look-ahead", path, index);
                    selectedUpdate = adjusted + lookAhead;
                    requireFinite(selectedUpdate, "Nesterov update", path, index);
                } else {
                    selectedUpdate = updatedMomentum;
                }
            }
            double scaledUpdate = learningRate * selectedUpdate;
            requireFinite(scaledUpdate, "scaled update", path, index);
            double candidate = parameterValue - scaledUpdate;
            requireFinite(candidate, "parameter candidate", path, index);
            parameterCandidate64[index] = candidate;
        }
    }

    void prepareParameterCandidate() {
        if (dataType == DataType.FLOAT16 || dataType == DataType.BFLOAT16) {
            for (int index = 0; index < elementCount; index++) {
                short narrowed = encodeLow(masterCandidate32[index]);
                requireFinite(
                        decodeLow(narrowed), "logical parameter candidate", path, index);
                parameterCandidate16[index] = narrowed;
            }
        }
    }

    void writeParameterCandidate() {
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_DOUBLE,
                        (long) index * Double.BYTES,
                        parameterCandidate64[index]);
            }
        } else if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_FLOAT,
                        (long) index * Float.BYTES,
                        parameterCandidate32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_SHORT,
                        (long) index * Short.BYTES,
                        parameterCandidate16[index]);
            }
            System.arraycopy(masterCandidate32, 0, master32, 0, elementCount);
        }
    }

    void rollbackParameter() {
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_DOUBLE,
                        (long) index * Double.BYTES,
                        parameterRollback64[index]);
            }
        } else if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_FLOAT,
                        (long) index * Float.BYTES,
                        parameterRollback32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_SHORT,
                        (long) index * Short.BYTES,
                        parameterRollback16[index]);
            }
            System.arraycopy(masterRollback32, 0, master32, 0, elementCount);
        }
    }

    /**
     * Publishes staged first and optional second optimizer slots after parameter writes, without
     * allocation. Exactly zero configured SGD momentum retains its established normalization
     * behavior for restored signed zero.
     */
    void commitOptimizerSlots() {
        if (adaptive != null) {
            swapMomentum();
            swapSecondMoment();
            return;
        }
        boolean configuredMomentum = ((Sgd) optimizer).momentum() != 0.0d;
        if (configuredMomentum) {
            swapMomentum();
        } else if (momentumNeedsZeroNormalization) {
            if (dataType != DataType.FLOAT64) {
                Arrays.fill(momentum32, 0.0f);
            } else {
                Arrays.fill(momentum64, 0.0d);
            }
            momentumNeedsZeroNormalization = false;
        }
    }

    /** Publishes the fully staged sum, retaining the former active slot for reuse. */
    void commitAccumulation() {
        if (dataType != DataType.FLOAT64) {
            float[] previous = accumulated32;
            accumulated32 = accumulatedCandidate32;
            accumulatedCandidate32 = previous;
        } else {
            double[] previous = accumulated64;
            accumulated64 = accumulatedCandidate64;
            accumulatedCandidate64 = previous;
        }
    }

    /** Clears only the active sum and releases inactive staging after a successful clear. */
    void clearAccumulation() {
        if (dataType != DataType.FLOAT64) {
            Arrays.fill(accumulated32, 0.0f);
            accumulatedCandidate32 = null;
        } else {
            Arrays.fill(accumulated64, 0.0d);
            accumulatedCandidate64 = null;
        }
    }

    void validateFiniteState() {
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                requireFiniteState(
                        segment.get(ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES),
                        "current parameter", index);
                requireFiniteState(momentum64[index], "momentum slot", index);
                if (adaptive != null) {
                    requireFiniteState(secondMoment64[index], "second moment slot", index);
                    if (secondMoment64[index] < 0.0d) {
                        throw new IllegalStateException("negative second moment at path " + path);
                    }
                }
                requireFiniteState(accumulated64[index], "accumulated gradient", index);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                if (dataType == DataType.FLOAT32) {
                    requireFiniteState(
                            segment.get(ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES),
                            "current parameter", index);
                } else {
                    short logicalBits = segment.get(
                            ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
                    requireFiniteState(
                            decodeLow(logicalBits), "current logical parameter", index);
                    requireFiniteState(master32[index], "master parameter", index);
                }
                requireFiniteState(momentum32[index], "momentum slot", index);
                if (adaptive != null) {
                    requireFiniteState(secondMoment32[index], "second moment slot", index);
                    if (secondMoment32[index] < 0.0f) {
                        throw new IllegalStateException("negative second moment at path " + path);
                    }
                }
                requireFiniteState(accumulated32[index], "accumulated gradient", index);
            }
        }
    }

    /**
     * Captures a detached canonical state from this validated session-owned parameter binding.
     *
     * <p>The fresh arrays are filled here and handed off exclusively to the resulting state;
     * neither this binding nor the session retains a mutable reference to them.</p>
     *
     * @return a non-null immutable state with exact logical and optimizer-width payloads
     * @throws IllegalStateException if the binding changed or current state is non-finite
     */
    TrainingState.ParameterState snapshot() {
        validateBinding();
        validateFiniteState();
        byte[] parameterBytes = new byte[Math.toIntExact(byteSize())];
        byte[] masterBytes = new byte[Math.toIntExact(masterByteSize())];
        byte[] momentumBytes = new byte[masterBytes.length];
        byte[] secondBytes = adaptive == null ? null : new byte[masterBytes.length];
        byte[] accumulatedBytes = new byte[masterBytes.length];
        ByteBuffer parameterBuffer = ByteBuffer.wrap(parameterBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer masterBuffer = ByteBuffer.wrap(masterBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBuffer = ByteBuffer.wrap(momentumBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer secondBuffer = secondBytes == null ? null
                : ByteBuffer.wrap(secondBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBuffer = ByteBuffer.wrap(accumulatedBytes).order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                double value = segment.get(
                        ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES);
                parameterBuffer.putDouble(value);
                masterBuffer.putDouble(value);
                momentumBuffer.putDouble(momentum64[index]);
                if (secondBuffer != null) secondBuffer.putDouble(secondMoment64[index]);
                accumulatedBuffer.putDouble(accumulated64[index]);
            }
        } else if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                float value = segment.get(
                        ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES);
                parameterBuffer.putFloat(value);
                masterBuffer.putFloat(value);
                momentumBuffer.putFloat(momentum32[index]);
                if (secondBuffer != null) secondBuffer.putFloat(secondMoment32[index]);
                accumulatedBuffer.putFloat(accumulated32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                parameterBuffer.putShort(segment.get(
                        ValueLayout.JAVA_SHORT, (long) index * Short.BYTES));
                masterBuffer.putFloat(master32[index]);
                momentumBuffer.putFloat(momentum32[index]);
                if (secondBuffer != null) secondBuffer.putFloat(secondMoment32[index]);
                accumulatedBuffer.putFloat(accumulated32[index]);
            }
        }
        return secondBytes == null
                ? TrainingState.ParameterState.adoptFreshSnapshotArrays(path, dataType, shape,
                        parameterBytes, masterBytes, momentumBytes, accumulatedBytes)
                : TrainingState.ParameterState.adoptFreshSnapshotArrays(path, dataType, shape,
                        parameterBytes, masterBytes, momentumBytes, secondBytes, accumulatedBytes);
    }

    /**
     * Validates this path's detached payload and assigned slot schema before restore.
     *
     * @param state non-null detached payload expected to match this path and schema
     * @throws IllegalArgumentException if schema, width, finiteness, second-moment sign, or
     *     disabled-momentum slot
     *     values are incompatible
     */
    void validateState(TrainingState.ParameterState state) {
        Objects.requireNonNull(state, "state");
        if (!path.equals(state.path())
                || dataType != state.dataType()
                || !shape.equals(state.shape())) {
            throw new IllegalArgumentException(
                    "training state parameter schema mismatch at path " + path);
        }
        int expectedLogicalBytes = Math.toIntExact(byteSize());
        int expectedMasterBytes = Math.toIntExact(masterByteSize());
        if (state.parameterBytesInternal().length != expectedLogicalBytes
                || state.masterParameterBytesInternal().length != expectedMasterBytes
                || state.momentumBytesInternal().length != expectedMasterBytes
                || (adaptive == null ? state.secondMomentBytesInternal() != null
                        : state.secondMomentBytesInternal() == null
                                || state.secondMomentBytesInternal().length != expectedMasterBytes)
                || state.accumulatedGradientBytesInternal().length != expectedMasterBytes) {
            throw new IllegalArgumentException(
                    "training state payload size mismatch at path " + path);
        }
        if (adaptive == null && ((Sgd) optimizer).momentum() == 0.0d
                && containsNonZeroMomentum(state)) {
            throw new IllegalArgumentException(
                    "momentum-disabled training state has a non-zero slot at path " + path);
        }
        validateFinitePayload(state);
    }

    /**
     * Stages all restored slots and parameter bytes before any installation.
     *
     * @param state non-null, fully validated detached state with this parameter's schema
     */
    void stageRestore(TrainingState.ParameterState state) {
        ensureAccumulationCandidate();
        ensureMomentumCandidate();
        if (adaptive != null) ensureSecondMomentCandidate();
        stagedMomentumHasNegativeZero = false;
        ByteBuffer parameterBuffer = ByteBuffer.wrap(state.parameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer masterBuffer = ByteBuffer.wrap(state.masterParameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBuffer = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer secondBuffer = adaptive == null ? null
                : ByteBuffer.wrap(state.secondMomentBytesInternal()).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBuffer = ByteBuffer.wrap(state.accumulatedGradientBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                parameterRollback64[index] = segment.get(
                        ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES);
                parameterCandidate64[index] = parameterBuffer.getDouble();
                masterBuffer.getDouble();
                double restoredMomentum = momentumBuffer.getDouble();
                momentumCandidate64[index] = restoredMomentum;
                if (secondBuffer != null) secondMomentCandidate64[index] = secondBuffer.getDouble();
                stagedMomentumHasNegativeZero |= Double.doubleToRawLongBits(restoredMomentum)
                        == Long.MIN_VALUE;
                accumulatedCandidate64[index] = accumulatedBuffer.getDouble();
            }
        } else if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                parameterRollback32[index] = segment.get(
                        ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES);
                parameterCandidate32[index] = parameterBuffer.getFloat();
                masterBuffer.getFloat();
                float restoredMomentum = momentumBuffer.getFloat();
                momentumCandidate32[index] = restoredMomentum;
                if (secondBuffer != null) secondMomentCandidate32[index] = secondBuffer.getFloat();
                stagedMomentumHasNegativeZero |= Float.floatToRawIntBits(restoredMomentum)
                        == Integer.MIN_VALUE;
                accumulatedCandidate32[index] = accumulatedBuffer.getFloat();
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                parameterRollback16[index] = segment.get(
                        ValueLayout.JAVA_SHORT, (long) index * Short.BYTES);
                masterRollback32[index] = master32[index];
                parameterCandidate16[index] = parameterBuffer.getShort();
                masterCandidate32[index] = masterBuffer.getFloat();
                float restoredMomentum = momentumBuffer.getFloat();
                momentumCandidate32[index] = restoredMomentum;
                if (secondBuffer != null) secondMomentCandidate32[index] = secondBuffer.getFloat();
                stagedMomentumHasNegativeZero |= Float.floatToRawIntBits(restoredMomentum)
                        == Integer.MIN_VALUE;
                accumulatedCandidate32[index] = accumulatedBuffer.getFloat();
            }
        }
    }

    /**
     * Publishes pre-staged optimizer slots after parameter writes, without allocation.
     *
     * <p>Restore needs both staging arrays before commit, but releases the former active
     * accumulation array when the restored sum becomes active. The former active momentum array
     * is retained for adaptive or nonzero configured SGD momentum and released for zero SGD
     * momentum. The adaptive second-moment staging array is retained for reuse.
     * Later staging reallocates released arrays as needed.</p>
     *
     */
    void commitRestore() {
        boolean configuredMomentum = adaptive != null
                || ((Sgd) optimizer).momentum() != 0.0d;
        swapMomentum();
        if (adaptive != null) swapSecondMoment();
        momentumNeedsZeroNormalization = !configuredMomentum && stagedMomentumHasNegativeZero;
        if (!configuredMomentum) {
            if (dataType != DataType.FLOAT64) {
                momentumCandidate32 = null;
            } else {
                momentumCandidate64 = null;
            }
        }
        commitAccumulation();
        if (dataType != DataType.FLOAT64) {
            accumulatedCandidate32 = null;
        } else {
            accumulatedCandidate64 = null;
        }
    }

    /**
     * Releases restore staging after a failed installation without changing active slots.
     *
     */
    void discardFailedRestoreStaging() {
        boolean configuredMomentum = adaptive != null
                || ((Sgd) optimizer).momentum() != 0.0d;
        if (dataType != DataType.FLOAT64) {
            accumulatedCandidate32 = null;
            if (!configuredMomentum) {
                momentumCandidate32 = null;
            }
        } else {
            accumulatedCandidate64 = null;
            if (!configuredMomentum) {
                momentumCandidate64 = null;
            }
        }
        stagedMomentumHasNegativeZero = false;
    }

    private void ensureAccumulationCandidate() {
        if (dataType != DataType.FLOAT64) {
            if (accumulatedCandidate32 == null) {
                accumulatedCandidate32 = new float[elementCount];
            }
        } else if (accumulatedCandidate64 == null) {
            accumulatedCandidate64 = new double[elementCount];
        }
    }

    private void ensureMomentumCandidate() {
        if (dataType != DataType.FLOAT64) {
            if (momentumCandidate32 == null) {
                momentumCandidate32 = new float[elementCount];
            }
        } else if (momentumCandidate64 == null) {
            momentumCandidate64 = new double[elementCount];
        }
    }

    private void ensureSecondMomentCandidate() {
        if (dataType == DataType.FLOAT64) {
            if (secondMomentCandidate64 == null) secondMomentCandidate64 = new double[elementCount];
        } else if (secondMomentCandidate32 == null) {
            secondMomentCandidate32 = new float[elementCount];
        }
    }

    private void swapSecondMoment() {
        if (dataType == DataType.FLOAT64) {
            double[] previous = secondMoment64;
            secondMoment64 = secondMomentCandidate64;
            secondMomentCandidate64 = previous;
        } else {
            float[] previous = secondMoment32;
            secondMoment32 = secondMomentCandidate32;
            secondMomentCandidate32 = previous;
        }
    }

    private void swapMomentum() {
        if (dataType != DataType.FLOAT64) {
            float[] previous = momentum32;
            momentum32 = momentumCandidate32;
            momentumCandidate32 = previous;
        } else {
            double[] previous = momentum64;
            momentum64 = momentumCandidate64;
            momentumCandidate64 = previous;
        }
    }

    private void validateFinitePayload(TrainingState.ParameterState state) {
        ByteBuffer parameterBytes = ByteBuffer.wrap(state.parameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer masterBytes = ByteBuffer.wrap(state.masterParameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBytes = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer secondBytes = adaptive == null ? null
                : ByteBuffer.wrap(state.secondMomentBytesInternal()).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBytes = ByteBuffer.wrap(state.accumulatedGradientBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        for (int index = 0; index < elementCount; index++) {
            if (dataType == DataType.FLOAT64) {
                requireFinitePayload(parameterBytes.getDouble(), "parameter", index);
                requireFinitePayload(masterBytes.getDouble(), "master parameter", index);
                requireFinitePayload(momentumBytes.getDouble(), "momentum", index);
                if (secondBytes != null) requireSecondMoment(secondBytes.getDouble(), index);
                requireFinitePayload(accumulatedBytes.getDouble(), "accumulated gradient", index);
            } else if (dataType == DataType.FLOAT32) {
                requireFinitePayload(parameterBytes.getFloat(), "parameter", index);
                requireFinitePayload(masterBytes.getFloat(), "master parameter", index);
                requireFinitePayload(momentumBytes.getFloat(), "momentum", index);
                if (secondBytes != null) requireSecondMoment(secondBytes.getFloat(), index);
                requireFinitePayload(accumulatedBytes.getFloat(), "accumulated gradient", index);
            } else {
                requireFinitePayload(
                        decodeLow(parameterBytes.getShort()), "logical parameter", index);
                requireFinitePayload(masterBytes.getFloat(), "master parameter", index);
                requireFinitePayload(momentumBytes.getFloat(), "momentum", index);
                if (secondBytes != null) requireSecondMoment(secondBytes.getFloat(), index);
                requireFinitePayload(accumulatedBytes.getFloat(), "accumulated gradient", index);
            }
        }
    }

    private void requireSecondMoment(double value, int index) {
        requireFinitePayload(value, "second moment", index);
        if (value < 0.0d) {
            throw new IllegalArgumentException("negative second moment at path " + path
                    + " index " + index);
        }
    }

    private void requireSecondMoment(float value, int index) {
        requireFinitePayload(value, "second moment", index);
        if (value < 0.0f) {
            throw new IllegalArgumentException("negative second moment at path " + path
                    + " index " + index);
        }
    }

    private boolean containsNonZeroMomentum(TrainingState.ParameterState state) {
        ByteBuffer bytes = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT64) {
            for (int index = 0; index < elementCount; index++) {
                if (bytes.getDouble() != 0.0d) {
                    return true;
                }
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                if (bytes.getFloat() != 0.0f) {
                    return true;
                }
            }
        }
        return false;
    }

    private float decodeLow(short bits) {
        return dataType == DataType.BFLOAT16
                ? BFloat16Bits.toFloat(bits)
                : Float16Bits.toFloat(bits);
    }

    private short encodeLow(float value) {
        return dataType == DataType.BFLOAT16
                ? BFloat16Bits.fromFloat(value)
                : Float16Bits.fromFloat(value);
    }

    /**
     * Rejects a non-finite represented value without changing the source bytes.
     *
     * @param path non-null diagnostic parameter path
     * @param segment non-null, live, accessible exact parameter region; alignment is not required
     * @param dataType non-null supported floating parameter type
     * @param elementCount positive element count bounded by the Java array limit
     * @throws IllegalArgumentException if a represented parameter value is NaN or infinite
     */
    static void validateFiniteStorage(
            String path, MemorySegment segment, DataType dataType, int elementCount) {
        for (int index = 0; index < elementCount; index++) {
            boolean finite;
            if (dataType == DataType.FLOAT64) {
                finite = Double.isFinite(segment.get(
                        ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) index * Double.BYTES));
            } else if (dataType == DataType.FLOAT32) {
                finite = Float.isFinite(segment.get(
                        ValueLayout.JAVA_FLOAT_UNALIGNED, (long) index * Float.BYTES));
            } else {
                short bits = segment.get(
                        ValueLayout.JAVA_SHORT_UNALIGNED, (long) index * Short.BYTES);
                float value = dataType == DataType.BFLOAT16
                        ? BFloat16Bits.toFloat(bits)
                        : Float16Bits.toFloat(bits);
                finite = Float.isFinite(value);
            }
            if (!finite) {
                throw new IllegalArgumentException(
                        "training parameter must be finite at path " + path
                                + ", element " + index);
            }
        }
    }

    private static void requireFinite(float value, String role, String path, int index) {
        if (!Float.isFinite(value)) {
            throw new ArithmeticException(
                    "non-finite " + role + " at path " + path + ", element " + index);
        }
    }

    private static void requireFinite(double value, String role, String path, int index) {
        if (!Double.isFinite(value)) {
            throw new ArithmeticException(
                    "non-finite " + role + " at path " + path + ", element " + index);
        }
    }

    private void requireFiniteState(float value, String role, int index) {
        if (!Float.isFinite(value)) {
            throw new IllegalStateException(
                    "non-finite " + role + " at path " + path + ", element " + index);
        }
    }

    private void requireFiniteState(double value, String role, int index) {
        if (!Double.isFinite(value)) {
            throw new IllegalStateException(
                    "non-finite " + role + " at path " + path + ", element " + index);
        }
    }

    private void requireFinitePayload(float value, String role, int index) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    "training state contains non-finite " + role + " at path " + path
                            + ", element " + index);
        }
    }

    private void requireFinitePayload(double value, String role, int index) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "training state contains non-finite " + role + " at path " + path
                            + ", element " + index);
        }
    }

    static void validateStorage(
            String role,
            HostTensorStorage storage,
            DataType expectedType,
            long expectedElements,
            boolean writable,
            boolean requireNative) {
        Objects.requireNonNull(storage, "storage");
        if (storage.dataType() != expectedType) {
            throw new IllegalArgumentException(
                    role + " storage data type mismatch: expected=" + expectedType
                            + ", actual=" + storage.dataType());
        }
        if (storage.elementCapacity() != expectedElements) {
            throw new IllegalArgumentException(
                    role + " storage capacity mismatch: expected=" + expectedElements
                            + ", actual=" + storage.elementCapacity());
        }
        long expectedBytes = Math.multiplyExact(expectedElements, expectedType.byteWidth());
        if (storage.byteSize() != expectedBytes
                || storage.segment().byteSize() != expectedBytes) {
            throw new IllegalArgumentException(role + " storage byte size mismatch");
        }
        if (writable && storage.isReadOnly()) {
            throw new IllegalArgumentException(role + " storage must be writable");
        }
        if (!storage.isAlive()) {
            throw new IllegalStateException(role + " storage is not alive");
        }
        if (!storage.segment().isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException(role + " storage is not accessible to current thread");
        }
        if (requireNative && storage.segment().heapBase().isPresent()) {
            throw new IllegalArgumentException(role + " storage must use native memory");
        }
        if (Math.floorMod(storage.segment().address(), expectedType.byteWidth()) != 0) {
            throw new IllegalArgumentException(role + " storage is not element aligned");
        }
    }
}
