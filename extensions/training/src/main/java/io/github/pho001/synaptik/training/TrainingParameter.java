package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/** Session-private stable parameter binding and reusable primitive update buffers. */
final class TrainingParameter {
    private final String path;
    private final Parameter parameter;
    private final Tensor tensor;
    private final HostTensorStorage storage;
    private final MemorySegment segment;
    private final DataType dataType;
    private final Shape shape;
    private final int elementCount;

    private float[] gradient32;
    private float[] accumulated32;
    private float[] accumulatedCandidate32;
    private float[] parameterCandidate32;
    private float[] parameterRollback32;
    private float[] momentum32;
    private float[] momentumCandidate32;

    private double[] gradient64;
    private double[] accumulated64;
    private double[] accumulatedCandidate64;
    private double[] parameterCandidate64;
    private double[] parameterRollback64;
    private double[] momentum64;
    private double[] momentumCandidate64;

    private TrainingParameter(
            String path,
            Parameter parameter,
            Tensor tensor,
            HostTensorStorage storage,
            int elementCount) {
        this.path = path;
        this.parameter = parameter;
        this.tensor = tensor;
        this.storage = storage;
        this.segment = storage.segment();
        this.dataType = tensor.descriptor().dataType();
        this.shape = tensor.descriptor().shape();
        this.elementCount = elementCount;
        if (dataType == DataType.FLOAT32) {
            gradient32 = new float[elementCount];
            accumulated32 = new float[elementCount];
            accumulatedCandidate32 = new float[elementCount];
            parameterCandidate32 = new float[elementCount];
            parameterRollback32 = new float[elementCount];
            momentum32 = new float[elementCount];
            momentumCandidate32 = new float[elementCount];
        } else {
            gradient64 = new double[elementCount];
            accumulated64 = new double[elementCount];
            accumulatedCandidate64 = new double[elementCount];
            parameterCandidate64 = new double[elementCount];
            parameterRollback64 = new double[elementCount];
            momentum64 = new double[elementCount];
            momentumCandidate64 = new double[elementCount];
        }
    }

    static TrainingParameter capture(String path, Parameter parameter) {
        String stablePath = Objects.requireNonNull(path, "path");
        if (stablePath.isBlank()) {
            throw new IllegalArgumentException("parameter path must not be blank");
        }
        Parameter stableParameter = Objects.requireNonNull(parameter, "parameter");
        Tensor stableTensor = Objects.requireNonNull(stableParameter.value(), "parameter.value()");
        var descriptor = stableTensor.descriptor();
        DataType type = descriptor.dataType();
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64) {
            throw new IllegalArgumentException(
                    "training parameter must use FLOAT32 or FLOAT64 at path "
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
                stablePath, stableParameter, stableTensor, stableStorage, (int) count);
    }

    String path() {
        return path;
    }

    Parameter parameter() {
        return parameter;
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

    void validateBinding() {
        if (parameter.value() != tensor) {
            throw new IllegalStateException(
                    "parameter Tensor binding changed while training session is open: " + path);
        }
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
        if (value.dataType() != dataType || !value.shape().equals(shape)) {
            throw new IllegalStateException(
                    "gradient schema does not match parameter at path " + path);
        }
        if (value.elementCount() != elementCount || value.byteSize() != byteSize()) {
            throw new IllegalStateException(
                    "gradient payload size does not match parameter at path " + path);
        }
        ByteBuffer bytes = value.bytes().order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                float gradient = bytes.getFloat();
                requireFinite(gradient, "decoded gradient", path, index);
                gradient32[index] = gradient;
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                double gradient = bytes.getDouble();
                requireFinite(gradient, "decoded gradient", path, index);
                gradient64[index] = gradient;
            }
        }
        if (bytes.hasRemaining()) {
            throw new IllegalStateException(
                    "gradient payload contains trailing bytes at path " + path);
        }
    }

    void stageAccumulation() {
        if (dataType == DataType.FLOAT32) {
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

    void stageUpdate(Sgd optimizer, long completedSteps, boolean useAccumulatedCandidate) {
        if (dataType == DataType.FLOAT32) {
            stageUpdate32(optimizer, completedSteps, useAccumulatedCandidate);
        } else {
            stageUpdate64(optimizer, completedSteps, useAccumulatedCandidate);
        }
    }

    private void stageUpdate32(
            Sgd optimizer, long completedSteps, boolean useAccumulatedCandidate) {
        float learningRate = (float) optimizer.learningRate();
        float momentum = (float) optimizer.momentum();
        float dampening = (float) optimizer.dampening();
        float weightDecay = (float) optimizer.weightDecay();
        float[] selected = useAccumulatedCandidate ? accumulatedCandidate32 : gradient32;
        boolean firstMomentumStep = completedSteps == 0;
        for (int index = 0; index < elementCount; index++) {
            long offset = (long) index * Float.BYTES;
            float parameterValue = segment.get(ValueLayout.JAVA_FLOAT, offset);
            requireFinite(parameterValue, "current parameter", path, index);
            parameterRollback32[index] = parameterValue;
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
            } else {
                momentumCandidate32[index] = 0.0f;
            }
            float scaledUpdate = learningRate * selectedUpdate;
            requireFinite(scaledUpdate, "scaled update", path, index);
            float candidate = parameterValue - scaledUpdate;
            requireFinite(candidate, "parameter candidate", path, index);
            parameterCandidate32[index] = candidate;
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
            long offset = (long) index * Double.BYTES;
            double parameterValue = segment.get(ValueLayout.JAVA_DOUBLE, offset);
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
            } else {
                momentumCandidate64[index] = 0.0d;
            }
            double scaledUpdate = learningRate * selectedUpdate;
            requireFinite(scaledUpdate, "scaled update", path, index);
            double candidate = parameterValue - scaledUpdate;
            requireFinite(candidate, "parameter candidate", path, index);
            parameterCandidate64[index] = candidate;
        }
    }

    void writeParameterCandidate() {
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_FLOAT,
                        (long) index * Float.BYTES,
                        parameterCandidate32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_DOUBLE,
                        (long) index * Double.BYTES,
                        parameterCandidate64[index]);
            }
        }
    }

    void rollbackParameter() {
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_FLOAT,
                        (long) index * Float.BYTES,
                        parameterRollback32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                segment.set(
                        ValueLayout.JAVA_DOUBLE,
                        (long) index * Double.BYTES,
                        parameterRollback64[index]);
            }
        }
    }

    void commitMomentum() {
        if (dataType == DataType.FLOAT32) {
            float[] previous = momentum32;
            momentum32 = momentumCandidate32;
            momentumCandidate32 = previous;
        } else {
            double[] previous = momentum64;
            momentum64 = momentumCandidate64;
            momentumCandidate64 = previous;
        }
    }

    void commitAccumulation() {
        if (dataType == DataType.FLOAT32) {
            float[] previous = accumulated32;
            accumulated32 = accumulatedCandidate32;
            accumulatedCandidate32 = previous;
        } else {
            double[] previous = accumulated64;
            accumulated64 = accumulatedCandidate64;
            accumulatedCandidate64 = previous;
        }
    }

    void clearAccumulation() {
        if (dataType == DataType.FLOAT32) {
            Arrays.fill(accumulated32, 0.0f);
            Arrays.fill(accumulatedCandidate32, 0.0f);
        } else {
            Arrays.fill(accumulated64, 0.0d);
            Arrays.fill(accumulatedCandidate64, 0.0d);
        }
    }

    void validateFiniteState() {
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                requireFiniteState(
                        segment.get(ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES),
                        "current parameter", index);
                requireFiniteState(momentum32[index], "momentum slot", index);
                requireFiniteState(accumulated32[index], "accumulated gradient", index);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                requireFiniteState(
                        segment.get(ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES),
                        "current parameter", index);
                requireFiniteState(momentum64[index], "momentum slot", index);
                requireFiniteState(accumulated64[index], "accumulated gradient", index);
            }
        }
    }

    TrainingState.ParameterState snapshot() {
        validateBinding();
        validateFiniteState();
        byte[] parameterBytes = new byte[(int) byteSize()];
        byte[] momentumBytes = new byte[parameterBytes.length];
        byte[] accumulatedBytes = new byte[parameterBytes.length];
        ByteBuffer parameterBuffer = ByteBuffer.wrap(parameterBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBuffer = ByteBuffer.wrap(momentumBytes).order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBuffer = ByteBuffer.wrap(accumulatedBytes).order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                parameterBuffer.putFloat(segment.get(
                        ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES));
                momentumBuffer.putFloat(momentum32[index]);
                accumulatedBuffer.putFloat(accumulated32[index]);
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                parameterBuffer.putDouble(segment.get(
                        ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES));
                momentumBuffer.putDouble(momentum64[index]);
                accumulatedBuffer.putDouble(accumulated64[index]);
            }
        }
        return new TrainingState.ParameterState(
                path, dataType, shape, parameterBytes, momentumBytes, accumulatedBytes);
    }

    void validateState(TrainingState.ParameterState state, Sgd optimizer) {
        Objects.requireNonNull(state, "state");
        if (!path.equals(state.path())
                || dataType != state.dataType()
                || !shape.equals(state.shape())) {
            throw new IllegalArgumentException(
                    "training state parameter schema mismatch at path " + path);
        }
        int expectedBytes = (int) byteSize();
        if (state.parameterBytesInternal().length != expectedBytes
                || state.momentumBytesInternal().length != expectedBytes
                || state.accumulatedGradientBytesInternal().length != expectedBytes) {
            throw new IllegalArgumentException(
                    "training state payload size mismatch at path " + path);
        }
        if (optimizer.momentum() == 0.0d && containsNonZeroMomentum(state)) {
            throw new IllegalArgumentException(
                    "momentum-disabled training state has a non-zero slot at path " + path);
        }
        validateFinitePayload(state);
    }

    void stageRestore(TrainingState.ParameterState state) {
        ByteBuffer parameterBuffer = ByteBuffer.wrap(state.parameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBuffer = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBuffer = ByteBuffer.wrap(state.accumulatedGradientBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                parameterRollback32[index] = segment.get(
                        ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES);
                parameterCandidate32[index] = parameterBuffer.getFloat();
                momentumCandidate32[index] = momentumBuffer.getFloat();
                accumulatedCandidate32[index] = accumulatedBuffer.getFloat();
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                parameterRollback64[index] = segment.get(
                        ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES);
                parameterCandidate64[index] = parameterBuffer.getDouble();
                momentumCandidate64[index] = momentumBuffer.getDouble();
                accumulatedCandidate64[index] = accumulatedBuffer.getDouble();
            }
        }
    }

    void commitRestore() {
        commitMomentum();
        commitAccumulation();
    }

    private void validateFinitePayload(TrainingState.ParameterState state) {
        ByteBuffer parameterBytes = ByteBuffer.wrap(state.parameterBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer momentumBytes = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        ByteBuffer accumulatedBytes = ByteBuffer.wrap(state.accumulatedGradientBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        for (int index = 0; index < elementCount; index++) {
            if (dataType == DataType.FLOAT32) {
                requireFinitePayload(parameterBytes.getFloat(), "parameter", index);
                requireFinitePayload(momentumBytes.getFloat(), "momentum", index);
                requireFinitePayload(accumulatedBytes.getFloat(), "accumulated gradient", index);
            } else {
                requireFinitePayload(parameterBytes.getDouble(), "parameter", index);
                requireFinitePayload(momentumBytes.getDouble(), "momentum", index);
                requireFinitePayload(accumulatedBytes.getDouble(), "accumulated gradient", index);
            }
        }
    }

    private boolean containsNonZeroMomentum(TrainingState.ParameterState state) {
        ByteBuffer bytes = ByteBuffer.wrap(state.momentumBytesInternal())
                .order(ByteOrder.BIG_ENDIAN);
        if (dataType == DataType.FLOAT32) {
            for (int index = 0; index < elementCount; index++) {
                if (bytes.getFloat() != 0.0f) {
                    return true;
                }
            }
        } else {
            for (int index = 0; index < elementCount; index++) {
                if (bytes.getDouble() != 0.0d) {
                    return true;
                }
            }
        }
        return false;
    }

    private static void validateFiniteStorage(
            String path, MemorySegment segment, DataType dataType, int elementCount) {
        for (int index = 0; index < elementCount; index++) {
            boolean finite = dataType == DataType.FLOAT32
                    ? Float.isFinite(segment.get(
                            ValueLayout.JAVA_FLOAT, (long) index * Float.BYTES))
                    : Double.isFinite(segment.get(
                            ValueLayout.JAVA_DOUBLE, (long) index * Double.BYTES));
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
