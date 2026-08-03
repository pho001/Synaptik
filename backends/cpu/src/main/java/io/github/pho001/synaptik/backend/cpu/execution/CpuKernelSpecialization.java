package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodType;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import jdk.incubator.vector.VectorSpecies;

/**
 * Immutable complete description of every fact allowed to affect one generated CPU class. It
 * fixes the ordered direct carrier signature, baked geometry, one of four portable execution
 * modes, Vector species when selected, byte order, loop structure, numerical mode, combine order,
 * lowering identity, schema, and class-file target. Runtime identities, physical addresses,
 * worker configuration, scheduling choices, and cache state are deliberately absent. Equality
 * and the derived fingerprint are structural and order-sensitive.
 */
final class CpuKernelSpecialization {
    /** Exact direct storage carrier selected before generation and exposed in the entry type. */
    enum Carrier {
        /** Direct {@code double[]} entry argument for FLOAT64 elements. */ DOUBLE_ARRAY,
        /** Direct {@code float[]} entry argument for FLOAT32 elements. */ FLOAT_ARRAY,
        /** Direct {@code short[]} entry argument for BFLOAT16 bits. */ SHORT_ARRAY,
        /** Direct {@code int[]} entry argument for INT32 elements. */ INT_ARRAY,
        /** Direct {@code long[]} entry argument for INT64 elements. */ LONG_ARRAY,
        /** Direct {@code byte[]} entry argument for BOOL storage. */ BYTE_ARRAY,
        /** Exact direct {@link MemorySegment} entry argument for any logical data type. */
        MEMORY_SEGMENT
    }
    /** Already-selected Vector-tail structure; scalar modes accept only {@link Tail#NONE}. */
    enum Tail {
        /** No remainder body is emitted. */ NONE,
        /** Remaining elements use a scalar body. */ SCALAR,
        /** Remaining elements use one masked Vector body. */ MASKED
    }
    /** Closed numerical contract available in schema version one. */
    enum NumericalMode {
        /** Current exact/default Model numerical contract with no relaxed-math permission. */
        EXACT_DEFAULT
    }
    /** Structural ordering contract for partial combination; it does not select the mathematics. */
    enum CombineOrder {
        /** Family combination receives partials in their fixed left-to-right order. */ FIXED,
        /** Family combination may receive the two structural operands in either order. */
        UNRESTRICTED
    }

    /**
     * Describes one ordered direct generated-method argument.
     *
     * @param dataType non-null logical element type
     * @param carrier non-null exact direct carrier form matching {@code dataType}
     * @param access non-null permitted generated access direction
     * @param byteOffsetBaked whether the array byte offset is embedded in generated code
     * @param bakedByteOffset non-negative embedded byte offset, or zero when not embedded
     * @param bakedElementStrides non-null ordered non-negative strides; copied defensively
     */
    record Argument(DataType dataType, Carrier carrier, PreparedExecutable.BufferAccess access,
            boolean byteOffsetBaked, long bakedByteOffset, List<Long> bakedElementStrides) {
        /**
         * Validates and snapshots one generated entry argument.
         *
         * @throws NullPointerException if a reference or stride entry is {@code null}
         * @throws IllegalArgumentException if offset state, a stride, carrier compatibility, or
         *     primitive-array alignment violates the specialization contract
         */
        Argument {
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(carrier, "carrier");
            Objects.requireNonNull(access, "access");
            if (!byteOffsetBaked && bakedByteOffset != 0) {
                throw new IllegalArgumentException(
                        "bakedByteOffset must be zero when byteOffsetBaked is false");
            }
            if (bakedByteOffset < 0) {
                throw new IllegalArgumentException("bakedByteOffset must be non-negative");
            }
            Objects.requireNonNull(bakedElementStrides, "bakedElementStrides");
            var copied = new ArrayList<Long>(bakedElementStrides.size());
            for (int index = 0; index < bakedElementStrides.size(); index++) {
                Long stride = Objects.requireNonNull(bakedElementStrides.get(index),
                        "bakedElementStrides[" + index + "]");
                if (stride < 0) throw new IllegalArgumentException(
                        "bakedElementStrides[" + index + "] must be non-negative");
                copied.add(stride);
            }
            bakedElementStrides = List.copyOf(copied);
            if (carrier == Carrier.MEMORY_SEGMENT) {
                if (!byteOffsetBaked || bakedByteOffset != 0) throw new IllegalArgumentException(
                        "MEMORY_SEGMENT requires a baked zero byte offset");
            } else if (carrierFor(dataType) != carrier) {
                throw new IllegalArgumentException("carrier does not match dataType");
            } else if (byteOffsetBaked && bakedByteOffset % dataType.byteWidth() != 0) {
                throw new IllegalArgumentException(
                        "bakedByteOffset must be aligned to dataType byte width");
            }
        }
    }

    /**
     * Identifies one exact supported Java Vector API species.
     *
     * @param laneType FLOAT64, FLOAT32, INT32, or INT64 lane type
     * @param vectorBitSize positive power-of-two supported vector size in bits
     * @param laneCount positive lane count exactly matching type and bit size
     */
    record VectorShape(DataType laneType, int vectorBitSize, int laneCount) {
        /**
         * Validates one exact Java 26 Vector API species shape.
         *
         * @throws NullPointerException if {@code laneType} is {@code null}
         * @throws IllegalArgumentException if the lane type, bit size, or lane count does not name
         *     an exact Java 26 Vector API species
         */
        VectorShape {
            Objects.requireNonNull(laneType, "laneType");
            if (vectorBitSize <= 0 || (vectorBitSize & (vectorBitSize - 1)) != 0) {
                throw new IllegalArgumentException("vectorBitSize must be a positive power of two");
            }
            if (laneCount <= 0) throw new IllegalArgumentException("laneCount must be positive");
            Class<?> laneClass = laneClass(laneType);
            VectorSpecies<?> species;
            try {
                species = VectorSpecies.of(laneClass,
                        jdk.incubator.vector.VectorShape.forBitSize(vectorBitSize));
            } catch (IllegalArgumentException failure) {
                throw new IllegalArgumentException("unsupported vector species", failure);
            }
            if (species.length() != laneCount || species.vectorBitSize() != vectorBitSize) {
                throw new IllegalArgumentException("laneCount does not match vector species");
            }
        }
    }

    private final int schemaVersion;
    private final CpuLoweringFingerprint loweringFingerprint;
    private final CpuPortableExecutionMode executionMode;
    private final List<Argument> arguments;
    private final List<Long> bakedExtents;
    private final int dynamicExtentCount;
    private final VectorShape vectorShape;
    private final ByteOrder byteOrder;
    private final int unrollFactor;
    private final long tileElementCount;
    private final Tail tail;
    private final NumericalMode numericalMode;
    private final CombineOrder combineOrder;
    private final int classFileMajorVersion;
    private final MethodType entryType;
    private final CpuLoweringFingerprint specializationFingerprint;

    /**
     * Validates and snapshots one complete bytecode specialization.
     *
     * @param schemaVersion must be {@link CpuGeneratorSchema#CURRENT_VERSION}
     * @param loweringFingerprint non-null family-owned lowering identity
     * @param executionMode non-null exact portable execution mode
     * @param arguments non-null ordered direct arguments; copied deeply
     * @param bakedExtents non-null ordered non-negative embedded extents; copied
     * @param dynamicExtentCount non-negative number of ordered runtime extents
     * @param vectorShape exact species for vector modes, or {@code null} for scalar modes
     * @param byteOrder non-null exact big- or little-endian segment order
     * @param unrollFactor positive embedded unroll factor
     * @param tileElementCount positive embedded tile size in elements
     * @param tail non-null selected tail structure
     * @param numericalMode non-null numerical contract
     * @param combineOrder non-null partial-combine structure
     * @param classFileMajorVersion Java 26 class-file major version
     * @throws NullPointerException if a required reference or list entry is null
     * @throws IllegalArgumentException if any value violates the schema-one contract
     */
    CpuKernelSpecialization(int schemaVersion, CpuLoweringFingerprint loweringFingerprint,
            CpuPortableExecutionMode executionMode, List<Argument> arguments,
            List<Long> bakedExtents, int dynamicExtentCount, VectorShape vectorShape,
            ByteOrder byteOrder, int unrollFactor, long tileElementCount, Tail tail,
            NumericalMode numericalMode, CombineOrder combineOrder, int classFileMajorVersion) {
        if (schemaVersion != CpuGeneratorSchema.CURRENT_VERSION) {
            throw new IllegalArgumentException("schemaVersion must equal CURRENT_VERSION");
        }
        this.schemaVersion = schemaVersion;
        this.loweringFingerprint = Objects.requireNonNull(loweringFingerprint, "loweringFingerprint");
        this.executionMode = Objects.requireNonNull(executionMode, "executionMode");
        Objects.requireNonNull(arguments, "arguments");
        var copiedArguments = new ArrayList<Argument>(arguments.size());
        for (int index = 0; index < arguments.size(); index++) copiedArguments.add(
                Objects.requireNonNull(arguments.get(index), "arguments[" + index + "]"));
        this.arguments = List.copyOf(copiedArguments);
        Objects.requireNonNull(bakedExtents, "bakedExtents");
        var copiedExtents = new ArrayList<Long>(bakedExtents.size());
        for (int index = 0; index < bakedExtents.size(); index++) {
            Long extent = Objects.requireNonNull(bakedExtents.get(index),
                    "bakedExtents[" + index + "]");
            if (extent < 0) throw new IllegalArgumentException(
                    "bakedExtents[" + index + "] must be non-negative");
            copiedExtents.add(extent);
        }
        this.bakedExtents = List.copyOf(copiedExtents);
        if (dynamicExtentCount < 0) throw new IllegalArgumentException(
                "dynamicExtentCount must be non-negative");
        this.dynamicExtentCount = dynamicExtentCount;
        if (executionMode.vectorized() && vectorShape == null) throw new IllegalArgumentException(
                "vectorShape is required for vector execution mode");
        if (!executionMode.vectorized() && vectorShape != null) throw new IllegalArgumentException(
                "vectorShape is forbidden for scalar execution mode");
        this.vectorShape = vectorShape;
        if (byteOrder != ByteOrder.LITTLE_ENDIAN && byteOrder != ByteOrder.BIG_ENDIAN) {
            throw new IllegalArgumentException("byteOrder must be LITTLE_ENDIAN or BIG_ENDIAN");
        }
        this.byteOrder = byteOrder;
        if (unrollFactor <= 0) throw new IllegalArgumentException("unrollFactor must be positive");
        this.unrollFactor = unrollFactor;
        if (tileElementCount <= 0) throw new IllegalArgumentException(
                "tileElementCount must be positive");
        this.tileElementCount = tileElementCount;
        this.tail = Objects.requireNonNull(tail, "tail");
        if (!executionMode.vectorized() && tail != Tail.NONE) throw new IllegalArgumentException(
                "scalar execution mode requires Tail.NONE");
        this.numericalMode = Objects.requireNonNull(numericalMode, "numericalMode");
        this.combineOrder = Objects.requireNonNull(combineOrder, "combineOrder");
        if (classFileMajorVersion != ClassFile.JAVA_26_VERSION) throw new IllegalArgumentException(
                "classFileMajorVersion must equal ClassFile.JAVA_26_VERSION");
        this.classFileMajorVersion = classFileMajorVersion;
        this.entryType = deriveEntryType();
        this.specializationFingerprint = CpuLoweringFingerprint.fromDigest(
                CpuLoweringFingerprint.of(canonicalBytes()).bytes());
    }

    /** Returns the schema version governing encoding and generator behavior.
     * @return schema version governing encoding and generator behavior */
    int schemaVersion() { return schemaVersion; }
    /** Returns the family-owned lowering identity.
     * @return exact non-null immutable identity of the family-owned lowering */
    CpuLoweringFingerprint loweringFingerprint() { return loweringFingerprint; }
    /** Returns the selected portable execution mode.
     * @return exact non-null portable execution mode */
    CpuPortableExecutionMode executionMode() { return executionMode; }
    /** Returns the ordered direct-argument signature facts.
     * @return immutable ordered direct-argument snapshot; never {@code null} */
    List<Argument> arguments() { return arguments; }
    /** Returns extents embedded in generated code.
     * @return immutable ordered baked-extent snapshot; never {@code null} */
    List<Long> bakedExtents() { return bakedExtents; }
    /** Returns the number of extents supplied at invocation.
     * @return number of ordered {@code long} extent values in the generated entry signature */
    int dynamicExtentCount() { return dynamicExtentCount; }
    /** Returns the optional exact Vector species shape.
     *
     * @return selected exact Vector API species shape for a vector mode, or an empty optional for
     *     a scalar mode; never {@code null}
     */
    Optional<VectorShape> vectorShape() { return Optional.ofNullable(vectorShape); }
    /** Returns the byte order embedded for segment access.
     * @return exact non-null byte order embedded for segment access */
    ByteOrder byteOrder() { return byteOrder; }
    /** Returns the baked unroll factor.
     * @return positive baked unroll factor */
    int unrollFactor() { return unrollFactor; }
    /** Returns the baked tile size.
     * @return positive baked tile size in logical elements */
    long tileElementCount() { return tileElementCount; }
    /** Returns the selected tail structure.
     * @return exact non-null selected tail structure */
    Tail tail() { return tail; }
    /** Returns the selected numerical contract.
     * @return exact non-null numerical contract */
    NumericalMode numericalMode() { return numericalMode; }
    /** Returns the partial-combine ordering contract.
     * @return exact non-null partial-combine ordering contract */
    CombineOrder combineOrder() { return combineOrder; }
    /** Returns the generated class-file target.
     * @return Java 26 class-file major version fixed for generated bytes */
    int classFileMajorVersion() { return classFileMajorVersion; }
    /** Returns the derived exact entry signature.
     *
     * @return exact immutable {@code void} entry type derived from ordered carriers, dynamic
     *     offsets and extents, and mode-specific range controls; never {@code null}
     */
    MethodType entryType() { return entryType; }
    /** Returns the complete specialization content fingerprint.
     *
     * @return immutable deterministic fingerprint of every structural component; never
     *     {@code null}
     */
    CpuLoweringFingerprint specializationFingerprint() { return specializationFingerprint; }

    @Override public boolean equals(Object other) {
        if (this == other) return true;
        if (!(other instanceof CpuKernelSpecialization that)) return false;
        return schemaVersion == that.schemaVersion && dynamicExtentCount == that.dynamicExtentCount
                && unrollFactor == that.unrollFactor && tileElementCount == that.tileElementCount
                && classFileMajorVersion == that.classFileMajorVersion
                && loweringFingerprint.equals(that.loweringFingerprint)
                && executionMode == that.executionMode && arguments.equals(that.arguments)
                && bakedExtents.equals(that.bakedExtents) && Objects.equals(vectorShape, that.vectorShape)
                && byteOrder.equals(that.byteOrder) && tail == that.tail
                && numericalMode == that.numericalMode && combineOrder == that.combineOrder;
    }

    @Override public int hashCode() {
        return Objects.hash(schemaVersion, loweringFingerprint, executionMode, arguments,
                bakedExtents, dynamicExtentCount, vectorShape, byteOrder, unrollFactor,
                tileElementCount, tail, numericalMode, combineOrder, classFileMajorVersion);
    }

    @Override public String toString() {
        return "CpuKernelSpecialization[" + specializationFingerprint + ", " + executionMode + "]";
    }

    private MethodType deriveEntryType() {
        var parameters = new ArrayList<Class<?>>();
        for (Argument argument : arguments) {
            parameters.add(carrierClass(argument.carrier));
            if (argument.carrier != Carrier.MEMORY_SEGMENT && !argument.byteOffsetBaked) {
                parameters.add(long.class);
            }
        }
        for (int index = 0; index < dynamicExtentCount; index++) parameters.add(long.class);
        if (executionMode.parallel()) {
            parameters.add(long.class); parameters.add(long.class); parameters.add(int.class);
        } else parameters.add(long.class);
        return MethodType.methodType(void.class, parameters);
    }

    private byte[] canonicalBytes() {
        try {
            var bytes = new ByteArrayOutputStream();
            var out = new DataOutputStream(bytes);
            out.writeInt(schemaVersion); out.write(loweringFingerprint.bytes());
            out.writeInt(modeTag(executionMode)); out.writeInt(arguments.size());
            for (Argument argument : arguments) {
                out.writeInt(dataTypeTag(argument.dataType)); out.writeInt(carrierTag(argument.carrier));
                out.writeInt(accessTag(argument.access)); out.writeByte(argument.byteOffsetBaked ? 1 : 0);
                out.writeLong(argument.bakedByteOffset); out.writeInt(argument.bakedElementStrides.size());
                for (long stride : argument.bakedElementStrides) out.writeLong(stride);
            }
            out.writeInt(bakedExtents.size()); for (long extent : bakedExtents) out.writeLong(extent);
            out.writeInt(dynamicExtentCount); out.writeByte(vectorShape == null ? 0 : 1);
            if (vectorShape != null) { out.writeInt(dataTypeTag(vectorShape.laneType));
                out.writeInt(vectorShape.vectorBitSize); out.writeInt(vectorShape.laneCount); }
            out.writeInt(byteOrder == ByteOrder.LITTLE_ENDIAN ? 1 : 2); out.writeInt(unrollFactor);
            out.writeLong(tileElementCount); out.writeInt(tailTag(tail));
            out.writeInt(numericalModeTag(numericalMode));
            out.writeInt(combineOrder == CombineOrder.FIXED ? 1 : 2); out.writeInt(classFileMajorVersion);
            out.flush(); return bytes.toByteArray();
        } catch (IOException impossible) { throw new AssertionError(impossible); }
    }

    private static Carrier carrierFor(DataType type) { return switch (type) {
        case FLOAT64 -> Carrier.DOUBLE_ARRAY; case FLOAT32 -> Carrier.FLOAT_ARRAY;
        case BFLOAT16 -> Carrier.SHORT_ARRAY; case INT32 -> Carrier.INT_ARRAY;
        case INT64 -> Carrier.LONG_ARRAY; case BOOL -> Carrier.BYTE_ARRAY; }; }
    private static Class<?> carrierClass(Carrier carrier) { return switch (carrier) {
        case DOUBLE_ARRAY -> double[].class; case FLOAT_ARRAY -> float[].class;
        case SHORT_ARRAY -> short[].class; case INT_ARRAY -> int[].class;
        case LONG_ARRAY -> long[].class; case BYTE_ARRAY -> byte[].class;
        case MEMORY_SEGMENT -> MemorySegment.class; }; }
    private static Class<?> laneClass(DataType type) { return switch (type) {
        case FLOAT64 -> double.class; case FLOAT32 -> float.class; case INT32 -> int.class;
        case INT64 -> long.class; default -> throw new IllegalArgumentException(
                "laneType must be FLOAT64, FLOAT32, INT32, or INT64"); }; }
    private static int dataTypeTag(DataType value) { return switch (value) {
        case FLOAT64 -> 1; case FLOAT32 -> 2; case BFLOAT16 -> 3; case INT32 -> 4;
        case INT64 -> 5; case BOOL -> 6; }; }
    private static int carrierTag(Carrier value) { return switch (value) {
        case DOUBLE_ARRAY -> 1; case FLOAT_ARRAY -> 2; case SHORT_ARRAY -> 3;
        case INT_ARRAY -> 4; case LONG_ARRAY -> 5; case BYTE_ARRAY -> 6;
        case MEMORY_SEGMENT -> 7; }; }
    private static int accessTag(PreparedExecutable.BufferAccess value) { return switch (value) {
        case READ_ONLY -> 1; case WRITE_ONLY -> 2; case READ_WRITE -> 3; }; }
    private static int modeTag(CpuPortableExecutionMode value) { return switch (value) {
        case SCALAR_SINGLE_THREAD -> 1; case SCALAR_PARALLEL -> 2;
        case VECTOR_API_SINGLE_THREAD -> 3; case VECTOR_API_PARALLEL -> 4; }; }
    private static int tailTag(Tail value) { return switch (value) {
        case NONE -> 1; case SCALAR -> 2; case MASKED -> 3; }; }
    private static int numericalModeTag(NumericalMode value) { return switch (value) {
        case EXACT_DEFAULT -> 1; }; }
}
