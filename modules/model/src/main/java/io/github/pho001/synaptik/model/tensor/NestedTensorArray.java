package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Array;
import java.util.Optional;

/**
 * Validates, infers, and flattens one supported nested Java primitive-array graph.
 *
 * <p>This package-private implementation boundary derives declared rank and ultimate primitive
 * carrier from the runtime array class, validates every reachable branch before allocating a flat
 * carrier, and then performs a second row-major traversal using typed bulk leaf copies. It creates
 * only fully static dense-contiguous descriptors and delegates final allocation, identity, label
 * behavior, BOOL normalization, and destination copying to {@link TensorFactory#fromFlatArray}.
 * Numeric values and raw BFLOAT16 shorts are preserved; BOOL bytes remain raw until the delegated
 * flat import canonicalizes them. It retains no source reference and owns no storage or mutable
 * static state.</p>
 */
final class NestedTensorArray {
    /**
     * Prevents construction because nested-array analysis is a stateless package operation.
     */
    private NestedTensorArray() {
    }

    /**
     * Imports one already non-null nested-array source through the matching flat import path.
     *
     * <p>Runtime class inspection precedes structural traversal. Traversal establishes one exact
     * size per axis, rejects null children, ragged branches, and empty non-final axes, and permits
     * an empty final primitive axis. Only after complete structural validation does this method
     * validate checked logical count and Java-array capacity, allocate and populate one fresh flat
     * carrier with typed {@link System#arraycopy(Object, int, Object, int, int) bulk leaf copies},
     * create static shape and dense layout metadata, and dispatch exactly once to a carrier-matched
     * flat overload. Source inspection and flattening are not synchronized with caller mutation;
     * callers must not mutate any source level concurrently, and no atomic deep snapshot is
     * promised.</p>
     *
     * @param source non-null caller-owned candidate source; the reference and its descendants are
     *     inspected but never retained or mutated
     * @param label non-null optional label delegated unchanged to flat import
     * @param requiresGrad whether the inferred descriptor requests gradient eligibility, subject
     *     to the descriptor's differentiable-data-type rule after flattening
     * @return a non-null fresh tensor containing a destination copy of the source values
     * @throws IllegalArgumentException if the source is not an array, has rank below two, has an
     *     unsupported leaf carrier, contains a null subarray, is ragged, has an empty non-final
     *     axis, exceeds Java flat-array capacity, requests gradients for a non-differentiable
     *     inferred type, or delegated label validation rejects blank text; exact diagnostics are
     *     part of {@link TensorFactory#fromNestedArray(Object, Optional, boolean)}
     * @throws ArithmeticException if checked logical-count or layout arithmetic overflows
     * @throws IllegalStateException if delegated tensor identifier allocation is exhausted
     * @throws OutOfMemoryError if intermediate or destination array allocation fails
     */
    static Tensor importArray(
            Object source,
            Optional<String> label,
            boolean requiresGrad) {
        Class<?> sourceClass = source.getClass();
        if (!sourceClass.isArray()) {
            throw new IllegalArgumentException(
                    "nested tensor source must be an array: actual=" + sourceClass.getName());
        }

        int rank = 0;
        Class<?> componentType = sourceClass;
        while (componentType.isArray()) {
            rank++;
            componentType = componentType.getComponentType();
        }
        if (rank < 2) {
            throw new IllegalArgumentException(
                    "nested tensor source must have rank at least 2: actual=" + rank);
        }
        DataType dataType = dataTypeFor(componentType);

        long[] sizes = new long[rank];
        boolean[] established = new boolean[rank];
        validateStructure(source, 0, "[]", rank, sizes, established);

        Shape shape = Shape.of(sizes);
        long required = shape.knownElementCount().orElseThrow();
        if (required > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "nested tensor element count exceeds Java array limit: required="
                            + required
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }

        Object flat = newFlatArray(dataType, (int) required);
        flatten(source, 0, rank, flat, 0);

        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor =
                new TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad);
        return importFlat(descriptor, label, dataType, flat);
    }

    /**
     * Maps one ultimate runtime component class to its exact model data type.
     *
     * @param componentType non-null ultimate, non-array runtime component class
     * @return exact supported model data type; never {@code null}
     * @throws IllegalArgumentException if the component class is not one of the six supported
     *     primitive carriers
     */
    private static DataType dataTypeFor(Class<?> componentType) {
        if (componentType == double.class) {
            return DataType.FLOAT64;
        }
        if (componentType == float.class) {
            return DataType.FLOAT32;
        }
        if (componentType == short.class) {
            return DataType.BFLOAT16;
        }
        if (componentType == int.class) {
            return DataType.INT32;
        }
        if (componentType == long.class) {
            return DataType.INT64;
        }
        if (componentType == byte.class) {
            return DataType.BOOL;
        }
        throw new IllegalArgumentException(
                "nested tensor source leaf carrier is unsupported: " + componentType.getName());
    }

    /**
     * Validates one array level and recursively visits every required child in row-major order.
     *
     * @param array non-null array at the current axis
     * @param axis zero-based current axis
     * @param path zero-based bracket path for diagnostics; root is {@code []}
     * @param rank declared runtime array rank
     * @param sizes mutable traversal metadata receiving one established non-negative size per axis
     * @param established mutable traversal metadata recording which axis sizes are known
     * @throws IllegalArgumentException if an established size differs, a required child is null,
     *     or a non-final axis is empty
     */
    private static void validateStructure(
            Object array,
            int axis,
            String path,
            int rank,
            long[] sizes,
            boolean[] established) {
        int length = Array.getLength(array);
        if (!established[axis]) {
            sizes[axis] = length;
            established[axis] = true;
        } else if (sizes[axis] != length) {
            throw new IllegalArgumentException(
                    "nested tensor source is ragged at axis "
                            + axis
                            + ", path "
                            + path
                            + ": expected="
                            + sizes[axis]
                            + ", actual="
                            + length);
        }

        if (axis == rank - 1) {
            return;
        }
        if (length == 0) {
            throw new IllegalArgumentException(
                    "nested tensor source cannot infer dimensions after empty axis "
                            + axis
                            + " at path "
                            + path);
        }

        for (int index = 0; index < length; index++) {
            Object child = Array.get(array, index);
            String childPath = childPath(path, index);
            if (child == null) {
                throw new IllegalArgumentException(
                        "nested tensor source contains null subarray at path " + childPath);
            }
            validateStructure(child, axis + 1, childPath, rank, sizes, established);
        }
    }

    /**
     * Creates the zero-based bracket path for one child of the current diagnostic location.
     *
     * @param parentPath non-null parent path, with root represented by {@code []}
     * @param childIndex non-negative child index
     * @return non-null child path such as {@code [1]} or {@code [1][2]}
     */
    private static String childPath(String parentPath, int childIndex) {
        if (parentPath.equals("[]")) {
            return "[" + childIndex + "]";
        }
        return parentPath + "[" + childIndex + "]";
    }

    /**
     * Allocates the single intermediate primitive carrier for a validated source.
     *
     * @param dataType exact non-null inferred model data type
     * @param length validated non-negative Java array length
     * @return non-null primitive array whose carrier exactly matches {@code dataType}
     */
    private static Object newFlatArray(DataType dataType, int length) {
        if (dataType == DataType.FLOAT64) {
            return new double[length];
        }
        if (dataType == DataType.FLOAT32) {
            return new float[length];
        }
        if (dataType == DataType.BFLOAT16) {
            return new short[length];
        }
        if (dataType == DataType.INT32) {
            return new int[length];
        }
        if (dataType == DataType.INT64) {
            return new long[length];
        }
        return new byte[length];
    }

    /**
     * Copies primitive leaves into the validated matching carrier in row-major encounter order.
     *
     * @param array non-null current source array
     * @param axis zero-based current array axis
     * @param rank declared runtime source rank
     * @param flat non-null fresh matching primitive destination array
     * @param offset non-negative next destination element offset
     * @return next destination offset after this subtree has been copied
     */
    private static int flatten(Object array, int axis, int rank, Object flat, int offset) {
        int length = Array.getLength(array);
        if (axis == rank - 1) {
            System.arraycopy(array, 0, flat, offset, length);
            return offset + length;
        }
        int nextOffset = offset;
        for (int index = 0; index < length; index++) {
            nextOffset = flatten(Array.get(array, index), axis + 1, rank, flat, nextOffset);
        }
        return nextOffset;
    }

    /**
     * Dispatches a validated flat carrier to exactly one matching public flat-import overload.
     *
     * @param descriptor non-null inferred dense-contiguous descriptor
     * @param label non-null optional label delegated unchanged
     * @param dataType exact inferred model data type
     * @param flat non-null matching primitive array populated in row-major order
     * @return exact non-null tensor returned by the selected flat import overload
     */
    private static Tensor importFlat(
            TensorDescriptor descriptor,
            Optional<String> label,
            DataType dataType,
            Object flat) {
        if (dataType == DataType.FLOAT64) {
            return TensorFactory.fromFlatArray(descriptor, label, (double[]) flat);
        }
        if (dataType == DataType.FLOAT32) {
            return TensorFactory.fromFlatArray(descriptor, label, (float[]) flat);
        }
        if (dataType == DataType.BFLOAT16) {
            return TensorFactory.fromFlatArray(descriptor, label, (short[]) flat);
        }
        if (dataType == DataType.INT32) {
            return TensorFactory.fromFlatArray(descriptor, label, (int[]) flat);
        }
        if (dataType == DataType.INT64) {
            return TensorFactory.fromFlatArray(descriptor, label, (long[]) flat);
        }
        return TensorFactory.fromFlatArray(descriptor, label, (byte[]) flat);
    }
}
