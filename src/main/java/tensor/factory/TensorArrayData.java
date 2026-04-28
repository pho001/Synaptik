package tensor.factory;

import java.lang.reflect.Array;

/**
 * Reflection helpers for constructing tensors from rectangular Java arrays.
 */
public final class TensorArrayData {
    private TensorArrayData() {
    }

    /**
     * Infers the shape of a nested rectangular array from its first element on each level.
     *
     * @param multiDimArray array object; must be non-null and non-empty at every nested level
     * @return inferred dimension sizes
     * @throws IllegalArgumentException if {@code multiDimArray} is not an array
     * @throws ArrayIndexOutOfBoundsException if any nested array level is empty
     */
    public static int[] inferShape(Object multiDimArray) {
        int[] dims = new int[getDepth(multiDimArray)];
        Object currentArray = multiDimArray;
        for (int i = 0; i < dims.length; i++) {
            dims[i] = Array.getLength(currentArray);
            if (Array.get(currentArray, 0).getClass().isArray()) {
                currentArray = Array.get(currentArray, 0);
            } else {
                break;
            }
        }
        return dims;
    }

    /**
     * Flattens nested double-array data into row-major order.
     *
     * @param multiDimArray nested data array; leaves must be {@code double[]}
     * @param flatSize expected number of output elements
     * @return newly allocated flat double array
     * @throws IllegalArgumentException if leaves are not double arrays or nested objects
     */
    public static double[] flattenToDouble(Object multiDimArray, int flatSize) {
        double[] flatArray = new double[flatSize];
        fillFlatArray(multiDimArray, flatArray, 0);
        return flatArray;
    }

    private static int getDepth(Object array) {
        int depth = 0;
        while (array.getClass().isArray()) {
            depth++;
            array = Array.get(array, 0);
        }
        return depth;
    }

    private static int fillFlatArray(Object multiDimArray, double[] flatArray, int startIndex) {
        if (multiDimArray instanceof double[] row) {
            System.arraycopy(row, 0, flatArray, startIndex, row.length);
            return startIndex + row.length;
        }
        if (multiDimArray instanceof Object[] objects) {
            int currentIndex = startIndex;
            for (Object element : objects) {
                currentIndex = fillFlatArray(element, flatArray, currentIndex);
            }
            return currentIndex;
        }
        throw new IllegalArgumentException("Multidimensional data must be either double, or n-dimensional object");
    }
}
