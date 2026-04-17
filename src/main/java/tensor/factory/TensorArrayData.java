package tensor.factory;

import java.lang.reflect.Array;

public final class TensorArrayData {
    private TensorArrayData() {
    }

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
