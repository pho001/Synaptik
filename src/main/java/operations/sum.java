package operations;

import backend.ComputeBackend;

public class sum implements Operation{
    private final int dimension;
    private final boolean keepDims;

    public sum(int dimension) {
        this(dimension, false);
    }

    public sum(int dimension, boolean keepDims){
        this.dimension = dimension;
        this.keepDims = keepDims;
    }

    // Sum in direction of dim
    // Dim is dimension of tensor which values you want to sum. Example for matrix (2x4) - if you sum around dimension 1 (4 columns),
    // you are summing up values from this dimension, meaning result will have 2 values: sums of all column values in each row.
    @Override
    public OpType opType() {
        return OpType.SUM;
    }

    @Override
    public String getExpression() {
        return "sum";
    }


    public int getDimension() {
        return dimension;
    }

    public boolean keepDims() {
        return keepDims;
    }


}
