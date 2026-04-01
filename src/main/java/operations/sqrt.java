package operations;

import backend.ComputeBackend;
import tensor.Tensor;
import java.util.List;

public class sqrt implements Operation {

    @Override
    public OpType opType() {
        return OpType.SQRT;
    }

    @Override
    public String getExpression() {
        // Použijeme název funkce pro generátor výrazů
        return "sqrt";
    }



    @Override
    public boolean isCheap() {
        // Jak jsme si řekli, sqrt je dražší, takže raději materiálizovat než počítat 2x
        return false;
    }
}
