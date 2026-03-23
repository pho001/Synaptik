package Graph.optimizer.rules;

import Graph.optimizer.OptimizationRule;
import Tensor.DataType;
import Tensor.Tensor;
import java.util.*;

public class MemoryOptimizerRule implements OptimizationRule {

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        if (sortedGraph.isEmpty()) return sortedGraph;
        DataType graphType = sortedGraph.get(0).getDataType();
        for (Tensor t : sortedGraph) {
            if (t.getDataType() != graphType) {
                return sortedGraph;
            }
        }
        if (graphType == DataType.FLOAT64) {
            return applyFloat64(sortedGraph);
        }
        if (graphType == DataType.FLOAT32) {
            return applyFloat32(sortedGraph);
        }
        return sortedGraph;
    }

    private List<Tensor> applyFloat64(List<Tensor> sortedGraph) {
        // 1. Liveness Analysis: Kdy je uzel naposledy použit?
        Map<Tensor, Integer> lastUseMap = new HashMap<>();
        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getPrevTensors() != null) {
                for (Tensor prev : t.getPrevTensors()) {
                    lastUseMap.put(prev, i);
                    consumerCounts.put(prev, consumerCounts.getOrDefault(prev, 0) + 1);
                }
            }
        }
        // Ochrana všech sink uzlů (nejen posledního v listu)
        for (Tensor t : sortedGraph) {
            if (consumerCounts.getOrDefault(t, 0) == 0) {
                lastUseMap.put(t, Integer.MAX_VALUE);
            }
        }

        Map<Integer, Stack<double[]>> freeBuffers = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getOperation() == null) continue;

            // --- IN-PLACE OPTIMALIZACE ---
            boolean inPlaceFound = false;
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    // Podmínky pro In-place:
                    // 1. Vstup má stejnou velikost jako potřebuje výstup (Broadcasting safe)
                    // 2. Vstup je v tomto kroku použit naposledy
                    // 3. Vstup není konstanta/váha (má operaci) - volitelné, ale bezpečnější
                    double[] inputData = input.getFloat64Data();
                    if (inputData != null &&
                            input.getFlatDataSize() == t.getFlatDataSize() &&
                            lastUseMap.get(input) == i &&
                            input.getOperation() != null) {

                        t.setData(inputData);
                        inPlaceFound = true;
                        break; // Našli jsme vhodný buffer, končíme hledání pro tento uzel
                    }
                }
            }

            // --- STANDARDNÍ ALOKACE (pokud In-place nevyšlo) ---
            if (!inPlaceFound) {
                int size = t.getFlatDataSize();
                double[] buffer = getFromPool(size, freeBuffers);
                if (buffer != null) {
                    t.setData(buffer);
                } else {
                    t.setData(new double[size]);
                }
            }

            // 2. Uvolnění nepoužívaných bufferů do poolu
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    // Pokud vstup umírá a jeho data nebyla převzata In-place (nejsou stejná jako výstup)
                    double[] inputData = input.getFloat64Data();
                    double[] outputData = t.getFloat64Data();
                    if (inputData != null &&
                            outputData != null &&
                            lastUseMap.get(input) == i &&
                            input.getOperation() != null &&
                            inputData != outputData) {

                        releaseToPool(inputData, freeBuffers);
                    }
                }
            }
        }
        return sortedGraph;
    }

    private List<Tensor> applyFloat32(List<Tensor> sortedGraph) {
        Map<Tensor, Integer> lastUseMap = new HashMap<>();
        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getPrevTensors() != null) {
                for (Tensor prev : t.getPrevTensors()) {
                    lastUseMap.put(prev, i);
                    consumerCounts.put(prev, consumerCounts.getOrDefault(prev, 0) + 1);
                }
            }
        }
        for (Tensor t : sortedGraph) {
            if (consumerCounts.getOrDefault(t, 0) == 0) {
                lastUseMap.put(t, Integer.MAX_VALUE);
            }
        }

        Map<Integer, Stack<float[]>> freeBuffers = new HashMap<>();

        for (int i = 0; i < sortedGraph.size(); i++) {
            Tensor t = sortedGraph.get(i);
            if (t.getOperation() == null) continue;

            boolean inPlaceFound = false;
            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    float[] inputData = input.getFloat32Data();
                    if (inputData != null &&
                            input.getFlatDataSize() == t.getFlatDataSize() &&
                            lastUseMap.get(input) == i &&
                            input.getOperation() != null) {
                        t.aliasRuntimeFrom(input);
                        inPlaceFound = true;
                        break;
                    }
                }
            }

            if (!inPlaceFound) {
                int size = t.getFlatDataSize();
                float[] buffer = getFromPoolF32(size, freeBuffers);
                if (buffer != null) {
                    t.setFloat32Data(buffer);
                } else {
                    t.setFloat32Data(new float[size]);
                }
            }

            if (t.getPrevTensors() != null) {
                for (Tensor input : t.getPrevTensors()) {
                    float[] inputData = input.getFloat32Data();
                    float[] outputData = t.getFloat32Data();
                    if (inputData != null &&
                            outputData != null &&
                            lastUseMap.get(input) == i &&
                            input.getOperation() != null &&
                            inputData != outputData) {
                        releaseToPoolF32(inputData, freeBuffers);
                    }
                }
            }
        }
        return sortedGraph;
    }

    private double[] getFromPool(int size, Map<Integer, Stack<double[]>> pool) {
        Stack<double[]> buffers = pool.get(size);
        return (buffers != null && !buffers.isEmpty()) ? buffers.pop() : null;
    }

    private void releaseToPool(double[] buffer, Map<Integer, Stack<double[]>> pool) {
        if (buffer == null) return;
        pool.computeIfAbsent(buffer.length, k -> new Stack<>()).push(buffer);
    }

    private float[] getFromPoolF32(int size, Map<Integer, Stack<float[]>> pool) {
        Stack<float[]> buffers = pool.get(size);
        return (buffers != null && !buffers.isEmpty()) ? buffers.pop() : null;
    }

    private void releaseToPoolF32(float[] buffer, Map<Integer, Stack<float[]>> pool) {
        if (buffer == null) return;
        pool.computeIfAbsent(buffer.length, k -> new Stack<>()).push(buffer);
    }
}
