package graph.optimizer.rules;

import config.optimizer.CseConfig;
import graph.optimizer.OptimizationRule;
import operations.*;
import tensor.Tensor;
import java.util.*;

public class CommonSubexpressionEliminationRule implements OptimizationRule {

    private final CseConfig config;

    public CommonSubexpressionEliminationRule(CseConfig config) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
    }

    public CseConfig config() {
        return config;
    }

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> optimized = new ArrayList<>();
        Map<String, Tensor> seenNodes = new HashMap<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();
        Map<Tensor, String> structuralSignatures = new HashMap<>();


        for (Tensor t : sortedGraph) {
            // 1. Aktualizujeme vstupy uzlu, pokud nějaký jeho předek už byl smazán/nahrazen
            updateInputs(t, replacements);

            // 2. Vygenerujeme unikátní podpis operace
            String signature = generateSignature(t, structuralSignatures);
            if (signature != null) {
                structuralSignatures.put(t, signature);
            } else {
                structuralSignatures.put(t, leafSignature(t));
            }

            if (signature != null) {
                if (seenNodes.containsKey(signature)) {
                    Tensor existing = seenNodes.get(signature);
                    // NAŠLI JSME DUPLIKÁT!
                    // Uzel 't' vůbec nepřidáme do 'optimized' grafu.
                    // Místo toho řekneme: Kdo chtěl 't', ať odteď používá ten původní.
                    if (t.isBackward()) {
                        existing.setBackward(true);
                    }
                    replacements.put(t, existing);
                    continue;
                } else {
                    // Je to nová operace, zapamatujeme si ji
                    seenNodes.put(signature, t);
                }
            }

            // Přidáme uzel do finálního grafu (pokud to není duplikát)
            optimized.add(t);
        }

        // Stejně jako u algebraic rewrites znovu uzavřeme graf topologicky od sinků.
        // Tím zajistíme, že po přepojení referencí v CSE v seznamu nechybí závislosti.
        return rebuildTopologicalClosure(optimized);
    }

    /**
     * Vytvoří unikátní řetězec, který reprezentuje výpočet daného uzlu.
     */
    private String generateSignature(Tensor t, Map<Tensor, String> structuralSignatures) {
        Operation op = t.getOperation();
        boolean strictSafety = config.strictSafety();
        // Konstanty a vstupy (bez operace) necháme být, ty se optimalizují jinde
        if (op == null) return null;

        // noop/fused uzly fungují jako kotvy nebo hranice clusterů,
        // proto je nechceme slučovat CSE.
        if (op instanceof noop) return null;
        if (op.opType() == Operation.OpType.FUSED) return null;

        String opName = op.getClass().getSimpleName().toLowerCase();

        // BEZPEČNOSTNÍ POJISTKA:
        // Náhodné operace (Dropout, RandomNormal) nesmíme NIKDY sloučit,
        // protože při každém zavolání musí vrátit jiná čísla!
        if (opName.contains("random") || opName.contains("dropout")) return null;

        List<Tensor> inputs = t.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) return null;

        // Sesbíráme identity vstupních tenzorů (použijeme System.identityHashCode
        // nebo přímo instanci, zde pro stringový podpis hash)
        List<String> inputKeys = new ArrayList<>();
        for (Tensor input : inputs) {
            inputKeys.add(structuralSignatures.getOrDefault(input, leafSignature(input)));
        }

        // KOMUTATIVITA:
        // U sčítání a násobení nezáleží na pořadí (Add(A, B) == Add(B, A)).
        // Seřadíme ID vstupů, aby obě varianty vygenerovaly stejný podpis.
        if (op instanceof add || op instanceof mul) {
            Collections.sort(inputKeys);
        }

        // Sestavení podpisu: např. "add_123456_789012"
        StringBuilder sig = new StringBuilder(opName)
                .append("_phase=").append(t.isBackward() ? "bw" : "fw")
                .append("_type=").append(op.opType());

        if (strictSafety) {
            sig.append("_reqGrad=").append(t.getRequiresGrad())
                    .append("_backend=").append(t.resolveBackend())
                    .append("_shape=");
            int[] shape = t.getShape();
            if (shape != null) {
                for (int i = 0; i < shape.length; i++) {
                    if (i > 0) sig.append('x');
                    sig.append(shape[i]);
                }
            } else {
                sig.append("null");
            }
        }
        sig.append("_in");
        for (String key : inputKeys) {
            sig.append("_").append(key);
        }

        // PARAMETRY OPERACÍ:
        // Pokud má operace nějaký vnitřní parametr (např. exponent u Pow),
        // MUSÍ být součástí podpisu, jinak by se sloučilo Pow(x, 2) a Pow(x, 3)!
        switch (op) {
            case pow p -> sig.append("_").append(p.getExponent());
            case mulScalar m -> sig.append("_").append(m.getScalar());
            case sum s -> sig.append("_dim=").append(s.getDimension());
            case minGrad mg -> sig.append("_forFirst=").append(mg.isForFirstInput());
            case maxGrad mg -> sig.append("_forFirst=").append(mg.isForFirstInput());
            default -> {

            }
        }


        return sig.toString();
    }

    private String leafSignature(Tensor t) {
        // Trainable leaf uzly nikdy neslučujeme podle hodnoty.
        if (t.getOperation() == null && t.getRequiresGrad()) {
            return "leaf_var@" + System.identityHashCode(t);
        }

        // Konstantní scalar leaf: strukturální podpis podle hodnoty.
        if (t.getOperation() == null && !t.getRequiresGrad() && t.getFlatDataSize() == 1) {
            long bits = Double.doubleToLongBits(t.scalarAsDouble());
            int[] shape = t.getShape();
            String shapeKey = (shape == null) ? "null" : Arrays.toString(shape);
            return "const_scalar#" + bits + "_shape=" + shapeKey;
        }

        // Ostatní leafy a fallback: identita.
        return "leaf@" + System.identityHashCode(t);
    }

    private void updateInputs(Tensor t, Map<Tensor, Tensor> replacements) {
        if (t.getPrevTensors() == null) return;
        List<Tensor> inputs = t.getPrevTensors();
        for (int i = 0; i < inputs.size(); i++) {
            Tensor currentInput = inputs.get(i);
            Tensor replacement = resolveReplacement(currentInput, replacements);
            if (replacement != null) {
                inputs.set(i, replacement);
            }
        }
    }

    private Tensor resolveReplacement(Tensor tensor, Map<Tensor, Tensor> replacements) {
        Tensor current = replacements.get(tensor);
        if (current == null) return null;
        while (replacements.containsKey(current)) {
            current = replacements.get(current);
        }
        return current;
    }

    private List<Tensor> rebuildTopologicalClosure(List<Tensor> graph) {
        if (graph.isEmpty()) return graph;

        Map<Tensor, Integer> consumerCounts = new HashMap<>();
        for (Tensor t : graph) {
            if (t.getPrevTensors() != null) {
                for (Tensor p : t.getPrevTensors()) {
                    consumerCounts.put(p, consumerCounts.getOrDefault(p, 0) + 1);
                }
            }
        }

        List<Tensor> sinks = new ArrayList<>();
        for (Tensor t : graph) {
            if (consumerCounts.getOrDefault(t, 0) == 0) {
                sinks.add(t);
            }
        }

        Set<Tensor> visited = new HashSet<>();
        List<Tensor> rebuilt = new ArrayList<>();
        for (Tensor sink : sinks) {
            dfsPostOrder(sink, visited, rebuilt);
        }
        return rebuilt;
    }

    private void dfsPostOrder(Tensor node, Set<Tensor> visited, List<Tensor> out) {
        if (node == null || visited.contains(node)) return;
        visited.add(node);

        if (node.getPrevTensors() != null) {
            for (Tensor p : node.getPrevTensors()) {
                dfsPostOrder(p, visited, out);
            }
        }

        out.add(node);
    }
}
