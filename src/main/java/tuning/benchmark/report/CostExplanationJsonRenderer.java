package tuning.benchmark.report;

import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostExplanation;

import java.util.Locale;

final class CostExplanationJsonRenderer {
    private CostExplanationJsonRenderer() {
    }

    static String render(CostExplanation explanation) {
        if (explanation == null) {
            return "{}";
        }
        return "{"
                + "\"model\": \"" + escape(explanation.modelName()) + "\", "
                + "\"input_kind\": \"" + escape(explanation.inputKind()) + "\", "
                + "\"reason\": \"" + escape(explanation.reasonCode()) + "\", "
                + "\"comparison\": \"" + escape(explanation.comparison().name()) + "\", "
                + "\"top_contributors\": " + componentsJson(explanation.topContributors()) + ", "
                + "\"components\": " + componentsJson(explanation.rawComponents())
                + "}";
    }

    private static String componentsJson(java.util.List<CostComponent> components) {
        if (components == null || components.isEmpty()) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < components.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            CostComponent component = components.get(i);
            sb.append("{")
                    .append("\"name\": \"").append(escape(component.name())).append("\", ")
                    .append("\"value\": ").append(format(component.value())).append(", ")
                    .append("\"direction\": \"").append(component.direction().name()).append("\", ")
                    .append("\"reason\": \"").append(escape(component.reason())).append("\"")
                    .append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String format(double value) {
        if (!Double.isFinite(value)) {
            return "null";
        }
        return String.format(Locale.US, "%.6f", value);
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }
}
