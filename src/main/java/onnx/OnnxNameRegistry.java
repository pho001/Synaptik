package onnx;

import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.HashSet;
import java.util.Set;

final class OnnxNameRegistry {
    private final IdentityHashMap<Tensor, String> names = new IdentityHashMap<>();
    private final Set<String> used = new HashSet<>();

    String nameFor(Tensor tensor, int fallbackId) {
        String existing = names.get(tensor);
        if (existing != null) {
            return existing;
        }
        String preferred = tensor.getLabel();
        if (preferred == null || preferred.isBlank() || preferred.startsWith(Tensor.SYSTEM_FORWARD_OUTPUT_LABEL)) {
            preferred = "value_" + fallbackId;
        }
        String name = unique(sanitize(preferred));
        names.put(tensor, name);
        return name;
    }

    String auxiliary(String prefix) {
        return unique(sanitize(prefix));
    }

    private String unique(String base) {
        String candidate = base.isBlank() ? "value" : base;
        int suffix = 1;
        while (!used.add(candidate)) {
            candidate = base + "_" + suffix++;
        }
        return candidate;
    }

    private static String sanitize(String name) {
        StringBuilder out = new StringBuilder(name.length());
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '_' || c == '.' || c == '/') {
                out.append(c);
            } else {
                out.append('_');
            }
        }
        return out.toString();
    }
}
