package onnx;

final class OnnxAttributeReader {
    private final OnnxProto.NodeProto node;

    OnnxAttributeReader(OnnxProto.NodeProto node) {
        this.node = node;
    }

    int intAttribute(String name, int defaultValue) {
        for (OnnxProto.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals(name)) {
                return Math.toIntExact(attr.getI());
            }
        }
        return defaultValue;
    }

    float floatAttribute(String name, float defaultValue) {
        for (OnnxProto.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals(name)) {
                return attr.getF();
            }
        }
        return defaultValue;
    }

    int[] intsAttribute(String name) {
        for (OnnxProto.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals(name)) {
                int[] out = new int[attr.getIntsCount()];
                for (int i = 0; i < out.length; i++) {
                    out[i] = Math.toIntExact(attr.getInts(i));
                }
                return out;
            }
        }
        return null;
    }

    OnnxProto.TensorProto tensorAttribute(String name) {
        for (OnnxProto.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals(name) && attr.hasT()) {
                return attr.getT();
            }
        }
        return null;
    }
}
