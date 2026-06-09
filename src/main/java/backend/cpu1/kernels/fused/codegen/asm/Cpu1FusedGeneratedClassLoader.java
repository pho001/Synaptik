package backend.cpu1.kernels.fused.codegen.asm;

import utils.CustomClassLoader;

/**
 * Defines generated cpu1 fused kernel classes.
 */
public final class Cpu1FusedGeneratedClassLoader extends CustomClassLoader {
    public Class<?> defineGenerated(String binaryName, byte[] bytecode) {
        if (binaryName == null || binaryName.isBlank()) {
            throw new IllegalArgumentException("binaryName cannot be blank");
        }
        if (bytecode == null || bytecode.length == 0) {
            throw new IllegalArgumentException("bytecode cannot be empty");
        }
        return define(binaryName, bytecode);
    }
}
