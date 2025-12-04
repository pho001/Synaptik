package Graph;

import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;
import Operations.Operation;

import java.util.List;

public class ByteCodeGenerator {

    double[][] inputs;

    ByteCodeGenerator(double[][] inputs, List<Operation> operations){

    }

    public interface SumInterface {
        int sum(int a, int b);
    }

    public Class<?> generateClass(double[][] inputs, List<Operation> operations) {
        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        String className = "GeneratedSumClass";
        String classPath = className.replace('.', '/');
        String interfacePath = SumInterface.class.getName().replace('.', '/');

        // Definice třídy
        cw.visit(Opcodes.V1_8, Opcodes.ACC_PUBLIC, classPath, null, "java/lang/Object", new String[]{interfacePath});

        // Konstruktor
        MethodVisitor mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null);
        mv.visitCode();
        mv.visitVarInsn(Opcodes.ALOAD, 0);
        mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();

        // Metoda sum
        mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "sum", "(II)I", null, null);
        mv.visitCode();
        mv.visitVarInsn(Opcodes.ILOAD, 1); // Načtení prvního parametru
        mv.visitVarInsn(Opcodes.ILOAD, 2); // Načtení druhého parametru
        mv.visitInsn(Opcodes.IADD); // Sečtení
        mv.visitInsn(Opcodes.IRETURN); // Vrácení výsledku
        mv.visitMaxs(2, 3);
        mv.visitEnd();

        cw.visitEnd();

        // Vytvoření třídy
        byte[] classBytes = cw.toByteArray();
        Class<?> dynamicClass = new ClassLoader() {
            public Class<?> defineClass(String name, byte[] b) {
                return defineClass(name, b, 0, b.length);
            }
        }.defineClass(className, classBytes);
        return dynamicClass;
    }

}
