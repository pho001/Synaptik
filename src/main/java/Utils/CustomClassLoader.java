package Utils;

public class CustomClassLoader extends ClassLoader {
    public Class<?> define(String name, byte[] bytes) {
        return defineClass(name, bytes, 0, bytes.length);
    }
}
