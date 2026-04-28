import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.lang.reflect.Modifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AppEntryPointTest {

    @Test
    void packagedTuningCliExistsAndExposesStandardMainMethod() throws Exception {
        Class<?> mainClass = Class.forName("synaptik.app.TuningCli");
        Method main = mainClass.getMethod("main", String[].class);
        assertEquals(void.class, main.getReturnType());
        assertTrue(Modifier.isStatic(main.getModifiers()));
        assertTrue(Modifier.isPublic(main.getModifiers()));
    }

    @Test
    void legacyMainLauncherStillExistsAndExposesStandardMainMethod() throws Exception {
        Class<?> mainClass = Class.forName("synaptik.app.Main");
        Method main = mainClass.getMethod("main", String[].class);
        assertEquals(void.class, main.getReturnType());
        assertTrue(Modifier.isStatic(main.getModifiers()));
        assertTrue(Modifier.isPublic(main.getModifiers()));
    }
}
