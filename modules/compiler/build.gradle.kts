import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.JavadocMemberLevel
import org.gradle.external.javadoc.StandardJavadocDocletOptions

dependencies {
    api(project(":modules:model"))
    api(project(":modules:config"))
    api(project(":modules:planning"))
    api(project(":modules:backend-contract"))
    implementation(project(":modules:trace"))
}

tasks.named<Javadoc>("javadoc") {
    (options as StandardJavadocDocletOptions).memberLevel = JavadocMemberLevel.PACKAGE
}
