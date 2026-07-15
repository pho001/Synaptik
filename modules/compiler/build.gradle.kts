import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.JavadocMemberLevel
import org.gradle.external.javadoc.StandardJavadocDocletOptions

dependencies {
    implementation(project(":modules:model"))
    implementation(project(":modules:config"))
    implementation(project(":modules:planning"))
    implementation(project(":modules:backend-contract"))
    implementation(project(":modules:trace"))
}

tasks.named<Javadoc>("javadoc") {
    (options as StandardJavadocDocletOptions).memberLevel = JavadocMemberLevel.PACKAGE
}
