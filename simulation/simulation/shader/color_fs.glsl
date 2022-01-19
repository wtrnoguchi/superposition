#version 330 core

in vec3 normalInterp;
in vec3 vertPos;
in vec2 uv;
in vec3 ambientColor;
in vec3 diffuseColor;
in vec3 specColor;
in float specIntensity;

const vec3 lightPos = vec3(0.0,0.0,2.0);
float lightIntensity = 1.0;

out vec4 color;


void main() {
    vec3 normal = normalize(normalInterp);
    vec3 lightDir = normalize(lightPos - vertPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    vec3 viewDir = normalize(-vertPos);


    vec3 _ambientColor = vec3(0.7, 0.7, 0.7) * diffuseColor;
    float lambertian = max(dot(lightDir,normal), 0.0);
    float specular = 0.0;

    if(lambertian > 0.0) {
       float specAngle = max(dot(reflectDir, viewDir), 0.0);
       specular = pow(specAngle, specIntensity);
    }

    color = vec4(_ambientColor +
                      lambertian*diffuseColor +
                      specular*specColor, 1.0);
}
