#version 330 core

layout(location = 0) in vec3 inputPosition;
layout(location = 1) in vec2 inputTexCoord;
layout(location = 2) in vec3 inputNormal;
layout(location = 3) in vec3 inputAmbientColor;
layout(location = 4) in vec3 inputDiffuseColor;
layout(location = 5) in vec3 inputSpecColor;
layout(location = 6) in float inputSpecIntensity;

//uniform mat4 projection, model, view, normalMat;
uniform mat4 projection, model, view;
//normalMat;


out vec3 normalInterp;
out vec3 vertPos;
out vec2 uv;
out vec3 ambientColor;
out vec3 diffuseColor;
out vec3 specColor;
out float specIntensity;

void main(){
    mat4 modelview = view * model;
    mat4 normalMat = transpose(inverse(modelview));
    gl_Position = projection * modelview * vec4(inputPosition, 1.0);
    vec4 vertPos4 = modelview * vec4(inputPosition, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    normalInterp = vec3(normalMat * vec4(inputNormal, 0.0));
    uv = inputTexCoord;
    ambientColor = inputAmbientColor;
    diffuseColor = inputDiffuseColor;
    specColor = inputSpecColor;
    specIntensity = inputSpecIntensity;
}

