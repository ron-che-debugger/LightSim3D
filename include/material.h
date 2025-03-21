#ifndef MATERIAL_H
#define MATERIAL_H

#include "math_utils.h"

struct Material {
    float3 albedo;   // Diffuse albedo color
    float metallic;  // [0,1]: 0 = dielectric (purely diffuse), 1 = metallic (purely specular)
    float roughness; // Roughness
    float3 emission; // Emissive color (for light sources)
};

#endif // MATERIAL_H