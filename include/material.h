#ifndef MATERIAL_H
#define MATERIAL_H

#include "math_utils.h"

/**
 * @brief Describes the material properties used for shading and lighting.
 *
 * This structure holds physical-based rendering (PBR) parameters for a surface,
 * including its base color, metallic factor, roughness, and any emissive light it produces.
 */
struct Material {
    float3 albedo;   /// Diffuse albedo color (base color of the surface)
    float metallic;  /// Metallic factor [0,1]: 0 = dielectric (diffuse), 1 = metallic (specular)
    float3 emission; /// Emissive color (non-zero values make the surface a light source)
};

#endif // MATERIAL_H