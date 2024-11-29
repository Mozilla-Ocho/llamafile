
// Reduce attack surface.
//
// "At this point, there are no plans to include more image formats in
//  stb_image; a higher priority is to get the existing ones stable."
//                      —Fabian Giesen (stb_image collaborator)
//
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
