
#ifndef __BLEND_IMAGES_CY_IMPL_H__
#define __BLEND_IMAGES_CY_IMPL_H__

void _blend_images_cy_impl(
  const float* mask_warped,
  const float* frame_warped,
  const unsigned char* frame_rgb,
  const int height,
  const int width,
  unsigned char* result);

#endif  // __BLEND_IMAGES_CY_IMPL_H__