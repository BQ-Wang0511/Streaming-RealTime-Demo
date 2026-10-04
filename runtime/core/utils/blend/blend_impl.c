#include <blend_impl.h>

void _blend_images_cy_impl(
  const float* mask_warped,
  const float* frame_warped,
  const unsigned char* frame_rgb,
  const int height,
  const int width,
  unsigned char* result) {

    const float* mask_pointer = mask_warped;
    const float* frame_warped_pointer = frame_warped;
    const unsigned char* frame_rgb_pointer = frame_rgb;
    unsigned char* result_pointer = result;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            float mask = *mask_pointer;
            float mask_inv = 1.0f - mask;

            float blended1 = mask * (*frame_warped_pointer) + mask_inv * (*frame_rgb_pointer);
            float blended2 = mask * (*(frame_warped_pointer+1)) + mask_inv * (*(frame_rgb_pointer+1));
            float blended3 = mask * (*(frame_warped_pointer+2)) + mask_inv * (*(frame_rgb_pointer+2));

            *(result_pointer++) = blended1 > 255 ? 255 : (blended1 < 0) ? 0 : (unsigned char)blended1;
            *(result_pointer++) = blended2 > 255 ? 255 : (blended2 < 0) ? 0 : (unsigned char)blended2;
            *(result_pointer++) = blended3 > 255 ? 255 : (blended3 < 0) ? 0 : (unsigned char)blended3;

            frame_warped_pointer+=3;
            frame_rgb_pointer+=3;
            mask_pointer++;
        }
    }
}