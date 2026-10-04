#cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "blend_impl.h":
    void _blend_images_cy_impl(
        const float* mask_warped,
        const float* frame_warped,
        const unsigned char* frame_rgb,
        const int height,
        const int width,
        unsigned char* result
    ) noexcept nogil

def blend_images_cy(
    np.ndarray[np.float32_t, ndim=2] mask_warped,
    np.ndarray[np.float32_t, ndim=3] frame_warped,
    np.ndarray[np.uint8_t, ndim=3] frame_rgb,
    np.ndarray[np.uint8_t, ndim=3] result
):
    cdef int h = mask_warped.shape[0]
    cdef int w = mask_warped.shape[1]

    if not mask_warped.flags['C_CONTIGUOUS']:
        mask_warped = np.ascontiguousarray(mask_warped)
    if not frame_warped.flags['C_CONTIGUOUS']:
        frame_warped = np.ascontiguousarray(frame_warped)
    if not frame_rgb.flags['C_CONTIGUOUS']:
        frame_rgb = np.ascontiguousarray(frame_rgb)

    with nogil:
        _blend_images_cy_impl(
            <const float*>mask_warped.data,
            <const float*>frame_warped.data,
            <const unsigned char*>frame_rgb.data,
            h, w,
            <unsigned char*>result.data
        )