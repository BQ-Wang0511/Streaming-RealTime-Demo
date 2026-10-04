import pyximport
pyximport.install()

from .blend import blend_images_cy as blend_images_cy  # noqa: E402
