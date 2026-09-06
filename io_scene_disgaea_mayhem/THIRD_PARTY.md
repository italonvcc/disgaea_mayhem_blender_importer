# Third-party code

The portable BC1/BC3/BC7 decoder in `tools/nltx_texture.py` and the
BC7 constants in `tools/bc7_tables.py` are adapted from
[Pillow 12.3.0, src/libImaging/BcnDecode.c](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/BcnDecode.c).
That source file explicitly places its contents in the public domain
under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
The original C decoder is not bundled; the Python adaptation and constants
are included in the add-on. Pillow itself is used only for development
comparison tests and the existing standalone texture command.
