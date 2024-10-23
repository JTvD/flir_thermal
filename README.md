# Flir Image Extractor
There are already quite some tools available to process FLIR images.  
This script is based on two of them:
- [Thermimage](https://github.com/gtatters/Thermimage/blob/master/R)
- [flirpy](https://github.com/LJMUAstroecology/flirpy/tree/main)

Both use the same conversion parameters, but the first is in R. The second one is in Python.
Now comes the question: why would we need another?
FLIR provided the answer, the file format and naming conventions change slightly between different camera's.

This implementation only offers the temperature conversion and uses exiftool to extract the metadata.
[Exiftool](https://exiftool.org/index.html#running)
Exiftool is not fast, but extracts all metadata. Making it rebust and flexible.
See main.py for an example.

To work with different parameter names, both:
`extract_thermal_image()` and `raw2temp()` have to be updated.

The code is tested with FLIR AX8 camera's

## Temerature conversion
Next to parameter naming it's advised to check the calculation in: `raw2temp()`.
It is based on various assumptions which do not generalize to every situation.

## Usage
This module can be used by importing it:

```python
import flir_image_extractor
fir = flir_image_extractor.FlirImageExtractor()

# Print all metadata
metadata = flir.extract_all_metadata('examples/ax8.jpg')
print(dumps(metadata, indent=4))

fir.process_image('examples/ax8.jpg')
fir.plot()
```

Both are RGB images, the original temperature array is available using the `get_thermal_np` or `export_thermal_to_csv` functions.

The functions `get_rgb_np` and `get_thermal_np` yield numpy arrays and can be called from your own script after importing this lib.


## Credits
Raw value to temperature conversion is ported from [Thermimage](https://github.com/gtatters/Thermimage/blob/master/R)
Original Python code from: [flirpy](https://github.com/LJMUAstroecology/flirpy/tree/main)
