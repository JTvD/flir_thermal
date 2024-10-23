import matplotlib.pyplot as plt
import flir_image_extractor
from json import dumps

flir = flir_image_extractor.FlirImageExtractor(exiftool_path='exiftool/exiftool')

# Print all metadata
metadata = flir.extract_all_metadata('examples/ax8.jpg')
print(dumps(metadata, indent=4))

# Process the images, points in the thermal plot are in degrees
flir.process_image('examples/ax8.jpg')
thermal = flir.get_thermal_np()
rgb = flir.get_rgb_np()
flir.plot()
plt.show()
