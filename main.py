import matplotlib.pyplot as plt
import flir_image_extractor
import numpy as np
import cv2

from json import dumps

flir = flir_image_extractor.FlirImageExtractor(exiftool_path='exiftool/exiftool')

# Print all metadata
metadata = flir.extract_all_metadata('examples/flir_e8xt_ir.jpg')
print(dumps(metadata, indent=4))

# Process the images, points in the thermal plot are in degrees
flir.process_image('examples/flir_e8xt_ir.jpg')

# Extract and save the results
thermal = flir.get_thermal_np()
thermal_colored = flir.get_thermal_false_color()
rgb = flir.get_rgb_np()

cv2.imwrite('e8xt_rgb.png', rgb)
cv2.imwrite('e8xt_falsecolor.png', thermal_colored)
np.savetxt('e8xt_thermal.csv', thermal, delimiter=',')

flir.plot()
plt.show()
