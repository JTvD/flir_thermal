#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import io
import json
import re
import csv
import subprocess
import numpy as np
from PIL import Image
from os import path
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt


class FlirImageExtractor:
    """FLIr image processing class based on the R package: https://github.com/gtatters/Thermimage
    """
    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.flir_img_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True
        self.rgb_image_np = None
        self.thermal_image_np = None

    def extract_all_metadata(self, flir_img_filename: str):
        """
        Given a valid image path, extract all metadata
        Args:
            flir_img_filename:
        Return:
            meta_dict: dict
        """
        if not path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        meta_json = subprocess.check_output(
            [self.exiftool_path, flir_img_filename, '-j'])
        #exiftool a.jpg --system:all -e
        meta_dict = json.loads(meta_json.decode())[0]

        return meta_dict

    def process_image(self, flir_img_filename: str):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        Args:
            flir_img_filename:
        Return:
            -
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()

    def get_image_type(self) -> str:
        """ Get the embedded thermal image type, generally can be TIFF or PNG
        Return
            imagetype: str
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]
        return meta['RawThermalImageType']

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        Return:

        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
        meta_dict = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, meta_dict))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, meta_dict: dict):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # and https://github.com/LJMUAstroecology/flirpy/tree/main
        # a detailed explanation of the calculation can be found there
        """

        # constants
        ATA1 = float(meta_dict.get("AtmosphericTransAlpha1") or meta_dict.get("Atmospheric Trans Alpha 1") or 0.006569)
        ATA2 = float(meta_dict.get("AtmosphericTransAlpha2") or meta_dict.get("Atmospheric Trans Alpha 2") or 0.01262)
        ATB1 = float(meta_dict.get("AtmosphericTransBeta1") or meta_dict.get("Atmospheric Trans Beta 1") or -0.002276)
        ATB2 = float(meta_dict.get("AtmosphericTransBeta2") or meta_dict.get("Atmospheric Trans Beta 2") or -0.00667)
        ATX = float(meta_dict.get("AtmosphericTransX") or meta_dict.get("Atmospheric Trans X") or 1.9)
        PR1 = float(meta_dict.get("PlanckR1") or meta_dict.get("Planck R1") or 21106.77)
        PR2 = float(meta_dict.get("PlanckR2") or meta_dict.get("Planck R2") or 0.012545258)
        PO = float(meta_dict.get("PlanckO") or meta_dict.get("Planck O") or -7340)
        PB = float(meta_dict.get("PlanckB") or meta_dict.get("Planck B") or 1501)
        PF = float(meta_dict.get("PlanckF") or meta_dict.get("Planck F") or 1)
        E = float(meta_dict.get("Emissivity", 1))
        IRT = float(meta_dict.get("IRWindowTransmission") or meta_dict.get("IR Window Transmission") or 1)
        # Drop the units
        if "IRWindowTemperature" in meta_dict:
            IRWTemp = float(meta_dict["IRWindowTemperature"].split()[0])
        else:
            IRWTemp = 20
        if "ObjectDistance" in meta_dict:
            OD = float(meta_dict["ObjectDistance"].split()[0])
        else:
            OD = 1
        if "AtmosphericTemperature" in meta_dict:
            ATemp = float(meta_dict["AtmosphericTemperature"].split()[0])
        else:
            ATemp = 20
        if "ReflectedApparentTemperature" in meta_dict:
            RTemp = float(meta_dict["ReflectedApparentTemperature"].split()[0])
        else:
            RTemp = 20
        if "RelativeHumidity" in meta_dict:
            RH = float(meta_dict["RelativeHumidity"].split()[0])
        else:
            RH = 50

        # Transmission through window (calibrated)
        window_emissivity = 1 - IRT
        window_reflectivity = 0

        # Converts relative humidity into water vapour pressure (mmHg)
        water = (RH/100.0)*exp(1.5587+0.06939*(ATemp)-0.00027816*(ATemp)**2+0.00000068455*(ATemp)**3)

        # tau1 = ATX*np.exp(-np.sqrt(OD/2))
        tau1 = ATX*np.exp(-np.sqrt(OD/2)*(ATA1+ATB1*np.sqrt(water)))+(1-ATX)*np.exp(-np.sqrt(OD/2)*(ATA2+ATB2*np.sqrt(water)))
        tau2 = tau1

        # Transmission through atmosphere - equations from Minkina and Dudzik's Infrared Thermography Book
        # This script assumes the thermal window is at the mid-point (OD/2) between the source and camera sensor

        # Radiance reflecting off the object before the window
        raw_refl = PR1/(PR2*(np.exp(PB/(RTemp+273.15))-PF))-PO
        # attn = the attenuated radiance (in raw units)
        raw_refl_attn = (1-E)/E*raw_refl
        # Radiance from the atmosphere (before the window)
        raw_atm1 = PR1/(PR2*(np.exp(PB/(ATemp+273.15))-PF))-PO
        # attn = the attenuated radiance (in raw units)
        raw_atm1_attn = (1-tau1)/E/tau1*raw_atm1

        raw_window = PR1/(PR2*(np.exp(PB/(IRWTemp+273.15))-PF))-PO
        einv = 1./E/tau1/IRT
        raw_window_attn = window_emissivity*einv*raw_window

        raw_refl2 = raw_refl
        raw_refl2_attn = window_reflectivity*einv*raw_refl2

        raw_atm2 = raw_atm1
        ediv = einv/tau2
        raw_atm2_attn = (1-tau2)*ediv*raw_atm2

        raw_sub = -raw_atm1_attn-raw_atm2_attn-raw_window_attn-raw_refl_attn - raw_refl2_attn
        raw_object = np.add(np.multiply(raw, ediv), raw_sub)
        raw_object = np.add(raw_object, PO)
        raw_object = np.multiply(raw_object, PR2)
        raw_object_inv = np.multiply(np.reciprocal(raw_object), PR1)
        raw_object_inv = np.add(raw_object_inv, PF)
        raw_object_log = np.log(raw_object_inv)
        temp = np.multiply(np.reciprocal(raw_object_log), PB)

        return temp - 273.15


    @staticmethod
    def extract_float(dirtystr: str) -> float:
        """ Extract the float value of a string, helpful for parsing the exiftool data
        Args:
            dirtystr: string with float value
        Return:
            float value without the units
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])

    def plot(self):
        """ Plot the rgb + thermal image
        Return:
            -
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()

        plt.subplot(1, 2, 1)
        plt.imshow(thermal_np, cmap='hot')
        plt.subplot(1, 2, 2)
        plt.imshow(rgb_np)
        plt.show()

    def save_images(self):
        """ Save the extracted images to files
        Return:
            -
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.extract_thermal_image()

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename: str):
        """ Save the thermal data in a csv
        Return:
            -
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])
            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])
            writer.writerows(pixel_values)
