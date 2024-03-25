"""
This code is licensed under GNU GENERAL PUBLIC LICENSE Version 3.
See the accompanying LICENSE file or 
https://www.gnu.org/licenses/gpl-3.0.en.html 
for details.

Copyright (c) 2023 Tim Jacobs

Python 3.11
RGBMatrix since 19.11.2023

class RGBMatrix to store and manipulate pixel data in RGB colorformat
central for the conversion of color images into grayscale images

short-term goal: 
    - improve control of the persistence of the image 
      (self.original_shape can be used, among other things)
"""

import numpy as np
import imageio
import concurrent.futures

class RGBMatrix:

    def __init__(self, image_path):
        self.image_path = image_path

        self.r_matrix = None
        self.g_matrix = None
        self.b_matrix = None

        self.r_matrix_for_layer = None
        self.g_matrix_for_layer = None
        self.b_matrix_for_layer = None

        self.layer_raster = None

        self.original_shape = None  # For control and later use
        self.input_image_datatype = None  # For control and later use

        # Working bitspace = 64bit
        self.maxWorkingRGBValue = 2 ** 64 - 1

    def __version__(self):
        return ("RGBMatrix---V-0.002---using-py3.9")
    
    def datatype(self):
        if self.matrixIsOnline:
            return(self.r_matrix.dtype)
        else:
            return("no image loaded yet")

    """
    will be used later to control the persistence of the image
    and and during export, when the three matrices are combined 
    into a 3D matrix. it is also used to check whether an image
    is loaded
    """
    def __set_original_shape(self, image):
        self.original_shape = image.shape

    def __set_input_image_datatype(self, image):
        self.input_image_datatype = image.dtype

    """
    to make the code as readable as possible, the channels
    are stored individually with meaningful names, therefore
    the function separates the 3d numpy matrix into three matrices.
    """
    def __separate_channels(self, array_3d):
        self.r_matrix = array_3d[:, :, 0]
        self.g_matrix = array_3d[:, :, 1]
        self.b_matrix = array_3d[:, :, 2]

    """
    is used during export
    """
    def __convert_to_one_matrix(self):
        array = np.ones(self.original_shape)
        array[:, :, 0] = self.r_matrix
        array[:, :, 1] = self.g_matrix
        array[:, :, 2] = self.b_matrix
        return array

    """
    it is not necessary to check whether the matrix is online
    because this function is only called from other functions
    that do this themselves
    """
    def __clip_all_matrix_values(self):
        self.__clip_r_matrix_values()
        self.__clip_g_matrix_values()
        self.__clip_b_matrix_values()

    def __clip_r_matrix_values(self):
        self.r_matrix = np.clip(self.r_matrix, self.minWorkingRGBValue,\
                                self.maxWorkingRGBValue)

    def __clip_g_matrix_values(self):
        self.g_matrix = np.clip(self.g_matrix, self.minWorkingRGBValue,\
                                self.maxWorkingRGBValue)

    def __clip_b_matrix_values(self):
        self.b_matrix = np.clip(self.b_matrix, self.minWorkingRGBValue,\
                                self.maxWorkingRGBValue)

    """
    is used during export
    """
    def __round_matrix(self):
        self.r_matrix = self.r_matrix.round()
        self.g_matrix = self.g_matrix.round()
        self.b_matrix = self.b_matrix.round()

    """
    because the std. working bitspace is 64 bit, an imported
    image must be scaled
    """
    def __adjust_input_image_bitspace(self):
        if self.input_image_datatype == "uint8":
            self.__scale_rgb(2 ** 8 - 1, self.maxWorkingRGBValue)
        if self.input_image_datatype == "uint16":
            self.__scale_rgb(2 ** 16 - 1, self.maxWorkingRGBValue)
        if self.input_image_datatype == "uint32":
            self.__scale_rgb(2 ** 32 - 1, self.maxWorkingRGBValue)

    """
    by calling the individual rounding functions in parallel,
    all channels are rounded in parallel
    """
    def __scale_rgb(self, start_bitspace, goal_bitspace):

        def scale_matrix(matrix):
            return np.interp(matrix, [0, start_bitspace], [0, goal_bitspace])

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(scale_matrix, matrix)
                for matrix in [self.r_matrix, self.g_matrix, self.b_matrix]
            ]
        results = [future.result() for future in futures]

        self.r_matrix, self.g_matrix, self.b_matrix = results[0], results[1],\
            results[2]        

    """
    permitted formats and properties are regulated by imageio.
    depending on success, return value is returned.
    """
    def load_image(self):
        try:
            image = imageio.imread(self.image_path)
            self.__separate_channels(image)
            self.__set_input_image_datatype(image)
            self.__adjust_input_image_bitspace()
            self.__set_original_shape(image)
            return 0

        except:
            return 1

    """
    permitted formats and properties are regulated by imageio.
    depending on success, return value is returned.
    at 16bit, tiff and png are particularly useful
    """
    def save_16bit_image(self, output_path):
        if self.matrixIsOnline():
            try:
                self.__round_matrix()
                array = self.__convert_to_one_matrix()

                # Scale back to 16 bit
                array = np.interp(array, [0, self.maxWorkingRGBValue], [0, 2 ** 16 - 1])
                imageio.imwrite(output_path, array.astype(np.uint16))
                return 0

            except:
                return 1

    def matrixIsOnline(self):
        return self.original_shape is not None

    """
    to make level-related adjustments, a copy of the current matrix
    is created (before starting level-related adjustments)

    as an example for transparency level: After all level-related 
    adjustments have been made, the working matrix (r_matrix, g_matrix, b_matrix)
    is combined with the temporarily saved version (r_matrix_for_layer, 
    g_matrix_for_layer, b_matrix_for_layer) according to transparency factor

    the storage requirement therefore increases through the use of layers
    """
    def set_layer(self):
        if self.matrixIsOnline():
            self.r_matrix_for_layer = self.r_matrix
            self.g_matrix_for_layer = self.g_matrix
            self.b_matrix_for_layer = self.b_matrix

    """
    the copy previously created by 'set_layer' is calculated with the transparency factor 
    and the current working matrices and then saved in the working matrices
    """
    def end_transparency_layer(self, transparency):
        if self.matrixIsOnline():
            self.r_matrix = (self.r_matrix * transparency) + (1 - transparency) * self.r_matrix_for_layer
            self.g_matrix = (self.g_matrix * transparency) + (1 - transparency) * self.g_matrix_for_layer
            self.b_matrix = (self.b_matrix * transparency) + (1 - transparency) * self.b_matrix_for_layer

    """
    parallel call of the channel-specific functions
    multiplies the pixel values by the factor, for each individual channel
    """
    def multiply_brightness_adjustment(self, factor):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_r = executor.submit(self.multiply_r_brightness_adjustment, factor)
            future_g = executor.submit(self.multiply_g_brightness_adjustment, factor)
            future_b = executor.submit(self.multiply_b_brightness_adjustment, factor)

            concurrent.futures.wait([future_r, future_g, future_b])

    def multiply_r_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.r_matrix = (self.r_matrix * factor)
            self.__clip_r_matrix_values()

    def multiply_g_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.g_matrix = self.g_matrix * factor
            self.__clip_g_matrix_values()

    def multiply_b_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.b_matrix = self.b_matrix * factor
            self.__clip_b_matrix_values()

    """
    factors do not have to add up to 1.
    normal grayscale images are calculated using the luminance
    formula with R = 0.299 G = 0.587 B = 0.114
    """
    def convert_to_grayscale(self, r_factor, g_factor, b_factor):
        if self.matrixIsOnline():
            r_factor = float(r_factor)
            g_factor = float(g_factor)
            b_factor = float(b_factor)
            self.r_matrix = (self.r_matrix * r_factor) + (self.g_matrix * g_factor) + (self.b_matrix * b_factor)
            self.b_matrix = self.r_matrix
            self.g_matrix = self.r_matrix

    """
    iterates over the matrix, looking at the contrast of
    the value with the value to the left above and to the
    right below
    """
    def normal_deviation_sharpening(self):
        if self.matrixIsOnline():
            num_rows, num_cols = self.original_shape
            num_rows += 1

            for i in range(num_rows - 2):
                for j in range(num_cols - 2):
                    value = self.r_matrix[i][j]
                    sidevalue = self.r_matrix[i][j-1]
                    deviation = value / sidevalue

                    self.layer_raster[i][j] =  deviation

if __name__ == "__main__":
    print("RGBMatrix--V-0.002")
