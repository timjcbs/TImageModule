"""
This code is licensed under GNU GENERAL PUBLIC LICENSE Version 3.
See the accompanying LICENSE file or https://www.gnu.org/licenses/gpl-3.0.en.html for details.
Copyright (c) 2023 Tim Jacobs

Python 3.11.2
RGBMatrix since 19.11.2023

class RGBMatrix to store and manipulate pixel data in RGB colorformat
"""

import numpy as np
import imageio

class RGBMatrix:

    def __init__(self, image_path):
        self.image_path = image_path

        self.r_matrix = None
        self.g_matrix = None
        self.b_matrix = None

        self.r_matrix_for_layer = None
        self.g_matrix_for_layer = None
        self.b_matrix_for_layer = None

        self.original_shape = None          #For control and later use
        self.input_image_datatype = None    #For control and later use

        #Working bitspace = 16bit
        self.minWorkingRGBValue = 0
        self.maxWorkingRGBValue = 2 ** 64 - 1

    def __version__(self):
        return("RGBMatrix--V-0.002")

    def __set_original_shape(self, image):
        self.original_shape = image.shape

    def __set_input_image_datatype(self, image):
        self.input_image_datatype = image.dtype

    """
    To make the code as readable as possible, the channels
    are stored individually with meaningful names, therefore
    the function separates the 3d numpy matrix into three matrices.
    """ 
    def __separate_channels(self, array_3d):
        self.r_matrix = array_3d[:,:,0]
        self.g_matrix = array_3d[:,:,1]
        self.b_matrix = array_3d[:,:,2]

    def __convert_to_one_matrix(self):
        array = np.ones(self.original_shape)
        array[:,:,0] = self.r_matrix
        array[:,:,1] = self.g_matrix
        array[:,:,2] = self.b_matrix
        return array

    def __clip_matrix_values(self):
        self.r_matrix = np.clip(self.r_matrix, self.minWorkingRGBValue, self.maxWorkingRGBValue)
        self.g_matrix = np.clip(self.g_matrix, self.minWorkingRGBValue, self.maxWorkingRGBValue)
        self.b_matrix = np.clip(self.b_matrix, self.minWorkingRGBValue, self.maxWorkingRGBValue)

    def __round_matrix(self):
        self.r_matrix = self.r_matrix.round()
        self.g_matrix = self.g_matrix.round()
        self.b_matrix = self.b_matrix.round()

    """
    Because the std. working bitspace is 64 bit, an imported
    image must be scaled
    """
    def __adjust_input_image_bitspace(self):
        if self.input_image_datatype == "uint8":
            self.__scale_rgb(2**8-1, self.maxWorkingRGBValue)
        if self.input_image_datatype == "uint16":
            self.__scale_rgb(2**16-1, self.maxWorkingRGBValue)
        if self.input_image_datatype == "uint32":
            self._scale_rgb_(2**32-1, self.maxWorkingRGBValue)

    def __scale_rgb(self, start_bitspace, goal_bitspace):
        self.r_matrix = np.interp(self.r_matrix, [0, start_bitspace],[0, goal_bitspace])
        self.g_matrix = np.interp(self.g_matrix, [0, start_bitspace],[0, goal_bitspace])
        self.b_matrix = np.interp(self.b_matrix, [0, start_bitspace],[0, goal_bitspace])

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

    def save_16bit_image(self, output_path):
        if self.matrixIsOnline():
            try:
                self.__round_matrix()
                array = self.__convert_to_one_matrix()
                #Remove alpha channel if it exists
                if array.shape[2] == 4:
                    array = array[:, :, :3]
                #Scale back to 16 bit
                array = np.interp(array, [0, self.maxWorkingRGBValue],[0, 2**16-1])
                imageio.imwrite(output_path, array.astype(np.uint16))
                return 0

            except:
                return 1
                
    def matrixIsOnline(self):
        return self.original_shape is not None

    """
    To make level-related adjustments, a copy of the current matrix
    is created (before starting level-related adjustments).

    As an example for transparency level: After all level-related 
    adjustments have been made, the working matrix (r_matrix, g_matrix, b_matrix)
    is combined with the temporarily saved version (r_matrix_for_layer, 
    g_matrix_for_layer, b_matrix_for_layer) according to transparency factor.
    """
    def set_layer(self):
        if self.matrixIsOnline():
            self.r_matrix_for_layer = self.r_matrix
            self.g_matrix_for_layer = self.g_matrix
            self.b_matrix_for_layer = self.b_matrix

    """
    The copy previously created by 'set_layer' is calculated with the transparency factor 
    and the current working matrices and then saved in the working matrices.
    """
    def end_transparency_layer(self, transparency):
        if self.matrixIsOnline():
            self.r_matrix = (self.r_matrix * transparency) + (1 - transparency) * self.r_matrix_for_layer
            self.g_matrix = (self.g_matrix * transparency) + (1 - transparency) * self.g_matrix_for_layer
            self.b_matrix = (self.b_matrix * transparency) + (1 - transparency) * self.b_matrix_for_layer

    """
    Multiplies the pixel values by the factor, for each individual channel.
    """
    def multiply_brightness_adjustment(self, factor):
        self.multiply_r_brightness_adjustment(factor)
        self.multiply_g_brightness_adjustment(factor)
        self.multiply_b_brightness_adjustment(factor)

    def multiply_r_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.r_matrix = self.r_matrix * factor
            self.__clip_matrix_values()

    def multiply_g_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.g_matrix = self.g_matrix * factor
            self.__clip_matrix_values()

    def multiply_b_brightness_adjustment(self, factor):
        if self.matrixIsOnline():
            self.b_matrix = self.b_matrix * factor
            self.__clip_matrix_values()

if __name__ == "__main__":
    print("RGBMatrix--V-0.002")
