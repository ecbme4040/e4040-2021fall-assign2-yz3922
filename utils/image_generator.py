#!/usr/bin/env/ python
# ECBM E4040 Fall 2020 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
import os

try:
    from scipy.ndimage.interpolation import rotate
except ModuleNotFoundError:
    os.system('pip install scipy')
    from scipy.ndimage.interpolation import rotate

class ImageGenerator(object):
    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to False.
        #
        # Hint: Since you may directly perform transformations on x and y, and don't want your original data to be contaminated 
        # by those transformations, you should use numpy array build-in copy() method. 
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.__init__() not implemented!') # delete me
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################
        
        
        # One way to use augmented data is to store them after transformation (and then combine all of them to form new data set).
        # Following variables (along with create_aug_data() function) is one kind of implementation.
        # You can either figure out how to use them or find out your own ways to create the augmented dataset.
        # if you have your own idea of creating augmented dataset, just feel free to comment any codes you don't need
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.bright = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N
    
    
    def create_aug_data(self):
        # If you want to use function create_aug_data() to generate new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1.store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2.increase self.N_aug by the number of transformed data,
        # 3.you should also return the transformed data in order to show them in task4 notebook
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        if self.bright:
            self.x_aug = np.vstack((self.x_aug,self.bright[0]))
            self.y_aug = np.hstack((self.y_aug,self.bright[1]))
            
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        ########################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.next_batch_gen() not implemented!') # delete me
                
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.show() not implemented!') # delete me
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """
        
        # TODO: Implement the translate() function. You may wonder what values to append to the edge after the shift. Here, use rolling instead. For
        # example, if you shift 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        
        # HINT: use np.roll (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.translate() not implemented!') # delete me
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        """
                
        # TODO: Implement the rotate() function. The angle of rotation should match the value specified by angle.
        # HINT: Use scipy.ndimage.interpolation.rotate (which is already imported for you)
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.rotate() not implemented!') # delete me
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################
        

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        assert mode == 'h' or 'v' or 'hv'
        if mode == 'h':
            flipped = np.flip(self.x.copy(), axis=2)
            self.is_horizontal_flip = not self.is_horizontal_flip
        elif mode == 'v':
            flipped = np.flip(self.x.copy(), axis=1)
            self.is_vertical_flip = not self.is_vertical_flip
        elif mode == 'hv':
            flipped = np.flip(np.flip(self.x.copy(), axis=0), axis=1)
            self.is_horizontal_flip = not self.is_horizontal_flip
            self.is_vertical_flip = not self.is_vertical_flip
        else:
            raise ValueError('Mode should be \'h\' or \'v\' or \'hv\'')
        print('Vertical flip: ', self.is_vertical_flip, 'Horizontal flip: ', self.is_horizontal_flip)
    
        self.flipped = (flipped,self.y.copy())
        self.N_aug += self.N
        return flipped

    
    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        # TODO: Implement the add_noise function. Remember to record the boolean value is_add_noise. Any noise function
        # is acceptable.
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        print('./utils/image_generator.ImageGenerator.add_noise() not implemented!') # delete me
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def brightness(self, factor):
        """
        Scale the pixel values to increase the brightness
        :param factor: A number greater than or equal to 1 that decides how each pixel in the image will be scaled. If factor is 2, then 
                       all pixel values will be doubled.
        :return bright: dataset with increased brightness
        """
        assert factor >= 1
        if not self.is_bright:
            self.is_bright = True
        bright = self.x.copy()
        for i in range(bright.shape[0]):
            bright[i, :, :, :] = (bright[i,:,:,:] * factor).astype(int)
            bright[i, :, :, :][bright[i,:,:,:] >= 255] = 255
            
        self.bright = (bright, self.y.copy())
        self.N_aug += self.N
        print("Brightness increased by a factor of:", factor)
        return bright

      