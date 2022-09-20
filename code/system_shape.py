"""
    Module with some definitions of the sytems shapes and special
    builders:

        - author:  Marcos de Medeiros
        - email: mhlmedeiros@gmail.com

"""
import numpy as np


class Rectangle:
    """
    Class to define callable objects to define the
    shape of the scattering region of a rectangular
    system.
    """
    def __init__(self, width, length):
        '''
        Calling the scattering region as strip:
        W = width of the strip
        L = length of the strip
        '''
        self.width = width
        self.length = length

    def __call__(self, pos):
        width, length = self.width, self.length
        x, y = pos
        return -width/2 < y < width/2 and -length/2 <= x <= length/2

    def leads(self, pos):
        width = self.width
        _, y = pos
        return -width/2 < y < width/2
