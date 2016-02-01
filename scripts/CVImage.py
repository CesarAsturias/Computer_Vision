import sys, os
import numpy as np
import cv2
import time
import Matcher

# @file CVImage.py
# @author Cesar
# @version 1.0
# Class CVImage. It read and store the image, and implements
# several methods.


class CVImage(object):
    def __init__(self, path):
        # Initialize a number of global variables
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.display_image = None
        self.grey = None
        self.prev_grey = None
        self.keystroke = None
        self.resize_window_width = 0
        self.resize_window_height = 0
        self.path = str(path)
        self.counter = 0
        self.new_image = None
        self.prev_image = None

        # Check the number of images in the path
        self.number_images = self.count_images(self.path)

    def read_image(self):
        # This function reads the new image from the path specified
        # @return img : new image
        # Firts, we have to detail the image that we will load.
        # This is done by appending the  the self.counter attribute to the
        # path provided. This will work only if the images are
        # named like:"xxxxxx.png"
        # where xxxxxx is the number of the image.
        if self.counter < 9 or self.counter == 0:
            number_image = '00000' + str(self.counter)
        elif self.counter < 100 and self.counter > 9:
            number_image = '0000' + str(self.counter)
        elif self.counter < 1000 and self.counter >= 100:
            number_image = '000' + str(self.counter)
        elif self.counter < 10000 and self.counter >= 1000:
            number_image = '00' + str(self.counter)
        elif self.counter < 100000 and self.counter >= 10000:
            number_image = '0' + str(self.counter)
        else:
            number_image = str(self.counter)

        file_name = self.path + '/' + number_image + '.png'

        # Now, read the image
        self.new_image = cv2.imread(file_name)

    def show_image(self, image=0):
        # Show the new image
        # @param image: if 0, show the new image
        # If image = 0, show the new image. Otherwise, show prev_image
        if image == 0:
            cv2.imshow(self.cv_window_name, self.new_image)
        else:
            cv2.imshow(self.cv_prev_window_name, self.prev_image)
        self.keystroke = cv2.waitKey(0)
        # If we pressed the q key, shut down
        if self.keystroke != 1:
            try:
                cc = chr(self.keystroke & 255).lower()
                if cc == 'q':
                    self.cleanup()
                elif cc == 'n':
                    self.acquire()
            except:
                pass

    def copy_image(self):
        # Copy the new image to the previous image
        self.prev_image = self.new_image

    def count_images(self, path):
        # Count the number of images in the specified path
        # @param path: string containing the path of the images
        # @return count: Number of images

        count = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        return count

    def acquire(self):
        # Acquire a new image and store it in new_image
        self.counter = self.counter + 1
        self.read_image()

    def crop_image(self, start, size, img):
        # Extract ROI in the image
        # @param start: pixel coordinates of the start position of the ROI (np
        # array, x and y)
        # @param size : height and width (in pixels) of the ROI (np array)
        # @param img: input image
        # @return roi: ROI
        roi = img[start[1, 0]:start[1, 0] + size[1, 0],
                  start[0, 0]: start[0, 0] + size[0, 0]]
        return roi

    def on_mouse_click(self, event, x, y, flags, param):
        # This function allows the user to selct a ROI using the mouse
        return 0

    def cleanup(self):
        print "Shuting down CVImage"
        cv2.DestroyAllWindows()


def main(args):
    try:

        img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')
        img.read_image()
        img.copy_image()
        img.acquire()
        img.show_image()
        img.show_image(1)
        matcher = Matcher.Matcher()
        matcher.match_flann(img.new_image, img.prev_image)

        # matcher.draw_matches(img.new_image)
        # img.show_image()
        print type(matcher.good_matches)
        print matcher.good_matches[1]

    except KeyboardInterrupt:
        print "Shutting down VisualOdometry"
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
