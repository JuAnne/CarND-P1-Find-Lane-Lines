"""Udacity Project: CarND-LaneLines-P1 
   Author: Juan Cheng
   Date: 2017.07.19
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

### Global parameters

# Gaussian smoothing
kernel_size = 5

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices using a trapezoid shape
#top_width = 50 # width for top edge of trapezoid, debug use.
#top_height = 325 # height for top edge the trapezoid, debug use.
bottom_width = 0.9  # width of bottom edge of trapezoid, expressed as percentage of image width
top_width = 0.08  # width for top edge of trapezoid, expressed as percentage of image width
top_height = 0.4  # height of the trapezoid, expressed as percentage of image height

# Hough transform 
rho = 2 # distance resolution in pixels of the Hough grid
theta = 2*np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15 # minimum number of votes (intersections in Hough grid cell)
min_line_len = 20 #minimum number of pixels making up a line
max_line_gap = 10 # maximum gap in pixels between connectable line segments


### Helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

	
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

	
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

	
def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """	
    # Error case, don't draw the lines
    if (len(lines) == 0):
        return
    
    # Split lines into left_lines and right_lines, filter out insane segments based on slope thresholds
    left_lines = []
    right_lines = []
    image_center = img.shape[1] / 2
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2 != x1):
                slope = (y2 - y1) / (x2 - x1)
                #Debug
                #length = np.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) )
                #print( x1, y1, x2, y2, slope, length )
                #TO-DO later: remove hard-coded slope thresholds
                if (slope < -0.5 and slope > -1.5 and x1 < image_center and x2 < image_center):
                    left_lines.append(line)
                elif (slope > 0.5 and slope < 1.5 and x1 > image_center and x2 > image_center):
                    right_lines.append(line)
            else:
                slope = 1000 # corner case, give a big number to represent infinite slope, avoiding division by 0 
            
    # Run linear regression to find best fit line for left and right lane lines
    
    # Left lane lines
    left_xvals = []
    left_yvals = []
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        left_xvals.append(x1)
        left_xvals.append(x2)
        left_yvals.append(y1)
        left_yvals.append(y2)
        #Debug: use the following line to see left line segments (green color) before polyfit.
        cv2.line(img, (x1, y1), (x2, y2), [0,255,0], thickness)
    if (len(left_xvals) == 0):
        left_m, left_b = 1,1
    else:
        # Using Least squares polynomial fit (1 degree) to extrapolate lines
        left_m, left_b = np.polyfit(left_xvals, left_yvals, 1)  # y = m*x + b
        # Draw left lines on image
        y1 = int(img.shape[0] * (1 - top_height))
        y2 = img.shape[0]
        x1 = int((y1 - left_b) / left_m)
        x2 = int((y2 - left_b) / left_m)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    
    # Right lane lines
    right_xvals = []
    right_yvals = []
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        right_xvals.append(x1)
        right_xvals.append(x2)
        right_yvals.append(y1)
        right_yvals.append(y2)
        #Debug: use the following line to see right line segments (blue color) before polyfit.
        cv2.line(img, (x1, y1), (x2, y2), [0,0,255], thickness)
    if (len(right_xvals) == 0):
        right_m, right_b = 1,1
    else: 
        # Using 1 degree Least squares polynomial fit to extrapolate lines
        right_m, right_b = np.polyfit(right_xvals, right_yvals, 1)  # y = m*x + b
        # Draw right lines on image
        y1 = int(img.shape[0] * (1 - top_height))
        y2 = img.shape[0]
        x1 = int((y1 - right_b) / right_m)
        x2 = int((y2 - right_b) / right_m)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    #Debug: uncomment the following block to show plot of raw image which helps to debug insane hough lines. Remember to add raw_img as function input.
    """
    test_img = raw_img.copy()
    for points in lines :
        for x1, y1, x2, y2 in points :
            cv2.line(test_img, (x1, y1), (x2, y2), [255, 0, 0], 2)
    plt.imshow(test_img)
    plt.show()
    """
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [255, 0, 0], 6)
    return line_img
   

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
def process_image(image):
    # Pipeline to build lane lines 

    # Convert to grayscale
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')
    #plt.show()

    # Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap='gray')
    #plt.show()
	
    # Apply Canny Edge Detector
    edges = canny(blur_gray, low_threshold, high_threshold)
    #plt.imshow(edges, cmap='Greys_r')
    #plt.show()
	
    # Create a masked edges using trapezoid-shaped region-of-interest (ROI)
    imshape = image.shape
    # the following ROI shape is too wide that it fails challenge video
    """
    low_left = (0,imshape[0])
    top_left = (imshape[1]/2 - top_width/2, top_height)
    top_right = (imshape[1]/2 + top_width/2, top_height)
    low_right = (imshape[1],imshape[0])
    """
    low_indent  = (imshape[1] * (1 - bottom_width)) / 2
    top_indent  = (imshape[1] * (1 - top_width)) / 2
    top_horizon = (imshape[0] * (1 - top_height))
    top_delta = 0 # adjustment for top_indent, set to 0 for normal videos, but need tuning for challenge video   
    low_left = (low_indent, imshape[0])
    top_left = (top_indent + top_delta, top_horizon)
    top_right = (imshape[1]-top_indent + top_delta, top_horizon)
    low_right = (imshape[1]-low_indent, imshape[0])
    vertices = np.array([[low_left, top_left, top_right, low_right]], dtype=np.int32)
    
    # Debug
    #print("LL:", low_left)
    #print("TL:", top_left)
    #print("TR:", top_right)
    #print("LR:", low_right)
    
    masked_edges = region_of_interest(edges, vertices)
    #plt.imshow(masked_edges)
    #plt.show()
    
    #Debug: uncomment following block to show ROI with grey color and see if it fits properly 
    """
    roi_img = image.copy()
    cv2.fillPoly(roi_img, vertices, [128,128,128])
    return roi_img
    """
    
    # Run Hough on edge detected image, return an image with hough lines drawn
    line_img= hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    #plt.imshow(line_img)
    #plt.show()
    
    # Draw lane lines on the original image
    w_image = weighted_img(line_img, image)
    #plt.imshow(w_image)
    #plt.show()
    
    return w_image
   
###End helper functions 
  
### Main script

## test images
#Debug: uncomment the following block to test one image
"""
fname = 'whiteCarLaneSwitch.jpg'
print('input file name ', fname)
in_image = mpimg.imread('test_images/'+fname)
out_image = process_image(in_image)
#Make copies into the test_images_output directory
mpimg.imsave('test_images_output/'+fname, out_image)
"""
#use the following block to test all images
for fname in os.listdir("test_images/"): 
    print('input file name ', fname)
    # Read in the image
    in_image = mpimg.imread('test_images/'+fname)
    out_image = process_image(in_image)
    #Make copies into the test_images_output directory
    mpimg.imsave('test_images_output/'+fname, out_image)


## test videos
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
#clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
#clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
