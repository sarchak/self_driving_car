

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
%matplotlib inline
```

### Helper Methods
Simple helper methods for plotting and saving to output images to output directories


```python
def draw_plots(img_1, title1, img_2, title2, gray=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if gray:
        ax1.imshow(img_1, cmap='gray')
    else:
        ax1.imshow(img_1)
    ax1.set_title(title1, fontsize=50)
    if gray:
        ax2.imshow(img_2, cmap='gray')
    else:
        ax2.imshow(img_2)
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

```


```python
def draw_img_variants(img, scheme = 'HLS'):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, : , 0]
    l_channel = hls[:, : , 1]
    s_channel = hls[:, : , 2]    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.set_title("Original")
    ax1.imshow(img)
    ax2.set_title("H Channel")
    ax2.imshow(h_channel)
    ax3.set_title("L Channel")
    ax3.imshow(l_channel)
    ax4.set_title("S Channel")
    ax4.imshow(s_channel)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


```python
def save_to_output(filename, img):
    cv2.imwrite('output_images/'+filename, img)
```

### Calibrate Camera

#### Most of the **immages actually have 6*9 corners**
The code for this step is contained in the code cell (5-7) of the IPython notebook .

Camera calibration is a process of identifying the parameters of the Camera lens and image sensors of an image.
We start by preparing the objects points which will correspond to the (x,y,z) cooordinates of the chessboard corners in real world ( With z being 0).

A normal calibration process involves taking a few samples ( like 15 to 20 images) for calibration process. The real world coordinates won't changes and hence we keep the object point the same for all images and only change the image points looking depending on whether corners were detected or no. We will keep an array of real world object points and image points when we detect the corners.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort()


```python
images = glob.glob('camera_cal/calibration*.jpg')
```


```python
def calibrate_camera():
    objpoints = []
    imgpoints = []
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    img_shape = None
    for img_file in images:
        img = mpimg.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape,None,None)
    return (mtx, dist)
```


```python
mtx, dist = calibrate_camera()
```


```python
img_test = mpimg.imread('camera_cal/calibration1.jpg')
uimage = cv2.undistort(img_test, mtx, dist, None, mtx)
draw_plots(img_test, "Original Image", uimage, "Undistored Image")
save_to_output('undistorted_image.jpg', uimage)
```


![png](output_10_0.png)



```python
img_test = mpimg.imread('test_images/test4.jpg')
uimage = cv2.undistort(img_test, mtx, dist, None, mtx)
draw_plots(img_test, "Original Image", uimage, "Undistored Image")
save_to_output('undistorted_road_image.jpg', uimage)
```


![png](output_11_0.png)


### Sobel, Magnitude 

   * Sobel threshold helper
   * Magnitude threshold helper
   * Unwrap helper
   * Direction threshold


```python
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    gradx = 1 if orient == 'x' else 0
    grady = 1 if orient == 'y' else 0
    sobel = cv2.Sobel(gray, cv2.CV_64F, gradx, grady)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8( 255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output
```


```python
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output
```


```python
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    print(ret)
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(gray, (nx, ny), corners, ret)
        plt.imshow(gray)
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)
                                     
        return warped, M

```


```python
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output
```

Apply different threshold methods and test pipeline


```python
img_test = mpimg.imread('outfile/out_1485870524.005789.jpg')
uimage = cv2.undistort(img_test, mtx, dist, None, mtx)
gradx = abs_sobel_thresh(uimage, orient='x', thresh_min=10, thresh_max=250)
mag_binary = mag_thresh(uimage, sobel_kernel=15, mag_thresh=(100, 200))
dir_binary = dir_threshold(uimage, sobel_kernel=15, thresh=(np.pi/1, np.pi/2))
combined = np.zeros_like(dir_binary)
combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
save_to_output('binary_thresholded.jpg', combined)
plt.imshow(combined, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x110d8f4a8>




![png](output_18_1.png)



```python

```


```python

```


```python
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped
```


```python

```

### Bird's Eye view (Perspective Transform)


```python
def draw_lines(img, points):
    points = [tuple(x) for x in points]
    cv2.line(img,points[0],points[1],(255,0,0),5)
    cv2.line(img,points[1],points[2],(255,0,0),5)
    cv2.line(img,points[2],points[3],(255,0,0),5)
    cv2.line(img,points[3],points[0],(255,0,0),5)
    return img
```


```python

img_test = mpimg.imread('test_images/straight_lines1.jpg')
img_size = (img_test.shape[1], img_test.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6)  - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 50, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

# src = np.float32([[800, 500], [950, 585], [450, 585], [550, 500]])
# dst = np.float32([[990, 500], [990, 650], [400, 650], [400, 500]])

warped = warper(img_test, src, dst)
draw_plots(draw_lines(img_test, src), 'Original', draw_lines(warped, dst), "Warped Bird's Eye View")
save_to_output('birds-eye.jpg', warped)
```


![png](output_25_0.png)


# Channel Exploration of images


```python
img_test = mpimg.imread('test_images/straight_lines1.jpg')
```


```python
draw_img_variants(img_test)
```


![png](output_28_0.png)


### Color and Threshold Gradient


```python
def color_gradient_threshold(img, s_thresh=(130, 255), sx_thresh=(45, 250)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.8, 1.2))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) & (dir_binary == 1)] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) | (sxbinary == 1))] = 1
    return combined_binary, color_binary
```


```python
# img_test = mpimg.imread('outfile/out_1485870518.219037.jpg')
# img_test = mpimg.imread('outfile/out_1485870371.647219.jpg')
# img_test = mpimg.imread('outfile/out_1485870375.444886.jpg')
img_test = mpimg.imread('test_images/test2.jpg')
gradient_image, colored_binary = color_gradient_threshold(img_test)
draw_plots(img_test, 'Original', gradient_image, 'Gradient Thresholded Image', True)
draw_plots(img_test, 'Original', colored_binary, 'Gradient Thresholded Colored Binary')

```


![png](output_31_0.png)



![png](output_31_1.png)



```python
binary_warped = warper(gradient_image, src, dst)
plt.imshow(binary_warped, cmap='gray')
print(binary_warped.shape)
```

    (720, 1280)



![png](output_32_1.png)


### Histogram Exploration


```python
def histogram(img):
    histogram_img = np.sum(img[img.shape[0]/2:,:], axis=0)
    plt.plot(histogram_img)
```


```python
histogram(binary_warped)
```


![png](output_35_0.png)


### Polynomial Fit

Fit the polynomial as described in the videos. Use the sliding window to identify the left and right pixel that correspond to lane lines.


```python
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_size = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_size
    win_y_high = binary_warped.shape[0] - window*window_size
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

```


```python
fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(fit_leftx, fity, color='yellow')
plt.plot(fit_rightx, fity, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
```




    (720, 0)




![png](output_38_1.png)



```python
yvals = fity
image  = img_test
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
Minv = cv2.getPerspectiveTransform(dst, src)
# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([fit_leftx, yvals]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, yvals])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(img_test, 1, newwarp, 0.3, 0)
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x1052f0518>




![png](output_39_1.png)



```python

```


```python
y_eval = np.max(yvals)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
print(left_curverad, right_curverad)
```

    2210.45999122 3942.67322031



```python
import datetime
```


```python
left_fit_avg = []
right_fit_avg = []
def pipeline(img):
    orig = np.copy(img)
    if img == None:
        print("NO DATA")
    gradient_image, colored_binary = color_gradient_threshold(img)
    binary_warped = warper(gradient_image, src, dst)
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_size = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_size
        win_y_high = binary_warped.shape[0] - window*window_size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Store fit coefficients
    left_fit_avg.append(left_fit)
    right_fit_avg.append(right_fit)
    
    # Mean of last 30 coefficients
    left_fit = np.mean(left_fit_avg[-30:], axis=0)
    right_fit = np.mean(right_fit_avg[-30:], axis=0)
    fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    yvals = fity
    image  = np.copy(img)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(orig, 1, newwarp, 0.3, 0)
    y_eval = np.max(yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])


    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    screen_middel_pixel = img.shape[1]/2
    left_lane_pixel = fit_leftx[-1]    # x position for left lane
    right_lane_pixel = fit_rightx[-1]   # x position for right lane
    car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
    screen_off_center = screen_middel_pixel-car_middle_pixel
    meters_off_center = np.absolute(xm_per_pix * screen_off_center)
    cv2.rectangle(result, (0, img.shape[0] - 130), (img.shape[1], img.shape[0]),(0, 255, 0), -1)
    cv2.putText(result, "Radius of Curvature (right)      = %d (m)" %(int(right_curverad)) , (10,img.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)    
    cv2.putText(result, "Radius of Curvature (left)       = %d (m)" %(int(left_curverad)) , (10,img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(result, "Vehicle position, left of center = %.2f" % (meters_off_center) , (10,img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    plt.imshow(result)
    return result
```


```python
img_test = mpimg.imread('outfile/out_1485870375.444886.jpg')
gradient_image, colored_binary = color_gradient_threshold(img_test)
binary_warped = warper(gradient_image, src, dst)
draw_plots(img_test, 'Original', gradient_image, 'Gradient Thresholded Image', True)
draw_plots(img_test, 'Original', binary_warped, 'Gradient Thresholded Colored Binary')
pipeline(img_test)
```




    array([[[ 95, 135, 184],
            [ 95, 135, 184],
            [ 95, 135, 184],
            ..., 
            [ 58, 116, 166],
            [ 58, 116, 166],
            [ 58, 116, 166]],
    
           [[ 95, 135, 184],
            [ 95, 135, 184],
            [ 95, 135, 184],
            ..., 
            [ 58, 116, 166],
            [ 58, 116, 166],
            [ 58, 116, 166]],
    
           [[ 95, 135, 184],
            [ 95, 135, 184],
            [ 95, 135, 184],
            ..., 
            [ 58, 116, 166],
            [ 58, 116, 166],
            [ 58, 116, 166]],
    
           ..., 
           [[  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0],
            ..., 
            [  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0]],
    
           [[  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0],
            ..., 
            [  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0]],
    
           [[  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0],
            ..., 
            [  0, 255,   0],
            [  0, 255,   0],
            [  0, 255,   0]]], dtype=uint8)




![png](output_44_1.png)



![png](output_44_2.png)



```python

```


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython.display import display
output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
%time output_clip.write_videofile(output, audio=False)
```

    [MoviePy] >>>> Building video output.mp4
    [MoviePy] Writing video output.mp4


    
    
      0%|          | 0/1261 [00:00<?, ?it/s][A[A
    
      0%|          | 1/1261 [00:00<04:21,  4.83it/s][A[A
    
      0%|          | 2/1261 [00:00<04:18,  4.87it/s][A[A
    
      0%|          | 3/1261 [00:00<04:24,  4.76it/s][A[A
    
      0%|          | 4/1261 [00:00<04:20,  4.82it/s][A[A
    
      0%|          | 5/1261 [00:01<04:15,  4.92it/s][A[A
    
      0%|          | 6/1261 [00:01<04:12,  4.97it/s][A[A
    
      1%|          | 7/1261 [00:01<04:09,  5.02it/s][A[A
    
      1%|          | 8/1261 [00:01<04:10,  4.99it/s][A[A
    
      1%|          | 9/1261 [00:01<04:10,  5.00it/s][A[A
    
      1%|          | 10/1261 [00:02<04:06,  5.08it/s][A[A
    
      1%|          | 11/1261 [00:02<04:05,  5.10it/s][A[A
    
      1%|          | 12/1261 [00:02<04:05,  5.08it/s][A[A
    
      1%|          | 13/1261 [00:02<04:09,  5.00it/s][A[A
    
      1%|          | 14/1261 [00:02<04:09,  5.00it/s][A[A
    
      1%|          | 15/1261 [00:03<04:09,  4.99it/s][A[A
    
      1%|▏         | 16/1261 [00:03<04:10,  4.97it/s][A[A
    
      1%|▏         | 17/1261 [00:03<04:11,  4.96it/s][A[A
    
      1%|▏         | 18/1261 [00:03<04:07,  5.03it/s][A[A
    
      2%|▏         | 19/1261 [00:03<04:06,  5.03it/s][A[A
    
      2%|▏         | 20/1261 [00:04<04:06,  5.04it/s][A[A
    
      2%|▏         | 21/1261 [00:04<04:07,  5.02it/s][A[A
    
      2%|▏         | 22/1261 [00:04<04:04,  5.08it/s][A[A
    
      2%|▏         | 23/1261 [00:04<04:06,  5.03it/s][A[A
    
      2%|▏         | 24/1261 [00:04<04:05,  5.04it/s][A[A
    
      2%|▏         | 25/1261 [00:04<04:02,  5.10it/s][A[A
    
      2%|▏         | 26/1261 [00:05<03:59,  5.15it/s][A[A
    
      2%|▏         | 27/1261 [00:05<04:01,  5.11it/s][A[A
    
      2%|▏         | 28/1261 [00:05<03:57,  5.18it/s][A[A
    
      2%|▏         | 29/1261 [00:05<03:56,  5.21it/s][A[A
    
      2%|▏         | 30/1261 [00:05<03:56,  5.20it/s][A[A
    
      2%|▏         | 31/1261 [00:06<03:59,  5.15it/s][A[A
    
      3%|▎         | 32/1261 [00:06<04:02,  5.08it/s][A[A
    
      3%|▎         | 33/1261 [00:06<04:01,  5.08it/s][A[A
    
      3%|▎         | 34/1261 [00:06<03:58,  5.14it/s][A[A
    
      3%|▎         | 35/1261 [00:06<03:57,  5.15it/s][A[A
    
      3%|▎         | 36/1261 [00:07<03:58,  5.14it/s][A[A
    
      3%|▎         | 37/1261 [00:07<04:00,  5.08it/s][A[A
    
      3%|▎         | 38/1261 [00:07<04:03,  5.02it/s][A[A
    
      3%|▎         | 39/1261 [00:07<04:02,  5.05it/s][A[A
    
      3%|▎         | 40/1261 [00:07<04:12,  4.84it/s][A[A
    
      3%|▎         | 41/1261 [00:08<04:18,  4.72it/s][A[A
    
      3%|▎         | 42/1261 [00:08<04:24,  4.60it/s][A[A
    
      3%|▎         | 43/1261 [00:08<04:22,  4.65it/s][A[A
    
      3%|▎         | 44/1261 [00:08<04:14,  4.79it/s][A[A
    
      4%|▎         | 45/1261 [00:09<04:08,  4.90it/s][A[A
    
      4%|▎         | 46/1261 [00:09<04:07,  4.91it/s][A[A
    
      4%|▎         | 47/1261 [00:09<04:05,  4.94it/s][A[A
    
      4%|▍         | 48/1261 [00:09<04:00,  5.04it/s][A[A
    
      4%|▍         | 49/1261 [00:09<04:02,  5.01it/s][A[A
    
      4%|▍         | 50/1261 [00:09<03:59,  5.06it/s][A[A
    
      4%|▍         | 51/1261 [00:10<03:58,  5.08it/s][A[A
    
      4%|▍         | 52/1261 [00:10<04:00,  5.03it/s][A[A
    
      4%|▍         | 53/1261 [00:10<03:59,  5.04it/s][A[A
    
      4%|▍         | 54/1261 [00:10<03:59,  5.03it/s][A[A
    
      4%|▍         | 55/1261 [00:10<03:59,  5.03it/s][A[A
    
      4%|▍         | 56/1261 [00:11<03:58,  5.06it/s][A[A
    
      5%|▍         | 57/1261 [00:11<03:59,  5.02it/s][A[A
    
      5%|▍         | 58/1261 [00:11<03:58,  5.05it/s][A[A
    
      5%|▍         | 59/1261 [00:11<03:58,  5.03it/s][A[A
    
      5%|▍         | 60/1261 [00:11<03:57,  5.06it/s][A[A
    
      5%|▍         | 61/1261 [00:12<03:58,  5.04it/s][A[A
    
      5%|▍         | 62/1261 [00:12<04:08,  4.82it/s][A[A
    
      5%|▍         | 63/1261 [00:12<04:07,  4.84it/s][A[A
    
      5%|▌         | 64/1261 [00:12<04:00,  4.99it/s][A[A
    
      5%|▌         | 65/1261 [00:12<04:00,  4.96it/s][A[A
    
      5%|▌         | 66/1261 [00:13<03:57,  5.03it/s][A[A
    
      5%|▌         | 67/1261 [00:13<03:57,  5.03it/s][A[A
    
      5%|▌         | 68/1261 [00:13<03:55,  5.07it/s][A[A
    
      5%|▌         | 69/1261 [00:13<03:53,  5.10it/s][A[A
    
      6%|▌         | 70/1261 [00:13<03:54,  5.07it/s][A[A
    
      6%|▌         | 71/1261 [00:14<03:51,  5.13it/s][A[A
    
      6%|▌         | 72/1261 [00:14<03:50,  5.15it/s][A[A
    
      6%|▌         | 73/1261 [00:14<03:52,  5.12it/s][A[A
    
      6%|▌         | 74/1261 [00:14<03:51,  5.12it/s][A[A
    
      6%|▌         | 75/1261 [00:14<03:49,  5.17it/s][A[A
    
      6%|▌         | 76/1261 [00:15<03:50,  5.13it/s][A[A
    
      6%|▌         | 77/1261 [00:15<03:54,  5.06it/s][A[A
    
      6%|▌         | 78/1261 [00:15<03:58,  4.96it/s][A[A
    
      6%|▋         | 79/1261 [00:15<03:56,  5.00it/s][A[A
    
      6%|▋         | 80/1261 [00:15<03:53,  5.07it/s][A[A
    
      6%|▋         | 81/1261 [00:16<03:52,  5.08it/s][A[A
    
      7%|▋         | 82/1261 [00:16<03:58,  4.95it/s][A[A
    
      7%|▋         | 83/1261 [00:16<03:59,  4.92it/s][A[A
    
      7%|▋         | 84/1261 [00:16<03:56,  4.98it/s][A[A
    
      7%|▋         | 85/1261 [00:16<03:54,  5.02it/s][A[A
    
      7%|▋         | 86/1261 [00:17<03:53,  5.04it/s][A[A
    
      7%|▋         | 87/1261 [00:17<03:53,  5.03it/s][A[A
    
      7%|▋         | 88/1261 [00:17<03:50,  5.09it/s][A[A
    
      7%|▋         | 89/1261 [00:17<03:50,  5.09it/s][A[A
    
      7%|▋         | 90/1261 [00:17<03:50,  5.09it/s][A[A
    
      7%|▋         | 91/1261 [00:18<03:52,  5.03it/s][A[A
    
      7%|▋         | 92/1261 [00:18<03:49,  5.08it/s][A[A
    
      7%|▋         | 93/1261 [00:18<03:50,  5.06it/s][A[A
    
      7%|▋         | 94/1261 [00:18<03:50,  5.06it/s][A[A
    
      8%|▊         | 95/1261 [00:18<03:51,  5.04it/s][A[A
    
      8%|▊         | 96/1261 [00:19<03:48,  5.10it/s][A[A
    
      8%|▊         | 97/1261 [00:19<03:51,  5.02it/s][A[A
    
      8%|▊         | 98/1261 [00:19<03:53,  4.97it/s][A[A
    
      8%|▊         | 99/1261 [00:19<03:49,  5.06it/s][A[A
    
      8%|▊         | 100/1261 [00:19<03:49,  5.07it/s][A[A
    
      8%|▊         | 101/1261 [00:20<03:49,  5.05it/s][A[A
    
      8%|▊         | 102/1261 [00:20<03:49,  5.04it/s][A[A
    
      8%|▊         | 103/1261 [00:20<03:45,  5.13it/s][A[A
    
      8%|▊         | 104/1261 [00:20<03:44,  5.16it/s][A[A
    
      8%|▊         | 105/1261 [00:20<03:44,  5.15it/s][A[A
    
      8%|▊         | 106/1261 [00:21<03:42,  5.20it/s][A[A
    
      8%|▊         | 107/1261 [00:21<03:39,  5.25it/s][A[A
    
      9%|▊         | 108/1261 [00:21<03:39,  5.25it/s][A[A
    
      9%|▊         | 109/1261 [00:21<03:41,  5.21it/s][A[A
    
      9%|▊         | 110/1261 [00:21<03:42,  5.18it/s][A[A
    
      9%|▉         | 111/1261 [00:22<03:40,  5.22it/s][A[A
    
      9%|▉         | 112/1261 [00:22<03:43,  5.14it/s][A[A
    
      9%|▉         | 113/1261 [00:22<03:41,  5.18it/s][A[A
    
      9%|▉         | 114/1261 [00:22<03:44,  5.11it/s][A[A
    
      9%|▉         | 115/1261 [00:22<03:44,  5.11it/s][A[A
    
      9%|▉         | 116/1261 [00:23<03:42,  5.14it/s][A[A
    
      9%|▉         | 117/1261 [00:23<03:42,  5.13it/s][A[A
    
      9%|▉         | 118/1261 [00:23<03:44,  5.09it/s][A[A
    
      9%|▉         | 119/1261 [00:23<03:43,  5.10it/s][A[A
    
     10%|▉         | 120/1261 [00:23<03:40,  5.17it/s][A[A
    
     10%|▉         | 121/1261 [00:23<03:39,  5.20it/s][A[A
    
     10%|▉         | 122/1261 [00:24<03:40,  5.17it/s][A[A
    
     10%|▉         | 123/1261 [00:24<03:38,  5.20it/s][A[A
    
     10%|▉         | 124/1261 [00:24<03:39,  5.17it/s][A[A
    
     10%|▉         | 125/1261 [00:24<03:37,  5.21it/s][A[A
    
     10%|▉         | 126/1261 [00:24<03:38,  5.19it/s][A[A
    
     10%|█         | 127/1261 [00:25<04:20,  4.35it/s][A[A
    
     10%|█         | 128/1261 [00:25<04:14,  4.46it/s][A[A
    
     10%|█         | 129/1261 [00:25<04:06,  4.58it/s][A[A
    
     10%|█         | 130/1261 [00:25<04:02,  4.67it/s][A[A
    
     10%|█         | 131/1261 [00:26<03:55,  4.79it/s][A[A
    
     10%|█         | 132/1261 [00:26<03:47,  4.96it/s][A[A
    
     11%|█         | 133/1261 [00:26<03:42,  5.06it/s][A[A
    
     11%|█         | 134/1261 [00:26<03:46,  4.98it/s][A[A
    
     11%|█         | 135/1261 [00:26<03:41,  5.08it/s][A[A
    
     11%|█         | 136/1261 [00:27<03:39,  5.12it/s][A[A
    
     11%|█         | 137/1261 [00:27<03:39,  5.13it/s][A[A
    
     11%|█         | 138/1261 [00:27<03:39,  5.13it/s][A[A
    
     11%|█         | 139/1261 [00:27<03:39,  5.12it/s][A[A
    
     11%|█         | 140/1261 [00:27<03:38,  5.13it/s][A[A
    
     11%|█         | 141/1261 [00:28<03:38,  5.12it/s][A[A
    
     11%|█▏        | 142/1261 [00:28<03:38,  5.12it/s][A[A
    
     11%|█▏        | 143/1261 [00:28<03:38,  5.13it/s][A[A
    
     11%|█▏        | 144/1261 [00:28<03:39,  5.08it/s][A[A
    
     11%|█▏        | 145/1261 [00:28<03:44,  4.97it/s][A[A
    
     12%|█▏        | 146/1261 [00:28<03:41,  5.03it/s][A[A
    
     12%|█▏        | 147/1261 [00:29<03:38,  5.11it/s][A[A
    
     12%|█▏        | 148/1261 [00:29<03:37,  5.11it/s][A[A
    
     12%|█▏        | 149/1261 [00:29<03:49,  4.85it/s][A[A
    
     12%|█▏        | 150/1261 [00:29<03:42,  4.98it/s][A[A
    
     12%|█▏        | 151/1261 [00:29<03:41,  5.01it/s][A[A
    
     12%|█▏        | 152/1261 [00:30<03:37,  5.10it/s][A[A
    
     12%|█▏        | 153/1261 [00:30<03:34,  5.16it/s][A[A
    
     12%|█▏        | 154/1261 [00:30<03:35,  5.13it/s][A[A
    
     12%|█▏        | 155/1261 [00:30<03:37,  5.09it/s][A[A
    
     12%|█▏        | 156/1261 [00:30<03:35,  5.13it/s][A[A
    
     12%|█▏        | 157/1261 [00:31<03:31,  5.21it/s][A[A
    
     13%|█▎        | 158/1261 [00:31<03:30,  5.23it/s][A[A
    
     13%|█▎        | 159/1261 [00:31<03:50,  4.78it/s][A[A
    
     13%|█▎        | 160/1261 [00:31<04:00,  4.57it/s][A[A
    
     13%|█▎        | 161/1261 [00:32<03:58,  4.61it/s][A[A
    
     13%|█▎        | 162/1261 [00:32<03:50,  4.76it/s][A[A
    
     13%|█▎        | 163/1261 [00:32<03:46,  4.86it/s][A[A
    
     13%|█▎        | 164/1261 [00:32<03:44,  4.89it/s][A[A
    
     13%|█▎        | 165/1261 [00:32<03:38,  5.01it/s][A[A
    
     13%|█▎        | 166/1261 [00:33<03:38,  5.02it/s][A[A
    
     13%|█▎        | 167/1261 [00:33<03:37,  5.03it/s][A[A
    
     13%|█▎        | 168/1261 [00:33<03:35,  5.07it/s][A[A
    
     13%|█▎        | 169/1261 [00:33<03:35,  5.07it/s][A[A
    
     13%|█▎        | 170/1261 [00:33<03:36,  5.04it/s][A[A
    
     14%|█▎        | 171/1261 [00:34<03:33,  5.10it/s][A[A
    
     14%|█▎        | 172/1261 [00:34<03:32,  5.12it/s][A[A
    
     14%|█▎        | 173/1261 [00:34<03:31,  5.16it/s][A[A
    
     14%|█▍        | 174/1261 [00:34<03:34,  5.07it/s][A[A
    
     14%|█▍        | 175/1261 [00:34<03:33,  5.08it/s][A[A
    
     14%|█▍        | 176/1261 [00:34<03:32,  5.11it/s][A[A
    
     14%|█▍        | 177/1261 [00:35<03:31,  5.11it/s][A[A
    
     14%|█▍        | 178/1261 [00:35<03:29,  5.16it/s][A[A
    
     14%|█▍        | 179/1261 [00:35<03:27,  5.20it/s][A[A
    
     14%|█▍        | 180/1261 [00:35<03:26,  5.24it/s][A[A
    
     14%|█▍        | 181/1261 [00:35<03:29,  5.15it/s][A[A
    
     14%|█▍        | 182/1261 [00:36<03:29,  5.16it/s][A[A
    
     15%|█▍        | 183/1261 [00:36<03:26,  5.22it/s][A[A
    
     15%|█▍        | 184/1261 [00:36<03:27,  5.18it/s][A[A
    
     15%|█▍        | 185/1261 [00:36<03:27,  5.18it/s][A[A
    
     15%|█▍        | 186/1261 [00:36<03:27,  5.19it/s][A[A
    
     15%|█▍        | 187/1261 [00:37<03:27,  5.19it/s][A[A
    
     15%|█▍        | 188/1261 [00:37<03:27,  5.16it/s][A[A
    
     15%|█▍        | 189/1261 [00:37<03:28,  5.14it/s][A[A
    
     15%|█▌        | 190/1261 [00:37<03:29,  5.11it/s][A[A
    
     15%|█▌        | 191/1261 [00:37<03:28,  5.13it/s][A[A
    
     15%|█▌        | 192/1261 [00:38<03:26,  5.18it/s][A[A
    
     15%|█▌        | 193/1261 [00:38<03:27,  5.15it/s][A[A
    
     15%|█▌        | 194/1261 [00:38<03:27,  5.14it/s][A[A
    
     15%|█▌        | 195/1261 [00:38<03:28,  5.12it/s][A[A
    
     16%|█▌        | 196/1261 [00:38<03:26,  5.15it/s][A[A
    
     16%|█▌        | 197/1261 [00:39<03:25,  5.19it/s][A[A
    
     16%|█▌        | 198/1261 [00:39<03:28,  5.11it/s][A[A
    
     16%|█▌        | 199/1261 [00:39<03:26,  5.14it/s][A[A
    
     16%|█▌        | 200/1261 [00:39<03:26,  5.14it/s][A[A
    
     16%|█▌        | 201/1261 [00:39<03:25,  5.17it/s][A[A
    
     16%|█▌        | 202/1261 [00:40<03:26,  5.12it/s][A[A
    
     16%|█▌        | 203/1261 [00:40<03:25,  5.15it/s][A[A
    
     16%|█▌        | 204/1261 [00:40<03:25,  5.14it/s][A[A
    
     16%|█▋        | 205/1261 [00:40<03:24,  5.15it/s][A[A
    
     16%|█▋        | 206/1261 [00:40<03:25,  5.12it/s][A[A
    
     16%|█▋        | 207/1261 [00:40<03:24,  5.15it/s][A[A
    
     16%|█▋        | 208/1261 [00:41<03:26,  5.10it/s][A[A
    
     17%|█▋        | 209/1261 [00:41<03:23,  5.18it/s][A[A
    
     17%|█▋        | 210/1261 [00:41<03:22,  5.19it/s][A[A
    
     17%|█▋        | 211/1261 [00:41<03:23,  5.16it/s][A[A
    
     17%|█▋        | 212/1261 [00:41<03:23,  5.17it/s][A[A
    
     17%|█▋        | 213/1261 [00:42<03:22,  5.18it/s][A[A
    
     17%|█▋        | 214/1261 [00:42<03:21,  5.19it/s][A[A
    
     17%|█▋        | 215/1261 [00:42<03:20,  5.22it/s][A[A
    
     17%|█▋        | 216/1261 [00:42<03:21,  5.19it/s][A[A
    
     17%|█▋        | 217/1261 [00:42<03:20,  5.21it/s][A[A
    
     17%|█▋        | 218/1261 [00:43<03:20,  5.19it/s][A[A
    
     17%|█▋        | 219/1261 [00:43<03:19,  5.22it/s][A[A
    
     17%|█▋        | 220/1261 [00:43<03:22,  5.13it/s][A[A
    
     18%|█▊        | 221/1261 [00:43<03:22,  5.13it/s][A[A
    
     18%|█▊        | 222/1261 [00:43<03:22,  5.13it/s][A[A
    
     18%|█▊        | 223/1261 [00:44<03:23,  5.10it/s][A[A
    
     18%|█▊        | 224/1261 [00:44<03:20,  5.17it/s][A[A
    
     18%|█▊        | 225/1261 [00:44<03:18,  5.21it/s][A[A
    
     18%|█▊        | 226/1261 [00:44<03:21,  5.14it/s][A[A
    
     18%|█▊        | 227/1261 [00:44<03:19,  5.18it/s][A[A
    
     18%|█▊        | 228/1261 [00:45<03:21,  5.13it/s][A[A
    
     18%|█▊        | 229/1261 [00:45<03:19,  5.18it/s][A[A
    
     18%|█▊        | 230/1261 [00:45<03:20,  5.14it/s][A[A
    
     18%|█▊        | 231/1261 [00:45<03:23,  5.07it/s][A[A
    
     18%|█▊        | 232/1261 [00:45<03:22,  5.08it/s][A[A
    
     18%|█▊        | 233/1261 [00:46<03:22,  5.08it/s][A[A
    
     19%|█▊        | 234/1261 [00:46<03:20,  5.12it/s][A[A
    
     19%|█▊        | 235/1261 [00:46<03:18,  5.18it/s][A[A
    
     19%|█▊        | 236/1261 [00:46<03:21,  5.09it/s][A[A
    
     19%|█▉        | 237/1261 [00:46<03:18,  5.17it/s][A[A
    
     19%|█▉        | 238/1261 [00:47<03:17,  5.18it/s][A[A
    
     19%|█▉        | 239/1261 [00:47<03:16,  5.21it/s][A[A
    
     19%|█▉        | 240/1261 [00:47<03:17,  5.18it/s][A[A
    
     19%|█▉        | 241/1261 [00:47<03:19,  5.12it/s][A[A
    
     19%|█▉        | 242/1261 [00:47<03:20,  5.09it/s][A[A
    
     19%|█▉        | 243/1261 [00:47<03:19,  5.11it/s][A[A
    
     19%|█▉        | 244/1261 [00:48<03:18,  5.13it/s][A[A
    
     19%|█▉        | 245/1261 [00:48<03:18,  5.12it/s][A[A
    
     20%|█▉        | 246/1261 [00:48<03:20,  5.06it/s][A[A
    
     20%|█▉        | 247/1261 [00:48<03:19,  5.08it/s][A[A
    
     20%|█▉        | 248/1261 [00:48<03:18,  5.10it/s][A[A
    
     20%|█▉        | 249/1261 [00:49<03:18,  5.11it/s][A[A
    
     20%|█▉        | 250/1261 [00:49<03:17,  5.11it/s][A[A
    
     20%|█▉        | 251/1261 [00:49<03:16,  5.15it/s][A[A
    
     20%|█▉        | 252/1261 [00:49<03:16,  5.14it/s][A[A
    
     20%|██        | 253/1261 [00:49<03:15,  5.17it/s][A[A
    
     20%|██        | 254/1261 [00:50<03:13,  5.20it/s][A[A
    
     20%|██        | 255/1261 [00:50<03:15,  5.16it/s][A[A
    
     20%|██        | 256/1261 [00:50<03:11,  5.24it/s][A[A
    
     20%|██        | 257/1261 [00:50<03:11,  5.24it/s][A[A
    
     20%|██        | 258/1261 [00:50<03:11,  5.23it/s][A[A
    
     21%|██        | 259/1261 [00:51<03:11,  5.24it/s][A[A
    
     21%|██        | 260/1261 [00:51<03:08,  5.30it/s][A[A
    
     21%|██        | 261/1261 [00:51<03:09,  5.27it/s][A[A
    
     21%|██        | 262/1261 [00:51<03:10,  5.23it/s][A[A
    
     21%|██        | 263/1261 [00:51<03:13,  5.15it/s][A[A
    
     21%|██        | 264/1261 [00:52<03:14,  5.13it/s][A[A
    
     21%|██        | 265/1261 [00:52<03:12,  5.18it/s][A[A
    
     21%|██        | 266/1261 [00:52<03:14,  5.11it/s][A[A
    
     21%|██        | 267/1261 [00:52<03:15,  5.08it/s][A[A
    
     21%|██▏       | 268/1261 [00:52<03:13,  5.13it/s][A[A
    
     21%|██▏       | 269/1261 [00:53<03:11,  5.19it/s][A[A
    
     21%|██▏       | 270/1261 [00:53<03:12,  5.15it/s][A[A
    
     21%|██▏       | 271/1261 [00:53<03:13,  5.13it/s][A[A
    
     22%|██▏       | 272/1261 [00:53<03:20,  4.93it/s][A[A
    
     22%|██▏       | 273/1261 [00:53<03:18,  4.98it/s][A[A
    
     22%|██▏       | 274/1261 [00:54<03:16,  5.02it/s][A[A
    
     22%|██▏       | 275/1261 [00:54<03:13,  5.08it/s][A[A
    
     22%|██▏       | 276/1261 [00:54<03:12,  5.11it/s][A[A
    
     22%|██▏       | 277/1261 [00:54<03:14,  5.07it/s][A[A
    
     22%|██▏       | 278/1261 [00:54<03:13,  5.08it/s][A[A
    
     22%|██▏       | 279/1261 [00:54<03:14,  5.05it/s][A[A
    
     22%|██▏       | 280/1261 [00:55<03:10,  5.14it/s][A[A
    
     22%|██▏       | 281/1261 [00:55<03:08,  5.21it/s][A[A
    
     22%|██▏       | 282/1261 [00:55<03:10,  5.15it/s][A[A
    
     22%|██▏       | 283/1261 [00:55<03:08,  5.19it/s][A[A
    
     23%|██▎       | 284/1261 [00:55<03:07,  5.22it/s][A[A
    
     23%|██▎       | 285/1261 [00:56<03:07,  5.20it/s][A[A
    
     23%|██▎       | 286/1261 [00:56<03:07,  5.21it/s][A[A
    
     23%|██▎       | 287/1261 [00:56<03:08,  5.17it/s][A[A
    
     23%|██▎       | 288/1261 [00:56<03:08,  5.16it/s][A[A
    
     23%|██▎       | 289/1261 [00:56<03:06,  5.20it/s][A[A
    
     23%|██▎       | 290/1261 [00:57<03:07,  5.19it/s][A[A
    
     23%|██▎       | 291/1261 [00:57<03:05,  5.22it/s][A[A
    
     23%|██▎       | 292/1261 [00:57<03:07,  5.16it/s][A[A
    
     23%|██▎       | 293/1261 [00:57<03:11,  5.05it/s][A[A
    
     23%|██▎       | 294/1261 [00:57<03:11,  5.04it/s][A[A
    
     23%|██▎       | 295/1261 [00:58<03:10,  5.08it/s][A[A
    
     23%|██▎       | 296/1261 [00:58<03:10,  5.06it/s][A[A
    
     24%|██▎       | 297/1261 [00:58<03:10,  5.07it/s][A[A
    
     24%|██▎       | 298/1261 [00:58<03:11,  5.03it/s][A[A
    
     24%|██▎       | 299/1261 [00:58<03:11,  5.02it/s][A[A
    
     24%|██▍       | 300/1261 [00:59<03:08,  5.10it/s][A[A
    
     24%|██▍       | 301/1261 [00:59<03:10,  5.04it/s][A[A
    
     24%|██▍       | 302/1261 [00:59<03:09,  5.07it/s][A[A
    
     24%|██▍       | 303/1261 [00:59<03:06,  5.13it/s][A[A
    
     24%|██▍       | 304/1261 [00:59<03:26,  4.63it/s][A[A
    
     24%|██▍       | 305/1261 [01:00<03:32,  4.50it/s][A[A
    
     24%|██▍       | 306/1261 [01:00<03:23,  4.69it/s][A[A
    
     24%|██▍       | 307/1261 [01:00<03:18,  4.80it/s][A[A
    
     24%|██▍       | 308/1261 [01:00<03:15,  4.88it/s][A[A
    
     25%|██▍       | 309/1261 [01:00<03:20,  4.74it/s][A[A
    
     25%|██▍       | 310/1261 [01:01<03:13,  4.92it/s][A[A
    
     25%|██▍       | 311/1261 [01:01<03:08,  5.03it/s][A[A
    
     25%|██▍       | 312/1261 [01:01<03:07,  5.06it/s][A[A
    
     25%|██▍       | 313/1261 [01:01<03:07,  5.05it/s][A[A
    
     25%|██▍       | 314/1261 [01:01<03:06,  5.08it/s][A[A
    
     25%|██▍       | 315/1261 [01:02<03:07,  5.06it/s][A[A
    
     25%|██▌       | 316/1261 [01:02<03:06,  5.06it/s][A[A
    
     25%|██▌       | 317/1261 [01:02<03:04,  5.11it/s][A[A
    
     25%|██▌       | 318/1261 [01:02<03:02,  5.17it/s][A[A
    
     25%|██▌       | 319/1261 [01:02<03:02,  5.18it/s][A[A
    
     25%|██▌       | 320/1261 [01:03<03:01,  5.19it/s][A[A
    
     25%|██▌       | 321/1261 [01:03<03:02,  5.14it/s][A[A
    
     26%|██▌       | 322/1261 [01:03<03:03,  5.13it/s][A[A
    
     26%|██▌       | 323/1261 [01:03<03:03,  5.12it/s][A[A
    
     26%|██▌       | 324/1261 [01:03<03:01,  5.17it/s][A[A
    
     26%|██▌       | 325/1261 [01:04<03:00,  5.18it/s][A[A
    
     26%|██▌       | 326/1261 [01:04<02:59,  5.20it/s][A[A
    
     26%|██▌       | 327/1261 [01:04<02:59,  5.19it/s][A[A
    
     26%|██▌       | 328/1261 [01:04<03:01,  5.15it/s][A[A
    
     26%|██▌       | 329/1261 [01:04<03:03,  5.09it/s][A[A
    
     26%|██▌       | 330/1261 [01:05<03:02,  5.09it/s][A[A
    
     26%|██▌       | 331/1261 [01:05<03:03,  5.08it/s][A[A
    
     26%|██▋       | 332/1261 [01:05<03:02,  5.08it/s][A[A
    
     26%|██▋       | 333/1261 [01:05<03:03,  5.07it/s][A[A
    
     26%|██▋       | 334/1261 [01:05<03:04,  5.03it/s][A[A
    
     27%|██▋       | 335/1261 [01:06<03:01,  5.11it/s][A[A
    
     27%|██▋       | 336/1261 [01:06<03:01,  5.09it/s][A[A
    
     27%|██▋       | 337/1261 [01:06<03:00,  5.12it/s][A[A
    
     27%|██▋       | 338/1261 [01:06<03:01,  5.10it/s][A[A
    
     27%|██▋       | 339/1261 [01:06<02:59,  5.14it/s][A[A
    
     27%|██▋       | 340/1261 [01:07<02:59,  5.13it/s][A[A
    
     27%|██▋       | 341/1261 [01:07<02:59,  5.12it/s][A[A
    
     27%|██▋       | 342/1261 [01:07<02:58,  5.15it/s][A[A
    
     27%|██▋       | 343/1261 [01:07<02:57,  5.17it/s][A[A
    
     27%|██▋       | 344/1261 [01:07<02:57,  5.18it/s][A[A
    
     27%|██▋       | 345/1261 [01:07<02:55,  5.21it/s][A[A
    
     27%|██▋       | 346/1261 [01:08<02:56,  5.18it/s][A[A
    
     28%|██▊       | 347/1261 [01:08<02:55,  5.21it/s][A[A
    
     28%|██▊       | 348/1261 [01:08<02:55,  5.19it/s][A[A
    
     28%|██▊       | 349/1261 [01:08<02:57,  5.14it/s][A[A
    
     28%|██▊       | 350/1261 [01:08<02:56,  5.15it/s][A[A
    
     28%|██▊       | 351/1261 [01:09<02:55,  5.20it/s][A[A
    
     28%|██▊       | 352/1261 [01:09<02:54,  5.21it/s][A[A
    
     28%|██▊       | 353/1261 [01:09<02:55,  5.17it/s][A[A
    
     28%|██▊       | 354/1261 [01:09<02:55,  5.17it/s][A[A
    
     28%|██▊       | 355/1261 [01:09<02:54,  5.20it/s][A[A
    
     28%|██▊       | 356/1261 [01:10<02:53,  5.22it/s][A[A
    
     28%|██▊       | 357/1261 [01:10<02:54,  5.19it/s][A[A
    
     28%|██▊       | 358/1261 [01:10<02:56,  5.12it/s][A[A
    
     28%|██▊       | 359/1261 [01:10<02:56,  5.11it/s][A[A
    
     29%|██▊       | 360/1261 [01:10<02:56,  5.10it/s][A[A
    
     29%|██▊       | 361/1261 [01:11<02:57,  5.06it/s][A[A
    
     29%|██▊       | 362/1261 [01:11<02:58,  5.05it/s][A[A
    
     29%|██▉       | 363/1261 [01:11<02:55,  5.12it/s][A[A
    
     29%|██▉       | 364/1261 [01:11<02:55,  5.12it/s][A[A
    
     29%|██▉       | 365/1261 [01:11<02:54,  5.14it/s][A[A
    
     29%|██▉       | 366/1261 [01:12<02:52,  5.18it/s][A[A
    
     29%|██▉       | 367/1261 [01:12<02:51,  5.21it/s][A[A
    
     29%|██▉       | 368/1261 [01:12<02:50,  5.23it/s][A[A
    
     29%|██▉       | 369/1261 [01:12<02:52,  5.18it/s][A[A
    
     29%|██▉       | 370/1261 [01:12<02:52,  5.16it/s][A[A
    
     29%|██▉       | 371/1261 [01:13<02:52,  5.15it/s][A[A
    
     30%|██▉       | 372/1261 [01:13<02:50,  5.21it/s][A[A
    
     30%|██▉       | 373/1261 [01:13<02:50,  5.22it/s][A[A
    
     30%|██▉       | 374/1261 [01:13<02:49,  5.22it/s][A[A
    
     30%|██▉       | 375/1261 [01:13<02:48,  5.26it/s][A[A
    
     30%|██▉       | 376/1261 [01:13<02:48,  5.26it/s][A[A
    
     30%|██▉       | 377/1261 [01:14<02:47,  5.28it/s][A[A
    
     30%|██▉       | 378/1261 [01:14<02:47,  5.28it/s][A[A
    
     30%|███       | 379/1261 [01:14<02:45,  5.32it/s][A[A
    
     30%|███       | 380/1261 [01:14<02:46,  5.28it/s][A[A
    
     30%|███       | 381/1261 [01:14<02:47,  5.24it/s][A[A
    
     30%|███       | 382/1261 [01:15<02:48,  5.21it/s][A[A
    
     30%|███       | 383/1261 [01:15<02:57,  4.95it/s][A[A
    
     30%|███       | 384/1261 [01:15<02:57,  4.94it/s][A[A
    
     31%|███       | 385/1261 [01:15<03:00,  4.85it/s][A[A
    
     31%|███       | 386/1261 [01:15<03:01,  4.81it/s][A[A
    
     31%|███       | 387/1261 [01:16<03:01,  4.82it/s][A[A
    
     31%|███       | 388/1261 [01:16<03:03,  4.76it/s][A[A
    
     31%|███       | 389/1261 [01:16<03:00,  4.83it/s][A[A
    
     31%|███       | 390/1261 [01:16<03:05,  4.70it/s][A[A
    
     31%|███       | 391/1261 [01:17<03:03,  4.75it/s][A[A
    
     31%|███       | 392/1261 [01:17<03:04,  4.71it/s][A[A
    
     31%|███       | 393/1261 [01:17<03:05,  4.67it/s][A[A
    
     31%|███       | 394/1261 [01:17<03:06,  4.64it/s][A[A
    
     31%|███▏      | 395/1261 [01:17<03:09,  4.57it/s][A[A
    
     31%|███▏      | 396/1261 [01:18<04:30,  3.19it/s][A[A
    
     31%|███▏      | 397/1261 [01:18<04:08,  3.47it/s][A[A
    
     32%|███▏      | 398/1261 [01:18<03:48,  3.78it/s][A[A
    
     32%|███▏      | 399/1261 [01:19<03:34,  4.01it/s][A[A
    
     32%|███▏      | 400/1261 [01:19<03:31,  4.07it/s][A[A
    
     32%|███▏      | 401/1261 [01:19<03:25,  4.18it/s][A[A
    
     32%|███▏      | 402/1261 [01:19<03:20,  4.28it/s][A[A
    
     32%|███▏      | 403/1261 [01:19<03:14,  4.41it/s][A[A
    
     32%|███▏      | 404/1261 [01:20<03:15,  4.37it/s][A[A
    
     32%|███▏      | 405/1261 [01:20<03:12,  4.45it/s][A[A
    
     32%|███▏      | 406/1261 [01:20<03:14,  4.40it/s][A[A
    
     32%|███▏      | 407/1261 [01:20<03:08,  4.54it/s][A[A
    
     32%|███▏      | 408/1261 [01:21<03:00,  4.73it/s][A[A
    
     32%|███▏      | 409/1261 [01:21<02:58,  4.76it/s][A[A
    
     33%|███▎      | 410/1261 [01:21<02:56,  4.81it/s][A[A
    
     33%|███▎      | 411/1261 [01:21<02:55,  4.83it/s][A[A
    
     33%|███▎      | 412/1261 [01:21<02:52,  4.93it/s][A[A
    
     33%|███▎      | 413/1261 [01:22<02:49,  4.99it/s][A[A
    
     33%|███▎      | 414/1261 [01:22<02:56,  4.81it/s][A[A
    
     33%|███▎      | 415/1261 [01:22<02:58,  4.75it/s][A[A
    
     33%|███▎      | 416/1261 [01:22<02:57,  4.77it/s][A[A
    
     33%|███▎      | 417/1261 [01:22<02:56,  4.78it/s][A[A
    
     33%|███▎      | 418/1261 [01:23<03:00,  4.67it/s][A[A
    
     33%|███▎      | 419/1261 [01:23<03:02,  4.61it/s][A[A
    
     33%|███▎      | 420/1261 [01:23<03:08,  4.47it/s][A[A
    
     33%|███▎      | 421/1261 [01:23<03:07,  4.49it/s][A[A
    
     33%|███▎      | 422/1261 [01:24<03:03,  4.57it/s][A[A
    
     34%|███▎      | 423/1261 [01:24<03:00,  4.65it/s][A[A
    
     34%|███▎      | 424/1261 [01:24<02:58,  4.70it/s][A[A
    
     34%|███▎      | 425/1261 [01:24<02:59,  4.65it/s][A[A
    
     34%|███▍      | 426/1261 [01:24<02:54,  4.78it/s][A[A
    
     34%|███▍      | 427/1261 [01:25<02:57,  4.69it/s][A[A
    
     34%|███▍      | 428/1261 [01:25<02:56,  4.73it/s][A[A
    
     34%|███▍      | 429/1261 [01:25<02:51,  4.86it/s][A[A
    
     34%|███▍      | 430/1261 [01:25<02:52,  4.82it/s][A[A
    
     34%|███▍      | 431/1261 [01:25<02:52,  4.81it/s][A[A
    
     34%|███▍      | 432/1261 [01:26<02:48,  4.93it/s][A[A
    
     34%|███▍      | 433/1261 [01:26<02:48,  4.93it/s][A[A
    
     34%|███▍      | 434/1261 [01:26<02:48,  4.92it/s][A[A
    
     34%|███▍      | 435/1261 [01:26<02:48,  4.91it/s][A[A
    
     35%|███▍      | 436/1261 [01:26<02:47,  4.92it/s][A[A
    
     35%|███▍      | 437/1261 [01:27<02:46,  4.95it/s][A[A
    
     35%|███▍      | 438/1261 [01:27<02:47,  4.92it/s][A[A
    
     35%|███▍      | 439/1261 [01:27<02:47,  4.90it/s][A[A
    
     35%|███▍      | 440/1261 [01:27<02:45,  4.97it/s][A[A
    
     35%|███▍      | 441/1261 [01:27<02:41,  5.08it/s][A[A
    
     35%|███▌      | 442/1261 [01:28<02:39,  5.13it/s][A[A
    
     35%|███▌      | 443/1261 [01:28<02:39,  5.13it/s][A[A
    
     35%|███▌      | 444/1261 [01:28<02:40,  5.10it/s][A[A
    
     35%|███▌      | 445/1261 [01:28<02:39,  5.11it/s][A[A
    
     35%|███▌      | 446/1261 [01:28<02:39,  5.10it/s][A[A
    
     35%|███▌      | 447/1261 [01:29<02:37,  5.17it/s][A[A
    
     36%|███▌      | 448/1261 [01:29<02:36,  5.19it/s][A[A
    
     36%|███▌      | 449/1261 [01:29<02:35,  5.22it/s][A[A
    
     36%|███▌      | 450/1261 [01:29<02:37,  5.14it/s][A[A
    
     36%|███▌      | 451/1261 [01:29<02:40,  5.06it/s][A[A
    
     36%|███▌      | 452/1261 [01:30<02:38,  5.11it/s][A[A
    
     36%|███▌      | 453/1261 [01:30<02:36,  5.16it/s][A[A
    
     36%|███▌      | 454/1261 [01:30<02:38,  5.10it/s][A[A
    
     36%|███▌      | 455/1261 [01:30<02:44,  4.91it/s][A[A
    
     36%|███▌      | 456/1261 [01:30<02:41,  4.97it/s][A[A
    
     36%|███▌      | 457/1261 [01:31<02:40,  5.02it/s][A[A
    
     36%|███▋      | 458/1261 [01:31<02:38,  5.07it/s][A[A
    
     36%|███▋      | 459/1261 [01:31<02:38,  5.07it/s][A[A
    
     36%|███▋      | 460/1261 [01:31<02:38,  5.05it/s][A[A
    
     37%|███▋      | 461/1261 [01:31<02:46,  4.81it/s][A[A
    
     37%|███▋      | 462/1261 [01:32<02:43,  4.89it/s][A[A
    
     37%|███▋      | 463/1261 [01:32<02:41,  4.96it/s][A[A
    
     37%|███▋      | 464/1261 [01:32<02:38,  5.02it/s][A[A
    
     37%|███▋      | 465/1261 [01:32<02:37,  5.06it/s][A[A
    
     37%|███▋      | 466/1261 [01:32<02:34,  5.14it/s][A[A
    
     37%|███▋      | 467/1261 [01:33<02:35,  5.12it/s][A[A
    
     37%|███▋      | 468/1261 [01:33<02:33,  5.16it/s][A[A
    
     37%|███▋      | 469/1261 [01:33<02:33,  5.17it/s][A[A
    
     37%|███▋      | 470/1261 [01:33<02:32,  5.19it/s][A[A
    
     37%|███▋      | 471/1261 [01:33<02:31,  5.20it/s][A[A
    
     37%|███▋      | 472/1261 [01:34<02:40,  4.91it/s][A[A
    
     38%|███▊      | 473/1261 [01:34<02:38,  4.96it/s][A[A
    
     38%|███▊      | 474/1261 [01:34<02:39,  4.95it/s][A[A
    
     38%|███▊      | 475/1261 [01:34<02:41,  4.87it/s][A[A
    
     38%|███▊      | 476/1261 [01:34<02:37,  4.98it/s][A[A
    
     38%|███▊      | 477/1261 [01:35<02:35,  5.05it/s][A[A
    
     38%|███▊      | 478/1261 [01:35<02:34,  5.08it/s][A[A
    
     38%|███▊      | 479/1261 [01:35<02:33,  5.10it/s][A[A
    
     38%|███▊      | 480/1261 [01:35<02:34,  5.07it/s][A[A
    
     38%|███▊      | 481/1261 [01:35<02:33,  5.10it/s][A[A
    
     38%|███▊      | 482/1261 [01:36<02:32,  5.12it/s][A[A
    
     38%|███▊      | 483/1261 [01:36<02:32,  5.09it/s][A[A
    
     38%|███▊      | 484/1261 [01:36<02:32,  5.09it/s][A[A
    
     38%|███▊      | 485/1261 [01:36<02:33,  5.05it/s][A[A
    
     39%|███▊      | 486/1261 [01:36<02:31,  5.12it/s][A[A
    
     39%|███▊      | 487/1261 [01:36<02:32,  5.09it/s][A[A
    
     39%|███▊      | 488/1261 [01:37<02:31,  5.09it/s][A[A
    
     39%|███▉      | 489/1261 [01:37<02:29,  5.15it/s][A[A
    
     39%|███▉      | 490/1261 [01:37<02:32,  5.05it/s][A[A
    
     39%|███▉      | 491/1261 [01:37<02:32,  5.04it/s][A[A
    
     39%|███▉      | 492/1261 [01:37<02:33,  5.01it/s][A[A
    
     39%|███▉      | 493/1261 [01:38<02:31,  5.08it/s][A[A
    
     39%|███▉      | 494/1261 [01:38<02:30,  5.10it/s][A[A
    
     39%|███▉      | 495/1261 [01:38<02:33,  4.99it/s][A[A
    
     39%|███▉      | 496/1261 [01:38<02:30,  5.08it/s][A[A
    
     39%|███▉      | 497/1261 [01:38<02:29,  5.11it/s][A[A
    
     39%|███▉      | 498/1261 [01:39<02:29,  5.11it/s][A[A
    
     40%|███▉      | 499/1261 [01:39<02:29,  5.09it/s][A[A
    
     40%|███▉      | 500/1261 [01:39<02:29,  5.09it/s][A[A
    
     40%|███▉      | 501/1261 [01:39<02:28,  5.11it/s][A[A
    
     40%|███▉      | 502/1261 [01:39<02:27,  5.14it/s][A[A
    
     40%|███▉      | 503/1261 [01:40<02:26,  5.18it/s][A[A
    
     40%|███▉      | 504/1261 [01:40<02:27,  5.14it/s][A[A
    
     40%|████      | 505/1261 [01:40<02:28,  5.09it/s][A[A
    
     40%|████      | 506/1261 [01:40<02:29,  5.06it/s][A[A
    
     40%|████      | 507/1261 [01:40<02:29,  5.04it/s][A[A
    
     40%|████      | 508/1261 [01:41<02:29,  5.04it/s][A[A
    
     40%|████      | 509/1261 [01:41<02:30,  5.01it/s][A[A
    
     40%|████      | 510/1261 [01:41<02:27,  5.11it/s][A[A
    
     41%|████      | 511/1261 [01:41<02:27,  5.07it/s][A[A
    
     41%|████      | 512/1261 [01:41<02:31,  4.95it/s][A[A
    
     41%|████      | 513/1261 [01:42<02:29,  4.99it/s][A[A
    
     41%|████      | 514/1261 [01:42<02:32,  4.91it/s][A[A
    
     41%|████      | 515/1261 [01:42<02:31,  4.93it/s][A[A
    
     41%|████      | 516/1261 [01:42<02:28,  5.00it/s][A[A
    
     41%|████      | 517/1261 [01:42<02:28,  4.99it/s][A[A
    
     41%|████      | 518/1261 [01:43<02:26,  5.06it/s][A[A
    
     41%|████      | 519/1261 [01:43<02:28,  5.00it/s][A[A
    
     41%|████      | 520/1261 [01:43<02:27,  5.03it/s][A[A
    
     41%|████▏     | 521/1261 [01:43<02:26,  5.07it/s][A[A
    
     41%|████▏     | 522/1261 [01:43<02:25,  5.09it/s][A[A
    
     41%|████▏     | 523/1261 [01:44<02:24,  5.10it/s][A[A
    
     42%|████▏     | 524/1261 [01:44<02:24,  5.09it/s][A[A
    
     42%|████▏     | 525/1261 [01:44<02:23,  5.14it/s][A[A
    
     42%|████▏     | 526/1261 [01:44<02:22,  5.15it/s][A[A
    
     42%|████▏     | 527/1261 [01:44<02:22,  5.16it/s][A[A
    
     42%|████▏     | 528/1261 [01:45<02:22,  5.15it/s][A[A
    
     42%|████▏     | 529/1261 [01:45<02:20,  5.21it/s][A[A
    
     42%|████▏     | 530/1261 [01:45<02:20,  5.21it/s][A[A
    
     42%|████▏     | 531/1261 [01:45<02:22,  5.13it/s][A[A
    
     42%|████▏     | 532/1261 [01:45<02:20,  5.19it/s][A[A
    
     42%|████▏     | 533/1261 [01:46<02:18,  5.27it/s][A[A
    
     42%|████▏     | 534/1261 [01:46<02:19,  5.22it/s][A[A
    
     42%|████▏     | 535/1261 [01:46<02:19,  5.19it/s][A[A
    
     43%|████▎     | 536/1261 [01:46<02:19,  5.20it/s][A[A
    
     43%|████▎     | 537/1261 [01:46<02:18,  5.24it/s][A[A
    
     43%|████▎     | 538/1261 [01:46<02:16,  5.29it/s][A[A
    
     43%|████▎     | 539/1261 [01:47<02:16,  5.30it/s][A[A
    
     43%|████▎     | 540/1261 [01:47<02:15,  5.31it/s][A[A
    
     43%|████▎     | 541/1261 [01:47<02:16,  5.28it/s][A[A
    
     43%|████▎     | 542/1261 [01:47<02:16,  5.29it/s][A[A
    
     43%|████▎     | 543/1261 [01:47<02:15,  5.29it/s][A[A
    
     43%|████▎     | 544/1261 [01:48<02:15,  5.29it/s][A[A
    
     43%|████▎     | 545/1261 [01:48<02:17,  5.20it/s][A[A
    
     43%|████▎     | 546/1261 [01:48<02:16,  5.25it/s][A[A
    
     43%|████▎     | 547/1261 [01:48<02:14,  5.30it/s][A[A
    
     43%|████▎     | 548/1261 [01:48<02:16,  5.24it/s][A[A
    
     44%|████▎     | 549/1261 [01:49<02:16,  5.22it/s][A[A
    
     44%|████▎     | 550/1261 [01:49<02:15,  5.23it/s][A[A
    
     44%|████▎     | 551/1261 [01:49<02:14,  5.28it/s][A[A
    
     44%|████▍     | 552/1261 [01:49<02:15,  5.23it/s][A[A
    
     44%|████▍     | 553/1261 [01:49<02:14,  5.27it/s][A[A
    
     44%|████▍     | 554/1261 [01:50<02:12,  5.32it/s][A[A
    
     44%|████▍     | 555/1261 [01:50<02:12,  5.32it/s][A[A
    
     44%|████▍     | 556/1261 [01:50<02:11,  5.34it/s][A[A
    
     44%|████▍     | 557/1261 [01:50<02:11,  5.37it/s][A[A
    
     44%|████▍     | 558/1261 [01:50<02:12,  5.29it/s][A[A
    
     44%|████▍     | 559/1261 [01:50<02:13,  5.26it/s][A[A
    
     44%|████▍     | 560/1261 [01:51<02:13,  5.26it/s][A[A
    
     44%|████▍     | 561/1261 [01:51<02:12,  5.28it/s][A[A
    
     45%|████▍     | 562/1261 [01:51<02:14,  5.22it/s][A[A
    
     45%|████▍     | 563/1261 [01:51<02:11,  5.31it/s][A[A
    
     45%|████▍     | 564/1261 [01:51<02:11,  5.29it/s][A[A
    
     45%|████▍     | 565/1261 [01:52<02:11,  5.29it/s][A[A
    
     45%|████▍     | 566/1261 [01:52<02:13,  5.23it/s][A[A
    
     45%|████▍     | 567/1261 [01:52<02:12,  5.24it/s][A[A
    
     45%|████▌     | 568/1261 [01:52<02:13,  5.17it/s][A[A
    
     45%|████▌     | 569/1261 [01:52<02:12,  5.23it/s][A[A
    
     45%|████▌     | 570/1261 [01:53<02:10,  5.30it/s][A[A
    
     45%|████▌     | 571/1261 [01:53<02:09,  5.32it/s][A[A
    
     45%|████▌     | 572/1261 [01:53<02:10,  5.29it/s][A[A
    
     45%|████▌     | 573/1261 [01:53<02:10,  5.27it/s][A[A
    
     46%|████▌     | 574/1261 [01:53<02:08,  5.34it/s][A[A
    
     46%|████▌     | 575/1261 [01:53<02:08,  5.34it/s][A[A
    
     46%|████▌     | 576/1261 [01:54<02:07,  5.37it/s][A[A
    
     46%|████▌     | 577/1261 [01:54<02:06,  5.40it/s][A[A
    
     46%|████▌     | 578/1261 [01:54<02:06,  5.42it/s][A[A
    
     46%|████▌     | 579/1261 [01:54<02:06,  5.39it/s][A[A
    
     46%|████▌     | 580/1261 [01:54<02:05,  5.41it/s][A[A
    
     46%|████▌     | 581/1261 [01:55<02:05,  5.42it/s][A[A
    
     46%|████▌     | 582/1261 [01:55<02:05,  5.40it/s][A[A
    
     46%|████▌     | 583/1261 [01:55<02:05,  5.42it/s][A[A
    
     46%|████▋     | 584/1261 [01:55<02:04,  5.42it/s][A[A
    
     46%|████▋     | 585/1261 [01:55<02:03,  5.46it/s][A[A
    
     46%|████▋     | 586/1261 [01:56<02:05,  5.38it/s][A[A
    
     47%|████▋     | 587/1261 [01:56<02:05,  5.39it/s][A[A
    
     47%|████▋     | 588/1261 [01:56<02:04,  5.42it/s][A[A
    
     47%|████▋     | 589/1261 [01:56<02:04,  5.40it/s][A[A
    
     47%|████▋     | 590/1261 [01:56<02:02,  5.46it/s][A[A
    
     47%|████▋     | 591/1261 [01:56<02:01,  5.52it/s][A[A
    
     47%|████▋     | 592/1261 [01:57<02:01,  5.50it/s][A[A
    
     47%|████▋     | 593/1261 [01:57<02:02,  5.45it/s][A[A
    
     47%|████▋     | 594/1261 [01:57<02:04,  5.38it/s][A[A
    
     47%|████▋     | 595/1261 [01:57<02:04,  5.37it/s][A[A
    
     47%|████▋     | 596/1261 [01:57<02:03,  5.37it/s][A[A
    
     47%|████▋     | 597/1261 [01:58<02:02,  5.42it/s][A[A
    
     47%|████▋     | 598/1261 [01:58<02:01,  5.47it/s][A[A
    
     48%|████▊     | 599/1261 [01:58<02:00,  5.51it/s][A[A
    
     48%|████▊     | 600/1261 [01:58<02:02,  5.40it/s][A[A
    
     48%|████▊     | 601/1261 [01:58<02:01,  5.42it/s][A[A
    
     48%|████▊     | 602/1261 [01:58<02:01,  5.45it/s][A[A
    
     48%|████▊     | 603/1261 [01:59<01:59,  5.50it/s][A[A
    
     48%|████▊     | 604/1261 [01:59<02:04,  5.29it/s][A[A
    
     48%|████▊     | 605/1261 [01:59<02:04,  5.28it/s][A[A
    
     48%|████▊     | 606/1261 [01:59<02:04,  5.27it/s][A[A
    
     48%|████▊     | 607/1261 [01:59<02:02,  5.32it/s][A[A
    
     48%|████▊     | 608/1261 [02:00<02:03,  5.27it/s][A[A
    
     48%|████▊     | 609/1261 [02:00<02:02,  5.30it/s][A[A
    
     48%|████▊     | 610/1261 [02:00<02:03,  5.26it/s][A[A
    
     48%|████▊     | 611/1261 [02:00<02:02,  5.31it/s][A[A
    
     49%|████▊     | 612/1261 [02:00<02:01,  5.36it/s][A[A
    
     49%|████▊     | 613/1261 [02:01<01:59,  5.41it/s][A[A
    
     49%|████▊     | 614/1261 [02:01<02:01,  5.31it/s][A[A
    
     49%|████▉     | 615/1261 [02:01<02:02,  5.26it/s][A[A
    
     49%|████▉     | 616/1261 [02:01<02:07,  5.06it/s][A[A
    
     49%|████▉     | 617/1261 [02:01<02:06,  5.10it/s][A[A
    
     49%|████▉     | 618/1261 [02:02<02:06,  5.10it/s][A[A
    
     49%|████▉     | 619/1261 [02:02<02:07,  5.03it/s][A[A
    
     49%|████▉     | 620/1261 [02:02<02:08,  5.00it/s][A[A
    
     49%|████▉     | 621/1261 [02:02<02:08,  4.98it/s][A[A
    
     49%|████▉     | 622/1261 [02:02<02:07,  5.02it/s][A[A
    
     49%|████▉     | 623/1261 [02:03<02:05,  5.08it/s][A[A
    
     49%|████▉     | 624/1261 [02:03<02:05,  5.07it/s][A[A
    
     50%|████▉     | 625/1261 [02:03<02:04,  5.11it/s][A[A
    
     50%|████▉     | 626/1261 [02:03<02:03,  5.15it/s][A[A
    
     50%|████▉     | 627/1261 [02:03<02:02,  5.19it/s][A[A
    
     50%|████▉     | 628/1261 [02:03<02:01,  5.19it/s][A[A
    
     50%|████▉     | 629/1261 [02:04<02:01,  5.18it/s][A[A
    
     50%|████▉     | 630/1261 [02:04<02:01,  5.19it/s][A[A
    
     50%|█████     | 631/1261 [02:04<02:01,  5.20it/s][A[A
    
     50%|█████     | 632/1261 [02:04<02:01,  5.16it/s][A[A
    
     50%|█████     | 633/1261 [02:04<02:00,  5.21it/s][A[A
    
     50%|█████     | 634/1261 [02:05<02:00,  5.21it/s][A[A
    
     50%|█████     | 635/1261 [02:05<02:01,  5.16it/s][A[A
    
     50%|█████     | 636/1261 [02:05<02:00,  5.17it/s][A[A
    
     51%|█████     | 637/1261 [02:05<02:00,  5.18it/s][A[A
    
     51%|█████     | 638/1261 [02:05<02:00,  5.16it/s][A[A
    
     51%|█████     | 639/1261 [02:06<01:59,  5.20it/s][A[A
    
     51%|█████     | 640/1261 [02:06<02:00,  5.15it/s][A[A
    
     51%|█████     | 641/1261 [02:06<02:00,  5.15it/s][A[A
    
     51%|█████     | 642/1261 [02:06<01:59,  5.17it/s][A[A
    
     51%|█████     | 643/1261 [02:06<01:58,  5.23it/s][A[A
    
     51%|█████     | 644/1261 [02:07<01:59,  5.16it/s][A[A
    
     51%|█████     | 645/1261 [02:07<01:59,  5.17it/s][A[A
    
     51%|█████     | 646/1261 [02:07<02:01,  5.04it/s][A[A
    
     51%|█████▏    | 647/1261 [02:07<02:01,  5.05it/s][A[A
    
     51%|█████▏    | 648/1261 [02:07<02:00,  5.07it/s][A[A
    
     51%|█████▏    | 649/1261 [02:08<01:59,  5.13it/s][A[A
    
     52%|█████▏    | 650/1261 [02:08<01:59,  5.13it/s][A[A
    
     52%|█████▏    | 651/1261 [02:08<01:59,  5.10it/s][A[A
    
     52%|█████▏    | 652/1261 [02:08<02:01,  5.01it/s][A[A
    
     52%|█████▏    | 653/1261 [02:08<01:59,  5.09it/s][A[A
    
     52%|█████▏    | 654/1261 [02:09<01:57,  5.16it/s][A[A
    
     52%|█████▏    | 655/1261 [02:09<01:57,  5.14it/s][A[A
    
     52%|█████▏    | 656/1261 [02:09<01:57,  5.15it/s][A[A
    
     52%|█████▏    | 657/1261 [02:09<01:58,  5.08it/s][A[A
    
     52%|█████▏    | 658/1261 [02:09<01:59,  5.05it/s][A[A
    
     52%|█████▏    | 659/1261 [02:10<01:58,  5.07it/s][A[A
    
     52%|█████▏    | 660/1261 [02:10<01:58,  5.09it/s][A[A
    
     52%|█████▏    | 661/1261 [02:10<01:57,  5.11it/s][A[A
    
     52%|█████▏    | 662/1261 [02:10<01:58,  5.05it/s][A[A
    
     53%|█████▎    | 663/1261 [02:10<01:57,  5.11it/s][A[A
    
     53%|█████▎    | 664/1261 [02:11<01:56,  5.13it/s][A[A
    
     53%|█████▎    | 665/1261 [02:11<01:55,  5.14it/s][A[A
    
     53%|█████▎    | 666/1261 [02:11<01:57,  5.07it/s][A[A
    
     53%|█████▎    | 667/1261 [02:11<01:58,  5.01it/s][A[A
    
     53%|█████▎    | 668/1261 [02:11<01:56,  5.08it/s][A[A
    
     53%|█████▎    | 669/1261 [02:11<01:55,  5.12it/s][A[A
    
     53%|█████▎    | 670/1261 [02:12<01:54,  5.15it/s][A[A
    
     53%|█████▎    | 671/1261 [02:12<01:54,  5.15it/s][A[A
    
     53%|█████▎    | 672/1261 [02:12<01:57,  5.01it/s][A[A
    
     53%|█████▎    | 673/1261 [02:12<01:56,  5.03it/s][A[A
    
     53%|█████▎    | 674/1261 [02:12<01:55,  5.06it/s][A[A
    
     54%|█████▎    | 675/1261 [02:13<01:54,  5.14it/s][A[A
    
     54%|█████▎    | 676/1261 [02:13<01:52,  5.20it/s][A[A
    
     54%|█████▎    | 677/1261 [02:13<01:52,  5.21it/s][A[A
    
     54%|█████▍    | 678/1261 [02:13<01:52,  5.17it/s][A[A
    
     54%|█████▍    | 679/1261 [02:13<01:52,  5.16it/s][A[A
    
     54%|█████▍    | 680/1261 [02:14<01:51,  5.23it/s][A[A
    
     54%|█████▍    | 681/1261 [02:14<01:51,  5.19it/s][A[A
    
     54%|█████▍    | 682/1261 [02:14<01:51,  5.18it/s][A[A
    
     54%|█████▍    | 683/1261 [02:14<01:50,  5.22it/s][A[A
    
     54%|█████▍    | 684/1261 [02:14<01:49,  5.26it/s][A[A
    
     54%|█████▍    | 685/1261 [02:15<01:50,  5.23it/s][A[A
    
     54%|█████▍    | 686/1261 [02:15<01:51,  5.16it/s][A[A
    
     54%|█████▍    | 687/1261 [02:15<01:51,  5.16it/s][A[A
    
     55%|█████▍    | 688/1261 [02:15<01:50,  5.18it/s][A[A
    
     55%|█████▍    | 689/1261 [02:15<01:50,  5.19it/s][A[A
    
     55%|█████▍    | 690/1261 [02:16<01:50,  5.15it/s][A[A
    
     55%|█████▍    | 691/1261 [02:16<01:51,  5.10it/s][A[A
    
     55%|█████▍    | 692/1261 [02:16<01:53,  5.01it/s][A[A
    
     55%|█████▍    | 693/1261 [02:16<01:51,  5.08it/s][A[A
    
     55%|█████▌    | 694/1261 [02:16<01:50,  5.15it/s][A[A
    
     55%|█████▌    | 695/1261 [02:17<01:49,  5.18it/s][A[A
    
     55%|█████▌    | 696/1261 [02:17<01:50,  5.13it/s][A[A
    
     55%|█████▌    | 697/1261 [02:17<01:49,  5.16it/s][A[A
    
     55%|█████▌    | 698/1261 [02:17<01:48,  5.18it/s][A[A
    
     55%|█████▌    | 699/1261 [02:17<01:48,  5.18it/s][A[A
    
     56%|█████▌    | 700/1261 [02:17<01:48,  5.17it/s][A[A
    
     56%|█████▌    | 701/1261 [02:18<01:48,  5.17it/s][A[A
    
     56%|█████▌    | 702/1261 [02:18<01:47,  5.21it/s][A[A
    
     56%|█████▌    | 703/1261 [02:18<01:47,  5.17it/s][A[A
    
     56%|█████▌    | 704/1261 [02:18<01:46,  5.23it/s][A[A
    
     56%|█████▌    | 705/1261 [02:18<01:46,  5.23it/s][A[A
    
     56%|█████▌    | 706/1261 [02:19<01:45,  5.25it/s][A[A
    
     56%|█████▌    | 707/1261 [02:19<01:45,  5.25it/s][A[A
    
     56%|█████▌    | 708/1261 [02:19<01:45,  5.23it/s][A[A
    
     56%|█████▌    | 709/1261 [02:19<01:46,  5.18it/s][A[A
    
     56%|█████▋    | 710/1261 [02:19<01:46,  5.16it/s][A[A
    
     56%|█████▋    | 711/1261 [02:20<01:46,  5.16it/s][A[A
    
     56%|█████▋    | 712/1261 [02:20<01:48,  5.08it/s][A[A
    
     57%|█████▋    | 713/1261 [02:20<01:47,  5.11it/s][A[A
    
     57%|█████▋    | 714/1261 [02:20<01:46,  5.14it/s][A[A
    
     57%|█████▋    | 715/1261 [02:20<01:45,  5.16it/s][A[A
    
     57%|█████▋    | 716/1261 [02:21<01:45,  5.19it/s][A[A
    
     57%|█████▋    | 717/1261 [02:21<01:44,  5.22it/s][A[A
    
     57%|█████▋    | 718/1261 [02:21<01:43,  5.24it/s][A[A
    
     57%|█████▋    | 719/1261 [02:21<01:44,  5.20it/s][A[A
    
     57%|█████▋    | 720/1261 [02:21<01:46,  5.09it/s][A[A
    
     57%|█████▋    | 721/1261 [02:22<01:46,  5.08it/s][A[A
    
     57%|█████▋    | 722/1261 [02:22<01:44,  5.14it/s][A[A
    
     57%|█████▋    | 723/1261 [02:22<01:45,  5.08it/s][A[A
    
     57%|█████▋    | 724/1261 [02:22<01:48,  4.94it/s][A[A
    
     57%|█████▋    | 725/1261 [02:22<01:46,  5.02it/s][A[A
    
     58%|█████▊    | 726/1261 [02:23<01:45,  5.05it/s][A[A
    
     58%|█████▊    | 727/1261 [02:23<01:44,  5.11it/s][A[A
    
     58%|█████▊    | 728/1261 [02:23<01:42,  5.20it/s][A[A
    
     58%|█████▊    | 729/1261 [02:23<01:43,  5.15it/s][A[A
    
     58%|█████▊    | 730/1261 [02:23<01:43,  5.15it/s][A[A
    
     58%|█████▊    | 731/1261 [02:24<01:43,  5.14it/s][A[A
    
     58%|█████▊    | 732/1261 [02:24<01:43,  5.12it/s][A[A
    
     58%|█████▊    | 733/1261 [02:24<01:42,  5.15it/s][A[A
    
     58%|█████▊    | 734/1261 [02:24<01:43,  5.10it/s][A[A
    
     58%|█████▊    | 735/1261 [02:24<01:42,  5.15it/s][A[A
    
     58%|█████▊    | 736/1261 [02:24<01:40,  5.22it/s][A[A
    
     58%|█████▊    | 737/1261 [02:25<01:40,  5.24it/s][A[A
    
     59%|█████▊    | 738/1261 [02:25<01:41,  5.18it/s][A[A
    
     59%|█████▊    | 739/1261 [02:25<01:40,  5.18it/s][A[A
    
     59%|█████▊    | 740/1261 [02:25<01:40,  5.21it/s][A[A
    
     59%|█████▉    | 741/1261 [02:25<01:40,  5.19it/s][A[A
    
     59%|█████▉    | 742/1261 [02:26<01:40,  5.18it/s][A[A
    
     59%|█████▉    | 743/1261 [02:26<01:40,  5.14it/s][A[A
    
     59%|█████▉    | 744/1261 [02:26<01:41,  5.11it/s][A[A
    
     59%|█████▉    | 745/1261 [02:26<01:40,  5.11it/s][A[A
    
     59%|█████▉    | 746/1261 [02:26<01:39,  5.18it/s][A[A
    
     59%|█████▉    | 747/1261 [02:27<01:39,  5.18it/s][A[A
    
     59%|█████▉    | 748/1261 [02:27<01:40,  5.12it/s][A[A
    
     59%|█████▉    | 749/1261 [02:27<01:40,  5.08it/s][A[A
    
     59%|█████▉    | 750/1261 [02:27<01:40,  5.07it/s][A[A
    
     60%|█████▉    | 751/1261 [02:27<01:39,  5.11it/s][A[A
    
     60%|█████▉    | 752/1261 [02:28<01:39,  5.11it/s][A[A
    
     60%|█████▉    | 753/1261 [02:28<01:40,  5.05it/s][A[A
    
     60%|█████▉    | 754/1261 [02:28<01:40,  5.07it/s][A[A
    
     60%|█████▉    | 755/1261 [02:28<01:38,  5.14it/s][A[A
    
     60%|█████▉    | 756/1261 [02:28<01:37,  5.19it/s][A[A
    
     60%|██████    | 757/1261 [02:29<01:36,  5.24it/s][A[A
    
     60%|██████    | 758/1261 [02:29<01:36,  5.22it/s][A[A
    
     60%|██████    | 759/1261 [02:29<01:35,  5.23it/s][A[A
    
     60%|██████    | 760/1261 [02:29<01:38,  5.10it/s][A[A
    
     60%|██████    | 761/1261 [02:29<01:37,  5.13it/s][A[A
    
     60%|██████    | 762/1261 [02:30<01:36,  5.19it/s][A[A
    
     61%|██████    | 763/1261 [02:30<01:36,  5.17it/s][A[A
    
     61%|██████    | 764/1261 [02:30<01:35,  5.20it/s][A[A
    
     61%|██████    | 765/1261 [02:30<01:39,  5.00it/s][A[A
    
     61%|██████    | 766/1261 [02:30<01:37,  5.08it/s][A[A
    
     61%|██████    | 767/1261 [02:31<01:37,  5.04it/s][A[A
    
     61%|██████    | 768/1261 [02:31<01:36,  5.08it/s][A[A
    
     61%|██████    | 769/1261 [02:31<01:43,  4.75it/s][A[A
    
     61%|██████    | 770/1261 [02:31<01:48,  4.54it/s][A[A
    
     61%|██████    | 771/1261 [02:31<01:46,  4.59it/s][A[A
    
     61%|██████    | 772/1261 [02:32<01:43,  4.73it/s][A[A
    
     61%|██████▏   | 773/1261 [02:32<01:40,  4.84it/s][A[A
    
     61%|██████▏   | 774/1261 [02:32<01:39,  4.91it/s][A[A
    
     61%|██████▏   | 775/1261 [02:32<01:37,  4.98it/s][A[A
    
     62%|██████▏   | 776/1261 [02:32<01:35,  5.09it/s][A[A
    
     62%|██████▏   | 777/1261 [02:33<01:34,  5.10it/s][A[A
    
     62%|██████▏   | 778/1261 [02:33<01:33,  5.15it/s][A[A
    
     62%|██████▏   | 779/1261 [02:33<01:33,  5.13it/s][A[A
    
     62%|██████▏   | 780/1261 [02:33<01:34,  5.08it/s][A[A
    
     62%|██████▏   | 781/1261 [02:33<01:34,  5.10it/s][A[A
    
     62%|██████▏   | 782/1261 [02:34<01:33,  5.14it/s][A[A
    
     62%|██████▏   | 783/1261 [02:34<01:32,  5.15it/s][A[A
    
     62%|██████▏   | 784/1261 [02:34<01:33,  5.11it/s][A[A
    
     62%|██████▏   | 785/1261 [02:34<01:34,  5.03it/s][A[A
    
     62%|██████▏   | 786/1261 [02:34<01:34,  5.05it/s][A[A
    
     62%|██████▏   | 787/1261 [02:35<01:32,  5.10it/s][A[A
    
     62%|██████▏   | 788/1261 [02:35<01:32,  5.14it/s][A[A
    
     63%|██████▎   | 789/1261 [02:35<01:31,  5.13it/s][A[A
    
     63%|██████▎   | 790/1261 [02:35<01:33,  5.06it/s][A[A
    
     63%|██████▎   | 791/1261 [02:35<01:33,  5.03it/s][A[A
    
     63%|██████▎   | 792/1261 [02:36<01:32,  5.05it/s][A[A
    
     63%|██████▎   | 793/1261 [02:36<01:34,  4.98it/s][A[A
    
     63%|██████▎   | 794/1261 [02:36<01:33,  4.99it/s][A[A
    
     63%|██████▎   | 795/1261 [02:36<01:35,  4.90it/s][A[A
    
     63%|██████▎   | 796/1261 [02:36<01:33,  5.00it/s][A[A
    
     63%|██████▎   | 797/1261 [02:37<01:32,  5.03it/s][A[A
    
     63%|██████▎   | 798/1261 [02:37<01:31,  5.04it/s][A[A
    
     63%|██████▎   | 799/1261 [02:37<01:30,  5.09it/s][A[A
    
     63%|██████▎   | 800/1261 [02:37<01:29,  5.13it/s][A[A
    
     64%|██████▎   | 801/1261 [02:37<01:29,  5.11it/s][A[A
    
     64%|██████▎   | 802/1261 [02:38<01:30,  5.07it/s][A[A
    
     64%|██████▎   | 803/1261 [02:38<01:30,  5.05it/s][A[A
    
     64%|██████▍   | 804/1261 [02:38<01:29,  5.10it/s][A[A
    
     64%|██████▍   | 805/1261 [02:38<01:29,  5.08it/s][A[A
    
     64%|██████▍   | 806/1261 [02:38<01:30,  5.00it/s][A[A
    
     64%|██████▍   | 807/1261 [02:39<01:30,  5.03it/s][A[A
    
     64%|██████▍   | 808/1261 [02:39<01:29,  5.05it/s][A[A
    
     64%|██████▍   | 809/1261 [02:39<01:28,  5.11it/s][A[A
    
     64%|██████▍   | 810/1261 [02:39<01:31,  4.92it/s][A[A
    
     64%|██████▍   | 811/1261 [02:39<01:31,  4.94it/s][A[A
    
     64%|██████▍   | 812/1261 [02:40<01:32,  4.83it/s][A[A
    
     64%|██████▍   | 813/1261 [02:40<01:31,  4.87it/s][A[A
    
     65%|██████▍   | 814/1261 [02:40<01:32,  4.86it/s][A[A
    
     65%|██████▍   | 815/1261 [02:40<01:34,  4.71it/s][A[A
    
     65%|██████▍   | 816/1261 [02:40<01:33,  4.77it/s][A[A
    
     65%|██████▍   | 817/1261 [02:41<01:32,  4.81it/s][A[A
    
     65%|██████▍   | 818/1261 [02:41<01:32,  4.78it/s][A[A
    
     65%|██████▍   | 819/1261 [02:41<01:32,  4.78it/s][A[A
    
     65%|██████▌   | 820/1261 [02:41<01:30,  4.85it/s][A[A
    
     65%|██████▌   | 821/1261 [02:41<01:29,  4.92it/s][A[A
    
     65%|██████▌   | 822/1261 [02:42<01:30,  4.87it/s][A[A
    
     65%|██████▌   | 823/1261 [02:42<01:30,  4.84it/s][A[A
    
     65%|██████▌   | 824/1261 [02:42<01:32,  4.73it/s][A[A
    
     65%|██████▌   | 825/1261 [02:42<01:30,  4.80it/s][A[A
    
     66%|██████▌   | 826/1261 [02:42<01:29,  4.88it/s][A[A
    
     66%|██████▌   | 827/1261 [02:43<01:31,  4.77it/s][A[A
    
     66%|██████▌   | 828/1261 [02:43<01:30,  4.77it/s][A[A
    
     66%|██████▌   | 829/1261 [02:43<01:28,  4.86it/s][A[A
    
     66%|██████▌   | 830/1261 [02:43<01:27,  4.92it/s][A[A
    
     66%|██████▌   | 831/1261 [02:43<01:26,  4.96it/s][A[A
    
     66%|██████▌   | 832/1261 [02:44<01:33,  4.58it/s][A[A
    
     66%|██████▌   | 833/1261 [02:44<01:30,  4.72it/s][A[A
    
     66%|██████▌   | 834/1261 [02:44<01:28,  4.80it/s][A[A
    
     66%|██████▌   | 835/1261 [02:44<01:34,  4.49it/s][A[A
    
     66%|██████▋   | 836/1261 [02:45<01:52,  3.77it/s][A[A
    
     66%|██████▋   | 837/1261 [02:45<01:48,  3.92it/s][A[A
    
     66%|██████▋   | 838/1261 [02:45<01:40,  4.21it/s][A[A
    
     67%|██████▋   | 839/1261 [02:45<01:35,  4.42it/s][A[A
    
     67%|██████▋   | 840/1261 [02:46<01:31,  4.62it/s][A[A
    
     67%|██████▋   | 841/1261 [02:46<01:47,  3.92it/s][A[A
    
     67%|██████▋   | 842/1261 [02:46<01:42,  4.07it/s][A[A
    
     67%|██████▋   | 843/1261 [02:46<01:36,  4.33it/s][A[A
    
     67%|██████▋   | 844/1261 [02:47<01:31,  4.55it/s][A[A
    
     67%|██████▋   | 845/1261 [02:47<01:29,  4.65it/s][A[A
    
     67%|██████▋   | 846/1261 [02:47<01:28,  4.70it/s][A[A
    
     67%|██████▋   | 847/1261 [02:47<01:24,  4.87it/s][A[A
    
     67%|██████▋   | 848/1261 [02:47<01:23,  4.95it/s][A[A
    
     67%|██████▋   | 849/1261 [02:48<01:27,  4.72it/s][A[A
    
     67%|██████▋   | 850/1261 [02:48<01:32,  4.44it/s][A[A
    
     67%|██████▋   | 851/1261 [02:48<01:32,  4.42it/s][A[A
    
     68%|██████▊   | 852/1261 [02:48<01:29,  4.57it/s][A[A
    
     68%|██████▊   | 853/1261 [02:48<01:26,  4.72it/s][A[A
    
     68%|██████▊   | 854/1261 [02:49<01:24,  4.81it/s][A[A
    
     68%|██████▊   | 855/1261 [02:49<01:24,  4.82it/s][A[A
    
     68%|██████▊   | 856/1261 [02:49<01:22,  4.92it/s][A[A
    
     68%|██████▊   | 857/1261 [02:49<01:24,  4.77it/s][A[A
    
     68%|██████▊   | 858/1261 [02:49<01:26,  4.66it/s][A[A
    
     68%|██████▊   | 859/1261 [02:50<01:27,  4.60it/s][A[A
    
     68%|██████▊   | 860/1261 [02:50<01:27,  4.60it/s][A[A
    
     68%|██████▊   | 861/1261 [02:50<01:25,  4.68it/s][A[A
    
     68%|██████▊   | 862/1261 [02:50<01:23,  4.77it/s][A[A
    
     68%|██████▊   | 863/1261 [02:51<01:21,  4.87it/s][A[A
    
     69%|██████▊   | 864/1261 [02:51<01:20,  4.92it/s][A[A
    
     69%|██████▊   | 865/1261 [02:51<01:19,  4.99it/s][A[A
    
     69%|██████▊   | 866/1261 [02:51<01:18,  5.03it/s][A[A
    
     69%|██████▉   | 867/1261 [02:51<01:17,  5.07it/s][A[A
    
     69%|██████▉   | 868/1261 [02:51<01:17,  5.10it/s][A[A
    
     69%|██████▉   | 869/1261 [02:52<01:16,  5.13it/s][A[A
    
     69%|██████▉   | 870/1261 [02:52<01:15,  5.16it/s][A[A
    
     69%|██████▉   | 871/1261 [02:52<01:15,  5.18it/s][A[A
    
     69%|██████▉   | 872/1261 [02:52<01:15,  5.18it/s][A[A
    
     69%|██████▉   | 873/1261 [02:52<01:15,  5.15it/s][A[A
    
     69%|██████▉   | 874/1261 [02:53<01:15,  5.15it/s][A[A
    
     69%|██████▉   | 875/1261 [02:53<01:14,  5.16it/s][A[A
    
     69%|██████▉   | 876/1261 [02:53<01:15,  5.07it/s][A[A
    
     70%|██████▉   | 877/1261 [02:53<01:15,  5.09it/s][A[A
    
     70%|██████▉   | 878/1261 [02:53<01:14,  5.16it/s][A[A
    
     70%|██████▉   | 879/1261 [02:54<01:14,  5.16it/s][A[A
    
     70%|██████▉   | 880/1261 [02:54<01:14,  5.10it/s][A[A
    
     70%|██████▉   | 881/1261 [02:54<01:15,  5.06it/s][A[A
    
     70%|██████▉   | 882/1261 [02:54<01:14,  5.10it/s][A[A
    
     70%|███████   | 883/1261 [02:54<01:12,  5.19it/s][A[A
    
     70%|███████   | 884/1261 [02:55<01:13,  5.15it/s][A[A
    
     70%|███████   | 885/1261 [02:55<01:13,  5.11it/s][A[A
    
     70%|███████   | 886/1261 [02:55<01:12,  5.15it/s][A[A
    
     70%|███████   | 887/1261 [02:55<01:11,  5.20it/s][A[A
    
     70%|███████   | 888/1261 [02:55<01:11,  5.19it/s][A[A
    
     70%|███████   | 889/1261 [02:56<01:11,  5.20it/s][A[A
    
     71%|███████   | 890/1261 [02:56<01:11,  5.17it/s][A[A
    
     71%|███████   | 891/1261 [02:56<01:11,  5.17it/s][A[A
    
     71%|███████   | 892/1261 [02:56<01:11,  5.18it/s][A[A
    
     71%|███████   | 893/1261 [02:56<01:10,  5.24it/s][A[A
    
     71%|███████   | 894/1261 [02:57<01:09,  5.27it/s][A[A
    
     71%|███████   | 895/1261 [02:57<01:09,  5.25it/s][A[A
    
     71%|███████   | 896/1261 [02:57<01:10,  5.21it/s][A[A
    
     71%|███████   | 897/1261 [02:57<01:11,  5.08it/s][A[A
    
     71%|███████   | 898/1261 [02:57<01:11,  5.09it/s][A[A
    
     71%|███████▏  | 899/1261 [02:57<01:10,  5.12it/s][A[A
    
     71%|███████▏  | 900/1261 [02:58<01:10,  5.09it/s][A[A
    
     71%|███████▏  | 901/1261 [02:58<01:10,  5.11it/s][A[A
    
     72%|███████▏  | 902/1261 [02:58<01:10,  5.07it/s][A[A
    
     72%|███████▏  | 903/1261 [02:58<01:10,  5.09it/s][A[A
    
     72%|███████▏  | 904/1261 [02:58<01:09,  5.12it/s][A[A
    
     72%|███████▏  | 905/1261 [02:59<01:09,  5.12it/s][A[A
    
     72%|███████▏  | 906/1261 [02:59<01:08,  5.17it/s][A[A
    
     72%|███████▏  | 907/1261 [02:59<01:08,  5.18it/s][A[A
    
     72%|███████▏  | 908/1261 [02:59<01:07,  5.20it/s][A[A
    
     72%|███████▏  | 909/1261 [02:59<01:08,  5.17it/s][A[A
    
     72%|███████▏  | 910/1261 [03:00<01:08,  5.14it/s][A[A
    
     72%|███████▏  | 911/1261 [03:00<01:08,  5.15it/s][A[A
    
     72%|███████▏  | 912/1261 [03:00<01:09,  5.04it/s][A[A
    
     72%|███████▏  | 913/1261 [03:00<01:08,  5.08it/s][A[A
    
     72%|███████▏  | 914/1261 [03:00<01:07,  5.12it/s][A[A
    
     73%|███████▎  | 915/1261 [03:01<01:07,  5.11it/s][A[A
    
     73%|███████▎  | 916/1261 [03:01<01:08,  5.06it/s][A[A
    
     73%|███████▎  | 917/1261 [03:01<01:07,  5.07it/s][A[A
    
     73%|███████▎  | 918/1261 [03:01<01:07,  5.07it/s][A[A
    
     73%|███████▎  | 919/1261 [03:01<01:07,  5.08it/s][A[A
    
     73%|███████▎  | 920/1261 [03:02<01:06,  5.10it/s][A[A
    
     73%|███████▎  | 921/1261 [03:02<01:07,  5.05it/s][A[A
    
     73%|███████▎  | 922/1261 [03:02<01:07,  5.01it/s][A[A
    
     73%|███████▎  | 923/1261 [03:02<01:07,  5.00it/s][A[A
    
     73%|███████▎  | 924/1261 [03:02<01:05,  5.12it/s][A[A
    
     73%|███████▎  | 925/1261 [03:03<01:06,  5.09it/s][A[A
    
     73%|███████▎  | 926/1261 [03:03<01:05,  5.15it/s][A[A
    
     74%|███████▎  | 927/1261 [03:03<01:04,  5.16it/s][A[A
    
     74%|███████▎  | 928/1261 [03:03<01:04,  5.16it/s][A[A
    
     74%|███████▎  | 929/1261 [03:03<01:03,  5.21it/s][A[A
    
     74%|███████▍  | 930/1261 [03:04<01:03,  5.21it/s][A[A
    
     74%|███████▍  | 931/1261 [03:04<01:03,  5.23it/s][A[A
    
     74%|███████▍  | 932/1261 [03:04<01:03,  5.21it/s][A[A
    
     74%|███████▍  | 933/1261 [03:04<01:03,  5.17it/s][A[A
    
     74%|███████▍  | 934/1261 [03:04<01:03,  5.17it/s][A[A
    
     74%|███████▍  | 935/1261 [03:05<01:02,  5.19it/s][A[A
    
     74%|███████▍  | 936/1261 [03:05<01:02,  5.20it/s][A[A
    
     74%|███████▍  | 937/1261 [03:05<01:02,  5.19it/s][A[A
    
     74%|███████▍  | 938/1261 [03:05<01:03,  5.09it/s][A[A
    
     74%|███████▍  | 939/1261 [03:05<01:02,  5.13it/s][A[A
    
     75%|███████▍  | 940/1261 [03:05<01:02,  5.13it/s][A[A
    
     75%|███████▍  | 941/1261 [03:06<01:01,  5.21it/s][A[A
    
     75%|███████▍  | 942/1261 [03:06<01:01,  5.19it/s][A[A
    
     75%|███████▍  | 943/1261 [03:06<01:01,  5.15it/s][A[A
    
     75%|███████▍  | 944/1261 [03:06<01:01,  5.18it/s][A[A
    
     75%|███████▍  | 945/1261 [03:06<01:01,  5.18it/s][A[A
    
     75%|███████▌  | 946/1261 [03:07<01:01,  5.16it/s][A[A
    
     75%|███████▌  | 947/1261 [03:07<01:01,  5.14it/s][A[A
    
     75%|███████▌  | 948/1261 [03:07<01:01,  5.07it/s][A[A
    
     75%|███████▌  | 949/1261 [03:07<01:01,  5.09it/s][A[A
    
     75%|███████▌  | 950/1261 [03:07<01:01,  5.04it/s][A[A
    
     75%|███████▌  | 951/1261 [03:08<01:00,  5.12it/s][A[A
    
     75%|███████▌  | 952/1261 [03:08<01:00,  5.13it/s][A[A
    
     76%|███████▌  | 953/1261 [03:08<00:59,  5.17it/s][A[A
    
     76%|███████▌  | 954/1261 [03:08<00:59,  5.15it/s][A[A
    
     76%|███████▌  | 955/1261 [03:08<00:59,  5.12it/s][A[A
    
     76%|███████▌  | 956/1261 [03:09<00:59,  5.10it/s][A[A
    
     76%|███████▌  | 957/1261 [03:09<00:59,  5.12it/s][A[A
    
     76%|███████▌  | 958/1261 [03:09<00:59,  5.13it/s][A[A
    
     76%|███████▌  | 959/1261 [03:09<00:59,  5.10it/s][A[A
    
     76%|███████▌  | 960/1261 [03:09<00:58,  5.14it/s][A[A
    
     76%|███████▌  | 961/1261 [03:10<00:59,  5.03it/s][A[A
    
     76%|███████▋  | 962/1261 [03:10<00:59,  5.00it/s][A[A
    
     76%|███████▋  | 963/1261 [03:10<00:58,  5.05it/s][A[A
    
     76%|███████▋  | 964/1261 [03:10<00:58,  5.06it/s][A[A
    
     77%|███████▋  | 965/1261 [03:10<00:58,  5.08it/s][A[A
    
     77%|███████▋  | 966/1261 [03:11<00:58,  5.08it/s][A[A
    
     77%|███████▋  | 967/1261 [03:11<00:59,  4.97it/s][A[A
    
     77%|███████▋  | 968/1261 [03:11<00:58,  5.04it/s][A[A
    
     77%|███████▋  | 969/1261 [03:11<00:57,  5.04it/s][A[A
    
     77%|███████▋  | 970/1261 [03:11<00:57,  5.07it/s][A[A
    
     77%|███████▋  | 971/1261 [03:12<00:57,  5.05it/s][A[A
    
     77%|███████▋  | 972/1261 [03:12<00:57,  5.06it/s][A[A
    
     77%|███████▋  | 973/1261 [03:12<00:57,  5.04it/s][A[A
    
     77%|███████▋  | 974/1261 [03:12<00:57,  4.97it/s][A[A
    
     77%|███████▋  | 975/1261 [03:12<00:56,  5.06it/s][A[A
    
     77%|███████▋  | 976/1261 [03:13<00:56,  5.06it/s][A[A
    
     77%|███████▋  | 977/1261 [03:13<00:56,  5.05it/s][A[A
    
     78%|███████▊  | 978/1261 [03:13<00:57,  4.95it/s][A[A
    
     78%|███████▊  | 979/1261 [03:13<00:56,  4.95it/s][A[A
    
     78%|███████▊  | 980/1261 [03:13<00:56,  4.93it/s][A[A
    
     78%|███████▊  | 981/1261 [03:14<00:56,  4.95it/s][A[A
    
     78%|███████▊  | 982/1261 [03:14<00:57,  4.87it/s][A[A
    
     78%|███████▊  | 983/1261 [03:14<00:55,  4.97it/s][A[A
    
     78%|███████▊  | 984/1261 [03:14<00:55,  4.95it/s][A[A
    
     78%|███████▊  | 985/1261 [03:14<00:55,  4.95it/s][A[A
    
     78%|███████▊  | 986/1261 [03:15<00:54,  5.02it/s][A[A
    
     78%|███████▊  | 987/1261 [03:15<00:54,  5.03it/s][A[A
    
     78%|███████▊  | 988/1261 [03:15<00:54,  4.98it/s][A[A
    
     78%|███████▊  | 989/1261 [03:15<01:03,  4.26it/s][A[A
    
     79%|███████▊  | 990/1261 [03:16<01:02,  4.31it/s][A[A
    
     79%|███████▊  | 991/1261 [03:16<01:01,  4.38it/s][A[A
    
     79%|███████▊  | 992/1261 [03:16<00:58,  4.58it/s][A[A
    
     79%|███████▊  | 993/1261 [03:16<00:56,  4.71it/s][A[A
    
     79%|███████▉  | 994/1261 [03:16<00:57,  4.65it/s][A[A
    
     79%|███████▉  | 995/1261 [03:17<01:52,  2.37it/s][A[A
    
     79%|███████▉  | 996/1261 [03:17<01:33,  2.82it/s][A[A
    
     79%|███████▉  | 997/1261 [03:18<01:23,  3.16it/s][A[A
    
     79%|███████▉  | 998/1261 [03:18<01:14,  3.52it/s][A[A
    
     79%|███████▉  | 999/1261 [03:18<01:12,  3.62it/s][A[A
    
     79%|███████▉  | 1000/1261 [03:18<01:05,  3.99it/s][A[A
    
     79%|███████▉  | 1001/1261 [03:19<01:00,  4.28it/s][A[A
    
     79%|███████▉  | 1002/1261 [03:19<01:01,  4.21it/s][A[A
    
     80%|███████▉  | 1003/1261 [03:19<00:59,  4.34it/s][A[A
    
     80%|███████▉  | 1004/1261 [03:19<00:56,  4.56it/s][A[A
    
     80%|███████▉  | 1005/1261 [03:19<00:54,  4.71it/s][A[A
    
     80%|███████▉  | 1006/1261 [03:20<00:53,  4.79it/s][A[A
    
     80%|███████▉  | 1007/1261 [03:20<00:52,  4.87it/s][A[A
    
     80%|███████▉  | 1008/1261 [03:20<00:51,  4.91it/s][A[A
    
     80%|████████  | 1009/1261 [03:20<00:50,  4.97it/s][A[A
    
     80%|████████  | 1010/1261 [03:20<00:49,  5.04it/s][A[A
    
     80%|████████  | 1011/1261 [03:21<00:49,  5.10it/s][A[A
    
     80%|████████  | 1012/1261 [03:21<00:48,  5.12it/s][A[A
    
     80%|████████  | 1013/1261 [03:21<00:48,  5.12it/s][A[A
    
     80%|████████  | 1014/1261 [03:21<00:48,  5.07it/s][A[A
    
     80%|████████  | 1015/1261 [03:21<00:48,  5.07it/s][A[A
    
     81%|████████  | 1016/1261 [03:22<00:48,  5.04it/s][A[A
    
     81%|████████  | 1017/1261 [03:22<00:48,  5.03it/s][A[A
    
     81%|████████  | 1018/1261 [03:22<00:48,  4.98it/s][A[A
    
     81%|████████  | 1019/1261 [03:22<00:49,  4.91it/s][A[A
    
     81%|████████  | 1020/1261 [03:22<00:48,  4.98it/s][A[A
    
     81%|████████  | 1021/1261 [03:23<00:49,  4.87it/s][A[A
    
     81%|████████  | 1022/1261 [03:23<00:49,  4.83it/s][A[A
    
     81%|████████  | 1023/1261 [03:23<00:49,  4.81it/s][A[A
    
     81%|████████  | 1024/1261 [03:23<00:48,  4.86it/s][A[A
    
     81%|████████▏ | 1025/1261 [03:23<00:48,  4.89it/s][A[A
    
     81%|████████▏ | 1026/1261 [03:24<00:47,  4.93it/s][A[A
    
     81%|████████▏ | 1027/1261 [03:24<00:47,  4.91it/s][A[A
    
     82%|████████▏ | 1028/1261 [03:24<00:47,  4.93it/s][A[A
    
     82%|████████▏ | 1029/1261 [03:24<00:48,  4.74it/s][A[A
    
     82%|████████▏ | 1030/1261 [03:24<00:48,  4.72it/s][A[A
    
     82%|████████▏ | 1031/1261 [03:25<00:49,  4.64it/s][A[A
    
     82%|████████▏ | 1032/1261 [03:25<00:51,  4.44it/s][A[A
    
     82%|████████▏ | 1033/1261 [03:25<00:50,  4.51it/s][A[A
    
     82%|████████▏ | 1034/1261 [03:25<00:49,  4.55it/s][A[A
    
     82%|████████▏ | 1035/1261 [03:26<00:48,  4.65it/s][A[A
    
     82%|████████▏ | 1036/1261 [03:26<00:47,  4.76it/s][A[A
    
     82%|████████▏ | 1037/1261 [03:26<00:48,  4.66it/s][A[A
    
     82%|████████▏ | 1038/1261 [03:26<00:47,  4.72it/s][A[A
    
     82%|████████▏ | 1039/1261 [03:26<00:46,  4.82it/s][A[A
    
     82%|████████▏ | 1040/1261 [03:27<00:44,  4.95it/s][A[A
    
     83%|████████▎ | 1041/1261 [03:27<00:44,  4.96it/s][A[A
    
     83%|████████▎ | 1042/1261 [03:27<00:43,  4.99it/s][A[A
    
     83%|████████▎ | 1043/1261 [03:27<00:43,  5.01it/s][A[A
    
     83%|████████▎ | 1044/1261 [03:27<00:45,  4.76it/s][A[A
    
     83%|████████▎ | 1045/1261 [03:28<00:45,  4.78it/s][A[A
    
     83%|████████▎ | 1046/1261 [03:28<00:44,  4.78it/s][A[A
    
     83%|████████▎ | 1047/1261 [03:28<00:44,  4.85it/s][A[A
    
     83%|████████▎ | 1048/1261 [03:28<00:43,  4.89it/s][A[A
    
     83%|████████▎ | 1049/1261 [03:28<00:45,  4.69it/s][A[A
    
     83%|████████▎ | 1050/1261 [03:29<00:43,  4.83it/s][A[A
    
     83%|████████▎ | 1051/1261 [03:29<00:42,  4.95it/s][A[A
    
     83%|████████▎ | 1052/1261 [03:29<00:41,  5.02it/s][A[A
    
     84%|████████▎ | 1053/1261 [03:29<00:40,  5.08it/s][A[A
    
     84%|████████▎ | 1054/1261 [03:29<00:41,  5.05it/s][A[A
    
     84%|████████▎ | 1055/1261 [03:30<00:40,  5.08it/s][A[A
    
     84%|████████▎ | 1056/1261 [03:30<00:41,  4.98it/s][A[A
    
     84%|████████▍ | 1057/1261 [03:30<00:40,  5.03it/s][A[A
    
     84%|████████▍ | 1058/1261 [03:30<00:40,  5.06it/s][A[A
    
     84%|████████▍ | 1059/1261 [03:30<00:39,  5.12it/s][A[A
    
     84%|████████▍ | 1060/1261 [03:31<00:39,  5.08it/s][A[A
    
     84%|████████▍ | 1061/1261 [03:31<00:39,  5.10it/s][A[A
    
     84%|████████▍ | 1062/1261 [03:31<00:38,  5.13it/s][A[A
    
     84%|████████▍ | 1063/1261 [03:31<00:40,  4.94it/s][A[A
    
     84%|████████▍ | 1064/1261 [03:31<00:39,  5.00it/s][A[A
    
     84%|████████▍ | 1065/1261 [03:32<00:39,  5.01it/s][A[A
    
     85%|████████▍ | 1066/1261 [03:32<00:38,  5.03it/s][A[A
    
     85%|████████▍ | 1067/1261 [03:32<00:39,  4.96it/s][A[A
    
     85%|████████▍ | 1068/1261 [03:32<00:38,  5.00it/s][A[A
    
     85%|████████▍ | 1069/1261 [03:32<00:37,  5.06it/s][A[A
    
     85%|████████▍ | 1070/1261 [03:33<00:37,  5.05it/s][A[A
    
     85%|████████▍ | 1071/1261 [03:33<00:37,  5.05it/s][A[A
    
     85%|████████▌ | 1072/1261 [03:33<00:37,  5.05it/s][A[A
    
     85%|████████▌ | 1073/1261 [03:33<00:37,  5.06it/s][A[A
    
     85%|████████▌ | 1074/1261 [03:33<00:38,  4.92it/s][A[A
    
     85%|████████▌ | 1075/1261 [03:34<00:37,  4.90it/s][A[A
    
     85%|████████▌ | 1076/1261 [03:34<00:37,  4.90it/s][A[A
    
     85%|████████▌ | 1077/1261 [03:34<00:37,  4.94it/s][A[A
    
     85%|████████▌ | 1078/1261 [03:34<00:36,  4.99it/s][A[A
    
     86%|████████▌ | 1079/1261 [03:34<00:36,  4.98it/s][A[A
    
     86%|████████▌ | 1080/1261 [03:35<00:35,  5.03it/s][A[A
    
     86%|████████▌ | 1081/1261 [03:35<00:36,  4.97it/s][A[A
    
     86%|████████▌ | 1082/1261 [03:35<00:36,  4.90it/s][A[A
    
     86%|████████▌ | 1083/1261 [03:35<00:35,  4.96it/s][A[A
    
     86%|████████▌ | 1084/1261 [03:35<00:35,  4.99it/s][A[A
    
     86%|████████▌ | 1085/1261 [03:36<00:35,  5.00it/s][A[A
    
     86%|████████▌ | 1086/1261 [03:36<00:34,  5.04it/s][A[A
    
     86%|████████▌ | 1087/1261 [03:36<00:35,  4.96it/s][A[A
    
     86%|████████▋ | 1088/1261 [03:36<00:34,  4.99it/s][A[A
    
     86%|████████▋ | 1089/1261 [03:36<00:34,  4.99it/s][A[A
    
     86%|████████▋ | 1090/1261 [03:37<00:34,  4.98it/s][A[A
    
     87%|████████▋ | 1091/1261 [03:37<00:33,  5.03it/s][A[A
    
     87%|████████▋ | 1092/1261 [03:37<00:33,  4.99it/s][A[A
    
     87%|████████▋ | 1093/1261 [03:37<00:33,  5.01it/s][A[A
    
     87%|████████▋ | 1094/1261 [03:37<00:33,  4.98it/s][A[A
    
     87%|████████▋ | 1095/1261 [03:38<00:33,  4.97it/s][A[A
    
     87%|████████▋ | 1096/1261 [03:38<00:33,  4.92it/s][A[A
    
     87%|████████▋ | 1097/1261 [03:38<00:33,  4.96it/s][A[A
    
     87%|████████▋ | 1098/1261 [03:38<00:33,  4.85it/s][A[A
    
     87%|████████▋ | 1099/1261 [03:38<00:32,  4.93it/s][A[A
    
     87%|████████▋ | 1100/1261 [03:39<00:32,  4.98it/s][A[A
    
     87%|████████▋ | 1101/1261 [03:39<00:32,  4.99it/s][A[A
    
     87%|████████▋ | 1102/1261 [03:39<00:31,  5.03it/s][A[A
    
     87%|████████▋ | 1103/1261 [03:39<00:31,  5.06it/s][A[A
    
     88%|████████▊ | 1104/1261 [03:39<00:31,  5.04it/s][A[A
    
     88%|████████▊ | 1105/1261 [03:40<00:31,  5.02it/s][A[A
    
     88%|████████▊ | 1106/1261 [03:40<00:30,  5.04it/s][A[A
    
     88%|████████▊ | 1107/1261 [03:40<00:30,  5.08it/s][A[A
    
     88%|████████▊ | 1108/1261 [03:40<00:30,  5.07it/s][A[A
    
     88%|████████▊ | 1109/1261 [03:40<00:29,  5.07it/s][A[A
    
     88%|████████▊ | 1110/1261 [03:41<00:38,  3.88it/s][A[A
    
     88%|████████▊ | 1111/1261 [03:41<00:35,  4.21it/s][A[A
    
     88%|████████▊ | 1112/1261 [03:41<00:33,  4.49it/s][A[A
    
     88%|████████▊ | 1113/1261 [03:41<00:31,  4.64it/s][A[A
    
     88%|████████▊ | 1114/1261 [03:42<00:30,  4.78it/s][A[A
    
     88%|████████▊ | 1115/1261 [03:42<00:30,  4.84it/s][A[A
    
     89%|████████▊ | 1116/1261 [03:42<00:29,  4.93it/s][A[A
    
     89%|████████▊ | 1117/1261 [03:42<00:29,  4.92it/s][A[A
    
     89%|████████▊ | 1118/1261 [03:42<00:28,  5.03it/s][A[A
    
     89%|████████▊ | 1119/1261 [03:43<00:28,  4.96it/s][A[A
    
     89%|████████▉ | 1120/1261 [03:43<00:28,  4.98it/s][A[A
    
     89%|████████▉ | 1121/1261 [03:43<00:27,  5.08it/s][A[A
    
     89%|████████▉ | 1122/1261 [03:43<00:27,  5.14it/s][A[A
    
     89%|████████▉ | 1123/1261 [03:43<00:26,  5.14it/s][A[A
    
     89%|████████▉ | 1124/1261 [03:44<00:26,  5.19it/s][A[A
    
     89%|████████▉ | 1125/1261 [03:44<00:25,  5.23it/s][A[A
    
     89%|████████▉ | 1126/1261 [03:44<00:26,  5.19it/s][A[A
    
     89%|████████▉ | 1127/1261 [03:44<00:26,  5.03it/s][A[A
    
     89%|████████▉ | 1128/1261 [03:44<00:26,  5.07it/s][A[A
    
     90%|████████▉ | 1129/1261 [03:45<00:25,  5.10it/s][A[A
    
     90%|████████▉ | 1130/1261 [03:45<00:25,  5.09it/s][A[A
    
     90%|████████▉ | 1131/1261 [03:45<00:26,  4.97it/s][A[A
    
     90%|████████▉ | 1132/1261 [03:45<00:26,  4.89it/s][A[A
    
     90%|████████▉ | 1133/1261 [03:45<00:25,  4.94it/s][A[A
    
     90%|████████▉ | 1134/1261 [03:46<00:25,  5.01it/s][A[A
    
     90%|█████████ | 1135/1261 [03:46<00:24,  5.06it/s][A[A
    
     90%|█████████ | 1136/1261 [03:46<00:24,  5.09it/s][A[A
    
     90%|█████████ | 1137/1261 [03:46<00:24,  4.97it/s][A[A
    
     90%|█████████ | 1138/1261 [03:46<00:24,  5.09it/s][A[A
    
     90%|█████████ | 1139/1261 [03:47<00:23,  5.11it/s][A[A
    
     90%|█████████ | 1140/1261 [03:47<00:23,  5.08it/s][A[A
    
     90%|█████████ | 1141/1261 [03:47<00:23,  5.08it/s][A[A
    
     91%|█████████ | 1142/1261 [03:47<00:23,  5.05it/s][A[A
    
     91%|█████████ | 1143/1261 [03:47<00:23,  5.02it/s][A[A
    
     91%|█████████ | 1144/1261 [03:48<00:22,  5.10it/s][A[A
    
     91%|█████████ | 1145/1261 [03:48<00:22,  5.12it/s][A[A
    
     91%|█████████ | 1146/1261 [03:48<00:22,  5.09it/s][A[A
    
     91%|█████████ | 1147/1261 [03:48<00:22,  5.07it/s][A[A
    
     91%|█████████ | 1148/1261 [03:48<00:22,  5.05it/s][A[A
    
     91%|█████████ | 1149/1261 [03:48<00:22,  5.08it/s][A[A
    
     91%|█████████ | 1150/1261 [03:49<00:21,  5.12it/s][A[A
    
     91%|█████████▏| 1151/1261 [03:49<00:21,  5.13it/s][A[A
    
     91%|█████████▏| 1152/1261 [03:49<00:21,  5.07it/s][A[A
    Exception in thread Thread-9:
    Traceback (most recent call last):
      File "/Users/shrikararchak/Anaconda/anaconda/envs/tensorflow/lib/python3.5/threading.py", line 914, in _bootstrap_inner
        self.run()
      File "/Users/shrikararchak/Anaconda/anaconda/envs/tensorflow/lib/python3.5/site-packages/tqdm/_tqdm.py", line 102, in run
        for instance in self.tqdm_cls._instances:
      File "/Users/shrikararchak/Anaconda/anaconda/envs/tensorflow/lib/python3.5/_weakrefset.py", line 60, in __iter__
        for itemref in self.data:
    RuntimeError: Set changed size during iteration
    
    
     91%|█████████▏| 1153/1261 [03:49<00:21,  5.07it/s][A
     92%|█████████▏| 1154/1261 [03:49<00:21,  5.03it/s][A
     92%|█████████▏| 1155/1261 [03:50<00:20,  5.06it/s][A
     92%|█████████▏| 1156/1261 [03:50<00:21,  4.94it/s][A
     92%|█████████▏| 1157/1261 [03:50<00:21,  4.85it/s][A
     92%|█████████▏| 1158/1261 [03:50<00:21,  4.86it/s][A
     92%|█████████▏| 1159/1261 [03:51<00:21,  4.83it/s][A
     92%|█████████▏| 1160/1261 [03:51<00:20,  4.89it/s][A
     92%|█████████▏| 1161/1261 [03:51<00:20,  4.93it/s][A
     92%|█████████▏| 1162/1261 [03:51<00:19,  4.96it/s][A
     92%|█████████▏| 1163/1261 [03:51<00:19,  4.96it/s][A
     92%|█████████▏| 1164/1261 [03:52<00:19,  4.96it/s][A
     92%|█████████▏| 1165/1261 [03:52<00:19,  4.94it/s][A
     92%|█████████▏| 1166/1261 [03:52<00:19,  4.95it/s][A
     93%|█████████▎| 1167/1261 [03:52<00:19,  4.92it/s][A
     93%|█████████▎| 1168/1261 [03:52<00:18,  5.01it/s][A
     93%|█████████▎| 1169/1261 [03:53<00:18,  5.09it/s][A
     93%|█████████▎| 1170/1261 [03:53<00:18,  5.00it/s][A
     93%|█████████▎| 1171/1261 [03:53<00:17,  5.03it/s][A
     93%|█████████▎| 1172/1261 [03:53<00:17,  5.05it/s][A
     93%|█████████▎| 1173/1261 [03:53<00:17,  4.98it/s][A
     93%|█████████▎| 1174/1261 [03:54<00:17,  4.98it/s][A
     93%|█████████▎| 1175/1261 [03:54<00:17,  5.02it/s][A
     93%|█████████▎| 1176/1261 [03:54<00:16,  5.04it/s][A
     93%|█████████▎| 1177/1261 [03:54<00:16,  4.95it/s][A
     93%|█████████▎| 1178/1261 [03:54<00:16,  4.97it/s][A
     93%|█████████▎| 1179/1261 [03:55<00:16,  5.04it/s][A
     94%|█████████▎| 1180/1261 [03:55<00:15,  5.10it/s][A
     94%|█████████▎| 1181/1261 [03:55<00:15,  5.11it/s][A
     94%|█████████▎| 1182/1261 [03:55<00:15,  5.11it/s][A
     94%|█████████▍| 1183/1261 [03:55<00:15,  5.14it/s][A
     94%|█████████▍| 1184/1261 [03:55<00:15,  5.13it/s][A
     94%|█████████▍| 1185/1261 [03:56<00:14,  5.12it/s][A
     94%|█████████▍| 1186/1261 [03:56<00:14,  5.07it/s][A
     94%|█████████▍| 1187/1261 [03:56<00:14,  5.02it/s][A
     94%|█████████▍| 1188/1261 [03:56<00:14,  4.99it/s][A
     94%|█████████▍| 1189/1261 [03:56<00:14,  5.05it/s][A
     94%|█████████▍| 1190/1261 [03:57<00:13,  5.10it/s][A
     94%|█████████▍| 1191/1261 [03:57<00:13,  5.12it/s][A
     95%|█████████▍| 1192/1261 [03:57<00:13,  5.11it/s][A
     95%|█████████▍| 1193/1261 [03:57<00:13,  5.04it/s][A
     95%|█████████▍| 1194/1261 [03:58<00:15,  4.27it/s][A
     95%|█████████▍| 1195/1261 [03:58<00:14,  4.50it/s][A
     95%|█████████▍| 1196/1261 [03:58<00:14,  4.58it/s][A
     95%|█████████▍| 1197/1261 [03:58<00:13,  4.60it/s][A
     95%|█████████▌| 1198/1261 [03:58<00:13,  4.72it/s][A
     95%|█████████▌| 1199/1261 [03:59<00:12,  4.87it/s][A
     95%|█████████▌| 1200/1261 [03:59<00:12,  4.87it/s][A
     95%|█████████▌| 1201/1261 [03:59<00:12,  4.89it/s][A
     95%|█████████▌| 1202/1261 [03:59<00:11,  4.99it/s][A
     95%|█████████▌| 1203/1261 [03:59<00:11,  5.05it/s][A
     95%|█████████▌| 1204/1261 [04:00<00:11,  5.10it/s][A
     96%|█████████▌| 1205/1261 [04:00<00:11,  4.98it/s][A
     96%|█████████▌| 1206/1261 [04:00<00:11,  4.88it/s][A
     96%|█████████▌| 1207/1261 [04:00<00:10,  4.99it/s][A
     96%|█████████▌| 1208/1261 [04:00<00:10,  4.98it/s][A
     96%|█████████▌| 1209/1261 [04:01<00:10,  4.93it/s][A
     96%|█████████▌| 1210/1261 [04:01<00:10,  4.92it/s][A
     96%|█████████▌| 1211/1261 [04:01<00:10,  4.96it/s][A
     96%|█████████▌| 1212/1261 [04:01<00:09,  4.91it/s][A
     96%|█████████▌| 1213/1261 [04:01<00:09,  4.84it/s][A
     96%|█████████▋| 1214/1261 [04:02<00:09,  4.93it/s][A
     96%|█████████▋| 1215/1261 [04:02<00:09,  4.96it/s][A
     96%|█████████▋| 1216/1261 [04:02<00:09,  4.95it/s][A
     97%|█████████▋| 1217/1261 [04:02<00:08,  4.97it/s][A
     97%|█████████▋| 1218/1261 [04:02<00:08,  4.94it/s][A
     97%|█████████▋| 1219/1261 [04:03<00:08,  4.98it/s][A
     97%|█████████▋| 1220/1261 [04:03<00:08,  4.81it/s][A
     97%|█████████▋| 1221/1261 [04:03<00:08,  4.89it/s][A
     97%|█████████▋| 1222/1261 [04:03<00:07,  4.95it/s][A
     97%|█████████▋| 1223/1261 [04:03<00:07,  5.00it/s][A
     97%|█████████▋| 1224/1261 [04:04<00:07,  5.04it/s][A
     97%|█████████▋| 1225/1261 [04:04<00:07,  5.00it/s][A
     97%|█████████▋| 1226/1261 [04:04<00:07,  4.95it/s][A
     97%|█████████▋| 1227/1261 [04:04<00:06,  4.95it/s][A
     97%|█████████▋| 1228/1261 [04:04<00:06,  4.98it/s][A
     97%|█████████▋| 1229/1261 [04:05<00:06,  5.04it/s][A
     98%|█████████▊| 1230/1261 [04:05<00:06,  5.05it/s][A
     98%|█████████▊| 1231/1261 [04:05<00:06,  4.98it/s][A
     98%|█████████▊| 1232/1261 [04:05<00:05,  4.99it/s][A
     98%|█████████▊| 1233/1261 [04:05<00:05,  5.01it/s][A
     98%|█████████▊| 1234/1261 [04:06<00:05,  5.06it/s][A
     98%|█████████▊| 1235/1261 [04:06<00:05,  5.02it/s][A
     98%|█████████▊| 1236/1261 [04:06<00:05,  4.97it/s][A
     98%|█████████▊| 1237/1261 [04:06<00:04,  5.05it/s][A
     98%|█████████▊| 1238/1261 [04:06<00:04,  5.04it/s][A
     98%|█████████▊| 1239/1261 [04:07<00:04,  5.06it/s][A
     98%|█████████▊| 1240/1261 [04:07<00:04,  5.07it/s][A
     98%|█████████▊| 1241/1261 [04:07<00:03,  5.02it/s][A
     98%|█████████▊| 1242/1261 [04:07<00:03,  5.05it/s][A
     99%|█████████▊| 1243/1261 [04:07<00:03,  5.00it/s][A
     99%|█████████▊| 1244/1261 [04:08<00:03,  5.08it/s][A
     99%|█████████▊| 1245/1261 [04:08<00:03,  5.04it/s][A
     99%|█████████▉| 1246/1261 [04:08<00:02,  5.04it/s][A
     99%|█████████▉| 1247/1261 [04:08<00:02,  5.07it/s][A
     99%|█████████▉| 1248/1261 [04:08<00:02,  5.10it/s][A
     99%|█████████▉| 1249/1261 [04:09<00:02,  5.15it/s][A
     99%|█████████▉| 1250/1261 [04:09<00:02,  5.09it/s][A
     99%|█████████▉| 1251/1261 [04:09<00:01,  5.07it/s][A
     99%|█████████▉| 1252/1261 [04:09<00:01,  5.10it/s][A
     99%|█████████▉| 1253/1261 [04:09<00:01,  5.08it/s][A
     99%|█████████▉| 1254/1261 [04:10<00:01,  5.11it/s][A
    100%|█████████▉| 1255/1261 [04:10<00:01,  5.06it/s][A
    100%|█████████▉| 1256/1261 [04:10<00:00,  5.05it/s][A
    100%|█████████▉| 1257/1261 [04:10<00:00,  5.00it/s][A
    100%|█████████▉| 1258/1261 [04:10<00:00,  5.01it/s][A
    100%|█████████▉| 1259/1261 [04:11<00:00,  5.02it/s][A
    100%|█████████▉| 1260/1261 [04:11<00:00,  4.97it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: output.mp4 
    
    CPU times: user 3min 57s, sys: 1min 15s, total: 5min 13s
    Wall time: 4min 12s



![png](output_46_3.png)


### Project Details

The pipeline works well when we have roads which are have clear demarcation on the road boundaries. In our case lets say if the road have the same material and the color of the tar its able to perform significantly well.

The pipeline fails to detect lines when we have road which have been patched or a new tar is laid on existing road and the lane markings have changed.  After working I personally think Computer vision projects are unforgiving when it comes to the parameters. A suitable and a robust approach would be to use deep learning with more data so that its able to learn from the data. Tuning computer vision problem to different condition and generalizing the parameter might be a bit more of work
