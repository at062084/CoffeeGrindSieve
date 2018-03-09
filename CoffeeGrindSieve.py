#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:41:14 2018

@author: at062084
"""

# cv2::AdaptiveThresholdTypes
# cv2.ximgproc.niBlackThreshold
# cv.ximgproc.thinning

import numpy as np
from matplotlib import pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

bPrint=False
bPlot=False

imgDir = "/home/at062084/DataEngineering/Python/images1/"
imgFile = "/home/at062084/DataEngineering/Python/PB070871.JPG"

def imgWrite(img, fName):
    f = imgDir + fName + ".png"
    cv2.imwrite(f, img)

def figWrite(fig, fName):
    f = imgDir + fName + ".png"
    fig.savefig(f)
    plt.close(fig)

def ccHistWrite(cc, fName):
    x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    h = np.histogram(cc, bins=2**x)
    fig = plt.figure()
    plt.bar(x[0:16]+0.5,h[0][0:16])
    plt.xticks(x,x)
    plt.grid(); plt.xlim(0,16)
    plt.title("openCV-CoffeeGrinderSieve")
    plt.xlabel("Size of particles [2**n pixels]")
    plt.ylabel("Number of particles")
    f = imgDir + fName + ".png"    
    fig.savefig(f)
    plt.close(fig)


def plot3dWrite(img, fName, s=1, e=75):
    # 3d plot of dist transform
    fig = plt.figure()
    ims = cv2.resize(img, None, fx=1/s, fy=1/s, interpolation = cv2.INTER_LINEAR)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=e)
    xx, yy = np.mgrid[0:ims.shape[0], 0:ims.shape[1]]
    ax.plot_surface(xx, yy, ims, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0)
    f = imgDir + fName + ".png"    
    fig.savefig(f)
    plt.close(fig)

# Load image as RGB
img = cv2.imread(imgFile,1)
if bPrint: print(img.shape, img.size, img.dtype)

# Split RGB channels
b,g,r = cv2.split(img)
imgWrite(b,"raw_b"); imgWrite(r,"raw_r"); imgWrite(g,"raw_g"); 

# best rgb channel. done convert rgb to grey
gray = b
fn = "gray"; imgWrite(gray,fn)

# scale down image to speed up further processing
k = 4
gray = cv2.resize(gray, None, fx=1/k, fy=1/k, interpolation = cv2.INTER_LINEAR)
# plt.imshow(gray, cmap='gray')
fn += ".scaled" + str(k); imgWrite(gray, fn)

# 3d DEM plot
plot3dWrite(-gray, fn+".3d", s=2)
 
# Edge presering smoothing
h = 3
smooth = cv2.medianBlur(gray, h)
fn += ".smooth" + str(h); imgWrite(smooth, fn)

# strel size experimental --> will be 25*8=200 for original size
t=np.int(500/k)
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
tophat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, strel)
fn += ".tophat" + str(t); imgWrite(tophat, fn)

# convert to BW. Threshold experimental
b = 32
bw = cv2.threshold(tophat,b, 255, cv2.THRESH_BINARY)[1]
fn += ".bw" + str(b); imgWrite(bw, fn)


# calculate smooth background
gb = 99
bg = smooth + tophat
bgs = cv2.GaussianBlur(bg,(gb,gb),gb)
plt.imshow(bgs, cmap='gray', vmin=0, vmax=255)
flat = (512+smooth)-bgs
flat = flat - flat.min()
flat = np.uint8(flat)
plt.imshow(flat, cmap='gray', vmin=0, vmax=255)

t=np.int(500/k)
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(t,t))
tophat2 = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, strel)
plt.imshow(tophat2, cmap='gray', vmin=0, vmax=255)
plt.imshow(tophat2>32, cmap='gray')
fn += ".tophat" + str(t); imgWrite(tophat, fn)


# fill small holes
c=3
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(c,c))
bw1 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, strel)

# remove small particles
o=3
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(o,o))
bw2 = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, strel)
fn += ".oc" + str(o); imgWrite(bw2, fn)

# distanceTransform
dt = cv2.distanceTransform(bw2, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
fn += ".dist"; imgWrite(dt/dt.max()*255, fn)

# 3d DEM plot
plot3dWrite(dt, fn+".3d", s=2, e=65)

# remember current file name
fnx = fn
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# Threshold 1: Low threshold on distance transform
t2=2
dt1 = cv2.threshold(dt,t2, 255, cv2.THRESH_BINARY)[1]
dt1 = cv2.morphologyEx(dt1, cv2.MORPH_OPEN, se)
dt1 = np.uint8(dt1)
fn += ".bw" + str(t2); imgWrite(dt1, fn)

# identify marker regions
n,cc1 = cv2.connectedComponents(dt1)
fn += ".cc"; imgWrite(cc1, fn)

# construct watershed from tophat and connected components
rgb = cv2.cvtColor(255-tophat, cv2.COLOR_GRAY2BGR)
ws1 = cv2.watershed(rgb,cc1)
seg1 = (ws1==-1)*255
fn += ".ws"; imgWrite(seg1, fn)

# restore filename
fn = fnx

# Threshold 2: High threshold on distance transform
t2=6
dt2 = cv2.threshold(dt,t2, 255, cv2.THRESH_BINARY)[1]
dt2 = cv2.morphologyEx(dt2, cv2.MORPH_OPEN, se)
dt2 = np.uint8(dt2)
fn += ".bw" + str(t2); imgWrite(dt2, fn)

# identify marker regions
n,cc2 = cv2.connectedComponents(dt2)
fn += ".cc"; imgWrite(cc2, fn)

# construct watershed from tophat and connected components
rgb = cv2.cvtColor(255-tophat, cv2.COLOR_GRAY2BGR)
ws2 = cv2.watershed(rgb,cc2)
seg2 = (ws2==-1)*255
fn += ".ws"; imgWrite(seg2, fn)

# Final connected components: Merge the watersheds

# Set watersheds to 0 an dconvert to uint8
ws1[ws1==-1]=0
ws2[ws2==-1]=0
ws1 = np.uint8(ws1)
ws2 = np.uint8(ws2)

# remove background.  different on every litte  change. weird
h1 = cv2.calcHist([ws1],[0],None,[256],[0,256])
b1 = np.where(h1==h1.max())[0][0]
ws1[ws1==b1]=0
h2 = cv2.calcHist([ws2],[0],None,[256],[0,256])
b2 = np.where(h2==h2.max())[0][0]
ws2[ws2==b2]=0

# set forground to 255
ws1[ws1>0]=1
ws2[ws2>0]=1
ws12 = ws1+ws2
ws12[ws12>0]=255
f = fn + ".ws12"; imgWrite(ws12, f)

# Merge Watersheds and make walls thicker
seg12 = seg1 | seg2
seg12 = 255-np.uint8(seg12)
seg12 = cv2.morphologyEx(seg12, cv2.MORPH_ERODE, se)
seg12 = cv2.morphologyEx(seg12, cv2.MORPH_CLOSE, se)
seg12 = cv2.morphologyEx(seg12, cv2.MORPH_OPEN, se)
seg12 = cv2.morphologyEx(seg12, cv2.MORPH_DILATE, se)
f = fn + ".seg12"; imgWrite(seg12, f)

# Impose watersheds on 
wsx = ws12 * seg12 * 255
wsx = np.uint8(wsx)
f = fn + ".final"; imgWrite(wsx, f)

# Impose on tophat
wsth = (wsx/255) * (tophat/tophat.max()*255)
fn += ".tophat"; imgWrite(wsth, fn)

# Final connected components
cc_n, cc_lbl, cc_stats, cc_cntr = cv2.connectedComponentsWithStats(wsx, connectivity=4)
f = fn + ".cc"; imgWrite(wsx, f)

# Area of cc's
cc_area = cc_stats[:,cv2.CC_STAT_AREA]

f = fn + ".hist"
ccHistWrite(cc_area, f)


