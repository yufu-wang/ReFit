#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#------------------------------------------------------------------------------
import logging
import math
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG)


def get_agora_intrinsic(imgPath):
    dslr_sens_width = 36
    dslr_sens_height = 20.25
    imgWidth = 1280
    imgHeight = 720
    
    if 'hdri' in imgPath:
        focalLength = 50
    elif 'cam00' in imgPath:
        focalLength = 18
    elif 'cam01' in imgPath:
        focalLength = 18
    elif 'cam02' in imgPath:
        focalLength = 18
    elif 'cam03' in imgPath:
        focalLength = 18
    elif 'ag2' in imgPath:
        focalLength = 28
    else:
        focalLength = 28
    
    cx=imgWidth / 2.
    cy=imgHeight / 2.
    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)
    
    img_focal = (focalLength_x + focalLength_y)/2
    img_center = [cx, cy]
    
    return img_focal, img_center


def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel


def toCamCoords(j3d, camPosWorld):
    # transform gt to camera coordinate frame
    j3d = j3d - camPosWorld
    return j3d


def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1, -1, 1])
    return points


def smpl2opencv(j3d):
    # change sign of axis 1 and axis 2
    j3d = j3d * np.array([1, -1, -1])
    return j3d


def project_point(joint, RT, KKK):

    P = np.dot(KKK, RT)
    joints_2d = np.dot(P, joint)
    joints_2d = joints_2d[0:2] / joints_2d[2]

    return joints_2d


def project_2d(
        df,
        i,
        pNum,
        joints3d,
        meanPose=False):

    dslr_sens_width = 36
    dslr_sens_height = 20.25
    imgWidth = 1280
    imgHeight = 720

    imgPath = df.iloc[i]['imgPath']
    if 'hdri' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 50
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 0

    elif 'cam00' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, -275, 265]
        camYaw = 135
        camPitch = 30
    elif 'cam01' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [400, 225, 265]
        camYaw = -135
        camPitch = 30
    elif 'cam02' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, 170, 265]
        camYaw = -45
        camPitch = 30
    elif 'cam03' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = True
        focalLength = 18
        camPosWorld = [-490, -275, 265]
        camYaw = 45
        camPitch = 30
    elif 'ag2' in imgPath:
        ground_plane = [0, 0, 0]
        scene3d = False
        focalLength = 28
        camPosWorld = [0, 0, 170]
        camYaw = 0
        camPitch = 15
    else:
        ground_plane = [0, -1.7, 0]
        scene3d = True
        focalLength = 28
        camPosWorld = [
            df.iloc[i]['camX'],
            df.iloc[i]['camY'],
            df.iloc[i]['camZ']]
        camYaw = df.iloc[i]['camYaw']
        camPitch = 0

    if meanPose:
        yawSMPL = 0
        trans3d = [0, 0, 0]
    else:
        yawSMPL = df.iloc[i]['Yaw'][pNum]
        trans3d = [df.iloc[i]['X'][pNum],
                   df.iloc[i]['Y'][pNum],
                   df.iloc[i]['Z'][pNum]]

    gt2d, gt3d_camCoord, focal = project2d(joints3d, focalLength=focalLength, scene3d=scene3d,
                                    trans3d=trans3d,
                                    dslr_sens_width=dslr_sens_width,
                                    dslr_sens_height=dslr_sens_height,
                                    camPosWorld=camPosWorld,
                                    cy=imgHeight / 2,
                                    cx=imgWidth / 2,
                                    yawSMPL=yawSMPL,
                                    ground_plane=ground_plane,
                                    ind=i,
                                    pNum=pNum,
                                    meanPose=meanPose, camPitch=camPitch, camYaw=camYaw)
    return gt2d, gt3d_camCoord, focal


def project2d(
        j3d,
        focalLength,
        scene3d,
        trans3d,
        dslr_sens_width,
        dslr_sens_height,
        camPosWorld,
        cy,
        cx,
        yawSMPL,
        ground_plane,
        ind=-1,
        pNum=-1,
        meanPose=False,
        camPitch=0,
        camYaw=0):

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    camMat = np.array([[focalLength_x, 0, cx],
                       [0, focalLength_y, cy],
                       [0, 0, 1]])

    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    if scene3d:
        camPosWorld = unreal2cv2(
            np.reshape(
                camPosWorld, (1, 3))) + np.array(ground_plane)
    else:
        camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3)))

    # get points in camera coordinate system
    j3d = smpl2opencv(j3d)

    # scans have a 90deg rotation, but for mean pose from vposer there is no
    # such rotation
    if meanPose:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, (yawSMPL) / 180 * np.pi, 0]], dtype=float))
    else:
        rotMat, _ = cv2.Rodrigues(
            np.array([[0, ((yawSMPL - 90) / 180) * np.pi, 0]], dtype=float))

    j3d = np.matmul(rotMat, j3d.T).T
    j3d = j3d + trans3d

    camera_rotationMatrix, _ = cv2.Rodrigues(
        np.array([0, ((-camYaw) / 180) * np.pi, 0]).reshape(3, 1))
    camera_rotationMatrix2, _ = cv2.Rodrigues(
        np.array([camPitch / 180 * np.pi, 0, 0]).reshape(3, 1))

    j3d_new = np.matmul(camera_rotationMatrix, j3d.T - camPosWorld.T).T
    j3d_new = np.matmul(camera_rotationMatrix2, j3d_new.T).T

    RT = np.concatenate((np.diag([1., 1., 1.]), np.zeros((3, 1))), axis=1)
    j2d = np.zeros((j3d_new.shape[0], 2))
    for i in range(j3d_new.shape[0]):
        j2d[i, :] = project_point(np.concatenate(
            [j3d_new[i, :], np.array([1])]), RT, camMat)


    return j2d, j3d_new, (focalLength_x + focalLength_y)/2
