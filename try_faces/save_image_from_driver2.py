import pygame
import numpy as np
from pygame.locals import *
import pygame.camera
from time import time


def image_cap(folder_path, face_encodings):
    pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()
    if camlist:
        cam = pygame.camera.Camera(camlist[0], (640, 480))
    else:
        raise ValueError("No camera found")
    cam.start()
    screen = pygame.display.set_mode((640, 480))
    while True:
        img = cam.get_image()
        screen.blit(img, (0,0))
        pygame.display.update()
        for event in pygame.event.get():
            if event.key == pygame.K_s:
                filename = "frame_" + str(int(time())) + ".jpg"
                new_pic_path = folder_path + "\\" + filename
                pygame.image.save(img, new_pic_path)
            if event.key == pygame.K_q:
                break

try_image_path = r"C:\talpiot\year2\armani\try_faces\blitz_faces"
image_cap(try_image_path,0)
