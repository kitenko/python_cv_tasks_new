import numpy as np
import cv2
import os

#image_final = np.zeros((10, 10, 3), np.uint8)

#image = np.random.randint(0, 255, (10, 10, 1))
#image_zeros = np.zeros((10, 10, 1), np.uint8)
#image_2 = np.zeros((10, 10, 1), np.uint8)

#ter = np.concatenate((image_zeros, image_zeros, image), axis=-1)
#print(ter)
#color = tuple(reversed(ter))

#image_final[:] = color
#image_final = image_final[:, :, ::-1]
#print(image_final)

#print(ter)
#cv2.imwrite('red2.jpg', ter)



"""print(image)
print(image_1)"""


path_dataset = "dataset"
path_red = path_dataset + "/red"
path_blue = path_dataset + "/blue"
path_green = path_dataset + "/green"


def create_dataset(path: str, number_of_images=100, width_image=(100, 300), height_image=(100, 300), colour=''):
    def create_directory(path: str):

        path_red = path + "/red"
        path_blue = path + "/blue"
        path_green = path + "/green"

        os.makedirs(path_red, exist_ok=True)
        os.makedirs(path_blue, exist_ok=True)
        os.makedirs(path_green, exist_ok=True)

        return

    create_directory(path)


    if colour == 'red':
        os.chdir(path_red)
        for i in range(number_of_images):

            # random height and width
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_red = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_zeros, image_red), axis=-1)
            cv2.imwrite('red_'+str(i)+'.jpg', final_image)

    elif colour == 'blue':
        os.chdir(path_blue)
        for i in range(number_of_images):

            # random height and width
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_blue = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_blue, image_zeros, image_zeros), axis=-1)
            cv2.imwrite('blue_' + str(i) + '.jpg', final_image)

    else:
        os.chdir(path_green)
        for i in range(number_of_images):

            # random wight and height
            width = np.random.randint(width_image[0], width_image[1])
            height = np.random.randint(height_image[0], height_image[1])

            # create and save image
            image_green = np.random.randint(0, 255, (width, height, 1))
            image_zeros = np.zeros((width, height, 1), np.uint8)
            final_image = np.concatenate((image_zeros, image_green, image_zeros), axis=-1)
            cv2.imwrite('green_' + str(i) + '.jpg', final_image)


create_dataset(path="dataset", number_of_images=100, colour='green', width_image=(300, 500), height_image=(300, 500))
