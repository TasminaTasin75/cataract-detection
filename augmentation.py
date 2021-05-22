import random
import os
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import imageio
#import imgaug as ia
import imgaug.augmenters as iaa

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def shear_image(image):
    shear = iaa.Affine(shear=(0,40))
    shear_image=shear.augment_image(image)
    
    return shear_image

def brightness_enhance(image):
    contrast=iaa.GammaContrast(gamma=2.0)
    contrast_image =contrast.augment_image(image)
    
    return contrast_image





# our folder path containing some images

cat_im_path = "cataract"
non_cat_im_path = "noncataract"
cataract_image_aug_path = "augmented/cataract"
non_cataract_image_aug_path = "augmented/non_cataract"
# the number of file to generate
num_files_desired = 1000

#num_files_desired_rand_noise = 2000

num_files_desired_brightness_enhance = 1500

num_files_horizontal_flip = 2500

#num_files_shear = 3500

# loop on all files of the folder and build a list of files paths
#images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
cat_images = [os.path.join(cat_im_path, f) for f in os.listdir(cat_im_path) if os.path.isfile(os.path.join(cat_im_path, f))]
non_cat_images = [os.path.join(non_cat_im_path, f) for f in os.listdir(non_cat_im_path) if os.path.isfile(os.path.join(non_cat_im_path, f))]


num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = random_rotation(image_to_transform)
    new_file_path = '%s/augmented_cat_image_%s.jpg' % (cataract_image_aug_path, num_generated_files)
    # write image to the disk
    print(image_path)
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files = num_generated_files+1    
   
num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(non_cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = random_rotation(image_to_transform)
    new_file_path = '%s/augmented_non_cat_image_%s.jpg' % (non_cataract_image_aug_path, num_generated_files)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files = num_generated_files+1
"""    
num_generated_files_rand_noise = 1001
while num_generated_files_rand_noise <= num_files_desired_rand_noise:
    # random image from the folder
    image_path = random.choice(cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = random_noise(image_to_transform)
    new_file_path = '%s/augmented_cat_image_%s.jpg' % (cataract_image_aug_path, num_generated_files_rand_noise)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_rand_noise = num_generated_files_rand_noise+1

num_generated_files_rand_noise = 1001
while num_generated_files_rand_noise <= num_files_desired_rand_noise:
    # random image from the folder
    image_path = random.choice(non_cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = random_noise(image_to_transform)
    new_file_path = '%s/augmented_non_cat_image_%s.jpg' % (non_cataract_image_aug_path, num_generated_files_rand_noise)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_rand_noise = num_generated_files_rand_noise+1
"""
num_generated_files_brightness_enhancement = 1001
while num_generated_files_brightness_enhancement <= num_files_desired_brightness_enhance:
    # random image from the folder
    image_path = random.choice(cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = brightness_enhance(image_to_transform)
    new_file_path = '%s/augmented_cat_image_%s.jpg' % (cataract_image_aug_path, num_generated_files_brightness_enhancement)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_brightness_enhancement = num_generated_files_brightness_enhancement+1

num_generated_files_brightness_enhancement = 1001
while num_generated_files_brightness_enhancement <= num_files_desired_brightness_enhance:
    # random image from the folder
    image_path = random.choice(non_cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = brightness_enhance(image_to_transform)
    new_file_path = '%s/augmented_non_cat_image_%s.jpg' % (non_cataract_image_aug_path, num_generated_files_brightness_enhancement)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_brightness_enhancement = num_generated_files_brightness_enhancement+1
    
num_generated_files_horizontal_flip = 1501
while num_generated_files_horizontal_flip <= num_files_horizontal_flip:
    # random image from the folder
    image_path = random.choice(cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = horizontal_flip(image_to_transform)
    new_file_path = '%s/augmented_cat_image_%s.jpg' % (cataract_image_aug_path, num_generated_files_horizontal_flip)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_horizontal_flip = num_generated_files_horizontal_flip+1

num_generated_files_horizontal_flip = 1501
while num_generated_files_horizontal_flip <= num_files_horizontal_flip:
    # random image from the folder
    image_path = random.choice(non_cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = horizontal_flip(image_to_transform)
    new_file_path = '%s/augmented_non_cat_image_%s.jpg' % (non_cataract_image_aug_path, num_generated_files_horizontal_flip)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_horizontal_flip = num_generated_files_horizontal_flip+1 


"""
num_generated_files_shear = 3001
while num_generated_files_shear <= num_files_shear:
    # random image from the folder
    image_path = random.choice(cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = shear_image(image_to_transform)
    new_file_path = '%s/augmented_cat_image_%s.jpg' % (cataract_image_aug_path, num_generated_files_shear)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_shear = num_generated_files_shear+1

num_generated_files_shear = 3001
while num_generated_files_shear <= num_files_shear:
    # random image from the folder
    image_path = random.choice(non_cat_images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    transformed_image = shear_image(image_to_transform)
    new_file_path = '%s/augmented_non_cat_image_%s.jpg' % (non_cataract_image_aug_path, num_generated_files_shear)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files_shear = num_generated_files_shear+1      
"""    