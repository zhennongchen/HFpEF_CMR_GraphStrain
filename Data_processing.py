import numpy as np
import nibabel as nb
import os
from scipy import ndimage
import random
import DeepStrain.Defaults as Defaults
import DeepStrain.functions_collection as ff


def crop_or_pad(array, target, value=0):
    """
    Symmetrically pad or crop along each dimension to the specified target dimension.
    :param array: Array to be cropped / padded.
    :type array: array-like
    :param target: Target dimension.
    :type target: `int` or array-like of length array.ndim
    :returns: Cropped/padded array. 
    :rtype: array-like
    """
    # Pad each axis to at least the target.
    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]

# function:center crop (need to provide the segmentation mask)
def center_crop(I, S, crop_size, according_to_which_class, centroid = None):
    # make sure S is integers
    S = S.astype(int)
    # Compute the centroid of the class 1 region in the mask
    assert isinstance(according_to_which_class, list), "according_to_which_class must be a list"
    assert I.shape == S.shape, "Image and mask must have the same shape"
    assert len(crop_size) == len(I.shape), "Crop size dimensions must match image dimensions"
    
    # Find the indices where the mask > 0
    if centroid is None:
        mask_indices = np.argwhere(np.isin(S, according_to_which_class))

        if len(mask_indices) == 0:
            raise ValueError("The mask does not contain any class 1 region")

        # Compute centroid
        
        centroid = np.mean(mask_indices, axis=0).astype(int)

    # Define the crop slices for each dimension
    slices = []
    for dim, size in enumerate(crop_size):
        start = max(centroid[dim] - size // 2, 0)
        end = start + size
        # Adjust the start and end if they are out of bounds
        if end > I.shape[dim]:
            end = I.shape[dim]
            start = max(end - size, 0)
        slices.append(slice(start, end))

    # Crop the image and the mask
    if len(I.shape) == 2:
        cropped_I = I[slices[0], slices[1]]
        cropped_S = S[slices[0], slices[1]]
    elif len(I.shape) == 3:
        cropped_I = I[slices[0], slices[1], slices[2]]
        cropped_S = S[slices[0], slices[1], slices[2]]
    else:
        raise ValueError("Image dimensions not supported")

    return cropped_I, cropped_S, centroid


def adapt(x, target, crop = True, expand_dims = True):
    x = nb.load(x).get_data()
    # clip the very high value
    if crop == True:
        x = crop_or_pad(x, target)
    if expand_dims == True:
        x = np.expand_dims(x, axis = -1)
    #   print('after adapt, shape of x is: ', x.shape)
    return x


def normalize_image(x):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    
    return x.astype(np.float32) / 1000

def cutoff_intensity(x,cutoff):
 
    x[x<cutoff] = cutoff
    return x

def relabel(x,original_label,new_label):
    x[x==original_label] = new_label
    return x

def one_hot(image, num_classes):
    # Reshape the image to a 2D array
    image_2d = image.reshape(-1)

    # Perform one-hot encoding using NumPy's eye function
    encoded_image = np.eye(num_classes, dtype=np.uint8)[image_2d]

    # Reshape the encoded image back to the original shape
    encoded_image = encoded_image.reshape(image.shape + (num_classes,))

    return encoded_image


# function: translate image
def translate_image(image, shift):
    assert len(shift) in [2, 3], "Shift must be a list of 2 elements for 2D or 3 elements for 3D"
    assert len(image.shape) in [2, 3], "Image must be either 2D or 3D"
    assert len(image.shape) == len(shift), "Shift dimensions must match image dimensions"

    fill_val = np.min(image)  # Fill value is the minimum value in the image
    translated_image = np.full_like(image, fill_val)  # Create an image filled with fill_val

    if image.ndim == 2:  # 2D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_i = i - shift[0]
                new_j = j - shift[1]
                if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1]:
                    translated_image[new_i, new_j] = image[i, j]
    elif image.ndim == 3:  # 3D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    new_i = i - shift[0]
                    new_j = j - shift[1]
                    new_k = k - shift[2]
                    if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1] and 0 <= new_k < image.shape[2]:
                        translated_image[new_i, new_j, new_k] = image[i, j, k]
    else:
        raise ValueError("Image dimensions not supported")

    return translated_image


# function: rotate image
def rotate_image(image, degrees, order, fill_val = None):

    if fill_val is None:
        fill_val = np.min(image)
        
    if image.ndim == 2:  # 2D image
        assert isinstance(degrees, (int, float)), "Degrees should be a single number for 2D rotation"
        rotated_img = ndimage.rotate(image, degrees, reshape=False, mode='constant', cval=fill_val, order = order)

    elif image.ndim == 3:  # 3D image
        assert len(degrees) == 3 and all(isinstance(deg, (int, float)) for deg in degrees), "Degrees should be a list of three numbers for 3D rotation"
        # Rotate around x-axis
        rotated_img = ndimage.rotate(image, degrees[0], axes=(1, 2), reshape=False, mode='constant', cval=fill_val, order  = order)
        # Rotate around y-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[1], axes=(0, 2), reshape=False, mode='constant', cval=fill_val, order = order)
        # Rotate around z-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[2], axes=(0, 1), reshape=False, mode='constant', cval=fill_val, order = order)
    else:
        raise ValueError("Image must be either 2D or 3D")

    return rotated_img

def random_rotate(i, z_rotate_degree = None, z_rotate_range = [-10,10], fill_val = None, order = 0):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        return rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))

    return translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate
