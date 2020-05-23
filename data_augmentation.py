'''
resulting  in two  correlated view of the same example 
denoted  x~_i and x~_j, which we consider as a positive pair.

we sequentially apply three simple augmentations:
   RANDOM CROPPING followed by 
   resize back to the original size (224 * 224) 
   RANDOM COLOR DISTORTIONS
   RANDOM GAUSSIAN BLUR 

'''

from  torchvision import transforms 
def get_color_distortion(s=1.0):
    #s is the  strength of color distortion
    color_jitter =  transforms.ColorJitter(0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = trnasforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort

