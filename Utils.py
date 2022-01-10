# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 20:02:44 2022

@author: Lawrence
"""

import numpy as np
import pandas as pd
import PIL.Image as Image
from Globals import *

def rle_decode(rle_string,Image_size)->np.ndarray:
    """
    

    Parameters
    ----------
    rle_string : The Run Length encoding string
    Image_size : the image size in the form (w,d)

    Returns
    -------
    np.array binary mask

    """
    def convert_pos(pos_1d,Image_size):
        """
        A helper function to a 1d index position into a 
        2d position
        """
        return [pos_1d%Image_size[0],pos_1d//Image_size[1]]
    
    start_positions=[int(val) for val in rle_string.split()[::2]]
    run_lengths=[int(val) for val in rle_string.split()[1::2]]
    
    mask=np.zeros(Image_size)
    for i, start_position in enumerate(start_positions):
        for j in range(run_lengths[i]+1):
            
            x,y=convert_pos(start_position+j, Image_size)
            mask[x,y]=1
    return mask

    


if __name__=="__main__":
    #testing
    import matplotlib.pyplot as plt
    Data_df=pd.read_csv(f"{Data_Dir}/train_ship_segmentations_v2.csv")
    i=3
    
    test_Str=Data_df.iloc[i,1]
    test_img=np.array(
        Image.open(f"{Data_Dir}/train_v2/{Data_df.iloc[i,0]}"))
    
    
    print(test_img.shape)
    mask=rle_decode(test_Str,test_img.shape[:-1])
    
    plt.imshow(test_img)
    plt.imshow(mask,alpha=0.5)
    plt.show()

