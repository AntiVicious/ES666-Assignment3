from src.Maharshi.func_folder.initializer import initializer
from src.Maharshi.func_folder import funcs

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        
        pimages , centerIdx = initializer(path)
        self.say_hi()
        
        # Return Final panaroma
        stitched_image, H_matrices = funcs.pan_creator(pimages,centerIdx)
        #####
        homography_matrix_list =H_matrices
        
        return stitched_image, homography_matrix_list

    def say_hi(self):
        print('Hi From satvik')
    
    