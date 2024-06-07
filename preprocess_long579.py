import os
import tempfile
import time
import sys

# from HD_BET.run import run_hd_bet

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from warnings import warn
import itk
import subprocess
from scipy.signal import medfilt
import skimage


golden_file_path7 = "data/golden_image/mni_templates/nihpd_asym_04.5-08.5_t1w.nii"
golden_file_path9 = "data/golden_image/mni_templates/nihpd_asym_07.5-13.5_t1w.nii"

#register to a template        
def register_to_template(input_image_path, output_path, fixed_image_path, create_subfolder=True):
    fixed_image = itk.imread(fixed_image_path, itk.F)

    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile('data/golden_image/mni_templates/Parameters_Rigid.txt')

    if "nii" in input_image_path and "._" not in input_image_path:
        print(input_image_path)

        # Call registration function
        try:        
            moving_image = itk.imread(input_image_path, itk.F)
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            
            if create_subfolder:
                new_dir = output_path+image_id.split(".")[0] 
                if not os.path.exists(new_dir):
                    os.mkdir(new_dir)
                itk.imwrite(result_image, new_dir+"/"+image_id)
            else:
                itk.imwrite(result_image, output_path+"/"+image_id)
                
            print("Registered ", image_id)
        except:
            print("Cannot transform", input_image_path.split("/")[-1])
  
def register_to_template_cmd(input_image_path, output_path, fixed_image_path,rename_id,create_subfolder=True):
    return_code = 1
    if "nii" in input_image_path and "._" not in input_image_path:
        try:  
            
            return_code = subprocess.call("elastix -f "+fixed_image_path+" -m "+input_image_path+" -out "+\
            output_path + " -p data/golden_image/mni_templates/Parameters_Rigid.txt", shell=True,\
            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            if return_code == 0:
                print("Registered ", rename_id)
                result_image = itk.imread(output_path+'/result.0.mhd',itk.F)
                itk.imwrite(result_image, output_path+"/"+rename_id+".nii.gz")
            else:
                print("Error registering ", rename_id)
                return_code = 1
        except:
            print("is elastix installed?")
            return_code = 1
    return return_code
          
def resample_image(image, new_spacing, interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate the new size based on the new spacing
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    # Create a resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)

    # Perform the resampling
    resampled_image = resampler.Execute(image)

    return resampled_image

def coreg(sitk_resampled, mask):
    fixed = sitk_resampled[0]
    moving = sitk_resampled[1]
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity3DTransform()
    )
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(fixed, moving)

    #print("-------")
    #print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    #print(f" Iteration: {R.GetOptimizerIteration()}")
    #print(f" Metric value: {R.GetMetricValue()}")

    #sitk.WriteTransform(outTx, args[3])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    
    # Use the // floor division operator so that the pixel type also becomes UInt8, and not Double
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0) 

    #compute the delta between the two images
    sitk_delta = sitk.RescaleIntensity(sitk.Abs(sitk.Subtract(simg1, simg2)))
    return [simg1, simg2]

def rescale_intensity_with_mask(img, mask, lower, upper, b_min=0.0, b_max=1.0, background_value=0, clip=False, relative=False):
    # Convert SimpleITK images to NumPy arrays
    img_np = sitk.GetArrayFromImage(img)
    mask_np = sitk.GetArrayFromImage(mask)
    
    a_min = np.percentile(img_np[img_np != background_value], lower)
    a_max = np.percentile(img_np[img_np != background_value], upper)
    b_min = b_min
    b_max = b_max

    if relative:
        b_min = ((b_max - b_min) * (lower / 100.0)) + b_min
        b_max = ((b_max - b_min) * (upper / 100.0)) + b_min

    if a_max - a_min == 0.0:
        warn("Divide by zero (a_min == a_max)", Warning)
        return sitk.GetImageFromArray(img_np - a_min + b_min)

    img_np = (img_np - a_min) / (a_max - a_min)
    img_np = img_np * (b_max - b_min) + b_min

    if clip:
        img_np = np.asarray(np.clip(img_np, b_min, b_max))
    img_np = np.where(mask_np == 1, img_np, background_value)

    # Convert NumPy array back to SimpleITK image
    rescaled_image = sitk.GetImageFromArray(img_np)

    return rescaled_image

def bias_field_correction(img):
    image = sitk.GetImageFromArray(img)
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    numberFittingLevels = 4

    corrector.SetMaximumNumberOfIterations([100] * numberFittingLevels)
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image_full_resolution = image / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected_image_full_resolution)

def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)

# rescale the intensity of the image and binning
def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    #remove background pixels by the otsu filtering
    t = skimage.filters.threshold_otsu(volume,nbins=6)
    volume[volume < t] = 0
    
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume

# equalize the histogram of the image
def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def preprocess(directory, output_path,path_to_hdbet,cuda_device):
    
    sys.path.insert(1, path_to_hdbet)
    try:
        from HD_BET.run import run_hd_bet
        print("Module imported successfully.")
    except ImportError as e:
        print(f"Error importing module: {e}")
        
    reg_output_path = os.path.join(output_path, 'registered')
    if not os.path.exists(reg_output_path):
        os.mkdir(reg_output_path)
        
    skull_stipped_output_path = os.path.join(output_path, 'skull_stripped')
    if not os.path.exists(skull_stipped_output_path):
        os.mkdir(skull_stipped_output_path)
        
    output_path = os.path.join(output_path, 'preprocessed')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    df = pd.read_csv(directory+'/participants.tsv', header=0, sep='\t')
    print(df)
    resampled_list =[]
    for i in range(len(df)):
        row = df.iloc[i]
        participant_id = str(row['participant_id'])
        if "sub" not in participant_id:
            participant_id= "sub-"+participant_id
        print(participant_id)
        # check if the 7-9 images are present
        ses_7_path = os.path.join(directory,participant_id,"ses-7","anat")
        ses_9_path = os.path.join(directory,participant_id,"ses-9","anat")
    
        if os.path.exists(ses_7_path) and os.path.exists(ses_9_path):
            files_in_ses_7 = os.listdir(ses_7_path)
            files_in_ses_9 = os.listdir(ses_9_path)
            ses_7_file = [f for f in files_in_ses_7 if "T1w.nii.gz" in f][0]
            ses_9_file = [f for f in files_in_ses_9 if "T1w.nii.gz" in f][0]
            
            if ses_7_file and ses_9_file:    
                return_code7=register_to_template_cmd(os.path.join(ses_7_path,ses_7_file), reg_output_path, golden_file_path7,participant_id+"_7",create_subfolder=False)
                return_code9=register_to_template_cmd(os.path.join(ses_9_path,ses_9_file), reg_output_path, golden_file_path9,participant_id+"_9",create_subfolder=False)
                #
                reg_7_path = os.path.join(reg_output_path,participant_id+"_7.nii.gz")
                reg_9_path = os.path.join(reg_output_path,participant_id+"_9.nii.gz")
                print("Registered")
                if return_code7==1 or return_code9==1:
                    print("Error registering")
                    continue
                
                #skull strip - run hdbet to calculate the mask - only on one image
                print("Running hdbet")
                
                run_hd_bet(reg_7_path,os.path.join(skull_stipped_output_path,participant_id+'_7.nii.gz'),
                            mode="accurate", 
                            config_file=os.path.join(path_to_hdbet,'HD_BET','config.py'),
                            device=int(cuda_device),
                            postprocess=False,
                            do_tta=True,
                            keep_mask=True, 
                            overwrite=True)
                        
                #coregister two
                print("Coregistering and denoising")
                sitk_image1 = sitk.ReadImage(reg_7_path)
                sitk_image2 = sitk.ReadImage(reg_9_path)
                
                image_array1  = sitk.GetArrayFromImage(sitk_image1)
                image_array2  = sitk.GetArrayFromImage(sitk_image2)
                kernel_size=3
                percentils=[0.5, 99.5]
                bins_num=256
                volume1 = bias_field_correction(image_array1)
                volume1 = denoise(volume1, kernel_size)
                volume1 = rescale_intensity(volume1, percentils, bins_num)
                volume1 = equalize_hist(volume1, bins_num)
                sitk_image1 = sitk.GetImageFromArray(volume1)
                
                volume2 = bias_field_correction(image_array2)
                volume2 = denoise(volume2, kernel_size)
                volume2 = rescale_intensity(volume2, percentils, bins_num)
                volume2 = equalize_hist(volume2, bins_num)
                sitk_image2 = sitk.GetImageFromArray(volume2)
                
                sitk_image_mask = sitk.ReadImage(os.path.join(skull_stipped_output_path,participant_id+'_7_mask.nii.gz'))
                sitk_resampled = coreg([sitk_image1,sitk_image2], sitk_image_mask) 
                sitk_resampled_fins = []
                flag=0
                for sitk_image in sitk_resampled:
                    # set background to zero by mask
                    try:
                        sitk_image = rescale_intensity_with_mask(sitk_image, sitk_image_mask, lower=0.5, upper=99.5, b_min=0, b_max=1, background_value=0,clip=True, relative=True)
                        #resample to (3., 3., 2.0) mm using bilinear interpolation with sitk
                        sitk_image = resample_image(sitk_image, (3., 3., 2.5))
                        #crop center to 64x64x64
                        # get image size
                        size = sitk_image.GetSize()
                        # get the center voxel index
                        center_index = (np.array(size) / 2).astype(int)
                        # get the start and end index for cropping
                        start_index = center_index - 32
                        end_index = center_index + 32
                        # crop the image
                        print(start_index, end_index)
                        sitk_image = sitk_image[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
                        # save the image to output_path
                        sitk_resampled_fins.append(sitk_image)
                    except:
                        print('Error')
                        flag = 1
                        continue
                    # coregister the images in the sitk_resampled list
                
                #save the images to output_path
                if flag ==0:
                    sitk.WriteImage(sitk_resampled_fins[0], os.path.join(output_path, participant_id+"_7.nii.gz"))
                    sitk.WriteImage(sitk_resampled_fins[1], os.path.join(output_path, participant_id+"_9.nii.gz"))
                    resampled_list.append([os.path.join(output_path, participant_id+"_7.nii.gz"), os.path.join(output_path, participant_id+"_9.nii.gz")])
                    #break

    output_df = pd.DataFrame(resampled_list)
    return output_df