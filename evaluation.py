# evaluation of the results

import numpy as np

from skimage.metrics import structural_similarity as ssim_fn

def nsme_fn(img1,img2):
    score = np.mean(np.sqrt(np.abs(img1 - img2)**2))
    return score

def error_compute(img1,img2):
    '''compute the nrmse, ssim'''
    # print(img1.shape,img2.shape)
    data_range = max(np.max(img1),np.max(img2)) - min(np.min(img1),np.min(img2))
    ssim_score = ssim_fn(img1,img2,data_range=data_range)
    nmse_score = nsme_fn(img1,img2)
    print(f'ssim: {ssim_score} , ')
    return nmse_score,ssim_score

if __name__=="__main__":
    # Knee results with pretrained model
    print('[knee pretrained]')
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_knee2_pretrained_acc=4/real/label/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_label)

    img_recon_4 = np.load(dir+'results_knee2_pretrained_acc=4/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_4)
    
    img_recon_8 = np.load(dir+'results_knee2_pretrained_acc=8/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_8)
    
    img_recon_16 = np.load(dir+'results_knee2_pretrained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_16)

    img_recon_32 = np.load(dir+'results_knee2_pretrained_acc=32/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_32)



    # Prostate results with pretrained model
    print('[prostate pretrained]')
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_prostate1_pretrained_acc=4/real/label/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_label)

    img_recon_4 = np.load(dir+'results_prostate1_pretrained_acc=4/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_4)
    
    img_recon_8 = np.load(dir+'results_prostate1_pretrained_acc=8/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_8)
    
    img_recon_16 = np.load(dir+'results_prostate1_pretrained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_16)

    img_recon_32 = np.load(dir+'results_prostate1_pretrained_acc=32/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_32)



    # Prostate results with trained model
    print('[prostate trained]')
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_prostate1_trained_acc=4/real/label/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_label)

    img_recon_4 = np.load(dir+'results_prostate1_trained_acc=4/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_4)
    
    img_recon_8 = np.load(dir+'results_prostate1_trained_acc=8/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_8)
    
    img_recon_16 = np.load(dir+'results_prostate1_trained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_16)

    img_recon_32 = np.load(dir+'results_prostate1_trained_acc=32/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_32)


    # Prostate results with trained model (different sample)
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_prostate2_trained_acc=4/real/label/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_label)

    img_recon_4 = np.load(dir+'results_prostate2_trained_acc=4/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_4)
    
    img_recon_8 = np.load(dir+'results_prostate2_trained_acc=8/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_8)
    
    img_recon_16 = np.load(dir+'results_prostate2_trained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_16)

    img_recon_32 = np.load(dir+'results_prostate2_trained_acc=32/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon_32)



    # 3D knee recon with pretrained model
    print('[3D knee recon]')
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_knee3d_pretrained_acc=16/real/label_3d.npy')
    
    img_recon3d = np.load(dir+'results_knee3d_pretrained_acc=16/real/recon_3d.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon3d)

    img_slice_label = img_label[10]
    img_slice_from_3d = img_recon3d[10]
    img_recon_slice = np.load(dir+'results_knee2_pretrained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_slice_from_3d,img_recon_slice)
    nmse_val,ssim_val = error_compute(img_slice_label,img_recon_slice)


    # 3D prostate recon with trained model
    print('[3d prostate recon]')
    dir = '/scratch/Projects/eecs598project_prostate_generative/'
    img_label = np.load(dir+'results_prostate3d_trained_acc=16/real/label_3d.npy')
    
    img_recon3d = np.load(dir+'results_prostate3d_trained_acc=16/real/recon_3d.npy')
    nmse_val,ssim_val = error_compute(img_label,img_recon3d)

    img_slice_label = img_label[15]
    img_slice_from_3d = img_recon3d[15]
    img_recon_slice = np.load(dir+'results_prostate2_trained_acc=16/real/recon/001.npy')
    nmse_val,ssim_val = error_compute(img_slice_from_3d,img_recon_slice)
    nmse_val,ssim_val = error_compute(img_slice_label,img_recon_slice)
