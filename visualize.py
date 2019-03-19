
import os
from dipy.io.image import load_nifti,save_nifti
from glob import glob
import matplotlib.pyplot as plt

file_path='latent/mean_features'


for i in range(32):
    filename_m=os.path.join(file_path,'m_feature_{}.nii'.format(i))
    filename_f=os.path.join(file_path,'f_feature_{}.nii'.format(i))
    data_m,affine_m=load_nifti(filename_m)
    data_f,affine_f=load_nifti(filename_f)
    
    axial_middle = data_m.shape[2] // 2
    
    main_m=data_m[:,:,axial_middle].T
    
    fig = plt.figure()
    #imshow(main_m)
#    subplot(321)
#    imshow(img1)
    #title('feature_{i}')
    #axis('off')
    
    pic_name='./latent/mean_features/feature_{}.png'.format(i)
    fig.tight_layout(pad=0)
    plt.savefig(pic_name, bbox_inches='tight')
    plt.close(fig)
