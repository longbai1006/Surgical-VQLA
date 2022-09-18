import os
import sys
from visual_bert.modeling_frcnn import GeneralizedRCNN
from visual_bert.utils import Config
import h5py
import torch
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch import nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(device)
        
    def forward(self, images_mine):
        output_dict = self.frcnn(
            images_mine,
            torch.tensor([[800, 1206]]),
            scales_yx=torch.tensor([[1.2800, 1.0614]]),
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        outputs = output_dict.get("roi_features")
        return outputs

# input data and IO folder location
filenames = []
seq = ['1','2','3','4','5','6','7','9','10','11','12','14','15','16']
folder_head = '/home/ren2/data3/Long/Surgical_VQLA/dataset/EndoVis-18-VQA/'
folder_tail = '/*.png'

# seq = ['1']
# folder_head = '/home/ren2/data3/Long/Surgical_VQLA/dataset/EndoVis-17-VQA/'
# folder_tail = '/*.jpg'

for curr_seq in seq: 
    filenames = filenames + glob(folder_head + 'seq_' + str(curr_seq) + '/left_frames' + folder_tail)
    # filenames = filenames + glob(folder_head + 'left_frames' + folder_tail)

new_filenames = []
for filename in filenames:
    frame_num = int(filename.split('/')[-1].split('.')[0].strip('frame'))
    if frame_num % 1 == 0: new_filenames.append(filename)
    # new_filenames.append(filename)

# declare fearure extraction model
feature_network = FeatureExtractor()

# Set data parallel based on GPU
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    device_ids = np.arange(num_gpu).tolist()
    feature_network = nn.DataParallel(feature_network, device_ids=device_ids)

# Use Cuda
feature_network = feature_network.cuda()
feature_network.eval()

for img_loc in tqdm(new_filenames):
    
    # get visual features
    img = Image.open(img_loc)
    raw_sizes = torch.tensor(np.array(img).shape[:2])
    sizes = torch.tensor([800, 1206])
    scales_yx = torch.true_divide(raw_sizes, sizes)
    pil_image = img.resize((1206, 800), Image.BILINEAR)
    normalizer = lambda x: (x - [102.9801, 115.9465, 122.7717]) / [1.0, 1.0, 1.0] 
    images_mine = torch.tensor(normalizer(np.array(pil_image))).double().permute(2, 0,1)[None]
    images_mine = images_mine.float()
    with torch.no_grad():
        visual_features = feature_network(images_mine)
        visual_features = visual_features.squeeze(0)
        visual_features = visual_features.data.cpu().numpy()

    # save extracted features
    img_loc = img_loc.split('/')
    print(img_loc)
    save_dir = '/' + os.path.join(img_loc[0],img_loc[1],img_loc[2],img_loc[3],img_loc[4],img_loc[5],img_loc[6],img_loc[7],'vqla/img_features','frcnn')
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    
    # save to file
    hdf5_file = h5py.File(os.path.join(save_dir, '{}.hdf5'.format(img_loc[-1].split('.')[0])),'w')
    print(os.path.join(save_dir, '{}.hdf5'.format(img_loc[-1].split('.')[0])))
    hdf5_file.create_dataset('visual_features', data=visual_features)
    hdf5_file.close()
    print('save_dir: ', save_dir, ' | visual_features: ', visual_features.shape)