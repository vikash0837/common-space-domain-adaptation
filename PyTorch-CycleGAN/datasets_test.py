import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
        print("len of files_a  =:",len(self.files_A))
        #print("files_b path=:",self.files_B)

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        filename = self.files_A[index]
        #print("1st file name",filename)
        filename_A = filename.split('/')[-1]
        #print("2nd filename=:",filename_A)

        # if self.unaligned:
        #     item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        # else:
        #     item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'filename':filename_A}#, 'B': item_B}

    def __len__(self):
        return len(self.files_A)
        #return max(len(self.files_A), len(self.files_B))