import torchvision
from torch.utils.data import random_split


class Minet(object):

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)), # (H,W)
        torchvision.transforms.ToTensor(),
    ])
    
    def __init__(self, args, ratio=0.8):
        self.image_root = args.dataset_dir
        self.ration = ratio

    def get_data(self):
        dataset = torchvision.datasets.ImageFolder(self.image_root, transform=self.transformation)

        train_set_size = int(len(dataset) * 0.8)
        test_set_size = len(dataset) - train_set_size
        train_set, valid_set = random_split(dataset, [train_set_size, test_set_size])
        return train_set, valid_set
        # all_data = []
        # for img in self.all_image:
        #     label = self.deal_label(img)
        #     all_data.append([os.path.join(self.image_root, img), label])
        # train, val = train_test_split(all_data, random_state=1, train_size=self.ration)
        # return train, val