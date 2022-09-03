import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # input data는 [batch size, 100, 1, 1]의 형태로 주어야합니다.
        return self.main(input)


class Discriminator(nn.Module):
    # 모델의 코드는 여기서 작성해주세요

    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, input):
        return output


if __name__ == "__main__":
    # 학습코드는 모두 여기서 작성해주세요

    data_path = 'training_data/'

    dataset = datasets.ImageFolder(root=data_path,
                                   transform=transforms.ToTensor()
                                   )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # FID score 측정에 사용할 fake 이미지를 생성하는 코드 입니다.
    # generator의 학습을 완료한 뒤 마지막에 실행하여 fake 이미지를 저장하시기 바랍니다.
    test_noise = torch.randn(3000, 100, 1, 1, device=device)
    with torch.no_grad():
        test_fake = generator(test_noise).detach().cpu()

        for index, img in enumerate(test_fake):
            fake = np.transpose(img.detach().cpu().numpy(), [1, 2, 0])
            fake = (fake * 127.5 + 127.5).astype(np.uint8)
            im = Image.fromarray(fake)
            im.save("./fake_img/fake_sample{}.jpeg".format(index))
