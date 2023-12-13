import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.module_photo2pixel import Photo2PixelModel
from utils import img_common_util


def convert():
    input = "C:/Users/fgtyf/PycharmProjects/pythonProject3/images/스크린샷 2023-04-10 215007.png"
    output = "./result.png"
    kernel_size = 50
    pixel_size = 50
    edge_thresh = 100

    img_input = Image.open(input)
    img_pt_input = img_common_util.convert_image_to_tensor(img_input)

    model = Photo2PixelModel()
    model.eval()
    with torch.no_grad():
        img_pt_output = model(
            img_pt_input,
            param_kernel_size=kernel_size,
            param_pixel_size=pixel_size,
            param_edge_thresh=edge_thresh
        )
    img_output = img_common_util.convert_tensor_to_image(img_pt_output)
    img_output.save(output)
    print(f"output is saved at {output}")

    plt.figure(figsize=(20, 20))
    plt.imshow(img_output)
    plt.show()

if __name__ == '__main__':
    convert()
