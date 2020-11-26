import os
import PIL.Image
from torchvision.transforms import transforms
import mnist

dataset = mnist.MNIST('../data', full=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.1307,), (0.3081,))
                               ]))

for i in range(len(dataset.train_data)):
    image_array, image_label = dataset.train_data[i], dataset.train_labels[i]  # 打印第i个
    image_array = PIL.Image.fromarray(image_array)
    filename = 'mnist_train_%d.jpg' % i
    filedir = os.path.join('./data', str(image_label))  # 保存文件的格式
    if not os.path.exists(filedir):
        os.mkdir(filedir)
    filepath = os.path.join(filedir, filename)
    print(filepath)
    image_array.save(filepath)  # 保存图像