from torchvision import transforms, datasets

DATA_FOLDER = './torch_data/DCGAN/MNIST'

def mnist_data():
    compose = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    out_dir = "{}/dataset/".format(DATA_FOLDER)
    return datasets.MNIST(root = out_dir, train=True, transform=compose, download=True)