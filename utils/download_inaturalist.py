from torchvision.datasets import INaturalist

train_dataset = INaturalist(
    root='/home/jroberto/INF492-TF/dataset/',
    version='2021_train_mini',
    target_type='kingdom',
    download=True
)

test_dataset = INaturalist(
    root='/home/jroberto/INF492-TF/dataset/',
    version='2021_valid',
    target_type='kingdom',
    download=True
)