# 50.039 Theory and Practice of Deep Learning | Coding Homework 4 - Data Loading and Augmentation with CNNs
## Joel Huang, 1002530

1. Run `ls imagespart | cat > paths.txt` to produce a file with all the image paths.
2. `preprocess_csv.py` is used to retrieve the pairs of image names and respective labels.
3. In `dataloader.py`, a custom `Dataset` class is written to load an image as a `torch.Tensor` and the corresponding label.
4. A full demonstration, with resulting accuracies, can be found in `data_augmentation.ipynb`.