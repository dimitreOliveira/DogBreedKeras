![](https://raw.githubusercontent.com/dimitreOliveira/DogBreedKeras/master/dog_breed_identification_header.png)

# Deep Learning image classification with Keras

## About the repository
The goal here is to practice with deep learning and images, in this case using CNNs and image transfer, the challenge here is the small dataset, the high number of classes and unbalanced data.

### What you will find
* Load and process image data to save as numpy matrices. [[link]](https://github.com/dimitreOliveira/DogBreedKeras/blob/master/dataset.py)
* Model architecture. [[link]](https://github.com/dimitreOliveira/DogBreedKeras/blob/master/model.py)
* Model training and prediction. [[link]](https://github.com/dimitreOliveira/DogBreedKeras/blob/master/main.py)

### Dog Breed Identification: Determine the breed of a dog in an image

link for the kaggle competition: https://www.kaggle.com/c/dog-breed-identification

datasets: https://www.kaggle.com/c/dog-breed-identification/data

### Overview
Who's a good dog? Who likes ear scratches? Well, it seems those fancy deep neural networks don't have all the answers. However, maybe they can answer that ubiquitous question we all ask when meeting a four-legged stranger: what kind of good pup is that?

In this playground competition, you are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more, err, ruff than you anticipated.

### Acknowledgments
We extend our gratitude to the creators of the Stanford Dogs Dataset for making this competition possible: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li.

### Dependencies:
* [Cv2](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
* [H5Py](https://www.h5py.org/)
* [Tqdm](https://tqdm.github.io/)
* [Keras](https://keras.io/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [Matplotlib](http://matplotlib.org/)

### To-Do:
* Efficiently use transfer learning to improve model predictions.
