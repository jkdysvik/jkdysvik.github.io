# Classifying Images Using FastAI and Deep Learning 🧠

In this post I will present how I used the Fastai library to create a simple program to determine whether an image has a dog, airplane, cat or car in it.

## Step 1: Getting the Data 📁
I started by gathering data to train the deep learning model. Using the `duckduckgo_search` library I searched for images from the five categories, including queries such as "sun photo" and "shade photo" to make sure that i get a varied collection of data. To get a good model you need high quality images, i also used `verify_images()` to detect and delete corrupted files.

## Step 2: Training a neural network ♟️
After collecting the images for my model, I used fastai's APIs to prepare the data for training. More specifically i used a `DataBlock` to load images, label them and split them into training and validation sets. After that I set up a ResNet-18 model which is pre-trained on millions of images, and applied it to my dataset.

`learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)`

## Step 3: Making Predicitions 🤖
After training the model, i tested it by giving it a new image. 
`is_it,_,probs = learn.predict(PILImage.create('dog1.JPG'))`
Dog1.jpg in this example was actually created by dall-e to make sure that the image wasnt already in the original database of dogs, the model said it was a dog with a 100% certainty.

## Step 4: Evaluating the Model 📊
To explore how well the model was working I used two main tools.

* Confusion Matrix: To see where the model was getting confused
* Top Losses: To examine the worst mistakes, often images that were truly amgiuous or poorly labeled

## Step 5: Visualizing the Models Understanding 👓
Lastly i i generated a t-SNE plot of the feature vectors from the second to last layer of the network. This allowed me to explore how the model grouped different images into classes internally.

