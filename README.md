# **BeMyEye**
BeMyEye is an app to help visually impaired people to do their everyday task independently.

## The Problem
It is estimated that 285 million people on this planet are visually impaired, out of which 39 million are blind. It is very difficult for them to do daily activities without the help of others. Some of the day to day problems they face are not being able to find a particular object they want, not being able to read any document or anything on any products, and not being able to know how far the objects around are.

## The Solution
So, to overcome the above stated problem, we have come up with a solution (a mobile app) to help visually impaired people do their day to day tasks independently. Our mobile app is known as BeMyEye.

### What it does?
It is a mobile app that:
* Detects all the object in the surrounding with their distance.
![objects](https://user-images.githubusercontent.com/46972935/100117406-c4692b00-2e9a-11eb-8721-6d88bd199611.png)
* Locates a particular object as stated by the user along with its distance.
![object with distance](https://user-images.githubusercontent.com/46972935/100106091-1fe0ec00-2e8e-11eb-9702-6980748c8362.jpg)
* Describes the surrounding to the user.
![image captioning](https://user-images.githubusercontent.com/46972935/100105888-ddb7aa80-2e8d-11eb-9c7b-6f86621a74a0.jpg)
* Detects and reads text from any image given by the user.

![text-recognition](https://user-images.githubusercontent.com/46972935/100105532-7b5eaa00-2e8d-11eb-8d35-9d43b2c39dc8.jpg)

### How to use?
![app](https://user-images.githubusercontent.com/46972935/100108239-9383f880-2e90-11eb-976d-3165df0517c6.jpg)

**Select any option from navigation in the app**

## The Implementation
Implementing it requires the following:
* **Object Detection:** Object Detection basically tells the user what objects are around in the visual. For this, we need to locate and classify objects in an image. There are many Object Detection models in Deep Learning that does the job for us like RCNN, faster RCNN, YOLO, etc. Here we used faster RCNN in our application.

* **Image Captioning:** Image Captioning is used to describe the view infront of the user. It generates a textual description of an image. This deep learning model includes the combination of both computer vision and natural language processing. We have used a model whose architecture includes connection between CNN and LSTM, and it is trained on flicker-8k dataset.

* **Text Detection and OCR:** Text Detection detects and recognizes text from anywhere in the image. For this we have used Tesseract OCR(Optical Character Recognition) Engine to localise and detect text from the images. It computes a bounding box on every region of text and recognize it.

* **Distance Measurement:** It is used to measure the probable distance of the object from the user, just with the help of an image. Here we have used a Geometrical Similarity method to measure the probable distance with just one single camera. This method is not very accurate, but accurate methods like finding distance with multiple cameras or depth sensing cameras can be taken.

* **Text to Speech and Speech to Text:** This feature is used for communication with visually impaired user. Here we have used Google's APIs for the text-to-speech and speech-to-text application.

## Future Scopes
It can be modified to include the following features: 
* Here we have the models ready which can be further deployed into an app in future or wearable device like glasses can be made instead of a Mobile App.
* Facial Recognition feature can also be added so that the user can recognise their dear ones.
* Depth sensing cameras or more cameras can measure the distance of objects more accurately.
* Maps can be added to help the user to navigate to different places.
