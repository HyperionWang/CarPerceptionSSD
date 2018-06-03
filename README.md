# MobileNet Encoder-Decoder Semantic Segmentation for Lyft Perception Challenge

[//]: # (Image References)
[image1]: ./Docs/00080.png
[image2]: ./Docs/00353.png
[image3]: ./Docs/00486.png
[image4]: ./Docs/DepthWiseConv.png
[image5]: ./Docs/MobileNetArch_Resize.png
[image6]: ./Docs/GT_car_img.png
[image7]: ./Docs/GT_road_img.png
[image8]: ./Docs/GT_bg_img.png
[image9]: ./Docs/ScoreCalculation.png
[image10]: ./Docs/TensorBoard.png
[image11]: ./Docs/00415.png

### Overview

The goal of this project is to address pixel-wise identification of car and road/lane objects in camera images or video captured from Carla simulation environment. In this project, the model training is done on the efficient neural network called MobileNets using depth-wise separable convolutions combined in a encoder-decoder structure for the Semantic Segmentation Detection (SSD).

With light-weighted network architecture, the model could be trained using single GTX1080Ti GPU, 15k images captured from Carla, and 20 epoches in 10 - 20 hours. After the training, the model is able to produce good results: 97% IOU and 0.9179 FScore with high frame rate of 15 - 16 FPS using Lyft Challenge platform.

 ![TestingResult][image1]

### Modified MobileNet Architecture
The backbone of encoder for SSD model uses [MobileNet](https://arxiv.org/abs/1704.04861) architecture with several modifications:

1. The depthwise separable convolution is kept but the 1-, 2-, 4- dilated convolutions are added. In this way, the context module aggregate multi-scale contextual information. The detailed info could be found in [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122).

 ![TestingResult][image4]

 [Depthwise Separable Convolution](https://arxiv.org/abs/1704.04861)

2. The layers with depth of 1024 in the original MobileNet are removed. The output of the encoder is in depth of 512, it is to increase the speed of the model.
3. The encoder is added following the MobileNet Encoder for the pixel-wise SSD. Also, the s16, s8, s4, and s2 layers are bypassed connected to the decoder.

 ![ModifiedMobileNet][image5]


### Data Collection and Preprocessing Data for Training

The stree camera data could be generated using Carla Simulation Tools with python script. More than 15k images with ground truth labels generated using provided Carla simulation tools.

The following code is the setting in python code to generate the camera images and ground truth for SSD.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('CameraRGB')
                # Set image resolution in pixels.
                camera0.set_image_size(800, 600)
                # Set its position relative to the car in meters.
                camera0.set_position(1.30, 0, 1.30)
                settings.add_sensor(camera0)

                camera1 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
                camera1.set_image_size(800, 600)
                camera1.set_position(1.30, 0, 1.30)
                settings.add_sensor(camera1)

To preprocess data, only car's label and road/lane label are used for SSD training and final testing results. Therefore, the Carla generate ground truth labels need to be processed. Also, the car itself's hood is also in the image, in order to avoid the mis-training to include that in the SSD, the car labels of own car's hood need to be removed, by using the following code:

    def preprocess_labels(label_image):
        road_id = 7
        lane_id = 6
        car_id = 10
        # Create a new single channel label image to modify
        labels_new = np.copy(label_image[:, :, 0])
        # Identify lane marking pixels (label is 6)
        lane_marking_pixels = (label_image[:, :, 0] == lane_id).nonzero()
        # Set lane marking pixels to road (label is 7)
        labels_new[lane_marking_pixels] = road_id

        # Identify all vehicle pixels
        vehicle_pixels = (label_image[:, :, 0] == car_id).nonzero()
        # Isolate vehicle pixels associated with the hood (y-position > 496)
        hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
        hood_pixels = (vehicle_pixels[0][hood_indices], \
                       vehicle_pixels[1][hood_indices])
        # Set hood pixel labels to 0
        labels_new[hood_pixels] = 0
        # Return the preprocessed label image
        return labels_new

The following are the ground truth images after the label processing.

Car Label

 ![GroundTruthLabel][image6]

Road Label

 ![GroundTruthLabel][image7]

Background Label

 ![GroundTruthLabel][image8]


Also, the augmentaions of flipping in horizontal and brightness changes are randomly added to the images for

### Model Training

The training data set is in size of 19k, and the training and validation sets are 80% and 20%, the batch size is 12 due to the limit of the GPU memory.

    N_train: 15302	N_val:3826	Train steps: 1275	Val_steps: 318	Batch size: 12

The cost function for the optimization is the combination of cross entropy and the modified IOU with different weight of car's F score and road's F score. The weight for car and road are 80% and 20%, it is because road's variation from image to image is small and F score is always much better than car's F score. By adding more weight on the car, the training would improve more on the car's SSD results.

The score is calculated based on the provided equations:

 ![ScoreCalculation][image9]

The following code is the calculation of the loss for optimization. Since the higher the score the better SSD, then, then use difference between the max F score and the calculated weighted F score for the minimization process.

    weights = [0.0, 0.8, 0.2]
    ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    loss = ce_loss + (F_max - (weights[1] * F_car + weights[2] * F_road))

Also, the training status is saved during the training for monitoring the loss improvement using the tensorboard.
The training paramteres are as following:

    learning_rate_val = 0.001
    epochs = 20
    decay = learning_rate_val / (2 * epochs)
    batch_size = 12

 ![TensorBoardMetric][image10]

After 20 epochs of learning with learning rate of 0.001, the fine tuning is followed, with 1/10th of learning rate, and the first 10 layers are locked. The performance of the model could be improved a little further, but not very much.


### Results
Here are a few predictions @~97% validation IoU. More testing results could be found in the runs folder. The SSD model is able to label most of the road pixels and car pixels accurately and the according to the grader program on the Challenge workspace, the FPS reaches to 16 FPS, which is fast enough for real-time SSD applications.

![alt text][image2]

![alt text][image3]

![alt text][image11]

    Your program has run, now scoring...

    Your program runs at 15.151 FPS

    Car F score: 0.848 | Car Precision: 0.803 | Car Recall: 0.860 | Road F score: 0.988 | Road Precision: 0.988 | Road Recall: 0.986| Averaged F score: 0.918

### Future Work

To improve the SSD performance better with trade-off on the FPS, several models could be implemented: including [MobileNetV2](https://arxiv.org/abs/1801.04381), [MobileNetV2 Keras](https://github.com/xiaochus/MobileNetV2), [deeplabV3+ with backbone on MobileNet V2 and Xception](https://github.com/tensorflow/models/tree/f798e4b5504b0b7ed08f7b7a03fc5a79f00b9f21/research/deeplab). I have tried to adapted those models into current training implementation, and I was able to train the model with very small batch size of 4 with current GTX1080Ti GPU. More simplifications on the models are needed for fast and high accuracy SSD.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [Keras](https://keras.io/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv.org/)


##### Dataset
To train the model, please save the data under the ./Train folder. ./Train/Test folder is for final test. ./Train/episode_* folders are for data captured from Carla program. Under each ./Train/episode_* folder, ./Train/episode_number/CameraRGB are images for training, and ./Train/episode_number/CameraSeg are data of ground truth labels.

##### Run the program

Command to train the model:

python main.py

The saved trained model is under foder ./checkpoint

To use the specific model for inference testing, change the 'model_path' variable in 'common.py' to the model for testing.

To run the model for grading:

grader 'python ./demo.py'

### References

1. Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    [link](https://arxiv.org/abs/1801.04381)

2. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

    Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    [link](https://arxiv.org/abs/1704.04861)

3. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.
     [link](https://arxiv.org/abs/1802.02611)

4. Xception: Deep Learning with Depthwise Separable Convolutions
    François Chollet
    [link](https://arxiv.org/abs/1610.02357)

4. [SSD Project from Self-Driving Car Nano Degree](https://github.com/see--/P12-Semantic-Segmentation)

5. [Modified MobileNet and Training](https://github.com/sagarbhokre/LyftChallengeV2)

6. [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122).