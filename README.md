## CNN Comparison Framework for Radio Galaxy Classification

This framework serves as a testbed to train and compare different CNNs for radio galaxy classification.

The quickstart guide will help you train ConvXpress, the novel classifier developed during this project. 

Given the same seeds for the random processes as was used during training, you _should_ see the same results as we got for ConvXpress. The results recorded in the article are averaged from three runs (seeds 8901, 1826 and 4915). If you do not reproduce the same average results, please email me the training logs: adolfburgerbecker@gmail.com

### Quickstart

You'll need at least 8.47 GB of free disk space and this assumes GPU availability. If no GPU is available, see the [Deepo CPU version](https://github.com/ufoym/deepo#cpu-version) guide.

0. Before you start, please follow these steps
	0. Make a new directory/folder for this project. We'll refer to this as the project directory.
	1. Download the FITS image data (*Warning: the uncompressed size is about 5.3GB*): https://drive.google.com/drive/folders/12AUrQ2aA1jpFD9ObRJQWSZ5If2Mm8Jfe?usp=sharing
	2. Clone this repo: 

```git clone https://github.com/BurgerBecker/rg-benchmarker.git```

1. Install [Docker](https://docs.docker.com/engine/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

2. Obtain the [Deepo](https://github.com/ufoym/deepo) Docker image:

```docker pull ufoym/deepo```

3. Then run the docker container with an interactive shell, change `<insert_custom_label>` to a unique label for the container and `<project_directory>` to the new directory discussed in item 0.:

```docker run -it --gpus=all --name="<insert_custom_name>" -l "<insert_custom_label>" -v "<project_directory>:/stored" ufoym/deepo bash```

4. Change directory into the `stored/rg-benchmarker/` directory.

5. Run the quickstart Python script. This will first augment your data and then start training ConvPress, after which the model will be tested.

Final results will be in the last lines of the log file. The argument after `-a` gives the file with the list of architectures, their learning rates and number of epochs to run them. The argument after `-s` gives the random seed to be used.

```python3 quickstart.py -a convxpress.txt -s 8901 >> log.txt```

To replicate the results exactly, run the above line with 1826 and 4915 as seed values and take the average of the three runs.

<!-- 
Build the Dockerfile[^1] with the following command (this might take a while):

```docker build --tag something-something```

This built a Docker image. You can now run an instance of this image (called a Docker container):
*UPDATE THIS*
```docker run ```

You should now see a Linux terminal. Change directory into `rg-benchmarker`. Run the following command:
*UPDATE THIS*
```python3 quickstart.py >> log.txt```

This will first augment your data and then start training, after which the results will be tested.

Final results will be in the last lines of the file.

[^1]: The Dockerfile will mount the FITS folder (so don't change the name) and will eventually use it to save augmentations on disk (which will use an additional 3.169 GB). This is useful when training all the models, since they reuse the same data and it saves quite a bit of time and compute. This is probably less useful when training a single model, since this will generate and save 24 augmented images for each of the 350 training/validation images. That's 24 x 350 x 377.3 KB = 3.169 GB on top of the existing 5.3GB. -->

## Training and Replicability
All training for our study was performed on a Nvidia Tesla V100 32GB. Our architectures were constructed using the Deep Learning framework, Keras. To ensure replicable results we provided a random seed to all non-deterministic processes. In addition to this step, Tensorflow requires you to set the `TF_CUDNN_DETERMINISTIC` environment variable to `1` or `TRUE` (alternatively, depending on the version of Tensorflow being used, either the Nvidia Tensorflow-Determinism patch can be applied or the `TF_DETERMINISTIC_OPS` environment variable must be set (see https://pypi.org/project/tensorflow-determinism/). A customized version of the Keras Data Generator class was used to load images during training, validation and testing.

Available at:

https://academic.oup.com/mnras/article-abstract/503/2/1828/6129789

https://arxiv.org/abs/2102.03780
