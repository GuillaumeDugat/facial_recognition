# Personal Project : Facial Recognition

This project consists in creating a tool able to recognize your name thanks to your face. It is inspired by a workshop created by the AI student association of CentraleSupélec, Automatants.

## Installation

The dependancies are managed by poetry. To install them, simply run:

```
make install
```

## Usage

### Creation of the dataset

The structure of the dataset must be the following: inside the `Dataset` folder, each subfolder is associated to one person and contains the photos of the person in .jpg format. The name of the subfolder is used as a label.

### Training of the model and recognition with the webcam

For this purpose, run:

```
make run
```
