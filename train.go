// 02-601: Programming for Scientist
// Final Project: Construct Convolutional Neural Network From Scratch Using GOlang
// This script contains methods for both "Forward" convolution and "Backward" gradient propogation
// It also contains all necessary helper function for the above two methods
// It has a function created to initialize a "Convolution" struct, which is a single convolution layer
// This script was developed by Zirui Chen

package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// This Train function would load model and use traing set to improve the parameter of model layers.
func Train(path string, learning_rate float32, num_epoch int, batch_size int) *Model {
	//load training image
	trainImages, err := LoadImagesFromFile(path + "/train-images-idx3-ubyte")
	if err != nil {
		panic("Load training image fail!")
	}

	trainLabels, err := LoadLabelsFromFile(path + "/train-labels-idx1-ubyte")
	if err != nil {
		panic("Load training label fail!")
	}

	//construct model
	data := []int{5, 5, 1, 6}
	kernel_1 := make([][][][]float32, data[0])
	for i := 0; i < data[0]; i++ {
		kernel_1[i] = make([][][]float32, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_1[i][j] = make([][]float32, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_1[i][j][k] = make([]float32, data[3])
			}
		}
	}
	conv_1 := InitializeConvolutionLayer(kernel_1, 0, 1, batch_size)

	//pool_1
	var pool_1 Pooling

	//relu
	var relu_1 Relu

	//conv2
	data = []int{5, 5, 6, 16}
	kernel_2 := make([][][][]float32, data[0])
	for i := 0; i < data[0]; i++ {
		kernel_2[i] = make([][][]float32, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_2[i][j] = make([][]float32, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_2[i][j][k] = make([]float32, data[3])
			}
		}
	}
	conv_2 := InitializeConvolutionLayer(kernel_2, 0, 1, batch_size)

	//relu_2
	var relu_2 Relu

	//pool_2
	var pool_2 Pooling

	//linear layer
	linear := NewLinear(256, 10)

	//softmax
	var softmax Softmax

	for epoch := 0; epoch < num_epoch; epoch++ {
		for i := 0; i < len(trainImages.Data); i += batch_size {
			//get batch data
			batchData := trainImages.Data[i : i+batch_size]
			batchLabel := trainLabels[i : i+batch_size]

			//forward pass
			conv_1_output := conv_1.Forward(batchData)
			relu_1.Forward(conv_1_output)
			pool_1_output := pool_1.Forward(conv_1_output)
			conv_2_output := conv_2.Forward(pool_1_output)
			relu_2.Forward(conv_2_output)
			pool_2_output := pool_2.Forward(conv_2_output)
			pool_2_output_reshaped := Reshape4Dto2D(pool_2_output)
			linear_output := linear.Forward(pool_2_output_reshaped)
			loss, delta := softmax.CalLoss(linear_output, batchLabel)

			//calculate loss
			delta = linear.Backward(delta, learning_rate)
			delta_1 := Reshape2Dto4D(delta, batch_size, 4, 4, 16)
			delta_2 := pool_2.Backward(delta_1)
			relu_2.Backward(delta_2)
			delta_3 := conv_2.Backward(delta_2, learning_rate)
			relu_1.Backward(delta_3)
			delta_4 := pool_1.Backward(delta_3)
			conv_1.Backward(delta_4, learning_rate)

			learning_rate *= float32(math.Pow(0.95, float64(epoch+1)))

			fmt.Printf("Epoch-%d-%05d : loss:%.4f\n", epoch, i, loss)
		}

	}

	//return model parameters
	m := Model {
		kernel_1: conv_1.Kernel,
		kernel_2: conv_2.Kernel,
		bias_1: conv_1.Bias,
		bias_2: conv_2.Bias,
		weight: linear.W,
		bias: linear.b,
	}
	return &m
}

// load training image data from ubyte file
// Would return tensor pointer, with tensor size [number of images][channel of image][image height][image width]
// For MNIST, would be [60000][1][28][28]
func LoadImagesFromFile(imageFile string) (*Tensor, error) {

	//\heck whether the path for image file is correct
	//
	file, err := os.Open(imageFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	// The first 4 bytes repmagic number，用于标识文件类型.discard
	reader.Discard(4)

	//because data type is unit 32, so each time binary.read would loaod 4 bytes
	//load the number of images, imgae rows, image column in sequence
	var numImages uint32
	binary.Read(reader, binary.BigEndian, &numImages)
	var numRows uint32
	binary.Read(reader, binary.BigEndian, &numRows)
	var numCols uint32
	binary.Read(reader, binary.BigEndian, &numCols)

	//begin loading image data
	//create tensor to store all Images
	tensorData := make([][][][]float32, numImages)

	for i := uint32(0); i < numImages; i++ {
		//set the channel number to 1 for gray scale image
		image := make([][][]float32, 1)
		image[0] = make([][]float32, numRows)

		//go through all rows
		for r := uint32(0); r < numRows; r++ {
			image[0][r] = make([]float32, numCols)

			//go through all columns
			for c := uint32(0); c < numCols; c++ {
				//read pixel information
				var pixel byte
				binary.Read(reader, binary.BigEndian, &pixel)
				image[0][r][c] = float32(pixel) / 255.0
			}
		}
		tensorData[i] = image
	}
	//create and return a new tensor object
	return &Tensor{Data: tensorData}, nil
}

// load training label data from ubyte file
// Would return tensor pointer, with tensor size [number of labels][channel of label(number of types))]
// for MNIST, would be [60000][10][1][1]
func LoadLabelsFromFile(labelFile string) ([][]float32, error) {
	file, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	//前四个字节是魔数magic number，用于标识文件类型.discard
	reader.Discard(4)

	var numLabels uint32
	binary.Read(reader, binary.BigEndian, &numLabels)

	labelData := make([][]float32, numLabels)

	for i := uint32(0); i < numLabels; i++ {
		labelByte, _ := reader.ReadByte()
		label := int(labelByte)
		for j := 0; j < 10; j++ {

			if j == label {
				labelData[i][j] = 1
			} else {
				labelData[i][j] = 0
			}
		}
	}

	return labelData, nil
}
