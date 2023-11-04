package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

func train(path string) {
	//load training image
	trainImages, err := loadImagesFromFile(path + "/train-images-idx3-ubyte")
	if err != nil {
		panic("Load training image fail!")
	}

	trainLabels, err := loadLabels(path + "/train-labels-idx1-ubyte")
	if err != nil {
		panic("Load training label fail!")
	}

	//construct model

	//set training parameter
	learning_rate := 0.01
	batch_size := 100
	num_epoch := 10

	for epoch := 0; epoch < num_epoch; epoch++ {
		for i := 0; i < len(trainImages.Data); i += batch_size {

			fmt.Printf("Epoch-%d-%05d : loss:%.4f\n", epoch, i, loss)
		}

		learning_rate *= 0.95 * *(epoch + 1)
	}

}

// load training image data from ubyte file
// Would return tensor pointer, with tensor size [number of images][channel of image][image height][image width]
// For MNIST, would be [60000][1][28][28]
func loadImagesFromFile(imageFile string) (*Tensor, error) {
	file, err := os.Open(imageFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	//前四个字节是魔数magic number，用于标识文件类型.discard
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
func loadLabels(labelFile string) (*Tensor, error) {
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

	tensorData := make([][][][]float32, numLabels)

	for i := uint32(0); i < numLabels; i++ {
		labelByte, _ := reader.ReadByte()
		label := int(labelByte)
		labelTensor := make([][][]float32, 10) // For MNIST, there is 10 types
		for j := 0; j < 10; j++ {

			if j == label {
				labelTensor[j] = [][]float32{{1}}
			} else {
				labelTensor[j] = [][]float32{{0}}
			}
		}
		tensorData[i] = labelTensor
	}

	return &Tensor{Data: tensorData}, nil
}
