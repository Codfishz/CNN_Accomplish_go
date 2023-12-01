package main

import (
	"fmt"
)

func main() {
	
	//training process
	fmt.Println("Start training process")

	learning_rate := float64(0.01)
	batch_size := 3
	num_epoch := 3
	path := "MINST"

	m := Train(path, learning_rate, num_epoch, batch_size)
	// Train(path, learning_rate, num_epoch, batch_size)

	//finish training
	fmt.Println("End training process")

	//evaluation process
	fmt.Println("Start evaluation process")
	Accuracy := Eval(path, batch_size, *m)
	fmt.Println("Accuracy is ", Accuracy)

	fmt.Println("End training process")
	
	/*
	fmt.Println("forward test starts here")
	// define a new tensor to feed our CNN
	input := make([][][][]float64, 3) // number of images, number of inChannel, height, width
	for b := 0; b < 3; b++ {
		input[b] = make([][][]float64, 1)
		input[b][0] = make([][]float64, 5)
	}
	// first image
	input[0][0][0] = []float64{1,2,3,4,5}
	input[0][0][1] = []float64{2,3,4,5,6}
	input[0][0][2] = []float64{3,4,5,6,7}
	input[0][0][3] = []float64{4,5,6,7,8}
	input[0][0][4] = []float64{5,6,7,8,9}
	// second image
	input[1][0][0] = []float64{1,2,3,4,5}
	input[1][0][1] = []float64{2,3,4,5,6}
	input[1][0][2] = []float64{3,4,5,6,7}
	input[1][0][3] = []float64{4,5,6,7,8}
	input[1][0][4] = []float64{5,6,7,8,9}
	// third image
	input[2][0][0] = []float64{1,2,3,4,5}
	input[2][0][1] = []float64{2,3,4,5,6}
	input[2][0][2] = []float64{3,4,5,6,7}
	input[2][0][3] = []float64{4,5,6,7,8}
	input[2][0][4] = []float64{5,6,7,8,9}
	fmt.Println("the input sample of three images as a batch:")
	fmt.Println(input[0])
	fmt.Println(input[1])
	fmt.Println(input[2])
	
	twoD := Reshape4Dto2D(input)
	fmt.Println("check input after first reshape:")
	fmt.Println(twoD[0])
	fmt.Println(twoD[1])
	fmt.Println(twoD[2])
	fourD := Reshape2Dto4D(twoD,3,1,5,5)
	fmt.Println("check input after two reshape functions:")
	fmt.Println(fourD[0])
	fmt.Println(fourD[1])
	fmt.Println(fourD[2])
	fmt.Println("")

	// initialize the kernel for a convolution layer
	kernelTest := make([][][][]float64, 2) // height, width, # inChanel = 1, # kernels(OutChannel)
	for i := 0; i < 2; i++ {
		kernelTest[i] = make([][][]float64, 2)
		for j := 0; j < 2; j++ {
			kernelTest[i][j] = make([][]float64, 1)
			kernelTest[i][j][0] = make([]float64, 4)
		}
	}
	kernelTest[0][0][0] = []float64{1,1,1,1}
	kernelTest[0][1][0] = []float64{2,2,2,2}
	kernelTest[1][0][0] = []float64{2,2,2,2}
	kernelTest[1][1][0] = []float64{3,3,3,3}
	fmt.Println("convolution layer's kernel is:")
	fmt.Println(kernelTest)
	conv_1 := InitializeConvAssigned(kernelTest, 0, 1, 3)
	conv_1.Bias = []float64{1,2,3,4}
	fmt.Println("")

	// forward convolution, manually checked for correction
	conv_1_output := conv_1.Forward(input)
	fmt.Println("input after forward convolution returns a feature:")
	fmt.Println("feature: 1st image:")
	fmt.Println(conv_1_output[0][0])
	fmt.Println(conv_1_output[0][1])
	fmt.Println(conv_1_output[0][2])
	fmt.Println(conv_1_output[0][3])
	fmt.Println("feature: 2nd image:")
	fmt.Println(conv_1_output[1][0])
	fmt.Println(conv_1_output[1][1])
	fmt.Println(conv_1_output[1][2])
	fmt.Println(conv_1_output[1][3])
	fmt.Println("feature: 3rd image:")
	fmt.Println(conv_1_output[2][0])
	fmt.Println(conv_1_output[2][1])
	fmt.Println(conv_1_output[2][2])
	fmt.Println(conv_1_output[2][3])
	fmt.Println("")

	// backward convolution
	deltaC := make([][][][]float64, 3)

	for b := 0; b < 3; b++ {
		deltaC[b] = make([][][]float64, 4)

		deltaC[b][0] = make([][]float64, 4)
		deltaC[b][0][0] = []float64{1,2,3,4}
		deltaC[b][0][1] = []float64{2,3,4,5}
		deltaC[b][0][2] = []float64{3,4,5,6}
		deltaC[b][0][3] = []float64{4,5,6,7}

		deltaC[b][1] = make([][]float64, 4)
		deltaC[b][1][0] = []float64{3,4,5,6}
		deltaC[b][1][1] = []float64{4,5,6,7}
		deltaC[b][1][2] = []float64{5,6,7,8}
		deltaC[b][1][3] = []float64{6,7,8,9}

		deltaC[b][2] = make([][]float64, 4)
		deltaC[b][2][0] = []float64{5,6,7,8}
		deltaC[b][2][1] = []float64{6,7,8,9}
		deltaC[b][2][2] = []float64{7,8,9,10}
		deltaC[b][2][3] = []float64{8,9,10,11}

		deltaC[b][3] = make([][]float64, 4)
		deltaC[b][3][0] = []float64{7,8,9,10}
		deltaC[b][3][1] = []float64{8,9,10,11}
		deltaC[b][3][2] = []float64{9,10,11,12}
		deltaC[b][3][3] = []float64{10,11,12,13}
	}
	convBack := conv_1.Backward(deltaC, 0.01)
	fmt.Println("after convolution backward, delta is:")
	fmt.Println(convBack[0])
	fmt.Println(convBack[1])
	fmt.Println(convBack[2])
	fmt.Println("after convolution backward, new kernel is:")
	fmt.Println(conv_1.Kernel)
	fmt.Println("after convolution backward, new bias is:")
	fmt.Println(conv_1.Bias)
	fmt.Println("")

	// start relu forward activation, manually checked for correction
	var relu_1 Relu
	relu_1.Forward(conv_1_output) // relu.Forward makes local changes
	fmt.Println("after forward convolution feature going through relu activation:")
	fmt.Println("feature: 1st image:")
	fmt.Println(conv_1_output[0][0])
	fmt.Println(conv_1_output[0][1])
	fmt.Println(conv_1_output[0][2])
	fmt.Println(conv_1_output[0][3])
	fmt.Println("feature: 2nd image:")
	fmt.Println(conv_1_output[1][0])
	fmt.Println(conv_1_output[1][1])
	fmt.Println(conv_1_output[1][2])
	fmt.Println(conv_1_output[1][3])
	fmt.Println("feature: 3rd image:")
	fmt.Println(conv_1_output[2][0])
	fmt.Println(conv_1_output[2][1])
	fmt.Println(conv_1_output[2][2])
	fmt.Println(conv_1_output[2][3])
	fmt.Println("")

	// pooling layer, manually checked for correction (both output and the memory of pool.FeatureMask.Data)
	var pool_1 Pooling
	pool_1_output := pool_1.Forward(conv_1_output)
	fmt.Println("max pooling layer of size 2*2:")
	fmt.Println("feature: 1st image:")
	fmt.Println(pool_1_output[0][0])
	fmt.Println(pool_1_output[0][1])
	fmt.Println(pool_1_output[0][2])
	fmt.Println(pool_1_output[0][3])
	fmt.Println("feature: 2nd image:")
	fmt.Println(pool_1_output[1][0])
	fmt.Println(pool_1_output[1][1])
	fmt.Println(pool_1_output[1][2])
	fmt.Println(pool_1_output[1][3])
	fmt.Println("feature: 3rd image:")
	fmt.Println(pool_1_output[2][0])
	fmt.Println(pool_1_output[2][1])
	fmt.Println(pool_1_output[2][2])
	fmt.Println(pool_1_output[2][3])
	fmt.Println("")

	// newLinear: fully-connected layer
	linear :=  NewLinearAssigned(16, 10) // all weight and bias equal float64(1)
	pool_1_output_reshaped := Reshape4Dto2D(pool_1_output) // has shape: batchSize by everythingElseFlattened
	linear_output := linear.Forward(pool_1_output_reshaped) // shape: 3 by 10
	fmt.Println("output from fully connected layer of size 3*10:")
	fmt.Println("1st batch:")
	fmt.Println(linear_output[0])
	fmt.Println("2nd batch:")
	fmt.Println(linear_output[1])
	fmt.Println("3rd batch:")
	fmt.Println(linear_output[2])
	fmt.Println("")

	// sofmax layer
	batchLabel := make([][]float64, 3)
	batchLabel[0] = []float64{1,0,0,0,0,0,0,0,0,0}
	batchLabel[1] = []float64{0,1,0,0,0,0,0,0,0,0}
	batchLabel[2] = []float64{0,0,1,0,0,0,0,0,0,0}
	loss, delta := SoftmaxCalLoss(linear_output, batchLabel)
	fmt.Println("output from softmax layer:")
	fmt.Println("loss:", loss)
	fmt.Println("delta from softmax:")
	fmt.Println(delta[0])
	fmt.Println(delta[1])
	fmt.Println(delta[2])
	fmt.Println("")

	// linear backward
	linearB := NewLinearAssigned(4, 3)
	linearB.x = make([][]float64, 2)
	linearB.x[0] = []float64{1,2,3,4}
	linearB.x[1] = []float64{2,3,4,5}
	linearB.W[0] = []float64{1,2,3}
	linearB.W[1] = []float64{2,3,4}
	linearB.W[2] = []float64{3,4,5}
	linearB.W[3] = []float64{4,5,6}
	linearB.b = []float64{1,2,3}
	deltab := make([][]float64, 2)
	deltab[0] = []float64{1,2,3}
	deltab[1] = []float64{2,3,4}
	deltabPrime := linearB.Backward(deltab, float64(0.01))
	fmt.Println("after linear backward:")
	fmt.Println(deltabPrime[0])
	fmt.Println(deltabPrime[1])
	fmt.Println("weight:")
	fmt.Println(linearB.W)
	fmt.Println("bias:")
	fmt.Println(linearB.b)
	fmt.Println(" ")
	*/

	

}

/*
func InitializeConvAssigned(kernel [][][][]float64, pad, stride, numImages int) *Convolution {

	var convLayer Convolution // one convolution layer

	// obtain kernelShape
	kernelShape := make([]int, 4)
	kernelShape[0] = len(kernel)
	kernelShape[1] = len(kernel[0])
	kernelShape[2] = len(kernel[0][0])
	kernelShape[3] = len(kernel[0][0][0])

	// copy over parameter values
	convLayer.Pad = pad
	convLayer.Stride = stride

	// initialize "Kernel"
	convLayer.Kernel = kernel

	// initialize "Bias"
	convLayer.Bias = make([]float64, kernelShape[3])
	for i := 0; i < len(convLayer.Bias); i++ {
		convLayer.Bias[i] = float64(0.1) * float64(i)
	}
	fmt.Println("convolution layer's bias is:")
	fmt.Println(convLayer.Bias)

	// initialize "KGradient"
	convLayer.KGradient = make([][][][]float64, kernelShape[0])
	for i := 0; i < kernelShape[0]; i++ {
		convLayer.KGradient[i] = make([][][]float64, kernelShape[1])
		for ii := 0; ii < kernelShape[1]; ii++ {
			convLayer.KGradient[i][ii] = make([][]float64, kernelShape[2])
			for iii := 0; iii < kernelShape[2]; iii++ {
				convLayer.KGradient[i][ii][iii] = make([]float64, kernelShape[3])
			}
		}
	}

	// initialize "BGradient"
	convLayer.BGradient = make([]float64, kernelShape[3])

	// initialize "ImageCol"
	convLayer.ImageCol = make([][][][]float64, numImages)

	pointer := &convLayer
	return pointer
}


func NewLinearAssigned(inChannel, outChannel int) *Linear {

	// Initialize the weights and biases with random values (gradients will be zero)
	W := make([][]float64, inChannel)
	WGradient := make([][]float64, inChannel)
	for i := range W {
		W[i] = make([]float64, outChannel)
		WGradient[i] = make([]float64, outChannel)
		for j := range W[i] {
			W[i][j] = float64(1)
		}
	}

	b := make([]float64, outChannel)
	bGradient := make([]float64, outChannel)
	for i := range b {
		b[i] = float64(1)
		bGradient[i] = 0
	}

	// Return the new layer
	return &Linear{
		W:           W,
		b:           b,
		WGradient:   WGradient,
		bGradient:   bGradient,
		inChannel:   inChannel,
		outChannel:  outChannel,
	}
}
*/