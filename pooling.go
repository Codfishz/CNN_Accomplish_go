package main

import (
	"math"
	//"fmt"
)

// Pooling is a struct for a max pooling layer.
type Pooling struct {
	// tensor.Data: batch size, channel, height, width
	// feature mask is used to record the max value index in each 2 * 2 window for backward max pooling
	FeatureMask *Tensor

}

// Forward takes the input data x and returns the output data after max pooling.
// It performs forward max pooling on the input data
func (pool *Pooling) Forward(x [][][][]float32) [][][][]float32 {
	// b: batch size, c: channel, h: height, w: width
	b, c, h, w := len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
	// stride = 2
	featureH, featureW := h/2, w/2

	// initialize feature maps, all zeros
	feature := NewTensor(b, c, featureH, featureW) // b * c * featureH * featureW

	// initialize feature mask, all zeros
	pool.FeatureMask = NewTensor(b, c, h, w) // b * c * h * w

	// dimension by dimension to initialize map and mask and compute max pooling
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			for i := 0; i < featureH; i++ {
				for j := 0; j < featureW; j++ {
					// initialize max value and max value index
					maxVal := float32(math.Inf(-1)) // negative infinity
					maxIndexI, maxIndexJ := 0, 0 // max value index

					// go through all2 * 2 window and compute max value
					for m := 0; m < 2; m++ {
						for n := 0; n < 2; n++ {
							val := x[bi][ci][i*2+m][j*2+n]
							if val > maxVal {
								maxVal = val
								maxIndexI, maxIndexJ = i*2+m, j*2+n
							}
						}
					}

					// set the max value to feature map
					feature.Data[bi][ci][i][j] = maxVal
					// set the max value index to 1 in feature mask
					pool.FeatureMask.Data[bi][ci][maxIndexI][maxIndexJ] = 1

				}
			}
			//fmt.Println(pool.FeatureMask.Data[bi][ci])
		}
	}
	return feature.Data
}

// Backward takes the gradient from the next layer delta and returns the gradient for the previous layer dx.
// delta: gradient from the next layer (b * c * featureH * featureW).
func (pool *Pooling) Backward(delta [][][][]float32) [][][][]float32 {
	b, c, featureH, featureW := len(delta), len(delta[0]), len(delta[0][0]), len(delta[0][0][0])
	h, w := len(pool.FeatureMask.Data[0][0]), len(pool.FeatureMask.Data[0][0][0])
	dx := NewTensor(b, c, h, w)
	restoredDelta := NewTensor(b, c, h, w)

	// restore the gradient from the next layer to the original size
	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			for i := 0; i < featureH; i++ {
				for j := 0; j < featureW; j++ {
					// duplicate each value in delta in 2 * 2 window
					for m := 0; m < 2; m++ {
						for n := 0; n < 2; n++ {
							restoredDelta.Data[bi][ci][i*2+m][j*2+n] = delta[bi][ci][i][j]
						}
					}
				}
			}
		}
	}

	for bi := 0; bi < b; bi++ {
		for ci := 0; ci < c; ci++ {
			for i := 0; i < h; i++ {
				for j := 0; j < w; j++ {
					// if the value in feature mask is 0, then the value in dx is 0
					// otherwise, the value in dx is the value in delta
					dx.Data[bi][ci][i][j] = restoredDelta.Data[bi][ci][i][j] * pool.FeatureMask.Data[bi][ci][i][j]
				}
			}
		}
	}

	return dx.Data
}

// // Testing max pooling
// // Put this under main.go
// import(
// 	"fmt"
// )

// func main() {
// 	// Create an instance of the pooling layer
// 	pool := &Pooling{}

// 	// Sample input data (batch size: 1, channels: 1, height: 4, width: 4)
// 	input := [][][][]float32{
// 		{
// 			{
// 				{1, 2, 1, 4},
// 				{2, 3, 2, 1},
// 				{4, 3, 1, 2},
// 				{1, 4, 3, 2},
// 			},
// 		},

// 		{
// 			{
// 				{3, 1, 0, 0},
// 				{1, 6, 0, 2},
// 				{1, 1, 1, 3},
// 				{1, 1, 3, 1},
// 			},
// 		},
// 	}

// 	// Perform forward max pooling
// 	output := pool.Forward(input)

// 	// Display the input and output
// 	fmt.Println("Input:")
// 	for i := 0; i < len(input); i++ {
// 		fmt.Println("Image", i+1)
// 		printData(input[i][0])
// 	}

// 	fmt.Println("\nOutput (Max Pooled):")
// 	for i := 0; i < len(output); i++ {
// 		fmt.Println("Image", i+1)
// 		printData(output[i][0])
// 	}

// 	// Display the feature mask
// 	fmt.Println("\nFeature Mask:")
// 	for i := 0; i < len(pool.FeatureMask.Data); i++ {
// 		fmt.Println("Image", i+1)
// 		printData(pool.FeatureMask.Data[i][0])
// 	}
	
// 	delta := [][][][]float32{
// 		{
// 			{
// 				{1, 2},
// 				{2, 3},
// 			},
// 		},

// 		{
// 			{
// 				{3, 1},
// 				{1, 6},
// 			},
// 		},
// 	}

// 	// Perform backward max pooling
// 	dx := pool.Backward(delta)

// 	// Display input delta
// 	fmt.Println("\nInput Delta:")
// 	for i := 0; i < len(delta); i++ {
// 		fmt.Println("Image", i+1)
// 		printData(delta[i][0])
// 	}

// 	// Display the gradient
// 	fmt.Println("\nGradient:")
// 	for i := 0; i < len(dx); i++ {
// 		fmt.Println("Image", i+1)
// 		printData(dx[i][0])
// 	}


// }

// // Function to print a 2D slice of data
// func printData(data [][]float32) {
// 	for i := 0; i < len(data); i++ {
// 		for j := 0; j < len(data[i]); j++ {
// 			fmt.Printf("%.1f ", data[i][j])
// 		}
// 		fmt.Println()
// 	}
// }