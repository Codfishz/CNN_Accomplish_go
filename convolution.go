// 02-601: Programming for Scientist
// Final Project: Construct Convolutional Neural Network From Scratch Using GOlang
// This script contains methods for both "Forward" convolution and "Backward" gradient propogation
// It also contains all necessary helper function for the above two methods
// It has a function created to initialize a "Convolution" struct, which is a single convolution layer
// This script was developed by Dunhan Jiang



package main



import (
	"math"
	"math/rand"
	// "fmt"
)



// This is a struct for a single convolution layer
type Convolution struct {
	Data     [][][][]float32
	Pad int
	Stride int
	Kernel [][][][]float32
	Bias []float32
	KGradient [][][][]float32
	BGradient []float32
	ImageCol [][][][]float32
}



// This InitializeConvolutionLayer function build a Convolution struct as one layer
// This function takes as input one slice of ints as the intended kernel's shape, 
// plus three int parameters for Pad and Stride fields, as well as a number of all Images (the batch size)
// It returns a pointer to the new convolution layer initialized
func InitializeConvolutionLayer(kernel [][][][]float32, pad, stride, numImages int) *Convolution {

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

	scale := float32(math.Sqrt(float64(3 * kernelShape[0] * kernelShape[1] * kernelShape[2] / kernelShape[3]))) // scaler

	// initialize "Kernel"
	convLayer.Kernel = kernel
	/*
	for i := 0; i < kernelShape[0]; i++ {
		convLayer.Kernel[i] = make([][][]float32, kernelShape[1])
		for ii := 0; ii < kernelShape[1]; ii++ {
			convLayer.Kernel[i][ii] = make([][]float32, kernelShape[2])
			for iii := 0; iii < kernelShape[2]; iii++ {
				convLayer.Kernel[i][ii][iii] = make([]float32, kernelShape[3])
				for iiii := 0; iiii < kernelShape[3]; iiii++ {
					convLayer.Kernel[i][ii][iii][iiii] = float32(rand.NormFloat64()) / scale
				}
			}
		}
	}
	*/
	
	// initialize "Bias"
	convLayer.Bias = make([]float32, kernelShape[3])
	for i := 0; i < len(convLayer.Bias); i++ {
		convLayer.Bias[i] = float32(rand.NormFloat64()) / scale
	}

	// initialize "KGradient"
	convLayer.KGradient = make([][][][]float32, kernelShape[0])
	for i := 0; i < kernelShape[0]; i++ {
		convLayer.KGradient[i] = make([][][]float32, kernelShape[1])
		for ii := 0; ii < kernelShape[1]; ii++ {
			convLayer.KGradient[i][ii] = make([][]float32, kernelShape[2])
			for iii := 0; iii < kernelShape[2]; iii++ {
				convLayer.KGradient[i][ii][iii] = make([]float32, kernelShape[3])
			}
		}
	}

	// initialize "BGradient"
	convLayer.BGradient = make([]float32, kernelShape[3])

	// initialize "ImageCol"
	convLayer.ImageCol = make([][][][]float32, numImages)

	pointer := &convLayer
	return pointer
}



// This forward method is defined for a Conv struct, namely a convolutional layer, to compute the forward convolution
// Regardless of the fact that it is a method, we have two input variable: 
// the first input is a data of all images in the current single batch;
// the second input is a counter of the index of the first image in this single batch, for purposes of storing image patch feature
// This method returns a feature layer which we obtained after the convolution
// This ouput has type [][][][]float32, whose shape is given by: (bx, imageNum) * (featureHeight) * (featureWidth) * (nk, kernelNum)
func (convL *Convolution) Forward(x [][][][]float32) [][][][]float32 {

	// copy over the input data to the convolution layer struct
	// make padding when necessary
	convL.Data = Copy4D(x)
	// forward
	if convL.Pad != 0 {
		convL.Data = PadLayer(convL.Data, convL.Pad) // a subrotine is neseccary to pad the convL.Data into a favored size
	}

	// obtain the shape of the data
	bx := len(convL.Data) // number of image
	cx := len(convL.Data[0]) // number of channels
	hx := len(convL.Data[0][0]) // height
	wx := len(convL.Data[0][0][0]) // width

	// obtain the shape of the specified kernel
	hk := len(convL.Kernel) // height
	wk := len(convL.Kernel[0]) // width
	ck := len(convL.Kernel[0][0]) // number of channels
	nk := len(convL.Kernel[0][0][0]) // number of kernels

	// fmt.Println(bx, cx, hx, wx, hk, wk, ck, nk)

	// fmt.Println(cx, ck)
	if cx != ck {
		panic("Error: the given image and kernel have different number of channels")
	}

	// obtain the shape of the feature map after convolution, specifically the width and height
	feature_h := int(math.Floor(float64(hx - hk) / float64(convL.Stride))) + 1 // height of feature
	feature_w := int(math.Floor(float64(wx - wk) / float64(convL.Stride))) + 1 // width of feature

	// initialize the feature slice for output
	feature := make([][][][]float32, bx)

	// construct the entire feature for potentially many images in the current batch
	// the outmost loop1: iterate through every signle image inside the entire batch ///////
	for b := 0; b < bx; b++ {

		// for every image, it to a 3-dimensional matrix
		// such matrix has a 2-dimensonal matrix construction but every pixel represents all data within an image patch
		
		image := ImageToColumn(convL.Data[b], feature_w, feature_h, ck, wk, hk, convL.Stride)
		// Note: cx and ck should be equal, this function will panic if not
		// also store this image inside the Conv struct
		convL.ImageCol[b] = image

		// Then do the actual convolution for every image patch
		// image has shape: (feature_h) by (feature_w) by (wk * hk * cx)
		// kernel has shape: (wk) by (hk) by (ck) by (nk), equivalently, make kernel into shape: (wk * hk * cx) by (nk)
		// (image .* kernel) has shape:  (feature_h * feature_w) by (nk)

		feature[b] = make([][][]float32, nk)
		// ? the second outmost loop2: iterate through every signle row of many image patches //////
		for i := 0; i < nk; i++ {
			feature[b][i] = make([][]float32, feature_w)
			// the third outmost loop3: iterate through every signle image patch at a specific ith row /////
			for ii := 0; ii < feature_w; ii++ {
				feature[b][i][ii] = make([]float32, feature_h)
				// now an image patch at ith row, iith column has been located
				// the fourth outmost loop4: iterate through every signle kernel for this specific image patch ////
				for iii := 0; iii < feature_h; iii++ {
					// do convolution for an image patch at [i][ii] location in the feature space by the [iii] kernel
					// initialize a summation variable
					patchConv := float32(0.0)
					// the fifth outmost loop5: iterate through the height of this kernel ///
					for h := 0; h < hk; h++ {
						// the sixth outmost loop6: iterate through the width of this kernel //
						for w := 0; w < wk; w++ {
							// the seventh outmost loop7: iterate through the channel of this kernel /
							for c := 0; c < ck; c++ {
								// sum add: (weight * value + bias)
								patchConv += image[ii][iii][h*hk + w + c] * convL.Kernel[w][h][c][i] + convL.Bias[i]
							}
						}
					}
					feature[b][i][ii][iii] = patchConv
				}
			}
		}
	}

	return feature
}



// The PadLayer function takes as input one 4-dimension data plus one int parameter
// The original 4-dimension data has shape: bx * cx * hx * wx
// This function adds number of Pad zeros to the outer rim of the hx * wx dimension, such that the data
// will be transformed and returned with shape: bx * cx * (Pad + hx + Pad) * (Pad + wx + Pad)
func PadLayer(data [][][][]float32, pad int) [][][][]float32 {

	// obtain the shape of the data
	bx := len(data) // number of image
	cx := len(data[0]) // number of channels
	hx := len(data[0][0]) // height
	wx := len(data[0][0][0]) // width

	dataNew := make([][][][]float32, bx) // for output
	// the outmost loop1: iterate through every signle image inside the entire batch ////
	for i := 0; i < bx; i++ {
		dataNew[i] = make([][][]float32, cx)
		// the second outmost loop2: iterate through every signle channel of a specific image ///
		for ii := 0; ii < cx; ii++ {
			dataNew[i][ii] = make([][]float32, hx)
			// the third outmost loop3: iterate through rows of a specific image channel //
			for iii := 0; iii < (pad + hx + pad); iii++ {
				// for the first and last "# of pad" rows, make slices of zeros
				if iii < pad || (pad + hx + pad - iii) <= pad{
					dataNew[i][ii][iii] = make([]float32, (pad + wx + pad))
				} else { // copy over original row to the center of new row, with leftmost and rightmmost "# of zeros" equal pad
					dataNew[i][ii][iii] = make([]float32, (pad + wx + pad))
					// the fourth outmost loop4: iterate through columns of a specific row of an image's specific channel /
					for iiii := 0; iiii < wx; iiii++ { // copy over
						dataNew[i][ii][iii][iiii+pad] = data[i][ii][iii][iiii]
					}
				}
			}
		}
	}

	return dataNew
}



// The ImageToColumn function takes as input some parameters, including the following:
// a single image of type [][][]float32, a feature map width of type int, a feature map heigt of type int,
// a number of input image channels of type int, a number of kernel channels of type int, 
// a number of kernel width of type int, a number of kernel heigt of type int, and a stride parameter of type int.
// This function reshape one input image data to a matrix for output, having feature_h rows and feature_w columns,
// each "pixel" of this matrix correspond to an image patch of the given image, whoes shape is given by (wk * hk * ck).
func ImageToColumn(image [][][]float32, feature_h, feature_w, ck, hk, wk, stride int) [][][]float32 {
	// check whether image and kernel have the same number of channels, panic otherwise:
	/*
	if cx != ck {
		panic("Error: the given image and kernel have different number of channels")
	}
	*/

	// first initialize a [][][]float32 variable for output
	imagePatches := make([][][]float32, feature_h)
	// the outmost loop1: iterate through every signle row of many image patches /////
	for i := 0; i < feature_h; i++ {
		imagePatches[i] = make([][]float32, feature_w)
		// the second outmost loop2: iterate through every signle column of one specific row of many image patches ////
		for ii := 0; ii < feature_w; ii++ {
			imagePatches[i][ii] = make([]float32, (wk * hk * ck))
			// the third outmost loop3: iterate through the height of the given image patch (or kernel) ///
			for h := 0; h < hk; h++ {
				// the fourth outmost loop4: iterate through the width of the given image patch (or kernel) //
				for w := 0; w < wk; w++ {
					// the fifth outmost loop5: iterate through the channel of the given image patch (or kernel) /
					for c := 0; c < ck; c++ {
						// index of current image patch pixel's row: i*stride+h
						// index of current image patch pixel's col: ii*stride+w
						// fmt.Println(i, ii, h*hk + w + c, c, i*stride+h, ii*stride+w)
						// fmt.Println(h, w, h*hk + w + c)
						imagePatches[i][ii][h*hk + w + c] = image[c][i*stride+h][ii*stride+w]
					}
				}
			}
		}
	}

	return imagePatches
}



// This backward method is defined for a Conv struct, namely a convolutional layer, to compute the back propagation
// Regardless of the fact that it is a method, we have two input variable: 
// the first input is a delta of type Tensor computed from the last returning layer;
// the second input is a float32 parameter representing the learning rate
// This method returns another updated delta by this current convolution layer
// This ouput has type [][][][]float32
func (convL *Convolution) Backward(delta [][][][]float32, lRate float32) [][][][]float32 {

	////////////////////////////// Module 0: Essential Shapes
	// obtain the shape of the data
	bx := len(convL.Data) // number of image
	cx := len(convL.Data[0]) // number of channels
	hx := len(convL.Data[0][0]) // height
	wx := len(convL.Data[0][0][0]) // width

	// obtain the shape of the kernel
	hk := len(convL.Kernel) // height
	wk := len(convL.Kernel[0]) // width
	ck := len(convL.Kernel[0][0]) // number of channels
	nk := len(convL.Kernel[0][0][0]) // number of kernels

	// obtain the shape of the delta
	bd := len(delta) // number of image
	cd := len(delta[0]) // number of outChannels
	hd := len(delta[0][0]) // height
	wd := len(delta[0][0][0]) // width

	
	////////////////////////////// Module 1: Compute kernel & bias gradients
	// first initialize both the KGradient and BGradient fields to be all zeros to record necessary changes:
	for i := 0; i < hk; i++ {
		// the second outmost loop2: iterate through all columns of the kernel ///
		for ii := 0; ii < wk; ii++ {
			// the third outmost loop3: iterate through all channels of this pixel //
			for iii := 0; iii < ck; iii++ {
				// the fourth outmost loop4: iterate through all different kernels at this specific channel of pixel /
				for iiii := 0; iiii < nk; iiii++ {
					// initialize "KGradient" to be all zeros
					convL.KGradient[i][ii][iii][iiii] = 0
					// initialize "BGradient" to be all zeros
					convL.BGradient[iiii] = 0
				}
			}
		}
	}

	// Then compute the weight of Kernel Gradient
	// This nested loop compute the gradient of kernel weights
	bxFloat32 := float32(bx)
	// the outmost loop1: iterate through every signle image inside the entire batch ///////
	for b := 0; b < bd; b++ {
		// the second outmost loop2: iterate through every signle row of delta //////
		for i := 0; i < hd; i++ {
			// the third outmost loop3: iterate through every signle image patch at a specific ith row of delta /////
			for ii := 0; ii < wd; ii++ {
				// the fourth outmost loop4: iterate through every signle outChannel (equivalently, nk = cd) ////
				for iii := 0; iii < cd; iii++ {
					// the fifth outmost loop5: iterate through the height of the given image patch (or kernel) ///
					for h := 0; h < hk; h++ {
						// the sixth outmost loop6: iterate through the width of the given image patch (or kernel) //
						for w := 0; w < wk; w++ {
							// the most inner loop7: iterate through all inChannels (number of image channels) /
							for c := 0; c < ck; c++ {
								convL.KGradient[h][w][c][iii] += convL.ImageCol[b][i][ii][h*hk + w + c] * delta[b][iii][i][ii] / bxFloat32
							}
						}
					}
				}
			}
		}
	}

	// Also compute the Bias Gradient
	// This nested loop compute the gradient of kernel biases, with updates for bias field
	// the outmost loop1: iterate through every outChannel (equivalently, nk = cd) ////
	for i := 0; i < cd; i++ {
		sumConc := float32(0.0)
		// the second outmost loop2: iterate through every image in the entire batch ///
		for ii := 0; ii < bd; ii++ {
			// the third outmost loop3: iterate through every single row of delta //
			for iii := 0; iii < hd; iii++ {
				// the fourth outmost loop4: iterate through every single column of delta /
				// now a pixel is located for add up
				for iiii := 0; iiii < wd; iiii++ {
					sumConc += delta[ii][i][iii][iiii]
				}
			}
		}
		convL.BGradient[i] = sumConc / bxFloat32
	}


	////////////////////////////// Module 2: Compute deltaBackward for output for the "next" shadower layer
	// Initialize a tensor for output as the "delta" after current convolution layer's back propagation
	deltaBackward := make([][][][]float32, bx)
	for i := 0; i < bx; i++ {
		deltaBackward[i] = make([][][]float32, cx)
		for ii := 0; ii < cx; ii++ {
			deltaBackward[i][ii] = make([][]float32, hx)
			for iii := 0; iii < hx; iii++ {
				deltaBackward[i][ii][iii] = make([]float32, wx)
			}
		}
	}

	// perform any padding if necessary
	var deltaPad [][][][]float32
	if hd-hk+1 != hx {
		pad := (hx - hd + hk - 1) / 2 // integer division
		deltaPad = PadLayer(delta, pad) // a subrotine is neseccary to pad the convL.Data into a favored size
	} else {
		deltaPad = delta
	}
	// Notice that after potential padding, the "deltaPad" slice should have the same shape as "convL.Data"

	// Finally, calculate the value for "deltaBackward" for output
	// the outmost loop1: iterate through every signle image inside the entire batch ///////
	for b := 0; b < bx; b++ {
		image := ImageToColumn(deltaPad[b], hx, wx, cd, wk, hk, convL.Stride)
		// image has shape: hx by wx by (wk * hk * nk)
		// ATTENTION HERE: cd (# of outChannels to be concatenated) dimension
		// is different from ck, when this function was last called
		// image * k_180_col (whose shape is (wk * hk * nk) by ck) gives a slice whose shape is given by
		// wx * hx * ck, corresponding to the shape in "deltaBackward"
		
		// the second outmost loop2: iterate through every signle row of "image" //////
		for i := 0; i < hx; i++ {
			// the third outmost loop3: iterate through every signle column of "image" /////
			for ii := 0; ii < wx; ii++ {
				// now that an image patch in "image" has been located
				// the fourth outmost loop4: iterate through every signle channel of the image (inChannel) ////
				for c := 0; c < cx; c++ {
					kernelSum := float32(0.0)
					// the fifth outmost loop5: iterate through every signle row of kernel ///
					for h := 0; h < hk; h++ {
						// the sixth outmost loop6: iterate through every signle column of kernel //
						for w := 0; w < wk; w++ {
							// the seventh outmost loop7: iterate through every signle kernel (nk) /
							for n := 0; n < nk; n++ {
								kernelSum += convL.Kernel[h][w][c][n] * image[i][ii][h*hk + w + n] // this n is the key
								// now in the third dimension of image, it has shape (wk * hk * nk)
								// here nk (or equivalently cd) has replaced the previous ck
							}
						}
					}
					deltaBackward[b][c][i][ii] = kernelSum
				}
			}
		}
	}


	////////////////////////////// Module 3: Back Propagation
	// Back Propagation: Where kernel and bias would be updated
	// The following nested loop update the kernel according to both the kernel gradient and the learning rate
	// the outmost loop1: iterate through all rows of the kernel ////
	for i := 0; i < hk; i++ {
		// the second outmost loop2: iterate through all columns of the kernel ///
		for ii := 0; ii < wk; ii++ {
			// the third outmost loop3: iterate through all channels of this pixel //
			for iii := 0; iii < ck; iii++ {
				// the fourth outmost loop4: iterate through all different kernels at this specific channel of pixel /
				for iiii := 0; iiii < nk; iiii++ {
					convL.Kernel[i][ii][iii][iiii] -= convL.KGradient[i][ii][iii][iiii] * lRate
				}
			}
		}
	}

	// This one last loop update the biases
	for i := 0; i < nk; i++ {
		// update the bias according to both the bias gradient and the learning rate
		convL.Bias[i] -= convL.BGradient[i] * lRate
	}

	////////////////////////////// Module 4: Return AT LAST
	return deltaBackward
}


