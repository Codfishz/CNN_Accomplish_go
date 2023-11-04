// 02-601: Programming for Scientist
// Final Project: Construct Convolutional Neural Network From Scratch Using GOlang
// This script contains methods for both "Forward" convolution and "Backward" gradient propogation
// It also contains all necessary helper function for the above two methods
// It has a function created to initialize a "Conv" struct, which is a single convolution layer
// It was developed by student: Dunhan Jiang



package main



import (
	"math"
	"math/rand"
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
// plus three int parameters for Pad and Stride fields, as well as a number of all Images
// It returns a pointer to the new convolution layer initialized
func InitializeConvolutionLayer(kernelShape []int, pad, stride, numImages int) *Convolution {

	var convLayer Convolution // one convolution layer

	// copy over parameter values
	convLayer.Pad = pad
	convLayer.Stride = stride

	scale := float32(math.Sqrt(float64(3 * kernelShape[0] * kernelShape[1] * kernelShape[2] / kernelShape[3]))) // scaler

	// initialize "Kernel"
	convLayer.Kernel = make([][][][]float32, kernelShape[0])
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
func (convL *Convolution) Forward(x [][][][]float32, imageIndex int) [][][][]float32 {

	// copy over the input data to the convolution layer struct
	// make padding when necessary
	convL.Data = x
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
		image := ImageToColumn(convL.Data[b], feature_w, feature_h, cx, ck, wk, hk, convL.Stride)
		// Note: cx and ck should be equal, this function will panic if not
		// also store this image inside the Conv struct
		convL.ImageCol[imageIndex + b] = image

		// Then do the actual convolution for every image patch
		// image has shape: (feature_h) by (feature_w) by (wk * hk * cx)
		// kernel has shape: (wk) by (hk) by (ck) by (nk), equivalently, make kernel into shape: (wk * hk * cx) by (nk)
		// (image .* kernel) has shape:  (feature_h * feature_w) by (nk)

		feature[b] = make([][][]float32, feature_h)
		// the second outmost loop2: iterate through every signle row of many image patches //////
		for i := 0; i < feature_h; i++ {
			feature[b][i] = make([][]float32, feature_w)
			// the third outmost loop3: iterate through every signle image patch at a specific ith row /////
			for ii := 0; ii < feature_w; ii++ {
				feature[b][i][ii] = make([]float32, nk)
				// now an image patch at ith row, iith column has been located
				// the fourth outmost loop4: iterate through every signle kernel for this specific image patch ////
				for iii := 0; iii < nk; iii++ {
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
								patchConv += image[i][ii][h*hk + w*wk + c] * convL.Kernel[w][h][c][iii] + convL.Bias[iii]
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
// each "pixel" of this matrix correspond to an image patch of the given image, whoes shape is given by (wk * hk * cx).
func ImageToColumn(image [][][]float32, feature_h, feature_w, cx, ck, hk, wk, stride int) [][][]float32 {

	// check whether image and kernel have the same number of channels, panic otherwise:
	if cx != ck {
		panic("Error: the given image and kernel have different number of channels")
	}

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
						imagePatches[i][ii][h*hk + w*wk + c] = image[c][i*stride+h][ii*stride+w]
					}
				}
			}
		}
	}

	return imagePatches
}



func (tensor *Convolution) backward() [][][][]float32 {

	return tensor.KGradient
}

