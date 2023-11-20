package main

type Tensor struct {
	// batch size, channel, height, width
	Data [][][][]float32
}

// Initialize a tensor with all zeros
func NewTensor(b, c, h, w int) *Tensor {
	data := make([][][][]float32, b)
	for i := 0; i < b; i++ {
		data[i] = make([][][]float32, c)
		for j := 0; j < c; j++ {
			data[i][j] = make([][]float32, h)
			for k := 0; k < h; k++ {
				data[i][j][k] = make([]float32, w)
			}
		}
	}
	return &Tensor{Data: data}
}

// Reshape a 4D tensor to a 2D tensor
func Reshape4Dto2D(matrix4D [][][][]float32) [][]float32 {
    batchSize := len(matrix4D)
    if batchSize == 0 {
        return nil
    }

    channel := len(matrix4D[0])
    height := len(matrix4D[0][0])
    width := len(matrix4D[0][0][0])
    flatSize := channel * height * width

    matrix2D := make([][]float32, batchSize)
    for i := range matrix2D {
        matrix2D[i] = make([]float32, flatSize)
        flatIndex := 0
        for j := 0; j < channel; j++ {
            for k := 0; k < height; k++ {
                for l := 0; l < width; l++ {
                    matrix2D[i][flatIndex] = matrix4D[i][j][k][l]
                    flatIndex++
                }
            }
        }
    }

    return matrix2D
}

// Reshape a 2D tensor to a 4D tensor
func Reshape2Dto4D(matrix2D [][]float32, batchSize, channel, height, width int) [][][][]float32 {
    if len(matrix2D) == 0 || len(matrix2D[0]) != channel*height*width {
        return nil // or handle this case as per your needs
    }

    matrix4D := make([][][][]float32, batchSize)
    for i := range matrix4D {
        matrix4D[i] = make([][][]float32, channel)
        for j := range matrix4D[i] {
            matrix4D[i][j] = make([][]float32, height)
            for k := range matrix4D[i][j] {
                matrix4D[i][j][k] = make([]float32, width)
                for l := range matrix4D[i][j][k] {
                    flatIndex := j*height*width + k*width + l
                    matrix4D[i][j][k][l] = matrix2D[i][flatIndex]
                }
            }
        }
    }

    return matrix4D
}
