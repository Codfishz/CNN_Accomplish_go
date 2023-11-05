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
