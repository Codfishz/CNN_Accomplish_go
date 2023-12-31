package main

// "fmt"

type Relu struct {
	FeatureMask *Tensor
}

func (relu *Relu) Forward(x [][][][]float64) {
	relu.FeatureMask = NewTensor(len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))

	for i := 0; i < len(x); i++ {
		for ii := 0; ii < len(x[0]); ii++ {
			for iii := 0; iii < len(x[0][0]); iii++ {
				for iiii := 0; iiii < len(x[0][0][0]); iiii++ {
					relu.FeatureMask.Data[i][ii][iii][iiii] = x[i][ii][iii][iiii]
					if x[i][ii][iii][iiii] < 0 {
						x[i][ii][iii][iiii] = 0
					}
				}
			}
		}
	}

}

func (relu *Relu) Backward(delta [][][][]float64) {
	feature := relu.FeatureMask.Data
	for i := 0; i < len(feature); i++ {
		for ii := 0; ii < len(feature[0]); ii++ {
			for iii := 0; iii < len(feature[0][0]); iii++ {
				for iiii := 0; iiii < len(feature[0][0][0]); iiii++ {
					if feature[i][ii][iii][iiii] < 0 {
						delta[i][ii][iii][iiii] = 0
					}
				}
			}
		}
	}
}
