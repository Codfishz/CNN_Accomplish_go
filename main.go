package main

import (
	"fmt"
)

func main() {

	// testing starts now
	Test()

	//training process for MNIST data set starts now
	fmt.Println("Start training process to recognize the MNIST data set, pending...")

	learning_rate := float64(0.01)
	batch_size := 3
	num_epoch := 3

	path := "MINST"

	m := Train(path, learning_rate, num_epoch, batch_size)
	Train(path, learning_rate, num_epoch, batch_size)

	//finish training
	fmt.Println("End training process")

	//evaluation process
	fmt.Println("Start evaluation process to recognize the MNIST data set, pending...")
	Accuracy := Eval(path, batch_size, *m)
	fmt.Println("Accuracy of the MNIST CNN is:", Accuracy)
	fmt.Println("End evaluation process")

	/*
	// Training our second model!
	pathBrainTumor := "Brain_tumor_modified"
	fmt.Println("Start brain tumor model training process, pending...")
	mBrainTumor := TrainBrainTumor(pathBrainTumor, learning_rate, num_epoch, batch_size)
	fmt.Println("End brain tumor model training")

	// Evaluation of our second model!
	fmt.Println("Start brain tumor model evaluation process, pendig...")
	AccuracyBrainTumor := EvaluateBrainTumor(pathBrainTumor, batch_size, *mBrainTumor)
	fmt.Println("Accuracy of the Brain Tumor CNN is:", AccuracyBrainTumor)
	fmt.Println("End brain tumor model training process")
	*/

}
