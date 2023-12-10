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
	num_epoch := 20

	// path := "MINST"

	// m, lossArr, accuracyArr, step := Train(path, learning_rate, num_epoch, batch_size)

	// //finish training
	// fmt.Println("End training process")

	// // store lossArr and accuracyArr
	// fmt.Println("Store lossArr and accuracyArr")
	// Store2DArray("lossArr", lossArr, step)
	// Store2DArray("accuracyArr", accuracyArr, step)
	// fmt.Println("Store lossArr and accuracyArr done")

	

	// //evaluation process
	// fmt.Println("Start evaluation process to recognize the MNIST data set, pending...")
	// Accuracy := Eval(path, batch_size, *m)
	// fmt.Println("Accuracy of the MNIST CNN is:", Accuracy)
	// fmt.Println("End evaluation process")


	// Training our second model!
	pathBrainTumor := "Brain_tumor_modified"
	fmt.Println("Start brain tumor model training process, pending...")
	mBrainTumor, lossArrBrainTumor, accuracyArrBrainTumor, stepBrainTumor := TrainBrainTumor(pathBrainTumor, learning_rate, num_epoch, batch_size)
	fmt.Println("End brain tumor model training")

	// store lossArr and accuracyArr
	fmt.Println("Store lossArr and accuracyArr for brain tumor model")
	Store2DArray("lossArr_Brain", lossArrBrainTumor, stepBrainTumor)
	Store2DArray("accuracyArr_Brain", accuracyArrBrainTumor, stepBrainTumor)
	fmt.Println("Store lossArr and accuracyArr done for brain tumor model")

	// Evaluation of our second model!
	fmt.Println("Start brain tumor model evaluation process, pending...")
	AccuracyBrainTumor := EvaluateBrainTumor(pathBrainTumor, batch_size, *mBrainTumor)
	fmt.Println("Accuracy of the Brain Tumor CNN is:", AccuracyBrainTumor)
	fmt.Println("End brain tumor model training process")

}
