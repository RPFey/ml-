# SVM OPENCV

## 基本操作

用的时候在装载上琢磨了好久

python :

```python
svm = cv2.ml.SVM_load(cv2.samples.findFile("...."))
```

cpp :

```c++
Ptr<SVM> svm;
svm = ml::SVM::load("...")
```

这样基本上就可以读入 .xml 的模型

然后在与 HOG 结合时，HOG 只接受参数：

```c++
vector<float> get_svm_detector(const Ptr<ml::SVM> &svm) {
		// get the support vectors
		Mat sv = svm->getSupportVectors();
		const int sv_total = sv.rows;
		cout << sv_total;
		// get the decision function
		Mat alpha, svidx;
		double rho = svm->getDecisionFunction(0, alpha, svidx);
		CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
		CV_Assert(
				(alpha.type() == CV_64F && alpha.at<double>(0) == 1.) || (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
		CV_Assert(sv.type() == CV_32F);
		vector<float> hog_detector(sv.cols + 1);
		memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
		hog_detector[sv.cols] = (float) -rho;
		return hog_detector;
	}

// 设置 hog descriptor
HOGDescriptor hog(Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9);
hog.setSVMDetector(get_svm_detector(svm));
// 一定要初始化 hog 的参数，否则载入svm 时大小匹配
```

## 训练

论文： histogram of oriented gradients for human detection

\1. hard example :

首先训练一个detector，在负样本图片上（图片中没有目标物）上测试。如果detector 预测出矩形框，则将这些矩形框中的图片作为负样本加入到训练集中。

