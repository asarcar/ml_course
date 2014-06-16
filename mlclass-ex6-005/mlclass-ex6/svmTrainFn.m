function [modelsvm] = svmTrainFn(X, Y, SvmType, KernelType, CostVal, GammaVal)
  % Wrapper fn on libsvm calls with vanilla matrix/vector arguments
  %
  % ---------------------------------------------------------------------------
  % svmtrain options
  % Usage: model = svmtrain(train_label_v, train_instance_m, 'libsvm_options');
  % libsvm_options:
  % -s svm_type : set type of SVM (default 0)
  % 	0 -- C-SVC		(multi-class classification)
  % 	1 -- nu-SVC		(multi-class classification)
  % 	2 -- one-class SVM
  % 	3 -- epsilon-SVR	(regression)
  % 	4 -- nu-SVR		(regression)
  % -t kernel_type : set type of kernel function (default 2)
  % 	0 -- linear: u'*v
  % 	1 -- polynomial: (gamma*u'*v + coef0)^degree
  % 	2 -- radial basis function: exp(-gamma*|u-v|^2)
  % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
  % 	4 -- precomputed kernel (kernel values in training_instance_matrix)
  % -g gamma : set gamma in kernel function (default 1/num_features)
  % -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
  % ---------------------------------------------------------------------------
  TrainFeatures = sparse(X);
  % libsvm recognizes labels as +1 and -1 
  TrainLabels = Y; TrainLabels(find(TrainLabels == 0)) = -1;
  TrainLabels = sparse(TrainLabels);
  
  % libsvmwrite('filename.txt', ys, Xs);
  % [TrainLabels, TrainFeatures] = libsvmread('filename.txt');
  
  svmoptstr = sprintf("-s %d -t %d -c %f -g %f",                    \
		      SvmType, KernelType, CostVal, GammaVal);

  modelsvm = svmtrain(TrainLabels, TrainFeatures, svmoptstr);

  % Populate model such that visualize routine works well
  % model.w = modelsvm.SVs'*modelsvm.sv_coef;
  % model.b = -modelsvm.rho;
  % if (modelsvm.Label(1) == -1)
  %  model.w = -model.w;
  %  model.b = -model.b;
  % end
end
