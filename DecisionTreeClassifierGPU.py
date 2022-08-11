import numpy as np
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as drv
from math import ceil

class Node():
    def __init__(self,feature_index = None,threshold=None,left = None,right = None, info_gain=None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifierGPU():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.source = source
        
    def build_tree(self, dataset, stream ,curr_depth=0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features,stream)
            if best_split["info_gain"]>0:
                left_stream = drv.Stream()
                left_subtree = self.build_tree(best_split["dataset_left"], left_stream,curr_depth+1)
                right_stream = drv.Stream()
                right_subtree = self.build_tree(best_split["dataset_right"], right_stream,curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features ,stream):
        X, Y = dataset[:,:-1], dataset[:,-1]
        get_best_gain = self.mod.get_function("get_best_gain")
        X_gpu = gpuarray.to_gpu_async(X.astype(np.float32),stream = stream)
        Y_gpu = gpuarray.to_gpu_async(Y.astype(np.int32),stream = stream)
        num_samples_int32 = np.int32(num_samples)
        RES = np.zeros(X.shape,dtype = np.float32)
        RES = gpuarray.to_gpu_async(RES,stream = stream)
        if num_features*num_samples<=1024:
            block = (num_features,num_samples,1)
            grid = (1,1)
        elif num_features>1024:
            block = (1024,1,1)
            grid = (ceil(num_features/1024),num_samples)
        elif num_features<=1024:
            block = (num_features,1024//num_features,1)
            grid = (1,ceil(num_samples/block[1]))
        get_best_gain(X_gpu,num_samples_int32,Y_gpu,RES,block=block,grid=grid,stream = stream)
        res = RES.get_async(stream = stream)
        best = res.max()
        best_index = np.where(res==best)
        best_feature = int(best_index[1][0])
        best_threshold_index = int(best_index[0][0])
        best_threshold = X[best_threshold_index,best_feature]
        dataset_left, dataset_right = self.split(dataset, best_feature, best_threshold)
        best_split = {}
        best_split["feature_index"] = best_feature
        best_split["threshold"] = best_threshold
        best_split["dataset_left"] = dataset_left
        best_split["dataset_right"] = dataset_right
        best_split["info_gain"] = best
        return best_split
          
    def split(self, dataset, feature_index, threshold):
        dataset_left = dataset[dataset[:,feature_index]<=threshold]
        dataset_right = dataset[dataset[:,feature_index]>threshold]
        return dataset_left, dataset_right
        
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        num_classes = len(np.unique(Y))
        num_features = np.shape(X)[1]
        self.source = self.source % {"num_classes":num_classes,"num_features":num_features}
        self.mod = SourceModule(self.source)
        dataset = np.concatenate((X, Y), axis=1)
        stream = drv.Stream()
        self.root = self.build_tree(dataset,stream)
    
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

source = """
__device__ const int num_classes = %(num_classes)s;
__device__ const int num_features = %(num_features)s;

__device__ float gini_index(int *Y,int num_samples,int size,int *left_right,int value)
{
    if(size==0)
        return 0;
    float gini = 0;
    int classes_counter[num_classes];
    for(int i=0;i<num_classes;i++)
        classes_counter[i] = 0;
    for(int i=0;i<num_samples;i++)
    {
        int index = Y[i];
        if(left_right[i] == value || value==-1)
        {
            classes_counter[index]++;
        }
    }
    for(int i=0;i<num_classes;i++)
    {
        float p = (float)classes_counter[i]/size;
        gini+= p*p;
    }
    gini = 1-gini;
    return gini;
}

__device__ int* split(float *features_values,int num_samples,float threshold,int col,int *size_left,int *size_right)
{
    int *left_right=(int *)malloc(num_samples*sizeof(int));
    for(int i=0;i<num_samples;i++)
    {
        if(features_values[i*num_features+col]>threshold)
        {
            left_right[i] = 1;
            (*size_right)++;
        }
        else
        {
            left_right[i] = 0;
            (*size_left)++;
        }
    }
    return left_right;
}

__global__ void get_best_gain(float *features_values,int num_samples,int *Y,float *RES)
{
    const int row = threadIdx.y + blockIdx.y*num_samples;
    const int col = threadIdx.x + blockIdx.x*num_features;
    if(row>=num_samples || col>=num_features)
        return;
    const int idx = row*num_features + col;
    const float threshold = features_values[idx];
    int size_left = 0;
    int size_right = 0;
    int *left_right = split(features_values,num_samples,threshold,col,&size_left,&size_right);
    float gini_parent = gini_index(Y,num_samples,num_samples,left_right,-1);
    float gini_left = gini_index(Y,num_samples,size_left,left_right,0);
    float gini_right = gini_index(Y,num_samples,size_right,left_right,1);
    float weight_left = (float)size_left/num_samples;
    float weight_right = (float)size_right/num_samples;
    float gain = gini_parent - (gini_left*weight_left + gini_right*weight_right);
    RES[idx]=gain;
    free(left_right);
}
"""