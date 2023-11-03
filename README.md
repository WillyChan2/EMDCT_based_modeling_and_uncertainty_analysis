# EMDCT_based_modeling_and_uncertainty_analysis

- Name of the code/library: EMDCT_based_modeling_and_uncertainty_analysis
- Contact: ccw2018@tju.edu.cn
- Hardware requirements: Intel(R) Core(TM) i7-7700 CPU, 12.0G RAM 
- Program language: Python
- Software required: Windows 10
- Program size: 264K

## File specification

### 0. data folder

The data folder contains a  part of the main example data and the outputs. Among them:

- Folder raw_borehole_data contains the example borehole data; 

- Folder dataset_example contains the example training sets, test set, and Delaunay-triangle vertex; 

- Folder EMDCT_result contains the final result.

The scripts contain 3 parts, including the pre-processing_on_investigation_data, main_process, and post-processing parts:

### 1. pre-processing_on_investigation_data folder
   
   - The GenerateBorehole_train1.py is used to generate sample points as a training set that consists of real boreholes. Given a specific real borehole location to sample, and label them. The format of the borehole must be the same as that of file data/raw_borehole_data/borehole6.csv. The training set looks like file data/raw_borehole_data/points_train_Boreholes6.csv
   
   - The GenerateBorehole_train2.py is used to generate sample points as a training set that consists of virtual boreholes. The drilling position is randomly selected, and virtual boreholes are taken to verify the approach in the paper.
   
   - The GenerateGrid_test.py is used to generate scattered points as points of unknown strata (test set). It includes the test set of the overall work area and the test set of the dam site area.

   - The PreparaData.py is used to generate training set and test set files.

   - The RhinoLabel.py is used to read the positions of the point geometries in the Rhino software.

### 2. main_process folder
   
   - The ModelingWorkFlow_EMDCT.py is the ensemble modeling workflow with the divide-and-conquer tactic. The result is in the folder EMDCT_result.
   
   - The CatBoost.py is common implicit modeling with CatBoost.
   
   - The DecisionTree.py is common implicit modeling with DecisionTree.
   
   - The DeepForest.py is common implicit modeling with DeepForest.
   
   - The KNN.py is common implicit modeling with KNN.
   
   - The RandomForest.py is common implicit modeling with RandomForest.
   
   - The SVM.py is common implicit modeling with SVM.
   
   - The XGB.py is common implicit modeling with XGBoost.
   
### 3. post-processing folder
   
   - The GeoVisualization.py is used to visualize stratigraphy or information entropy.
   
   - The PostWorkFlow.py is used to extract scattered points at the interface of strata.
   
   - The ScatterSupplement.py is used to increase the dispersion at the formation interface.

   - The Similarity.py is used to calculate root mean square error and correlation coefficient at the interface.

## Brief usage

- 1) One should write the drilling information as data/raw_borehole_data/borehole6.csv, and run GenerateBorehole_train1.py.

- 2) Run the GenerateGrid_test.py to generate the test set.

- 3) Run the ModelingWorkFlow_EMDCT.py to learn the distribution of the sample points, and build models with scattered points. Each scatter carries the corresponding underlying information and information entropy information.

- 4) Run the GeoVisualization.py to analysis on the spatial distribution of the information entropy. Analysis on the spatial distribution of the information entropy. Artificially adding boreholes where information entropy is high.

- 5)  Increased modeling accuracy after adding boreholes. Run ScatterSupplement.py and PostWorkFlow.py to obtain Dense scattered points at each interface. 

- 6) Using scattered points at the interface, curved surfaces and solids are formed in the Rhino.

## Notice

- The authors do not have permission to share data, so the data provided is only intended to show the data format at each step of the implicit modeling approach presented in this paper.

- The zip files is encrypted. Please contact Willy Chan to get the password.
