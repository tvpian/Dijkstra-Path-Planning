# Dijkstra Path Planning for Point Robots
Name: Tharun Vadakke Puthanveettil

UID: 119069516

Course:  ENPM661 - Planning for Autonomous Robots

## Installing the dependencies

*To install Matplotlib*
```
pip install matplotlib
```
*To install Pandas*
```
pip install pandas
```
*To install Numpy*
```
pip install numpy
```
*To install OpenCV*
```
pip install opencv-python
```

## Running the code
```
cd proj2_tharun_puthanveettil/code
python dijkstra_tharun_puthanveettil.py
```
Note:
Give the input in the following format:
```
Starting state: x_i,y_i Note: No spaces between x_i,y_i and brackets
Goal state: x_g,y_g Note: No spaces between x_g,y_g and brackets
```
Note: Due to resolution issue in the visuallization tools, kindly be cautious while giving extreme cases as inputs. The code might run in cases where the start node is the wall beacuase of the resolution issue even though its not expected to be the case.

## Outputs
The output files - Final.avi will be stored in the folder proj2_tharun_puthanveettil/code


## Test Case Results - To show successfull implementation of Searching and Backtracking
The video result for a random case: [200,200] stored in the folder proj2_tharun_puthanveettil/results
https://github.com/tvpian/Dijkstra-Path-Planning/assets/41953267/433f63dc-c6d7-4bb6-9023-a8b15a3c7a76

## References
- https://github.com/WuStangDan/pathplanning
