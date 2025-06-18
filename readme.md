# Automatic Semaphor Recognition System

## pose_2d_recognizer.py
Graphical tool that detects Semaphore letters based on 2D pose detection.

## collector
Graphical tool that detects Semaphore letters based on 3D pose detection. Can also save frames for easy data collection.

Call ```collector.py``` for graphical feedback of which letter is detected.

Call ```collector.py --save-data``` for saving the detected images in corresponding folder, saving only starts after letter was held for one second.

Call ```collector.py --save-data --save-letter X``` for saving all images in the X folder, saving only starts after X was held for one second.