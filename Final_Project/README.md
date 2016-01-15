Digital Photogrammetry Final Project
==========

##Description
This project contains the following programs:  
- **distortionProcessing:** Analyze result of camera calibration, show symmetric and decentering distortion distribution plots.
- **SIFTMatching:** Detect conjugate points with SIFT and brute force matching methods.
- **LSMatching:** Refine the result from SIFT method with least square matching.
- **showImgPair:** Display the matching result of SIFT or LSM method.
- **pixel2fiducial:** Transform image point (row, col) to fiducial coordinate system.
- **relativeOrientation:** Solve relative orientation with given matching points.
- **RO_for_PyPy:** An automatic batch script and relativeOrientation for PyPy interpreter.
- **3Dconf:** Perform 3D conformal transformation to link independent models.
- **show3D:** Display 3D object points in arbitrary model.
- **measureCP:** A program used to measure common pass point between two independent models.
- **autoProcess1:** A script which perform the following process automatically: SIFTMatching->LSMatching->pixel2fiducial.
- **autoProcess2:** Combine all independent models together with 3D conformal transformation automatically.

Testing photos are in the folder 'photos'.
