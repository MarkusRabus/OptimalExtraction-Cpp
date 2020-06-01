#include "Marsh.h"

namespace py = pybind11;

Eigen::MatrixXd ObtainP(Eigen::MatrixXd &inputImage, 
                          Eigen::MatrixXd &imageMask, 
                          Eigen::VectorXd &traceCenter,
                          int aperture,
                          double polySpacing,
                          int polyDegree,
                          double RON,
                          double GAIN,
                          double RejectSigma) 
{

  //size_t fitWidth {0}; //fitWidth goes over i =  columns

  colsInput = inputImage.cols(); //80
  rowsInput = inputImage.rows(); //500 nlam

  npoly = (int)(2*(int)((aperture/polySpacing)+0.5)+1);
  //2 * (int)((2.0 * aperture) / polySpacing / 2.) + 1;
  C_const = -1 * ( polySpacing * (1.0 + (( (double)npoly - 1.0 ) / 2.0 )) );

  Eigen::MatrixXd polyCenters(rowsInput,(size_t)npoly);
  Eigen::VectorXd minAperture(rowsInput);
  Eigen::VectorXd maxAperture(rowsInput);
  Eigen::MatrixXd imageVariance(rowsInput, colsInput);
  Eigen::VectorXd standardSpectrum(rowsInput);
  Eigen::MatrixXd JMatrix( rowsInput, (size_t)(2 * (polyDegree+1) - 1) );
  Eigen::MatrixXd QMatrix( npoly, colsInput * rowsInput );
  Eigen::MatrixXd invEvariance(rowsInput, colsInput);
  Eigen::MatrixXd weightedE(rowsInput, colsInput);
  Eigen::VectorXd XVector( (polyDegree+1)* (size_t)npoly);
  Eigen::MatrixXd CMatrix( (polyDegree+1) * (size_t)npoly,(polyDegree+1) * (size_t)npoly );
  Eigen::VectorXd Bsoln;
  Eigen::MatrixXd Acoeffs( (size_t)npoly, (polyDegree+1) );
  Eigen::MatrixXd GkjMatrix((size_t)npoly, rowsInput);
  Eigen::MatrixXd PMatrix(rowsInput, colsInput);
  Eigen::MatrixXd newSpectrum(rowsInput, colsInput);
  Eigen::MatrixXd newImageVariance(rowsInput, colsInput);
  

  for (size_t id_j = 0; id_j < rowsInput; id_j++){
    for (size_t id_k = 0; id_k < (size_t)npoly; id_k++){
      polyCenters(id_j,id_k) = traceCenter(id_j) + 
                              (((-npoly/2+ 1) + (double)id_k) * polySpacing) - polySpacing/2;
    }
  }


  for (size_t id_j {0}; id_j < rowsInput; id_j++){
      minAperture(id_j) = traceCenter(id_j) + C_const + polySpacing + 1.0; // min for row
      maxAperture(id_j) = traceCenter(id_j) + C_const + (polySpacing*npoly) ;
  }

  
  calculateImageVariance(imageVariance, inputImage, imageMask, GAIN, RON);


  
  calculateStandardSpectrum(standardSpectrum, inputImage, imageMask, 
                              minAperture, maxAperture);


  
  calculateJmatrix(JMatrix);


  
  calculateQmatrix( QMatrix, polyCenters, minAperture, maxAperture, 
                    polySpacing, rowsInput, colsInput, npoly );


  do{

    
    calculateInvEvariance(invEvariance, imageVariance, imageMask, standardSpectrum, 
                          minAperture, maxAperture);


    
    calculateWeightedE(weightedE, inputImage, imageMask, imageVariance, standardSpectrum,
                        minAperture, maxAperture);


    
    calculateXVector(XVector, imageMask, weightedE, QMatrix, JMatrix, 
                      minAperture, maxAperture, polyDegree, npoly);


    
    calculateCmatrix(CMatrix, QMatrix, invEvariance, JMatrix, imageMask, 
                      minAperture, maxAperture, polyDegree, npoly);


    
    Bsoln = CMatrix.ldlt().solve(XVector);


    
    calculateAcoeffs(Acoeffs, Bsoln, polyDegree, npoly);


    
    calculateGkjMatrix(GkjMatrix, imageMask, Acoeffs, JMatrix, polyDegree);


    
    calculatePMatrix(PMatrix, imageMask, QMatrix, GkjMatrix, minAperture, maxAperture, npoly);


    normalizeP(PMatrix, minAperture, maxAperture);


    
    calculateNewSpectrum(newSpectrum, PMatrix, standardSpectrum, 
                          minAperture, maxAperture, rowsInput, colsInput);

    
    calculateImageVariance(newImageVariance, newSpectrum, imageMask, GAIN, RON);

    nrBadPixels = outlierRejection( imageMask, inputImage, newSpectrum, newImageVariance, 
                                    minAperture, maxAperture, RejectSigma);

    nrIterations += 1;

    std::cout << "Iteration " << nrIterations << " finished! \n";
    std::cout << "Found " << nrBadPixels << " bad pixels\n";

  }
  while(nrBadPixels > 0); 


  return PMatrix;
}






/*







*/







Eigen::MatrixXd getSpectrum(Eigen::MatrixXd &inputImage, 
                          Eigen::MatrixXd &imageMask, 
                          Eigen::MatrixXd &PMatrix, 
                          Eigen::VectorXd &traceCenter,
                          int aperture,
                          double polySpacing,
                          int polyDegree,
                          double RON,
                          double GAIN,
                          double cosmicSigma) 
{

  //size_t fitWidth {0}; //fitWidth goes over i =  columns

  colsInput = inputImage.cols(); //80
  rowsInput = inputImage.rows(); //500 nlam

  npoly = (int)(2*(int)((aperture/polySpacing)+0.5)+1);
  //2 * (int)((2.0 * aperture) / polySpacing / 2.) + 1;
  C_const = -1 * ( polySpacing * (1.0 + (( (double)npoly - 1.0 ) / 2.0 )) );

  Eigen::MatrixXd polyCenters(rowsInput,(size_t)npoly);
  Eigen::VectorXd minAperture(rowsInput);
  Eigen::VectorXd maxAperture(rowsInput);
  Eigen::MatrixXd imageVariance(rowsInput, colsInput);
  Eigen::VectorXd standardSpectrum(rowsInput);
  Eigen::VectorXd weightNormFactor(rowsInput);
  Eigen::MatrixXd normalizedWeight(rowsInput, colsInput);
  Eigen::VectorXd weightedFlux(rowsInput);
  Eigen::VectorXd weightedVariance(rowsInput);

  for (size_t id_j = 0; id_j < rowsInput; id_j++){
    for (size_t id_k = 0; id_k < (size_t)npoly; id_k++){
      polyCenters(id_j,id_k) = traceCenter(id_j) + 
                              (((-npoly/2+ 1) + (double)id_k) * polySpacing) - polySpacing/2;
    }
  }


  for (size_t id_j {0}; id_j < rowsInput; id_j++){
      minAperture(id_j) = traceCenter(id_j) + C_const + polySpacing + 1.0; // min for row
      maxAperture(id_j) = traceCenter(id_j) + C_const + (polySpacing*npoly) ;
  }


  calculateStandardSpectrum(standardSpectrum, inputImage, imageMask, 
                              minAperture, maxAperture);

  do{

    calculateImageVariance(imageVariance, inputImage, imageMask, GAIN, RON);

    calculateWNormFactor(weightNormFactor, imageVariance, imageMask, 
                          PMatrix, minAperture, maxAperture);

    calculateWeight(normalizedWeight, weightNormFactor, imageVariance, imageMask, 
                    PMatrix, minAperture, maxAperture);

    calculateWeightedFlux(weightedFlux, normalizedWeight, inputImage, imageMask, 
                          minAperture, maxAperture);

    calculateWeightedVar(weightedVariance, normalizedWeight, imageVariance, imageMask, 
                          minAperture, maxAperture);

    nrBadPixels = CosmicRayRejection( imageMask, inputImage, imageVariance,
                                      PMatrix, weightedFlux, weightedVariance, 
                                      minAperture, maxAperture, cosmicSigma);

    nrIterations += 1;

    std::cout << "Iteration " << nrIterations << " finished! \n";

  }
  while(nrBadPixels > 0); 

  Eigen::MatrixXd Spectrum(3, rowsInput);
  for(size_t id_j {0}; id_j < rowsInput; id_j++){
    Spectrum(0, id_j) = (double)id_j;
    Spectrum(1, id_j) = weightedFlux(id_j);
    Spectrum(2, id_j) = (double)1. / weightedVariance(id_j);
  }
  
  return Spectrum;

}



PYBIND11_MODULE(Marsh, m) {
   m.doc() = "Utities to extract spectra"; // optional module docstring
   m.def("ObtainP", &ObtainP, 
            "Method which returns the spatial light fractions.");
   m.def("getSpectrum", &getSpectrum, 
            "Method which returns the optimal extracted spectrum.");
}
