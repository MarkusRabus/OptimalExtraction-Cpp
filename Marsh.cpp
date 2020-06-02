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

  colsInput = inputImage.cols(); 
  rowsInput = inputImage.rows(); 

  npoly = (int)(2*(int)((aperture/polySpacing)+0.5)+1);
  //2 * (int)((2.0 * aperture) / polySpacing / 2.) + 1;
  C_const = -1 * ( polySpacing * (1.0 + (( (double)npoly - 1.0 ) / 2.0 )) );

  // Define Matrices used for computation:

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
  Eigen::MatrixXd CMatrix( (polyDegree+1) * (size_t)npoly,
                           (polyDegree+1) * (size_t)npoly );
  Eigen::VectorXd Bsoln;
  Eigen::MatrixXd Acoeffs( (size_t)npoly, (polyDegree+1) );
  Eigen::MatrixXd GkjMatrix((size_t)npoly, rowsInput);
  Eigen::MatrixXd PMatrix(rowsInput, colsInput);
  Eigen::MatrixXd newSpectrum(rowsInput, colsInput);
  Eigen::MatrixXd newImageVariance(rowsInput, colsInput);
  


  //create an array of trace centers equation (9) in Marsh et al. (1989)
  for (size_t id_j = 0; id_j < rowsInput; id_j++){
    for (size_t id_k = 0; id_k < (size_t)npoly; id_k++){
      polyCenters(id_j,id_k) = traceCenter(id_j) + 
              (((-npoly/2+ 1) + (double)id_k) * polySpacing) - polySpacing/2;
    }
  }


  for (size_t id_j {0}; id_j < rowsInput; id_j++){
      minAperture(id_j) = traceCenter(id_j) + C_const + polySpacing + 1.0; 
      maxAperture(id_j) = traceCenter(id_j) + C_const + (polySpacing*npoly) ;
  }


  // Calculate image variance Vij  
  calculateImageVariance(imageVariance, inputImage, imageMask, GAIN, RON);


  // Calculate the first estimate Sum_i Dij
  calculateStandardSpectrum(standardSpectrum, inputImage, imageMask, 
                              minAperture, maxAperture);

  // Calculate Ji
  calculateJmatrix(JMatrix);

  // Q matrix, compare equations (6) and (11) in Marsh et al. 1989
  calculateQmatrix( QMatrix, polyCenters, minAperture, maxAperture, 
                    polySpacing, rowsInput, colsInput, npoly );


  // Start the outlier rejection iteration

  do{

    calculateInvEvariance(invEvariance, imageVariance, imageMask, 
                          standardSpectrum, minAperture, maxAperture);

    calculateWeightedE( weightedE, inputImage, imageMask, imageVariance, 
                        standardSpectrum, minAperture, maxAperture);

    calculateXVector(XVector, imageMask, weightedE, QMatrix, JMatrix, 
                      minAperture, maxAperture, polyDegree, npoly);

    calculateCmatrix(CMatrix, QMatrix, invEvariance, JMatrix, imageMask, 
                      minAperture, maxAperture, polyDegree, npoly);

    // Solve linear system C_qp * B_q = X_q
    Bsoln = CMatrix.ldlt().solve(XVector);
    
    // reformat solution of lineas system to fit A_nk
    calculateAcoeffs(Acoeffs, Bsoln, polyDegree, npoly);

    calculateGkjMatrix(GkjMatrix, imageMask, Acoeffs, JMatrix, polyDegree);
    
    calculatePMatrix( PMatrix, imageMask, QMatrix, GkjMatrix, 
                      minAperture, maxAperture, npoly);

    normalizeP(PMatrix, minAperture, maxAperture);

    calculateNewSpectrum(newSpectrum, PMatrix, standardSpectrum, 
                          minAperture, maxAperture, rowsInput, colsInput);

    
    calculateImageVariance(newImageVariance, newSpectrum, imageMask, GAIN, RON);

    nrBadPixels = outlierRejection( imageMask, inputImage, newSpectrum, 
                                    newImageVariance, minAperture, maxAperture, 
                                    RejectSigma);

    nrIterations += 1;

    std::cout << "Iteration " << nrIterations << " finished! \n";
    std::cout << "Found " << nrBadPixels << " bad pixels\n";

  }
  // iterate through while rejected pixels are found
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


  colsInput = inputImage.cols(); //
  rowsInput = inputImage.rows(); // nlam

  npoly = (int)(2*(int)((aperture/polySpacing)+0.5)+1);
  //2 * (int)((2.0 * aperture) / polySpacing / 2.) + 1;
  C_const = -1 * ( polySpacing * (1.0 + (( (double)npoly - 1.0 ) / 2.0 )) );

  // Define Matrices used for computation:

  Eigen::MatrixXd polyCenters(rowsInput,(size_t)npoly);
  Eigen::VectorXd minAperture(rowsInput);
  Eigen::VectorXd maxAperture(rowsInput);
  Eigen::MatrixXd imageVariance(rowsInput, colsInput);
  Eigen::VectorXd standardSpectrum(rowsInput);
  Eigen::VectorXd weightNormFactor(rowsInput);
  Eigen::MatrixXd normalizedWeight(rowsInput, colsInput);
  Eigen::VectorXd weightedFlux(rowsInput);
  Eigen::VectorXd weightedVariance(rowsInput);

  //create an array of trace centers equation (9) in Marsh et al. (1989)
  for (size_t id_j = 0; id_j < rowsInput; id_j++){
    for (size_t id_k = 0; id_k < (size_t)npoly; id_k++){
      polyCenters(id_j,id_k) = traceCenter(id_j) + 
                (((-npoly/2+ 1) + (double)id_k) * polySpacing) - polySpacing/2;
    }
  }

  for (size_t id_j {0}; id_j < rowsInput; id_j++){
      minAperture(id_j) = traceCenter(id_j) + C_const + polySpacing + 1.0; 
      maxAperture(id_j) = traceCenter(id_j) + C_const + (polySpacing*npoly) ;
  }

  calculateStandardSpectrum(standardSpectrum, inputImage, imageMask, 
                              minAperture, maxAperture);

  // Start the iterative cosmic ray rejection.
  do{

    calculateImageVariance(imageVariance, inputImage, imageMask, GAIN, RON);

    calculateWNormFactor(weightNormFactor, imageVariance, imageMask, 
                          PMatrix, minAperture, maxAperture);

    calculateWeight(normalizedWeight, weightNormFactor, imageVariance, 
                    imageMask, PMatrix, minAperture, maxAperture);

    calculateWeightedFlux(weightedFlux, normalizedWeight, inputImage, imageMask, 
                          minAperture, maxAperture);

    calculateWeightedVar( weightedVariance, normalizedWeight, imageVariance, 
                          imageMask, minAperture, maxAperture);

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
            "Method which returns the spatial light fractions. We assume for 
            all Matrices: rows are the disperion direction and columns are
            the spatial direction.");
   m.def("getSpectrum", &getSpectrum, 
            "Method which returns the optimal extracted spectrum.We assume for 
            all Matrices: rows are the disperion direction and columns are
            the spatial direction.");
}
