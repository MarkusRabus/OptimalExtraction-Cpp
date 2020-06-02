#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <math.h>
#include <Eigen/Eigen>


double C_const {0}; 
int nrBadPixels {0};
int nrIterations {0};
int npoly {0};

size_t colsInput {0}; 
size_t rowsInput {0}; 

void calculateImageVariance( Eigen::MatrixXd &imageVariance,
                             const Eigen::MatrixXd &inputImage, 
                             const Eigen::MatrixXd &maskImage, 
                             const Eigen::VectorXd &minAperture, 
                             const Eigen::VectorXd &maxAperture,
                             const double GAIN, const double READNOISE);

void calculateStandardSpectrum( Eigen::VectorXd &standardSpectrum,
                                const Eigen::MatrixXd &inputImage, 
                                const Eigen::MatrixXd &imageMask, 
                                const Eigen::VectorXd &minAperture, 
                                const Eigen::VectorXd &maxAperture);

void calculateInvEvariance( Eigen::MatrixXd &invEvariance,
                            const Eigen::MatrixXd &imageVariance, 
                            const Eigen::MatrixXd &imageMask, 
                            const Eigen::VectorXd &standardSpectrum,
                            const Eigen::VectorXd &minAperture, 
                            const Eigen::VectorXd &maxAperture);

void calculateWeightedE( Eigen::MatrixXd &weightedE,
                         const Eigen::MatrixXd &inputImage, 
                         const Eigen::MatrixXd &imageMask, 
                         const Eigen::MatrixXd &imageVariance, 
                         const Eigen::VectorXd &standardSpectrum,
                         const Eigen::VectorXd &minAperture, 
                         const Eigen::VectorXd &maxAperture);

void calculateJmatrix(Eigen::MatrixXd &JMatrix);

void calculateQmatrix( Eigen::MatrixXd &QMatrix,
                       const Eigen::MatrixXd &polyCenters, 
                       const Eigen::VectorXd &minAperture, 
                       const Eigen::VectorXd &maxAperture,
                       double polySpacing, 
                       size_t rowsInput, 
                       size_t colsInput,
                       size_t npoly);

void calculateXVector(Eigen::VectorXd &XVector, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &weightedE, 
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &JMatrix, 
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t polyDegree, 
                      size_t npoly);

void calculateCmatrix(Eigen::MatrixXd &CMatrix,
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &invEvariance, 
                      const Eigen::MatrixXd &JMatrix, 
                      const Eigen::MatrixXd &imageMask,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t polyDegree, 
                      size_t npoly);

void calculateAcoeffs(Eigen::MatrixXd &Acoeffs,
                      const Eigen::VectorXd &Bsoln, 
                      size_t polyDegree, 
                      size_t npoly);

void calculateGkjMatrix(Eigen::MatrixXd &GkjMatrix, 
                        const Eigen::MatrixXd &imageMask, 
                        const Eigen::MatrixXd &Acoeffs, 
                        const Eigen::MatrixXd &JMatrix, 
                        size_t polyDegree);

void calculatePMatrix(Eigen::MatrixXd &PMatrix, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &GkjMatrix,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t npoly);

void normalizeP(Eigen::MatrixXd &PMatrix,
                const Eigen::VectorXd &minAperture, 
                const Eigen::VectorXd &maxAperture);

void calculateNewSpectrum(Eigen::MatrixXd &newSpectrum, 
                          const Eigen::MatrixXd &PMatrix, 
                          const Eigen::VectorXd &standardSpectrum,
                          const Eigen::VectorXd &minAperture, 
                          const Eigen::VectorXd &maxAperture,
                          size_t rowsInput, 
                          size_t colsInput );


int outlierRejection( Eigen::MatrixXd &imageMask,
                      const Eigen::MatrixXd &inputImage,
                      const Eigen::MatrixXd &newSpectrum,
                      const Eigen::MatrixXd &newImageVariance,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      double RejectSigma);

void calculateWNormFactor( Eigen::VectorXd &weightNormFactor, 
                           const Eigen::MatrixXd &imageVariance, 
                           const Eigen::MatrixXd &imageMask, 
                           const Eigen::MatrixXd &PMatrix,
                           const Eigen::VectorXd &minAperture, 
                           const Eigen::VectorXd &maxAperture);

void calculateWeight( Eigen::MatrixXd &normalizedWeight,
                      const Eigen::VectorXd &weightNormFactor, 
                      const Eigen::MatrixXd &imageVariance, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &PMatrix,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture);

void calculateWeightedFlux( Eigen::VectorXd &weightedFlux,
                            const Eigen::MatrixXd &normalizedWeight,
                            const Eigen::MatrixXd &inputImage, 
                            const Eigen::MatrixXd &imageMask, 
                            const Eigen::VectorXd &minAperture, 
                            const Eigen::VectorXd &maxAperture);

void calculateWeightedVar( Eigen::VectorXd &weightedVariance,
                      const Eigen::MatrixXd &normalizedWeight,
                      const Eigen::MatrixXd &imageVariance, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture);


int CosmicRayRejection( Eigen::MatrixXd &imageMask, 
                        const Eigen::MatrixXd &inputImage, 
                        const Eigen::MatrixXd &imageVariance,
                        const Eigen::MatrixXd &PMatrix,
                        const Eigen::VectorXd &weightedFlux, 
                        const Eigen::VectorXd &weightedVariance, 
                        const Eigen::VectorXd &minAperture, 
                        const Eigen::VectorXd &maxAperture,
                        float cosmicSigma);




/*




*/






void calculateImageVariance( Eigen::MatrixXd &imageVariance,
                             const Eigen::MatrixXd &inputImage, 
                             const Eigen::MatrixXd &maskImage, 
                             const Eigen::VectorXd &minAperture, 
                             const Eigen::VectorXd &maxAperture,                             
                             const double GAIN, const double READNOISE){

/*
Calculates the variance of an image: 
      D_ij/GAIN + (READNOISE/GAIN)**2
*/

  for(size_t id_j {0}; id_j < (size_t)inputImage.rows(); id_j++){
    //for (size_t id_i {0}; id_i < (size_t)inputImage.cols(); id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if ( maskImage(id_j, id_i) ){
        imageVariance(id_j,id_i) = std::abs(inputImage(id_j,id_i) / GAIN) 
                                   + std::pow(READNOISE / GAIN, 2.0);
      }
      else{
        // Many times variances is in the deonminator, to avoid division by 0
        imageVariance(id_j,id_i) = 1e-9;
      }
    }
  }
}




void calculateStandardSpectrum( Eigen::VectorXd &standardSpectrum,
                                const Eigen::MatrixXd &inputImage, 
                                const Eigen::MatrixXd &imageMask, 
                                const Eigen::VectorXd &minAperture, 
                                const Eigen::VectorXd &maxAperture){

/*
Simple sum over spatial direction for each wavelenth row.
Sum_i D_ij
*/

  double Sum {0};
  for(size_t id_j {0}; id_j < (size_t)inputImage.rows(); id_j++){
    Sum=0;
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if ( imageMask(id_j, id_i) ){
        Sum += inputImage(id_j,id_i);
      }
    }
    standardSpectrum(id_j) = Sum; 
  }
}


void calculateInvEvariance( Eigen::MatrixXd &invEvariance,
                            const Eigen::MatrixXd &imageVariance, 
                            const Eigen::MatrixXd &imageMask, 
                            const Eigen::VectorXd &standardSpectrum,
                            const Eigen::VectorXd &minAperture, 
                            const Eigen::VectorXd &maxAperture){

/*
Calculate the inverse of the variance of E_ij. 
E_ij is defined by equation (5) in Marsh (1989)
E_ij = D_ij/( Sum_i D_ij )

*/

  for(size_t id_j {0}; id_j < (size_t)imageVariance.rows(); id_j++){
    //for (size_t id_i {0}; id_i < (size_t)imageVariance.cols(); id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
        if ( imageMask(id_j, id_i) ){
          invEvariance(id_j, id_i) = std::pow(standardSpectrum(id_j),2) 
                                    / imageVariance(id_j, id_i);
        }
        else{
          invEvariance(id_j, id_i) = 0.0;
        }
    }
  }
}


void calculateWeightedE( Eigen::MatrixXd &weightedE,
                         const Eigen::MatrixXd &inputImage, 
                         const Eigen::MatrixXd &imageMask, 
                         const Eigen::MatrixXd &imageVariance, 
                         const Eigen::VectorXd &standardSpectrum,
                         const Eigen::VectorXd &minAperture, 
                         const Eigen::VectorXd &maxAperture){

/*
Weight E_ij with image variance.
E_ij is defined by equation (5) in Marsh (1989)
E_ij = D_ij/( Sum_i D_ij )
*/

  for(size_t id_j {0}; id_j < (size_t)inputImage.rows(); id_j++){
    //for (size_t id_i {0}; id_i < (size_t)inputImage.cols(); id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
        if ( imageMask(id_j, id_i) ){
          weightedE(id_j, id_i) = ( inputImage(id_j, id_i) 
            * standardSpectrum(id_j) ) / imageVariance(id_j, id_i);
        }
        else{
          weightedE(id_j, id_i) = 0.0;
        }
    }
  }
}


void calculateJmatrix(Eigen::MatrixXd &JMatrix){

/*
Calculates the J polynomial, last term in equation (8) Marsh (1989)

J = j**(n-1)

*/


  Eigen::VectorXd JNormalised(JMatrix.rows());
  // use normalized X coordinated to calculate the J polynomial.
  JNormalised =  Eigen::ArrayXd::LinSpaced(JMatrix.rows(),-1., 1.);

  for(size_t id_n {0}; id_n < (size_t)JMatrix.cols(); id_n++){
    for(size_t id_j {0}; id_j < (size_t)JMatrix.rows(); id_j++){
      JMatrix(id_j,id_n) = std::pow((double)JNormalised(id_j), id_n);
    }
  }

}


void calculateQmatrix( Eigen::MatrixXd &QMatrix,
                       const Eigen::MatrixXd &polyCenters,
                       const Eigen::VectorXd &minAperture, 
                       const Eigen::VectorXd &maxAperture,
                       double polySpacing, 
                       size_t rowsInput, 
                       size_t colsInput,
                       size_t npoly){

/*
Calculates the Q Matrix, by comparing equations (6) and (11) in 
Marsh (1989), we get for Q:

Q_kij = Sum_k max[ 0, min(S, (S+1)/2 - |x_kj - i|) ]

*/



  double minTerm {0};

  for (size_t id_k {0}; id_k < npoly; id_k++){
    for (size_t id_j {0}; id_j < rowsInput; id_j++){
      //for (size_t id_i {0}; id_i < colsInput; id_i++){
      for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
        minTerm = std::min((double)polySpacing,0.5 * (double)(polySpacing + 1) 
                  - std::abs(polyCenters(id_j,id_k) - id_i));
        QMatrix(id_k, id_i*rowsInput+id_j) = std::max( (double)0,minTerm );
      }
    }
  }
}



void calculateXVector(Eigen::VectorXd &XVector, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &weightedE, 
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &JMatrix, 
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t polyDegree, 
                      size_t npoly){

/*
Calculates the X vector, as defined in equation A3 (upper equation)
in Marsh (1989):

X_q = Sum_ij ( ( E_ij * Q_kij * j**(n-1) )/sigma_ij**2 )

*/


  double XSum {0};
  int XCounter {0};

  for (size_t id_n {1}; id_n <= polyDegree+1; id_n++){
    for (size_t id_k {1}; id_k <= npoly; id_k++){
      XSum = 0;
      for(size_t id_j {0}; id_j < (size_t)weightedE.rows(); id_j++){
        //for (size_t id_i {0}; id_i < (size_t)weightedE.cols(); id_i++){
        for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
          if( (QMatrix( (id_k-1), id_i * weightedE.rows()+id_j) != 0.0) 
              && ( imageMask(id_j, id_i) )) { 
            XSum +=  weightedE(id_j, id_i) * QMatrix( (id_k-1), id_i 
                      * weightedE.rows()+id_j) * JMatrix(id_j, (id_n - 1));
          }
        }
      }
      XVector(XCounter) = XSum;
      XCounter += 1;
    }
  }

}


void calculateCmatrix(Eigen::MatrixXd &CMatrix,
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &invEvariance, 
                      const Eigen::MatrixXd &JMatrix, 
                      const Eigen::MatrixXd &imageMask,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t polyDegree, 
                      size_t npoly){


/*
Calculates the C Matrix, as defined in equation A3 (lower equation)
in Marsh (1989):

C_qp = Sum_ij ( ( Q_kij * Q_lij * j**(n+m-2) )/sigma_ij**2 )

*/


  
  double CSum {0};
  double Qkij {0};
  double Qlij {0};
  size_t CCounter {0};

  for(size_t id_n {1}; id_n <= polyDegree+1; id_n++){
    for(size_t id_k {1}; id_k <= npoly; id_k++){
      for(size_t id_m {1}; id_m <= polyDegree+1; id_m++){
        for(size_t id_l {1}; id_l <= npoly; id_l++){

            CSum = 0;

            for(size_t id_j {0}; id_j < (size_t)invEvariance.rows(); id_j++){
              //for (size_t id_i {0}; id_i < (size_t)invEvariance.cols(); id_i++){
              for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){

                Qkij = QMatrix( (id_k-1), id_i*invEvariance.rows()+id_j);
                Qlij = QMatrix( (id_l-1), id_i*invEvariance.rows()+id_j);

                if( (Qkij != 0.0) && (Qlij != 0.0) && ( imageMask(id_j, id_i) ) ){  
                  CSum += ( Qkij * Qlij * JMatrix( id_j, (id_n + id_m - 2)) *
                            invEvariance(id_j, id_i) );
                }
              }
            }

          *(CMatrix.data() + CCounter) = CSum;
          CCounter += 1;

          }
      }
    }
  }


}


void calculateAcoeffs(Eigen::MatrixXd &Acoeffs,
                      const Eigen::VectorXd &Bsoln, 
                      size_t polyDegree, 
                      size_t npoly){

/*
Calculate A coefficient, which are the solution of the linear equation:
C_qp * B_q = X_q
and where
A_kn = B_q

*/


  int ACounter {0};

  for(size_t id_n {0}; id_n < polyDegree+1; id_n++){
    for(size_t id_k {0}; id_k < npoly; id_k++){
        Acoeffs(id_k,id_n)  = *(Bsoln.data() + ACounter);
        ACounter += 1;

      }
  }

}


void calculateGkjMatrix(Eigen::MatrixXd &GkjMatrix, 
                        const Eigen::MatrixXd &imageMask, 
                        const Eigen::MatrixXd &Acoeffs, 
                        const Eigen::MatrixXd &JMatrix, 
                        size_t polyDegree){

/*
Calculate polynomials G_kj equation (8) Marsh 1989:

G_kj = Sum_n ( A_nk * j**(n-1) )

*/


  double GkjSum {0};

  for(size_t id_k {0}; id_k < (size_t)GkjMatrix.rows(); id_k++){
    for(size_t id_j {0}; id_j < (size_t)GkjMatrix.cols(); id_j++){
        GkjSum = Acoeffs(id_k,0);
        for(size_t id_n = 1; id_n <= polyDegree; id_n++){
          GkjSum += Acoeffs(id_k,id_n) * JMatrix( id_j, (id_n) );
        }
        GkjMatrix(id_k,id_j) = GkjSum;
    }
  }

}


void calculatePMatrix(Eigen::MatrixXd &PMatrix, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &QMatrix, 
                      const Eigen::MatrixXd &GkjMatrix,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      size_t npoly){

/*
Calulcate profile P_ij equation (6) in Marsh (1989):

P_ij = Sum_k ( Q_kij * G_kj )

*/


  double PSum {0};
  
  for(size_t id_j {0}; id_j < (size_t)PMatrix.rows(); id_j++){
    //for (size_t id_i {0}; id_i < (size_t)PMatrix.cols(); id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      PSum = 0;
      for(size_t id_k {0}; id_k < npoly; id_k++){
        PSum += QMatrix( (id_k), id_i*PMatrix.rows() + id_j) 
                * GkjMatrix(id_k,id_j);
      }
      PMatrix(id_j,id_i) = PSum;
    }
  }

}



void normalizeP(Eigen::MatrixXd &PMatrix,
                const Eigen::VectorXd &minAperture, 
                const Eigen::VectorXd &maxAperture){

/*
Normalize profile P_ij equation (6) in Marsh (1989):

P_ij = P_ij/ Sum_k ( P_ij )

*/


  double NormSum {0};
  PMatrix = (PMatrix.array() < 0).select(0, PMatrix);

  for(size_t id_j {0}; id_j < (size_t)PMatrix.rows(); id_j++){
    NormSum=1e-9;
    //for (size_t id_i {0}; id_i < (size_t)PMatrix.cols(); id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      NormSum += PMatrix(id_j,id_i);
    }
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      PMatrix(id_j,id_i) = PMatrix(id_j,id_i) / NormSum;
    }
  }

}


void calculateNewSpectrum(Eigen::MatrixXd &newSpectrum, 
                          const Eigen::MatrixXd &PMatrix, 
                          const Eigen::VectorXd &standardSpectrum,
                          const Eigen::VectorXd &minAperture, 
                          const Eigen::VectorXd &maxAperture,
                          size_t rowsInput, 
                          size_t colsInput ){

/*
Get new estimate by weighting the standard Spectrum with spatial light fraction

*/

  for(size_t id_j {0}; id_j < rowsInput; id_j++){
    //for (size_t id_i {0}; id_i < colsInput; id_i++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
          newSpectrum(id_j, id_i) = PMatrix(id_j, id_i) 
                                    * standardSpectrum(id_j);
    }
  }

}


int outlierRejection( Eigen::MatrixXd &imageMask,
                      const Eigen::MatrixXd &inputImage,
                      const Eigen::MatrixXd &newSpectrum,
                      const Eigen::MatrixXd &newImageVariance,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture,
                      double RejectSigma){

/*
Outlier rejection, return the number of rejected pixels for P_ij estimate

*/


  int nrBadPixels {0};
  double SigmaSquared, Ratio;
  SigmaSquared=pow(RejectSigma,2.0);
  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){

    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j,id_i) ){
        Ratio = std::pow(inputImage(id_j,id_i) - newSpectrum(id_j,id_i), 2.0) 
                / newImageVariance(id_j,id_i);
        if(Ratio >= SigmaSquared){
          imageMask(id_j,id_i) = 0;
          nrBadPixels++;
        }
      }
    }
  }
  return nrBadPixels;
}




void calculateWNormFactor( Eigen::VectorXd &weightNormFactor, 
                           const Eigen::MatrixXd &imageVariance, 
                           const Eigen::MatrixXd &imageMask, 
                           const Eigen::MatrixXd &PMatrix,
                           const Eigen::VectorXd &minAperture, 
                           const Eigen::VectorXd &maxAperture){

/*

Calculate the the deonminator in equation (4)

Sum_i ( P_i**2/V_i )

*/


  double weightSum;
  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){
    weightSum = 1e-9;
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j, id_i) ){
          weightSum += (pow(PMatrix(id_j, id_i), 2.0)/imageVariance(id_j, id_i));
      }
    }
    weightNormFactor(id_j) = weightSum;
  }
}



void calculateWeight( Eigen::MatrixXd &normalizedWeight,
                      const Eigen::VectorXd &weightNormFactor, 
                      const Eigen::MatrixXd &imageVariance, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::MatrixXd &PMatrix,
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture){

/*
Caluclate the weight as defined in equation (4) Marsh (1989).
W_i = (P_i/V_i)/Sum_i ( P_i**2/V_i )
 */

  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j, id_i) ){
          normalizedWeight(id_j, id_i) = ( PMatrix(id_j, id_i) / imageVariance(id_j,id_i) )
                                    / weightNormFactor(id_j);
      }
    }
  }
}


void calculateWeightedFlux( Eigen::VectorXd &weightedFlux,
                               const Eigen::MatrixXd &weight,
                               const Eigen::MatrixXd &inputImage, 
                               const Eigen::MatrixXd &imageMask, 
                               const Eigen::VectorXd &minAperture, 
                               const Eigen::VectorXd &maxAperture){

/*
Weighted sum across the profile, equation (1) in Marsh (1989)
*/

  double FluxSum;
  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){
    FluxSum = 0;
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j, id_i) ){
          FluxSum += ( weight(id_j, id_i) * inputImage(id_j,id_i) );
      }
    }
    weightedFlux(id_j) = FluxSum;
  }
}


void calculateWeightedVar( Eigen::VectorXd &weightedVariance,
                      const Eigen::MatrixXd &weight,
                      const Eigen::MatrixXd &imageVariance, 
                      const Eigen::MatrixXd &imageMask, 
                      const Eigen::VectorXd &minAperture, 
                      const Eigen::VectorXd &maxAperture){

/*
Weighted variance across the profile.
*/


  double varSum;
  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){
    varSum = 0;
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j, id_i) ){
          varSum += ( std::pow(weight(id_j, id_i),2) 
                          * imageVariance(id_j,id_i) );
      }
    }
    weightedVariance(id_j) = varSum;
  }
}


int CosmicRayRejection( Eigen::MatrixXd &imageMask, 
                        const Eigen::MatrixXd &inputImage, 
                        const Eigen::MatrixXd &imageVariance,
                        const Eigen::MatrixXd &PMatrix,
                        const Eigen::VectorXd &weightedFlux, 
                        const Eigen::VectorXd &weightedVariance, 
                        const Eigen::VectorXd &minAperture, 
                        const Eigen::VectorXd &maxAperture,
                        float cosmicSigma){

/*
Estimate cosmic rays, if cosmic ray found mask it and return 1. If there is now CR found
return 0.
*/


  int detection {0}, ii {0}, jj {0}, counter {0};
  double Ratio {0};
  double DummyRatio {0};
  double count {0};
  double fluxcount {0};
  double countsigma {0};
  double fluxcountsigma {0};

  for(size_t id_j {0}; id_j < (size_t)imageMask.rows(); id_j++){
    for(size_t id_i = minAperture(id_j); id_i <= maxAperture(id_j); id_i++){
      if( imageMask(id_j, id_i) ){
          count = inputImage(id_j,id_i);
          countsigma = std::sqrt( imageVariance(id_j,id_i) );
          fluxcount = weightedFlux(id_j) * PMatrix(id_j, id_i);
          fluxcountsigma = std::sqrt(weightedVariance(id_j)) 
                          * PMatrix(id_j, id_i);
          if(count >= fluxcount){
            Ratio = (count - cosmicSigma*countsigma)
                  - (fluxcount + cosmicSigma*fluxcountsigma) ;
            if(Ratio > 0){
              detection = 1;
            }
          }
          else{
            Ratio = (fluxcount - cosmicSigma*fluxcountsigma) 
                    - (count + cosmicSigma*countsigma);
            if(Ratio > 0){
              detection = 1;
            }            
          }
          if (detection == 1){
            if(DummyRatio < Ratio){
                ii = id_i;
                jj = id_j;
                DummyRatio = Ratio;
                counter = 1;
                detection = 0;
            }
          }
      }
      
    }
  }

  if(counter == 0){
    return 0;
  }
  else{
    imageMask(jj,ii) = 0;
    return 1;
  }

}
