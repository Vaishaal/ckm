 /** @internal
 ** @file     FisherExtractor.cxx
 ** @brief    JNI Wrapper for enceval GMM and Fisher Vector
 **/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdlib.h>
#include <iostream>
#include <armadillo>


#include "fht_header_only.h"
#include "NativeRoutines.h"


using namespace Eigen;
using namespace std;
using namespace arma;


static inline jint imageToVectorCoords(jint x, jint y, jint c, jint yDim, jint xDim) {
  return y + x * yDim + c * yDim * xDim;
}

JNIEXPORT jfloat JNICALL Java_utils_external_NativeRoutines_cosine(
    JNIEnv* env,
    jobject obj,
    jfloat input)
{
  return cos(input);
}

JNIEXPORT jdoubleArray JNICALL Java_utils_external_NativeRoutines_fwht(
    JNIEnv* env,
    jobject obj,
    jdoubleArray input,
    jint length)
{
  jdouble* inputVector = env->GetDoubleArrayElements(input, 0);
  jdouble* outVector;
  posix_memalign((void**) &outVector, 32, length*sizeof(double));
  memcpy(outVector, inputVector, length*sizeof(double));
  /* TODO: Don't hard code this? */
  FHTDouble(outVector, length, 2048);
  jdoubleArray result = env->NewDoubleArray(length);
  env->SetDoubleArrayRegion(result, 0, length, outVector);
  free(outVector);
  return result;
}

JNIEXPORT jdoubleArray JNICALL Java_utils_external_NativeRoutines_fastfood(
    JNIEnv* env,
    jobject obj,
    jdoubleArray gaussian,
    jdoubleArray radamacher,
    jdoubleArray uniform,
    jdoubleArray chiSquared,
    jdoubleArray patchMatrix,
    jint seed,
    jint outSize,
    jint inSize,
    jint numPatches)
{
  double* out;

  int ret = posix_memalign((void**) &out, 32, outSize*numPatches*sizeof(double));
  if (ret != 0) {
    throw std::runtime_error("Ran out of memory!");
  }

  jdouble* patchMatrixV = env->GetDoubleArrayElements(patchMatrix, 0);
  jdouble* radamacherV = env->GetDoubleArrayElements(radamacher, 0);
  jdouble* uniformV= env->GetDoubleArrayElements(uniform, 0);
  jdouble* gaussianV = env->GetDoubleArrayElements(gaussian, 0);
  jdouble* chiSquaredV = env->GetDoubleArrayElements(chiSquared, 0);

  /* (outSize x numPatches) matrix */
  Map< Array<double, Dynamic, Dynamic> > outM(out, outSize, numPatches);
  Map< Array<double, Dynamic, Dynamic> > mf(patchMatrixV, inSize, numPatches);
  Map< Array<double, Dynamic, 1> > radamacherVector(radamacherV, outSize);
  Map< Array<double, Dynamic, 1> > uniformVector(uniformV, outSize);
  Map< Array<double, Dynamic, 1> > gaussianVector(gaussianV, outSize);
  Map< Array<double, Dynamic, 1> > chisquaredVector(chiSquaredV, outSize);
  for (int i = 0; i < outSize; i += inSize) {
    outM.block(i, 0, inSize, numPatches) = mf;
    outM.block(i, 0, inSize, numPatches).colwise() *=  radamacherVector.segment(i, inSize);
    for (int j = 0; j < numPatches; j += 1) {
      double* patch = out + (j*outSize) + i;
      FHTDouble(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= gaussianVector.segment(i, inSize);
    for (int j = 0; j < numPatches; j += 1) {
      double* patch = out + (j*outSize) + i;
      FHTDouble(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= chisquaredVector.segment(i, inSize);
  }

  jdoubleArray result = env->NewDoubleArray(outSize*numPatches);
  env->SetDoubleArrayRegion(result, 0, outSize*numPatches, out);
  free(out);
  env->ReleaseDoubleArrayElements(patchMatrix, patchMatrixV, JNI_ABORT);
  env->ReleaseDoubleArrayElements(radamacher, radamacherV, JNI_ABORT);
  env->ReleaseDoubleArrayElements(uniform, uniformV, JNI_ABORT);
  env->ReleaseDoubleArrayElements(gaussian, gaussianV, JNI_ABORT);
  env->ReleaseDoubleArrayElements(chiSquared, chiSquaredV, JNI_ABORT);
  return result;
}

JNIEXPORT jdoubleArray JNICALL Java_utils_external_NativeRoutines_poolAndRectify (
    JNIEnv* env,
    jobject obj,
    jint stride,
    jint poolSize,
    jint numChannels,
    jint xDim,
    jint yDim,
    jdouble maxVal,
    jdouble alpha,
    jdoubleArray image)
{
  int strideStart = stride / 2;
  int numSourceChannels = numChannels;
  int numOutChannels = numChannels * 2;
  int numPoolsX = ceil((xDim - strideStart)/(stride*1.0));
  int numPoolsY = ceil((yDim - strideStart)/(stride*1.0));
  jdouble* imageVector = env->GetDoubleArrayElements(image, 0);
  int outSize = numPoolsX * numPoolsY * numOutChannels;
  jdouble* patch = (jdouble*) calloc(outSize, sizeof(double));
  for (int x = strideStart; x <= xDim; x += stride) {
    for (int y = strideStart; y<= yDim; y += stride) {
     int startX = x - poolSize/2;
     int endX = fmin(x + poolSize/2, xDim);
     int startY = y - poolSize/2;
     int endY = fmin(y + poolSize/2, yDim);

     int output_offset = (x - strideStart)/stride * numOutChannels +
     (y - strideStart)/stride * numPoolsX * numOutChannels;
     for(int s = startX; s < endX; ++s) {
        for(int b = startY; b < endY; ++b) {
          for (int c = 0; c < numSourceChannels; ++c) {
            int idx = imageToVectorCoords(s, b, c, yDim, xDim);
            jdouble pix = imageVector[idx];
            jdouble pix_pos = fmax(maxVal, pix - alpha);
            jdouble pix_neg = fmax(maxVal, -pix - alpha);
            int pos_position = c + output_offset;
            patch[pos_position] += pix_pos;

            int neg_position = c + numSourceChannels + output_offset;
            patch[neg_position] += pix_neg;
          }
        }
      }
    }
  }
  jdoubleArray result = env->NewDoubleArray(outSize);
  env->SetDoubleArrayRegion(result, 0, outSize, patch);
  free(patch);
  return result;
}


struct DualLeastSquaresEstimator
{
  mat K;
  mat model;
  int numExamples;
  int numClasses;
  int dim;
  DualLeastSquaresEstimator(int n, int d, int k, double lambda);
  ~DualLeastSquaresEstimator(void);
  void AccumulateGram(mat& X);
  void solve(mat& y);
};

typedef struct DualLeastSquaresEstimator DualLeastSquaresEstimator;

/* n is number of data points, k is number of classes */

DualLeastSquaresEstimator::DualLeastSquaresEstimator(int n, int d, int k, double reg)
    : K(n,n,fill::eye), model(n,k,fill::zeros)
{

  /* Initialize empty gram matrix */
  K *= reg;

  numExamples = n;
  dim = d;
  numClasses = k;
}

void DualLeastSquaresEstimator::AccumulateGram(mat &X) {
  printf("XXT COMPUTATION \n ");
  fflush(stdout);
  K += X * X.t();
  printf("XXT COMPUTATION END \n ");
}

void DualLeastSquaresEstimator::solve(mat &y) {
  printf("SOLVING in JNI\n");

  model = arma::solve(K, y,  arma::solve_opts::fast);

  fflush(stdout);
}

JNIEXPORT jlong JNICALL Java_utils_external_NativeRoutines_NewDualLeastSquaresEstimator(
    JNIEnv* env,
    jobject obj,
    jint n,
    jint k,
    jint d,
    jdouble lambda)
{
  printf("Creating empty gram matrix \n");
  DualLeastSquaresEstimator *est =  new DualLeastSquaresEstimator(n, d, k, lambda);
  printf("EST pointer: %p\n", (void*) est);
  fflush(stdout);
  return (jlong) est;
}

JNIEXPORT void JNICALL Java_utils_external_NativeRoutines_DeleteDualLeastSquaresEstimator(
    JNIEnv* env,
    jobject obj,
    uint64_t ptr)
{
  DualLeastSquaresEstimator *est = (DualLeastSquaresEstimator *) ptr;
  delete est;
}

JNIEXPORT void JNICALL Java_utils_external_NativeRoutines_DualLeastSquaresEstimatorAccumulateGram(
    JNIEnv* env,
    jobject obj,
    jlong ptr,
    jdoubleArray data)
{
  printf("ACCUMULATING GRAM in JNI \n");
  fflush(stdout);
  DualLeastSquaresEstimator *est = (DualLeastSquaresEstimator *) ptr;
  printf("EST pointer: %p\n", (void*) est);
  jdouble* Xraw = env->GetDoubleArrayElements(data, 0);
  mat X = mat(Xraw, est->numExamples, est->dim, false);
  est->AccumulateGram(X);
  printf("X FIRST VALUE %f\n", X(0,0));
  printf("JNI K FIRST VALUE %f\n", est->K(0, 0));
  printf("JNI K RANDOM VALUE %f\n", est->K(273, 137));
  fflush(stdout);
}

JNIEXPORT jdoubleArray JNICALL Java_utils_external_NativeRoutines_DualLeastSquaresEstimatorSolve(
    JNIEnv* env,
    jobject obj,
    jlong ptr,
    jdoubleArray labels)
{
  DualLeastSquaresEstimator *est = (DualLeastSquaresEstimator *) ptr;
  jdouble* Yraw = env->GetDoubleArrayElements(labels, 0);

  mat Y = mat(Yraw, est->numExamples, est->numClasses, false);
  printf("RIGHT BEFORE SOLVE CALL TO EIGEN\n");
  fflush(stdout);
  est->solve(Y);
  int modelSize = est->numClasses*est->numExamples;
  jdoubleArray model = env->NewDoubleArray(modelSize);
  env->SetDoubleArrayRegion(model, 0, modelSize, est->model.memptr());
  return model;
}

