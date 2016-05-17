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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdlib.h>


#include "fht_header_only.h"
#include "NativeRoutines.h"

using namespace Eigen;
using namespace std;


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
  jdouble* outVector = (jdouble*) malloc(length*sizeof(double));
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
  posix_memalign((void**) &out, 32, outSize*numPatches*sizeof(double));

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
      double* patch = out + (j*inSize);
      FHTDouble(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= gaussianVector.segment(i, inSize);
    for (int j = 0; j < numPatches; j += 1) {
      double* patch = out + (j*inSize);
      FHTDouble(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= chisquaredVector.segment(i, inSize);
    // outM.block(i, 0, inSize, numPatches).colwise() += uniformVector.segment(i, inSize);
  }

  /* Cosine
  for (int i = 0; i < outSize*numPatches; i++) {
    out[i] = cos(out[i]);
  }
  */

  jdoubleArray result = env->NewDoubleArray(outSize*numPatches);
  env->SetDoubleArrayRegion(result, 0, outSize*numPatches, out);

  free(out);
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
