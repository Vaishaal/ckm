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

#include "NativeRoutines.h"

static inline jint imageToVectorCoords(jint x, jint y, jint c, jint yDim, jint xDim) {
  return y + x * yDim + c * yDim * xDim;
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
