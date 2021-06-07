//
// Created by kouushou on 2021/6/6.
//

#ifndef REDUCTION_REDUCTIONGPU_H
#define REDUCTION_REDUCTIONGPU_H
double calReductionGPU_Neighbored(const double *a,unsigned siz,double *times);

double calReductionGPU_NeighboredNoDivided(const double *a, unsigned siz, double *time);

double calReductionGPU_NeighboredLessToRight(const double *a, unsigned siz, double *time);

double calReductionGPU_NeighboredReverse(const double *a, unsigned siz, double *time);
#endif //REDUCTION_REDUCTIONGPU_H
