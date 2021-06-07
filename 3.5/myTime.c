//
// Created by kouushou on 2021/6/6.
//
#include <sys/time.h>
#include <stddef.h>
#include "Mytime.h"

struct timeval t1,t2;
void MyTimeStart(){
    gettimeofday(&t1,NULL);
}

double MyTimePassed(){

    gettimeofday(&t2,NULL);
    return (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000000.0;
}