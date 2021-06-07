#include <iostream>
#include "Mytime.h"
#include "reductionGPU.h"

using namespace std;
double calReductionCPU_Roll2(const double *a,unsigned level,double *time){
    double res[2] = {0};
    MyTimeStart();
    int nnz = 1<<level;
//#pragma omp parallel for reduction(+=:res)
    for(int i = 0 ; i < nnz ; i+=2){
        res[0] += a[i];
        res[1] += a[i+1];
    }
    *time = MyTimePassed();
    return res[0]+res[1];
}
double calReductionCPU_Roll4(const double *a,unsigned level,double *time){
    double res[4] = {0};
    MyTimeStart();
    int nnz = 1<<level;
//#pragma omp parallel for reduction(+=:res)
    for(int i = 0 ; i < nnz ; i+=4){
        res[0] += a[i];
        res[1] += a[i+1];
        res[2] += a[i+2];
        res[3] += a[i+3];
    }
    *time = MyTimePassed();
    return res[0]+res[1]+res[2]+res[3];
}
double calReductionCPU_Roll8(const double *a,unsigned level,double *time){
    double res[8] = {0};
    MyTimeStart();
    int nnz = 1<<level;
//#pragma omp parallel for reduction(+=:res)
    for(int i = 0 ; i < nnz ; i+=8){
        res[0] += a[i];
        res[1] += a[i+1];
        res[2] += a[i+2];
        res[3] += a[i+3];
        res[4] += a[i+4];
        res[5] += a[i+5];
        res[6] += a[i+6];
        res[7] += a[i+7];
    }
    *time = MyTimePassed();
    return res[0]+res[1]+res[2]+res[3]
    +res[4]+res[5]+res[6]+res[7];
}
double calReductionCPU(const double *a,unsigned level,double *time){
    double res = 0;
    MyTimeStart();
    int nnz = 1<<level;
//#pragma omp parallel for reduction(+=:res)
    for(int i = 0 ; i < nnz ; ++i){
        res += a[i];

    }
    *time = MyTimePassed();
    return res;
}
void tester(const char *name ,
            double( *calReduction)(const double *,unsigned ,double *) ,
            const double *val,unsigned siz,double &golden){
    double times;
    int nnz = 1u<<siz;
    if(golden == 0){
        golden = calReductionCPU(val,siz,&times);
    }
    double cmp = calReduction(val,siz,&times);
    if(cmp!=golden){
        cout<<name<<" Not pass!"<<endl;
        cout<<"Golden: "<<golden<<" "<<name<<": "<<cmp<<endl;
    }else {
        cout << name << " has passed with " << nnz / times/1e9 << "\t gflops in " << times << "\t ms" << endl;
    }

}
int main(int argc,char **argv) {
    unsigned siz = atoi(argv[1]);
    int tot = 1u<<siz;
    auto a =(double *) malloc(sizeof(double )*tot);
    for(int i = 0 ; i < tot ; ++i){
        a[i] = rand()%2*0.125;
    }
    double golden = 0;
    tester("[CPU                       ]",calReductionCPU,a,siz,golden);
    tester("[CPU_Roll2                 ]",calReductionCPU_Roll2,a,siz,golden);
    tester("[CPU_Roll4                 ]",calReductionCPU_Roll4,a,siz,golden);
    tester("[CPU_Roll8                 ]",calReductionCPU_Roll8,a,siz,golden);
    tester("[GPU_Neighbored            ]",calReductionGPU_Neighbored,a,siz,golden);
    tester("[GPU_NeighboredNoDivided   ]",calReductionGPU_NeighboredNoDivided,a,siz,golden);
    tester("[GPU_NeighboredLessToRight ]",calReductionGPU_NeighboredLessToRight,a,siz,golden);
    tester("[GPU_NeighboredReverse     ]",calReductionGPU_NeighboredReverse,a,siz,golden);
    return 0;
}
