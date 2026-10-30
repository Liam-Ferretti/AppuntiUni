#include <stdio.h>
#include <math.h>
#define _USE_MATH_DEFINES

#define N 100000

int main(){
    double sum = 0;
    double real = M_PI * M_PI / 6;

    for(int i = N; i > 0; i--){
        sum += 1.0 / ((double)i*i);
    }
    printf("%15.14f | %15.14f | %15.14f \n",sum, real, sum - real);

    sum = 0;

    for(int i = 1; i <= N; i++){
        sum += 1.0 / ((double)i*i);
    }
    printf("%15.14f | %15.14f | %15.14f \n",sum, real, sum - real);
}