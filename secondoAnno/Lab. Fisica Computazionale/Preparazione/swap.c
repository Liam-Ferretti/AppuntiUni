#include <stdio.h>

void scambia(double *a, double *b){
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

int main(){
    double a = 10.3;
    double b = 3.10;

    printf("%f %f \n", a, b);

    scambia(&a, &b);

    printf("%f %f \n", a, b);
}