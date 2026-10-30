#define _USE_MATH_DEFINES

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void polare(double x, double y, double *r, double *theta){
    *r = hypot(x, y);
    *theta = atan2(y, x) * 180 / M_PI;
}

int main(int argc, char *args[]){
    if (argc < 3) return 1;
    double x = atoi(args[1]);
    double y = atoi(args[2]);

    double r, theta;

    polare(x, y, &r, &theta);

    printf("x: %f | y: %f | r: %f | theta: %f \n", x, y, r, theta);
}