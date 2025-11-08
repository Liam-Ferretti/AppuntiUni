#include<stdio.h>
#include<math.h>

int main(){
  float R = 6.2;
  float w = 0.1;
  float t[150] = {};

  for(int i = 0; i < 150; i++){
    t[i] = 0.5 * i;
  }

  for(int i = 0; i < 150; i++){
    printf("%f %f %f\n", t[i], R * cos(w * t[i]), R * sin(w * t[i]));
  }
}
