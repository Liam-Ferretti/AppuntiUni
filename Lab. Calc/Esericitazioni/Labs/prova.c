#include<stdio.h>
#include<math.h>

int main(){
  int n = -1, cifra = 0, rev = 0;
  scanf("%i", &n);
  while (n > 0){
    cifra = n % 10;
    rev = rev * 10 + cifra;
    n /= 10;
  }
  printf("%i \n", rev);
  
}
